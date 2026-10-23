"""Real HTTP transfers against a local server; no external downloads."""
import hashlib
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from fetal.transfer import DownloadError, download_file

PAYLOAD = bytes(range(256)) * 1024
CHECKSUM = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def server():
    state = {"requests": [], "failures": [], "ignore_range": False, "disconnect": False, "bad_range": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            header = self.headers.get("Range")
            state["requests"].append(header)
            if state["failures"]:
                self.send_error(state["failures"].pop(0))
                return
            start = int(header.removeprefix("bytes=").removesuffix("-")) if header else 0
            if start >= len(PAYLOAD):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(PAYLOAD)}")
                self.end_headers()
                return
            if state["ignore_range"]:
                start = 0
            resumed = header is not None and not state["ignore_range"]
            self.send_response(206 if resumed else 200)
            body = PAYLOAD[start:]
            if resumed:
                returned_start = 0 if state["bad_range"] else start
                state["bad_range"] = False
                self.send_header("Content-Range", f"bytes {returned_start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if state["disconnect"]:
                state["disconnect"] = False
                self.wfile.write(body[:70000])
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            self.wfile.write(body)

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}/file", state
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def test_dropped_connection_retries_from_saved_offset(server, tmp_path):
    url, state = server
    state["disconnect"] = True
    path = tmp_path / "dataset.zip"
    assert download_file(url, path, checksum=CHECKSUM, retry_delay=0) == path
    assert path.read_bytes() == PAYLOAD
    assert state["requests"] == [None, "bytes=70000-"]
    assert not path.with_suffix(".part").exists()


def test_failed_invocation_keeps_partial_and_next_invocation_resumes(server, tmp_path):
    url, state = server
    state["disconnect"] = True
    path = tmp_path / "weights.pth"
    with pytest.raises(DownloadError, match="after 1 attempts"):
        download_file(url, path, checksum=CHECKSUM, attempts=1)
    partial = path.with_suffix(".part")
    assert partial.read_bytes() == PAYLOAD[:70000]
    assert not path.exists()
    download_file(url, path, checksum=CHECKSUM, retry_delay=0)
    assert state["requests"][-1] == "bytes=70000-"
    assert path.read_bytes() == PAYLOAD


@pytest.mark.parametrize("algorithm", ["md5", "sha256"])
def test_server_ignoring_range_restarts_without_appending(server, tmp_path, algorithm):
    url, state = server
    state["ignore_range"] = True
    path = tmp_path / "archive.zip"
    path.with_suffix(".part").write_bytes(PAYLOAD[:12345])
    checksum = hashlib.new(algorithm, PAYLOAD).hexdigest()
    download_file(url, path, checksum=checksum, algorithm=algorithm)
    assert state["requests"] == ["bytes=12345-"]
    assert path.read_bytes() == PAYLOAD


def test_verified_complete_cache_and_partial_need_no_network(server, tmp_path):
    url, state = server
    path = tmp_path / "weights.pth"
    path.write_bytes(PAYLOAD)
    download_file(url, path, checksum=CHECKSUM[:8])
    path.rename(path.with_suffix(".part"))
    download_file(url, path, checksum=CHECKSUM[:8])
    assert path.read_bytes() == PAYLOAD
    assert not state["requests"]


def test_corrupt_complete_cache_is_replaced(server, tmp_path):
    url, state = server
    path = tmp_path / "weights.pth"
    path.write_bytes(b"corrupt cached bytes")
    download_file(url, path, checksum=CHECKSUM, retry_delay=0)
    assert path.read_bytes() == PAYLOAD
    assert state["requests"] == [None]


def test_oversized_partial_restarts_after_http_416(server, tmp_path):
    url, state = server
    path = tmp_path / "archive.zip"
    path.with_suffix(".part").write_bytes(PAYLOAD + b"stale")
    download_file(url, path, checksum=CHECKSUM, retry_delay=0)
    assert state["requests"] == [f"bytes={len(PAYLOAD) + 5}-", None]
    assert path.read_bytes() == PAYLOAD


def test_wrong_resume_range_is_discarded(server, tmp_path):
    url, state = server
    state["bad_range"] = True
    path = tmp_path / "archive.zip"
    path.with_suffix(".part").write_bytes(PAYLOAD[:100])
    download_file(url, path, checksum=CHECKSUM, retry_delay=0)
    assert state["requests"] == ["bytes=100-", None]
    assert path.read_bytes() == PAYLOAD


def test_transient_http_errors_retry_with_backoff(server, tmp_path, monkeypatch):
    url, state = server
    state["failures"] = [503, 429]
    delays = []
    monkeypatch.setattr("fetal.transfer.time.sleep", delays.append)
    download_file(url, tmp_path / "weights.pth", checksum=CHECKSUM)
    assert delays == [2, 4]
    assert len(state["requests"]) == 3


def test_permanent_http_error_does_not_retry(server, tmp_path):
    from urllib.error import HTTPError

    url, state = server
    state["failures"] = [404]
    with pytest.raises(HTTPError) as caught:
        download_file(url, tmp_path / "weights.pth", checksum=CHECKSUM, retry_delay=0)
    assert caught.value.code == 404
    assert len(state["requests"]) == 1


def test_checksum_mismatch_has_bounded_retries_and_never_publishes(server, tmp_path):
    url, state = server
    path = tmp_path / "weights.pth"
    with pytest.raises(DownloadError, match="after 2 attempts"):
        download_file(url, path, checksum="0" * 64, attempts=2, retry_delay=0)
    assert state["requests"] == [None, None]
    assert not path.exists()
    assert not path.with_suffix(".part").exists()


def test_weight_download_uses_torchvision_cache_and_published_hash(tmp_path, monkeypatch):
    from fetal import models

    calls = []
    monkeypatch.setattr(models.torch.hub, "get_dir", lambda: str(tmp_path))

    def download(url, destination, **options):
        calls.append((url, destination, options))
        return destination

    monkeypatch.setattr(models, "download_file", download)
    path = models.download_weights("resnet50")
    assert path == tmp_path / "checkpoints/resnet50-11ad3fa6.pth"
    assert calls == [(models.WEIGHTS["resnet50"].url, path, {"checksum": "11ad3fa6"})]


def test_stale_partial_bytes_trigger_clean_verified_download(server, tmp_path):
    url, state = server
    path = tmp_path / "archive.zip"
    path.with_suffix(".part").write_bytes(b"x" * 100)
    download_file(url, path, checksum=CHECKSUM, retry_delay=0)
    assert state["requests"] == ["bytes=100-", None]
    assert path.read_bytes() == PAYLOAD


def test_weight_download_failure_does_not_stop_other_architectures(tmp_path, monkeypatch):
    from fetal import pipeline

    checkpoint = tmp_path / "synthetic.pth"
    checkpoint.write_bytes(b"synthetic verified checkpoint")

    def download(arch):
        if arch == "resnet50":
            raise DownloadError("simulated exhausted retries")
        return checkpoint

    monkeypatch.setattr(pipeline, "download_weights", download)
    results = pipeline.cache_weights(["resnet50", "densenet121"], tmp_path)
    assert results["resnet50"]["status"] == "failed"
    assert results["densenet121"]["status"] == "completed"
