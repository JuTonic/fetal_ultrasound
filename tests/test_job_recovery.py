"""Exercise real child-process failure, retry and result reuse without models."""
import json
import sys

from fetal.files import sha256
from fetal.jobs import execute_job


WORKER = '''
import hashlib, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
request = json.loads(path.read_text())
out = path.parent
counter = out / "attempts.txt"
counter.write_text(str(int(counter.read_text()) + 1) if counter.exists() else "1")
if request["job"].get("fail"):
    sys.exit(9)
checkpoint = out / "model.pt"
checkpoint.write_bytes(b"test checkpoint")
result = {"status": "trained", "checkpoint": str(checkpoint),
          "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
(out / "train.json").write_text(json.dumps(result))
'''


def test_failure_does_not_stop_next_job_and_completed_job_is_skipped(tmp_path):
    command = [sys.executable, "-c", WORKER]
    failed = execute_job({"id": "first", "fail": True}, "train", tmp_path, {}, "cpu", worker_command=command)
    job = {"id": "second", "config": {"epochs": 2}}
    successful = execute_job(job, "train", tmp_path, {}, "cpu", worker_command=command)
    reused = execute_job(job, "train", tmp_path, {}, "cpu", worker_command=command)
    assert failed["status"] == "failed"
    assert successful["status"] == "trained"
    assert reused["reused"]
    assert (tmp_path / "second/attempts.txt").read_text() == "1"
    assert set(json.loads((tmp_path / "progress.json").read_text())) == {"first/train", "second/train"}
    retry = execute_job({"id": "first", "fail": False}, "train", tmp_path, {}, "cpu", worker_command=command)
    assert retry["status"] == "trained"
    assert (tmp_path / "first/attempts.txt").read_text() == "2"


def test_configuration_change_or_damaged_checkpoint_cannot_reuse_success(tmp_path):
    command = [sys.executable, "-c", WORKER]
    job = {"id": "model", "config": {"epochs": 2}}
    execute_job(job, "train", tmp_path, {}, "cpu", worker_command=command)
    changed = {"id": "model", "config": {"epochs": 3}}
    assert not execute_job(changed, "train", tmp_path, {}, "cpu", worker_command=command)["reused"]
    (tmp_path / "model/model.pt").write_bytes(b"damaged")
    assert not execute_job(changed, "train", tmp_path, {}, "cpu", worker_command=command)["reused"]
    assert (tmp_path / "model/attempts.txt").read_text() == "3"


def test_evaluation_reuse_requires_same_checkpoint_and_successful_profiling(tmp_path):
    out = tmp_path / "model"
    out.mkdir()
    checkpoint = out / "model.pt"
    checkpoint.write_bytes(b"weights")
    job = {"id": "model", "checkpoint": str(checkpoint)}
    request = {"job": job, "action": "evaluate", "dataset": {}, "device": "cpu",
               "checkpoint_sha256": sha256(checkpoint)}
    result = {"status": "completed", "cost": {"status": "completed"}, "cache_request": request}
    (out / "evaluate.json").write_text(json.dumps(result))
    command = [sys.executable, "-c", "raise SystemExit(12)"]
    assert execute_job(job, "evaluate", tmp_path, {}, "cpu", worker_command=command)["reused"]
    result["cost"]["status"] = "failed"
    (out / "evaluate.json").write_text(json.dumps(result))
    assert not execute_job(job, "evaluate", tmp_path, {}, "cpu", worker_command=command)["reused"]
    result["cost"]["status"] = "completed"
    (out / "evaluate.json").write_text(json.dumps(result))
    checkpoint.write_bytes(b"new weights")
    assert not execute_job(job, "evaluate", tmp_path, {}, "cpu", worker_command=command)["reused"]


def test_malformed_worker_output_is_a_failed_job(tmp_path):
    command = [sys.executable, "-c", "import pathlib,sys; pathlib.Path(sys.argv[1]).with_name('train.json').write_text('{broken')"]
    result = execute_job({"id": "broken"}, "train", tmp_path, {}, "cpu", worker_command=command)
    assert result["status"] == "failed"
    assert "JSONDecodeError" in result["error"]
