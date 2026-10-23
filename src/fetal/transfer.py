"""Verified downloads with bounded retries and persistent HTTP range recovery."""
from __future__ import annotations

import hashlib
import http.client
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path


class DownloadError(RuntimeError):
    pass


def _matches(path: Path, checksum: str, algorithm: str) -> bool:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest().startswith(checksum)


def download_file(
    url: str,
    destination: Path,
    *,
    checksum: str,
    algorithm: str = "sha256",
    attempts: int = 5,
    timeout: float = 60,
    retry_delay: float = 2,
) -> Path:
    """Keep partial bytes across attempts/invocations; publish only verified files."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".part")
    if destination.exists():
        if _matches(destination, checksum, algorithm):
            return destination
        print(f"Replacing corrupt cached file: {destination}", flush=True)
        destination.unlink()
    if partial.exists() and _matches(partial, checksum, algorithm):
        partial.replace(destination)
        return destination

    for attempt in range(1, attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "neuroml-pipeline", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        print(f"Downloading {destination.name}: attempt {attempt}/{attempts}, offset {offset}", flush=True)
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", response.headers.get("Content-Range", ""))
                    if match is None or int(match[1]) != offset or int(match[2]) < offset:
                        partial.unlink(missing_ok=True)
                        raise DownloadError("Server returned an invalid resume range; restarting")
                    expected_size = int(match[3]) if match[3] != "*" else int(match[2]) + 1
                    mode = "ab"
                elif response.status == 200:
                    # Servers may ignore Range. Replace the partial file instead of appending.
                    length = response.headers.get("Content-Length")
                    expected_size = int(length) if length is not None else None
                    mode = "wb"
                else:
                    raise DownloadError(f"Unexpected HTTP status: {response.status}")
                with partial.open(mode) as stream:
                    while chunk := response.read(64 * 1024):
                        stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise DownloadError(f"Incomplete transfer: {partial.stat().st_size}/{expected_size} bytes")
            if not _matches(partial, checksum, algorithm):
                partial.unlink()
                raise DownloadError("Downloaded checksum mismatch; restarting")
            partial.replace(destination)
            return destination
        except urllib.error.HTTPError as error:
            if error.code == 416:
                # A complete partial file needs no network data; a stale/oversized one must restart.
                if partial.exists() and _matches(partial, checksum, algorithm):
                    partial.replace(destination)
                    error.close()
                    return destination
                partial.unlink(missing_ok=True)
            elif error.code not in (408, 429) and not 500 <= error.code < 600:
                error.close()
                raise
            failure = error
            retry_after = error.headers.get("Retry-After", "")
            error.close()
        except (urllib.error.URLError, TimeoutError, ConnectionError, ssl.SSLError, http.client.HTTPException, DownloadError) as error:
            failure = error
            retry_after = ""
        if attempt == attempts:
            raise DownloadError(f"Download failed after {attempts} attempts: {url}; partial file (if any): {partial}") from failure
        delay = min(retry_delay * 2 ** (attempt - 1), 30)
        if retry_after.isdigit():
            delay = min(max(delay, int(retry_after)), 30)
        print(f"{failure}; retrying in {delay:g}s", flush=True)
        time.sleep(delay)
    raise ValueError("Download attempts must be positive")
