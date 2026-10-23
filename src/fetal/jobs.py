"""Run isolated model jobs and reuse only matching, intact completed results."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fetal.files import sha256, write_json


def execute_job(
    job: dict,
    action: str,
    root: Path,
    dataset: dict,
    device: str,
    *,
    worker_command: list[str] | None = None,
) -> dict:
    out = root / job["id"]
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / f"{action}.json"
    log = out / f"{action}.log"
    request = {"job": job, "action": action, "dataset": dataset, "device": device}
    try:
        if action == "evaluate":
            checkpoint = Path(job.get("checkpoint") or f"checkpoints/{job['id']}.pt")
            request["checkpoint_sha256"] = sha256(checkpoint) if checkpoint.is_file() else None
        if result_path.exists():
            try:
                cached = json.loads(result_path.read_text())
            except (ValueError, OSError):
                cached = {}
            usable = cached.get("cache_request") == request
            if action == "train":
                checkpoint = Path(cached.get("checkpoint", out / "model.pt"))
                usable = usable and cached.get("status") == "trained" and checkpoint.is_file()
                if usable:
                    usable = sha256(checkpoint) == cached.get("checkpoint_sha256")
            else:
                usable = usable and cached.get("status") == "completed" and cached.get("cost", {}).get("status") == "completed"
            if usable:
                print(f"[{job['id']}] reuse completed {action}", flush=True)
                return {**job, **cached, "reused": True, "log": str(log.relative_to(root)), "directory": job["id"]}
        request_path = out / f"{action}.request.json"
        write_json(request_path, request)
        # Replace stale success before starting: a killed process must never reuse it.
        write_json(result_path, {"status": "failed", "error": "Model process interrupted"})
        command = worker_command or [sys.executable, "-m", "fetal.pipeline", "--job"]
        with log.open("a") as stream:
            stream.write(f"\nStarting {action}\n")
            stream.flush()
            process = subprocess.run([*command, str(request_path)], stdout=stream, stderr=subprocess.STDOUT)
        result = json.loads(result_path.read_text())
        if process.returncode:
            if result.get("error") == "Model process interrupted" or result["status"] != "failed":
                result = {"status": "failed", "error": f"Model process exited {process.returncode}; see log"}
        result["exit_code"] = process.returncode
        result["cache_request"] = request
    except Exception as error:
        # This is the model-job boundary: record the failure and allow the next job.
        result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    write_json(result_path, result)
    progress_path = root / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    progress[f"{job['id']}/{action}"] = {"status": result["status"], "error": result.get("error"), "log": str(log.relative_to(root))}
    write_json(progress_path, progress)
    return {**job, **result, "reused": False, "log": str(log.relative_to(root)), "directory": job["id"]}
