"""Exercise the shared scheduler with synthetic jobs, not remote machines."""
import threading
import time
from types import SimpleNamespace

import pytest

from fetal.config import AppConfig
from fetal.distributed import choose_job, execute_remote, finalist_jobs, train_distributed, validate_resume_identity


def config():
    return AppConfig(_env_file=None, architectures=["resnet50", "densenet121", "convnext_tiny", "swin_t"],
                     training={"epochs": 1, "batch_size": 8}, cluster={"workers": [
                         {"host": "pc", "speed": "fast"}, {"host": "laptop", "speed": "small"}],
                         "profiling_host": "pc"})


def test_heavier_jobs_go_to_fast_worker_and_failed_worker_is_avoided():
    settings = config()
    jobs = [{"id": arch, "arch": arch, "stage": "screen"} for arch in settings.architectures]
    pc, laptop = settings.cluster.workers
    assert choose_job(pc, jobs, {})["arch"] == "swin_t"
    assert choose_job(laptop, jobs, {})["arch"] == "resnet50"
    assert choose_job(pc, jobs, {"swin_t": {"pc"}})["arch"] == "convnext_tiny"
    assert choose_job(pc, jobs, {}, "densenet121")["arch"] == "densenet121"
    assert choose_job(pc, jobs, {}, costs={"resnet50": 100})["arch"] == "resnet50"


def test_finalists_require_all_screen_results_and_use_validation_only():
    settings = config()
    trained = {arch: {"arch": arch, "status": "trained", "contract": {"validation_macro_f1": score}}
               for arch, score in zip(settings.architectures, [0.7, 0.6, 0.9, 0.8])}
    # IDs in a real plan encode the recipe; this test only exercises the selection rule.
    assert finalist_jobs(settings, trained, set(trained) | {"missing"}) == ([], [])
    finalists, _ = finalist_jobs(settings, trained, set(trained))
    assert finalists == ["convnext_tiny", "swin_t"]


def test_scheduler_runs_eleven_unique_jobs_and_never_overlaps_a_workers_jobs(tmp_path, monkeypatch):
    settings = config()
    lock = threading.Lock()
    active = set()
    calls = []
    scores = {"resnet50": 0.7, "densenet121": 0.6, "convnext_tiny": 0.9, "swin_t": 0.8}

    def execute(worker, job, action, *args):
        assert action == "train"
        with lock:
            assert worker.host not in active
            active.add(worker.host)
            calls.append(job["id"])
        time.sleep(0.001)
        with lock:
            active.remove(worker.host)
        return {**job, "status": "trained", "worker": worker.host, "training_seconds": 1,
                "contract": {"validation_macro_f1": scores[job["arch"]]}}

    monkeypatch.setattr("fetal.distributed.execute_remote", execute)
    jobs, trained, finalists = train_distributed(settings, tmp_path, {})
    assert len(jobs) == len(calls) == len(set(calls)) == 11
    assert all(result["status"] == "trained" for result in trained.values())
    assert finalists == ["convnext_tiny", "swin_t"]


def test_failed_model_retries_on_other_worker_and_remaining_jobs_continue(tmp_path, monkeypatch):
    settings = config()
    calls = []

    def execute(worker, job, action, *args):
        calls.append((worker.host, job["id"]))
        if job["arch"] == "swin_t" and worker.host == "pc":
            return {**job, "status": "failed", "worker": worker.host, "error": "synthetic failure"}
        return {**job, "status": "trained", "worker": worker.host, "training_seconds": 1,
                "contract": {"validation_macro_f1": 0.8}}

    monkeypatch.setattr("fetal.distributed.execute_remote", execute)
    jobs, trained, _ = train_distributed(settings, tmp_path, {})
    assert len(jobs) == 11
    key = "swin_t-split_grouped-imagenet-seed0"
    assert ("pc", key) in calls and ("laptop", key) in calls
    assert trained[key]["status"] == "trained"


@pytest.mark.parametrize("message", ["error: no such object: task", "Error: No such container: task"])
def test_missing_container_starts_a_task_with_either_docker_error_case(tmp_path, monkeypatch, message):
    monkeypatch.setattr("fetal.distributed.subprocess.run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr=message))
    monkeypatch.setattr("fetal.distributed.upload", lambda *args, **kwargs: None)
    monkeypatch.setattr("fetal.distributed.container_args", lambda *args, **kwargs: ["docker", "run"])

    def remote(worker, command):
        if command[:2] == ["docker", "run"]:
            raise RuntimeError("Task launch reached")
        return ""

    monkeypatch.setattr("fetal.distributed.remote", remote)
    with pytest.raises(RuntimeError, match="Task launch reached"):
        execute_remote(config().cluster.workers[1], {"id": "test-job"}, "train", tmp_path, {}, "image")


def test_resume_allows_controller_fix_but_rejects_scientific_or_recipe_changes():
    saved = {"config": {"epochs": 6}, "files": {"src/fetal/distributed.py": "old", "src/fetal/experiment.py": "science"}}
    current = {"config": {"epochs": 6}, "files": {"src/fetal/distributed.py": "new", "src/fetal/experiment.py": "science"}}
    validate_resume_identity(saved, current)
    with pytest.raises(ValueError, match="Cannot resume"):
        validate_resume_identity(saved, {**current, "config": {"epochs": 7}})
    with pytest.raises(ValueError, match="Cannot resume"):
        validate_resume_identity(saved, {**current, "files": {**current["files"], "src/fetal/experiment.py": "changed"}})
