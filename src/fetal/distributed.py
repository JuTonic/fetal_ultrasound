"""Schedule independent Docker jobs over SSH and collect a portable final report."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path

from fetal.config import AppConfig, load_config
from fetal.config.models import WorkerConfig
from fetal.data import CLASSES9
from fetal.files import sha256, write_json
from fetal.package import overall_status, package_report
from fetal.pipeline import make_job

SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]
COST = {"resnet50": 1, "densenet121": 2, "convnext_tiny": 3, "swin_t": 4}


def remote(worker: WorkerConfig, command: list[str], *, timeout: float | None = 90) -> str:
    result = subprocess.run([*SSH, worker.host, shlex.join(command)], check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    return result.stdout.strip()


def remote_shell(worker: WorkerConfig, script: str) -> None:
    subprocess.run([*SSH, worker.host, script], check=True)


def upload(source: Path, worker: WorkerConfig, target: str, *, checksum: bool = False) -> None:
    options = ["--checksum"] if checksum else []
    subprocess.run(["rsync", "-az", "--partial", *options, "-e", shlex.join(SSH),
                    str(source), f"{worker.host}:{target}"], check=True)


def download(worker: WorkerConfig, source: str, target: Path, *, recovery: bool = False, checksum: bool = False) -> None:
    target.mkdir(parents=True, exist_ok=True)
    command = ["rsync", "-az", "--partial", "-e", shlex.join(SSH)]
    if checksum:
        command.append("--checksum")
    if not recovery:
        command.extend(["--exclude=resume.pt", "--exclude=*.tmp"])
    subprocess.run([*command, f"{worker.host}:{source}", str(target) + "/"], check=True)


def container_args(worker: WorkerConfig, image: str, *, gpu: bool = False) -> list[str]:
    directory = worker.directory
    uid = remote(worker, ["id", "-u"])
    gid = remote(worker, ["id", "-g"])
    command = ["docker", "run", "--user", f"{uid}:{gid}", "--shm-size", "2g", "-e", "HOME=/tmp",
               "-v", f"{directory}/data/raw:/app/data/raw:ro",
               "-v", f"{directory}/data/downloads:/downloads",
               "-v", f"{directory}/.cache:/cache", "-v", f"{directory}/results:/app/results",
               "-v", f"{directory}/checkpoints:/app/checkpoints"]
    if gpu:
        command += ["--gpus", "all"]
    return command


def prepare(config: AppConfig, builder: WorkerConfig) -> dict:
    workers = config.cluster.workers
    image = config.cluster.image
    sources = ["Dockerfile", "pyproject.toml", "uv.lock", ".python-version", ".dockerignore", "src/", "configs/", "data/processed/"]
    for worker in workers:
        if worker.storage_mount is not None:
            remote(worker, ["mountpoint", "-q", worker.storage_mount])
        remote(worker, ["mkdir", "-p", *[f"{worker.directory}/{path}" for path in
                ("data/raw", "data/downloads", ".cache", "results", "checkpoints")]])
        subprocess.run(["rsync", "-azR", "--exclude=__pycache__", "-e", shlex.join(SSH),
                        *sources, f"{worker.host}:{worker.directory}/"], check=True)
    print(f"Building one image on {builder.host}", flush=True)
    remote_shell(builder, f"cd {shlex.quote(builder.directory)} && docker build -t {shlex.quote(image)} .")
    image_id = remote(builder, ["docker", "image", "inspect", "--format", "{{.Id}}", image])
    # Content-addressed OCI blobs let rsync reuse CUDA layers after small code changes.
    image_directory = f"{builder.directory}/.cache/image-oci"
    remote(builder, ["mkdir", "-p", image_directory])
    remote_shell(builder, f"set -o pipefail; docker save {shlex.quote(image)} | tar -xf - -C {shlex.quote(image_directory)}")
    cache = Path(".cache/distributed")
    cache.mkdir(parents=True, exist_ok=True)
    download(builder, image_directory + "/", cache / "image-oci", recovery=True, checksum=True)
    for worker in workers:
        if worker.host == builder.host:
            continue
        upload(cache / "image-oci", worker, f"{worker.directory}/.cache/", checksum=True)
        remote_shell(worker, f"set -o pipefail; tar -cf - -C {shlex.quote(worker.directory + '/.cache/image-oci')} . | docker load")
    # Verify the actual CUDA operation, not just nvidia-smi, before any training.
    for worker in workers:
        probe = "import torch; assert torch.cuda.is_available(); x=torch.ones(16,device='cuda'); print(torch.cuda.get_device_name(0),float((x*x).sum()))"
        print(worker.host, remote(worker, [*container_args(worker, image, gpu=True), "--rm", "--entrypoint", "/app/.venv/bin/python", image, "-c", probe], timeout=180), flush=True)
    setup = [*container_args(builder, image), "--rm"]
    setup = [part.replace("/app/data/raw:ro", "/app/data/raw") for part in setup]
    remote_shell(builder, shlex.join([*setup, image, "--setup"]))
    prefetch = "from fetal.models import download_weights; " + "; ".join(f"download_weights({arch!r})" for arch in config.architectures)
    remote_shell(builder, shlex.join([*container_args(builder, image), "--rm", "--entrypoint", "/app/.venv/bin/python", image, "-c", prefetch]))
    download(builder, f"{builder.directory}/data/downloads/FETAL_PLANES_ZENODO.zip", cache)
    download(builder, f"{builder.directory}/.cache/torch/", cache / "torch", recovery=True)
    for worker in workers:
        if worker.host == builder.host:
            continue
        upload(cache / "FETAL_PLANES_ZENODO.zip", worker, f"{worker.directory}/data/downloads/")
        upload(cache / "torch" / "", worker, f"{worker.directory}/.cache/")
        setup = [part.replace("/app/data/raw:ro", "/app/data/raw") for part in container_args(worker, image)]
        remote_shell(worker, shlex.join([*setup, "--rm", image, "--setup"]))
    dataset = None
    environments = {}
    for worker in workers:
        describe = "import json; from fetal.download import validate_data; print(json.dumps(validate_data()))"
        actual = json.loads(remote(worker, [*container_args(worker, image), "--rm", "--entrypoint", "/app/.venv/bin/python", image, "-c", describe], timeout=300))
        if dataset is not None and dataset != actual:
            raise ValueError(f"Dataset mismatch on {worker.host}")
        dataset = actual
        actual_image = remote(worker, ["docker", "image", "inspect", "--format", "{{.Id}}", image])
        if actual_image != image_id:
            raise ValueError(f"Image mismatch on {worker.host}")
        environments[worker.host] = {"image": actual_image, "directory": worker.directory}
    return {"dataset": dataset, "workers": environments}


def choose_job(worker: WorkerConfig, pending: list[dict], attempted: dict[str, set[str]], preferred: str | None = None, costs: dict[str, float] | None = None) -> dict | None:
    eligible = [job for job in pending if worker.host not in attempted.get(job["id"], set())]
    if not eligible:
        return None
    for job in eligible:
        if job["id"] == preferred:
            return job
    costs = COST if costs is None else costs
    screen = [job for job in eligible if job.get("stage") == "screen"]
    choices = screen or eligible
    return sorted(choices, key=lambda job: (costs.get(job["arch"], 1), job["id"]), reverse=worker.speed == "fast")[0]


def finalist_jobs(config: AppConfig, trained: dict[str, dict], screen_ids: set[str]) -> tuple[list[str], list[dict]]:
    if not all(trained.get(key, {}).get("status") == "trained" for key in screen_ids):
        return [], []
    ranked = sorted((trained[key] for key in screen_ids), key=lambda result: (-result["contract"]["validation_macro_f1"], result["arch"]))
    finalists = [result["arch"] for result in ranked[:2]]
    jobs = []
    for arch in finalists:
        for pretrained in (True, False):
            for seed in config.finalist_seeds:
                job = make_job(arch, "split_grouped", seed, pretrained, config.training, False)
                if job["id"] not in screen_ids:
                    jobs.append({**job, "stage": "transfer"})
    return finalists, jobs


def execute_remote(worker: WorkerConfig, job: dict, action: str, root: Path, dataset: dict, image: str) -> dict:
    out = root / job["id"]
    out.mkdir(parents=True, exist_ok=True)
    task_path = out / f"task-{action}.json"
    write_json(task_path, {"job": job, "action": action, "dataset": dataset, "device": "cuda"})
    remote_out = f"{worker.directory}/{out}"
    remote(worker, ["mkdir", "-p", remote_out])
    name = f"neuroml-{root.name[:10]}-{action}-{job['id']}"
    inspect = subprocess.run([*SSH, worker.host, shlex.join(["docker", "inspect", "--format", "{{json .State}}", name])], capture_output=True, text=True, timeout=90)
    running = inspect.returncode == 0 and json.loads(inspect.stdout)["Running"]
    if inspect.returncode and "no such" not in inspect.stderr.lower():
        raise ConnectionError(f"Could not inspect worker {worker.host}: {inspect.stderr}")
    finished = False
    if inspect.returncode == 0 and not running:
        result_path = remote_out + f"/{action}.result.json"
        result_check = subprocess.run([*SSH, worker.host, shlex.join(["test", "-f", result_path])], timeout=90)
        if result_check.returncode == 0:
            finished = json.loads(remote(worker, ["cat", result_path])).get("status") in ("trained", "completed")
    if not running and not finished:
        # Never overwrite recovery state while a detached task is still running.
        recovery = out / "resume.pt"
        if action == "train" and inspect.returncode == 0:
            remote_recovery = remote_out + "/resume.pt"
            recovery_check = subprocess.run([*SSH, worker.host, shlex.join(["test", "-f", remote_recovery])], timeout=90)
            if recovery_check.returncode == 0:
                download(worker, remote_recovery, out, recovery=True)
        if action == "train" and not recovery.exists() and job.get("resume_from"):
            recovery = Path(job["resume_from"])
        if action == "train" and recovery.exists():
            upload(recovery, worker, remote_out + "/resume.pt")
        upload(task_path, worker, remote_out + "/" + task_path.name)
        if inspect.returncode == 0:
            remote(worker, ["docker", "rm", name])
        command = [*container_args(worker, image, gpu=True), "-d", "--name", name,
                   "--label", f"neuroml.plan={root.name}", "--label", f"neuroml.worker={worker.host}",
                   "--entrypoint", "/app/.venv/bin/python", image, "-m", "fetal.task", str(task_path)]
        remote(worker, command)
    while True:
        state = json.loads(remote(worker, ["docker", "inspect", "--format", "{{json .State}}", name]))
        if not state["Running"]:
            break
        time.sleep(5)
    try:
        download(worker, remote_out + "/", out)
        result = json.loads((out / f"{action}.result.json").read_text())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as error:
        logs = remote(worker, ["docker", "logs", "--tail", "100", name])
        (out / f"{action}-container.log").write_text(logs)
        result = {**job, "status": "failed", "error": f"Task exited {state['ExitCode']}: {error}"}
    result["worker"] = worker.host
    result["container_image"] = remote(worker, ["docker", "inspect", "--format", "{{.Image}}", name])
    if action == "train" and result["status"] == "trained":
        if sha256(out / "model.pt") != result["checkpoint_sha256"]:
            raise ValueError(f"Collected checkpoint is corrupt: {job['id']}")
    if action == "train" and result["status"] == "failed":
        download(worker, remote_out + "/", out, recovery=True)
    write_json(out / f"{action}.collected.json", result)
    print(f"[{worker.host}] {action} {job['id']}: {result['status']}", flush=True)
    return result


def train_distributed(
    config: AppConfig,
    root: Path,
    dataset: dict,
    jobs_override: list[dict] | None = None,
    finalists_override: list[str] | None = None,
) -> tuple[list[dict], dict[str, dict], list[str]]:
    if jobs_override is None:
        jobs = [{**make_job(arch, "split_grouped", 0, True, config.training, False), "stage": "screen"}
                for arch in config.architectures]
        screen_ids = {job["id"] for job in jobs}
        jobs.append({**make_job("resnet50", "split_official", 0, True, config.training, False), "stage": "protocol"})
    else:
        jobs = jobs_override.copy()
        screen_ids = set()
    trained = {}
    for path in root.glob("*/train.collected.json"):
        result = json.loads(path.read_text())
        checkpoint = path.parent / "model.pt"
        if result["status"] == "trained" and checkpoint.is_file() and sha256(checkpoint) == result["checkpoint_sha256"]:
            trained[result["id"]] = result
    if finalists_override is None:
        finalists, additional = finalist_jobs(config, trained, screen_ids)
        jobs.extend(additional)
    else:
        finalists, additional = finalists_override, []
    pending = [job for job in jobs if job["id"] not in trained]
    attempted: dict[str, set[str]] = {}
    state_path = root / "scheduler.json"
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    assignments = previous.get("active", {})
    active = {}
    futures = {}
    unavailable = set()
    costs = dict(COST)
    for result in trained.values():
        if "training_seconds" in result:
            source = next(worker for worker in config.cluster.workers if worker.host == result["worker"])
            costs[result["arch"]] = result["training_seconds"] * (2 if source.speed == "fast" else 1)
    with ThreadPoolExecutor(max_workers=len(config.cluster.workers)) as pool:
        while pending or futures:
            busy = {worker.host for worker, _ in futures.values()}
            for worker in config.cluster.workers:
                if worker.host in busy or worker.host in unavailable:
                    continue
                job = choose_job(worker, pending, attempted, assignments.get(worker.host), costs)
                if job is None:
                    continue
                pending.remove(job)
                attempted.setdefault(job["id"], set()).add(worker.host)
                active[worker.host] = job["id"]
                print(f"[{worker.host}] assigned {job['id']}", flush=True)
                futures[pool.submit(execute_remote, worker, job, "train", root, dataset, config.cluster.image)] = (worker, job)
            write_json(state_path, {"active": active, "pending": pending, "finalists": finalists})
            if not futures:
                for job in pending:
                    trained[job["id"]] = {**job, "status": "failed", "error": "No reachable worker remains; rerun to reattach/retry"}
                break
            done, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                worker, job = futures.pop(future)
                active.pop(worker.host, None)
                try:
                    result = future.result()
                except Exception as error:
                    # A disconnected task may still run. Do not duplicate it on another GPU.
                    unavailable.add(worker.host)
                    active[worker.host] = job["id"]
                    result = {**job, "status": "failed", "worker": worker.host, "error": f"{type(error).__name__}: {error}"}
                trained[job["id"]] = result
                if result["status"] == "trained":
                    costs[job["arch"]] = result["training_seconds"] * (2 if worker.speed == "fast" else 1)
                if result["status"] != "trained" and worker.host not in unavailable and len(attempted[job["id"]]) < len(config.cluster.workers):
                    pending.append(job)
                if not finalists:
                    finalists, additional = finalist_jobs(config, trained, screen_ids)
                    jobs.extend(additional)
                    pending.extend(item for item in additional if item["id"] not in trained)
            write_json(state_path, {"active": active, "pending": pending, "finalists": finalists})
    return jobs, trained, finalists


def run(
    config: AppConfig,
    root: Path,
    preparation: dict,
    jobs_override: list[dict] | None = None,
    finalists_override: list[str] | None = None,
    parent_report: dict | None = None,
) -> int:
    dataset = preparation["dataset"]
    jobs, trained, finalists = train_distributed(config, root, dataset, jobs_override, finalists_override)
    profiler = next(worker for worker in config.cluster.workers if worker.host == config.cluster.profiling_host)
    runs = []
    for job in jobs:
        result = trained[job["id"]]
        if result["status"] != "trained":
            runs.append(result)
            continue
        out = root / job["id"]
        remote(profiler, ["mkdir", "-p", f"{profiler.directory}/{out}"])
        upload(out / "model.pt", profiler, f"{profiler.directory}/{out}/model.pt")
        evaluated_job = {**job, "checkpoint": str(out / "model.pt")}
        try:
            runs.append(execute_remote(profiler, evaluated_job, "evaluate", root, dataset, config.cluster.image))
        except Exception as error:
            runs.append({**job, "status": "failed", "error": f"{type(error).__name__}: {error}"})
    runs.append({"id": "team-model", "status": "unavailable", "required": False, "error": "Future team model is disabled"})
    if parent_report is not None:
        source_id_by_continuation = {job["id"]: job["source_job_id"] for job in jobs}
        updated_training = []
        for previous in parent_report["training"]:
            continuation_id = next((key for key, source in source_id_by_continuation.items()
                                    if source == previous["id"]), None)
            if continuation_id is None or trained[continuation_id]["status"] != "trained":
                updated_training.append(previous)
                continue
            result = {**trained[continuation_id], "id": previous["id"],
                      "continuation_run_id": continuation_id, "continued_from_epoch": 6,
                      "target_epochs": jobs_override[0]["target_epochs"]}
            updated_training.append(result)
        updated_runs = []
        for previous in parent_report["runs"]:
            continuation_id = next((key for key, source in source_id_by_continuation.items()
                                    if source == previous["id"]), None)
            if continuation_id is None:
                updated_runs.append(previous)
                continue
            result = next((item for item in runs if item["id"] == continuation_id), None)
            if result is None or result["status"] != "completed":
                updated_runs.append(previous)
                continue
            updated_runs.append({**result, "id": previous["id"], "continuation_run_id": continuation_id,
                                 "total_epochs": jobs_override[0]["target_epochs"]})
        parent_report = {**parent_report, "run_id": root.name, "training": updated_training, "runs": updated_runs,
                         "plan": {**parent_report["plan"], "continued_from_run": parent_report["run_id"],
                                  "target_epochs": jobs_override[0]["target_epochs"]},
                         "limitations": [item for item in parent_report["limitations"]
                                         if "Limited six-epoch" not in item]}
        runs = updated_runs
    status = overall_status(runs)
    if len(finalists) != 2:
        status = "partial" if any(run["status"] == "completed" for run in runs) else "no_evaluable_models"
    report = parent_report or {
        "schema_version": 1, "run_id": root.name, "overall_status": status, "requested_mode": "train",
        "classes": CLASSES9, "dataset": dataset, "environment": preparation["workers"],
        "plan": {"configuration": config.model_dump(mode="json"), "finalists": finalists,
                 "requested_experiment_ids": [job["id"] for job in jobs],
                 "selection_policy": "top two validation macro-F1; architecture name breaks ties",
                 "profiling_worker": profiler.host},
        "training": list(trained.values()), "runs": runs, "comparisons": {},
        "required_experiments_complete": status == "complete", "scientific_project_complete": False,
        "limitations": ["Limited six-epoch optimization budget; scratch models may not converge.",
                        "Training used different GPUs; per-run hardware is recorded. All final timing uses one GPU.",
                        "Grouped and official tests differ in cohort and training size; their difference is not a causal leakage estimate.",
                        "Duplicate/conflict frames retained with separate sensitivity analyses; patient links are heuristic.",
                        "Bootstrap intervals, Grad-CAM and finalist data-efficiency experiments are not included.",
                        "Human interpretation and scientific review remain required."],
    }
    if parent_report is not None:
        report["overall_status"] = status
        report["required_experiments_complete"] = status == "complete"
        report["scientific_project_complete"] = False
    package_report(report, root)
    print(f"Distributed plan {status}: results/final.json", flush=True)
    return int(status != "complete")


def continuation_jobs(
    config: AppConfig,
    parent: Path,
    root: Path,
    target_epochs: int,
    models: list[str] | None = None,
) -> tuple[list[dict], dict, list[str]]:
    previous = json.loads((parent / "final.json").read_text())
    if previous["overall_status"] != "complete":
        raise ValueError("The source benchmark must be complete before extending its finalists")
    finalists = previous["plan"]["finalists"]
    trained = {result["id"]: result for result in previous["training"]}
    jobs = []
    sources = []
    checkpoint_parent = (Path("results/distributed") / previous["plan"]["continued_from_run"]
                         if models and "continued_from_run" in previous["plan"] else parent)
    for arch in (models if models is not None else finalists):
        for pretrained in ((True,) if models is not None else (True, False)):
            for seed in ((0,) if models is not None else config.finalist_seeds):
                source = make_job(arch, "split_grouped", seed, pretrained, config.training, False)
                result = trained.get(source["id"])
                checkpoint = checkpoint_parent / source["id"] / "model.pt"
                if result is None or result["status"] != "trained" or not checkpoint.is_file():
                    raise ValueError(f"Missing completed source checkpoint: {source['id']}")
                if sha256(checkpoint) != result["checkpoint_sha256"]:
                    raise ValueError(f"Source checkpoint checksum mismatch: {source['id']}")
                continuation_id = f"{source['id']}-to{target_epochs}"
                recovery = root / "source-checkpoints" / source["id"]
                jobs.append({**source, "id": continuation_id, "stage": "continuation",
                             "source_job_id": source["id"], "resume_from": str(recovery / "resume.pt"),
                             "source_checkpoint_sha256": result["checkpoint_sha256"],
                             "target_epochs": target_epochs})
                worker = next(worker for worker in config.cluster.workers if worker.host == result["worker"])
                sources.append((worker, source["id"], recovery))
    with ThreadPoolExecutor(max_workers=len(config.cluster.workers)) as pool:
        futures = [pool.submit(download, worker,
                               f"{worker.directory}/results/distributed/{checkpoint_parent.name}/{job_id}/resume.pt",
                               recovery, recovery=True)
                   for worker, job_id, recovery in sources]
        for future in futures:
            future.result()
    for job in jobs:
        job["source_resume_sha256"] = sha256(Path(job["resume_from"]))
    return jobs, previous, finalists


def validate_resume_identity(saved: dict, current: dict) -> None:
    # Controller fixes may resume an existing image; scientific code and recipe must match.
    controller = "src/fetal/distributed.py"
    saved_files = {path: checksum for path, checksum in saved["files"].items() if path != controller}
    current_files = {path: checksum for path, checksum in current["files"].items() if path != controller}
    if saved["config"] != current["config"] or saved_files != current_files:
        raise ValueError("Cannot resume: scientific code, configuration or metadata changed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Build once, copy the image/data, validate both workers")
    parser.add_argument("--resume-plan", help="Resume a prepared plan after controller-only fixes, retaining its image")
    parser.add_argument("--continue-from", help="Extend trained models from a completed distributed plan")
    parser.add_argument("--continue-models", help="Comma-separated model names; continue their ImageNet seed-0 runs")
    parser.add_argument("--target-epochs", type=int, default=20, help="Total epochs after continuation (default: 20)")
    args = parser.parse_args()
    models = args.continue_models.split(",") if args.continue_models else None
    if models and (not args.continue_from or len(set(models)) != len(models) or any(not model for model in models)):
        parser.error("--continue-models requires --continue-from and unique model names")
    if args.resume_plan and (args.prepare or args.continue_from or len(args.resume_plan) != 20 or any(char not in "0123456789abcdef" for char in args.resume_plan)):
        parser.error("--resume-plan requires a 20-character plan ID and cannot be combined with --prepare")
    if args.continue_from and (args.prepare or args.resume_plan or len(args.continue_from) != 20
                               or any(char not in "0123456789abcdef" for char in args.continue_from)
                               or args.target_epochs <= 6):
        parser.error("--continue-from requires a completed plan ID and target epochs greater than 6")
    config = load_config()
    if len(config.cluster.workers) < 2 or config.team_model.enabled:
        raise ValueError("Distributed mode requires at least two workers and the future team model disabled")
    sources = [*sorted(Path("src/fetal").rglob("*.py")), Path("pyproject.toml"), Path("uv.lock"), Path("Dockerfile"), *sorted(Path("data/processed").glob("*"))]
    identity = {"config": config.model_dump(mode="json"), "files": {str(path): sha256(path) for path in sources if path.is_file()}}
    parent = Path("results/distributed") / args.continue_from if args.continue_from else None
    continuation_identity = {"parent": args.continue_from, "target_epochs": args.target_epochs,
                             "models": models,
                             "config": identity["config"], "files": identity["files"]}
    plan_id = args.resume_plan or hashlib.sha256(json.dumps(
        continuation_identity if args.continue_from else identity, sort_keys=True).encode()).hexdigest()[:20]
    root = Path("results/distributed") / plan_id
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.continue_from:
            if models and any(model not in config.architectures for model in models):
                raise ValueError("Continuation models must be configured architectures")
            parent_identity = json.loads((parent / "identity.json").read_text())
            changed_paths = {"src/fetal/distributed.py", "src/fetal/experiment.py"}
            old_files = {key: value for key, value in parent_identity["files"].items() if key not in changed_paths}
            new_files = {key: value for key, value in identity["files"].items() if key not in changed_paths}
            if parent_identity["config"] != identity["config"] or old_files != new_files:
                raise ValueError("Finalist continuation requires the original recipe and all other scientific files")
            jobs, previous, finalists = continuation_jobs(config, parent, root, args.target_epochs, models)
            write_json(root / "identity.json", continuation_identity)
            write_json(root / "continuation.json", {"source_run": args.continue_from,
                       "target_epochs": args.target_epochs, "finalists": finalists,
                       "continued_models": models or finalists,
                       "optimizer_state_resumed": True, "extension_learning_rate": "0.1 * configured lr",
                       "checkpoint_sha256": {job["source_job_id"]: job["source_resume_sha256"] for job in jobs}})
            baseline_archive = Path("results/final-six-epoch.zip")
            if not baseline_archive.exists() and Path("results/final.zip").exists():
                import shutil
                shutil.copy2("results/final.zip", baseline_archive)
            builder = next(worker for worker in config.cluster.workers if worker.speed == "fast")
            preparation = prepare(config, builder)
            write_json(root / "preparation.json", preparation)
            return run(config, root, preparation, jobs, finalists, previous)
        if args.resume_plan:
            validate_resume_identity(json.loads((root / "identity.json").read_text()), identity)
            write_json(root / "controller-resume.json", {"controller_sha256": sha256(Path(__file__)),
                       "reason": "Controller-only recovery; original image and scientific identity retained"})
        else:
            write_json(root / "identity.json", identity)
        preparation_path = root / "preparation.json"
        if args.prepare:
            builder = next(worker for worker in config.cluster.workers if worker.speed == "fast")
            write_json(preparation_path, prepare(config, builder))
        preparation = json.loads(preparation_path.read_text())
        if args.resume_plan:
            for worker in config.cluster.workers:
                actual_image = remote(worker, ["docker", "image", "inspect", "--format", "{{.Id}}", config.cluster.image])
                if actual_image != preparation["workers"][worker.host]["image"]:
                    raise ValueError(f"Prepared image changed on {worker.host}; original image is required to resume")
            continuation_path = root / "continuation.json"
            if continuation_path.exists():
                continuation = json.loads(continuation_path.read_text())
                selected = continuation.get("continued_models", continuation["finalists"])
                models = None if selected == continuation["finalists"] else selected
                jobs, previous, finalists = continuation_jobs(
                    config, Path("results/distributed") / continuation["source_run"],
                    root, continuation["target_epochs"], models)
                return run(config, root, preparation, jobs, finalists, previous)
        return run(config, root, preparation)


if __name__ == "__main__":
    raise SystemExit(main())
