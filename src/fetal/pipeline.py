"""Sequential Docker pipeline with isolated model processes and ordinary files."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import redirect_stdout
import fcntl
import hashlib
import json
import os
import platform
import traceback
from pathlib import Path

import numpy as np
import torch
import torchvision

from fetal.config import AppConfig, TrainingConfig, load_config
from fetal.data import CLASSES9
from fetal.download import download_data, validate_data
from fetal.evaluate import evaluate_checkpoint
from fetal.experiment import checkpoint_contract, train_checkpoint
from fetal.files import sha256, write_json
from fetal.jobs import execute_job
from fetal.models import WEIGHTS, download_weights, load_checkpoint
from fetal.package import overall_status, package_report
from fetal.runtime import configure_torch, pick_device


def make_job(arch: str, protocol: str, seed: int, pretrained: bool, config: TrainingConfig, smoke: bool) -> dict:
    initialization = "imagenet" if pretrained else "scratch"
    return {
        "id": f"{arch}-{protocol}-{initialization}-seed{seed}", "arch": arch,
        "protocol": protocol, "seed": seed, "pretrained": pretrained,
        "config": config.model_dump(mode="json"), "smoke": smoke,
    }


def run_worker(path: Path) -> int:
    request = json.loads(path.read_text())
    job, action = request["job"], request["action"]
    out = path.parent
    result_path = out / f"{action}.json"
    # Overwrite on every invocation; never accidentally reuse an earlier success.
    write_json(result_path, {"status": "failed", "error": "Model process interrupted"})
    try:
        checkpoint = Path(job.get("checkpoint") or f"checkpoints/{job['id']}.pt")
        if action == "train":
            training_path = train_checkpoint(job, request["dataset"], out, request["device"])
            contract = load_checkpoint(training_path, checkpoint_contract(job, request["dataset"]))[1]
            result = {"status": "trained", "contract": contract, "checkpoint": str(training_path),
                      "training_seconds": float((out / "training_seconds.txt").read_text()),
                      "checkpoint_sha256": sha256(training_path)}
        elif not checkpoint.is_file():
            result = {"status": "unavailable", "error": f"No compatible fetal checkpoint at {checkpoint}"}
        else:
            result = evaluate_checkpoint(job, request["dataset"], checkpoint, out, request["device"])
        write_json(result_path, result)
        return int(result["status"] == "failed")
    except Exception as error:
        traceback.print_exc()
        write_json(result_path, {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        return 1


def cache_weights(architectures: Sequence[str], root: Path) -> dict:
    results = {}
    for arch in architectures:
        log = root / f"weights-{arch}.log"
        try:
            with log.open("a") as stream, redirect_stdout(stream):
                path = download_weights(arch)
            checksum = sha256(path)
            results[arch] = {"status": "completed", "url": WEIGHTS[arch].url, "sha256": checksum}
        except Exception as error:
            results[arch] = {"status": "failed", "log": log.name, "error": f"{type(error).__name__}: {error}"}
    return results


def run(args: argparse.Namespace) -> int:
    config = load_config()
    smoke = args.mode == "smoke"
    # Content identity survives a restart, but a changed recipe/data/code gets a new directory.
    sources = [*sorted(Path("src/fetal").rglob("*.py")), Path("pyproject.toml"), Path("uv.lock"), Path("Dockerfile")]
    metadata = sorted(Path("data/processed").glob("*"))
    identity = {
        "configuration": config.model_dump(mode="json"), "mode": args.mode, "device": args.device,
        "image": os.getenv("PIPELINE_IMAGE"), "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "hardware": torch.cuda.get_device_name(0) if args.device == "cuda" and torch.cuda.is_available() else "cpu",
        "sources": {str(path): sha256(path) for path in [*sources, *metadata] if path.is_file()},
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    root = Path("results") / ("smoke" if smoke else "pipeline") / run_id
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"This experiment is already running: {root}") from None
        write_json(root / "identity.json", identity)
        print(f"Results and recovery state: {root}", flush=True)
        return run_plan(args, config, root, run_id)


def run_plan(args: argparse.Namespace, config: AppConfig, root: Path, run_id: str) -> int:
    smoke = args.mode == "smoke"
    report = {
        "schema_version": 1, "run_id": run_id, "requested_mode": args.mode,
        "overall_status": "setup_failure", "classes": CLASSES9, "runs": [], "comparisons": {},
        "plan": {"configuration": config.model_dump(mode="json"), "selection_policy": "top two validation macro-F1; architecture name breaks ties"},
        "environment": {
            "code_commit": os.getenv("PIPELINE_COMMIT"), "code_dirty": bool(os.getenv("PIPELINE_DIRTY")),
            "image_identity": os.getenv("PIPELINE_IMAGE"), "python": platform.python_version(),
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "numpy": np.__version__, "precision": "fp32", "device": args.device,
            "determinism": "Seeded Python/NumPy/torch/CUDA; cuDNN benchmark off; TF32 off; no cross-device bitwise guarantee",
        },
        "limitations": [
            "No scientific results exist until trained fetal checkpoints are evaluated.",
            "Six epochs is a limited optimization budget, not convergence evidence.",
            "Grouped scanner balance does not test unseen-scanner generalization.",
            "Official/grouped scores differ in patients, training size and scanner mix; not a causal leakage estimate.",
            "Exact duplicates and label conflicts are retained; sensitivity metrics are separate.",
            "Patient linking is heuristic; stored pixel hashes were not recomputed on every run.",
            "Published initialization recipes differ between architectures.",
            "Bootstrap intervals, Grad-CAM and finalist data-efficiency experiments are not implemented.",
        ],
    }
    try:
        if args.setup_failed:
            raise RuntimeError("Shared dataset download/validation failed; see results/setup.log")
        report["dataset"] = validate_data()
        dev = pick_device(args.device)
        report["environment"]["hardware"] = torch.cuda.get_device_name(dev) if dev.type == "cuda" else platform.processor()
        architectures = ["resnet18"] if smoke else config.architectures
        cfg = config.training
        if smoke:
            cfg = cfg.model_copy(update={"epochs": 1, "batch_size": 8, "num_workers": 0})
            report["limitations"].append("SMOKE TEST: random initialization, 32 images per partition; excluded from scientific summaries.")
        jobs = [make_job(arch, "split_grouped", 0, not smoke, cfg, smoke) for arch in architectures]
        needed_weights = architectures if args.mode == "benchmark" else [
            job["arch"] for job in jobs
            if job["pretrained"] and not (root / job["id"] / "resume.pt").is_file()
        ]
        report["initialization_weights"] = cache_weights(needed_weights, root)
        training_results = []
        if args.mode in ("train", "smoke"):
            for job in jobs:
                training_results.append(execute_job(job, "train", root, report["dataset"], args.device))
            if not smoke:
                # Selection reads validation only. No test prediction has happened yet.
                screen = [r for r in training_results if r["status"] == "trained"]
                screen.sort(key=lambda r: (-r["contract"]["validation_macro_f1"], r["arch"]))
                finalists = [r["arch"] for r in screen[:2]] if len(screen) == len(architectures) else []
                report["plan"]["finalists"] = finalists
                additional = []
                for arch in finalists:
                    for pretrained in (True, False):
                        for seed in config.finalist_seeds:
                            job = make_job(arch, "split_grouped", seed, pretrained, cfg, False)
                            if job["id"] not in {j["id"] for j in jobs}:
                                additional.append(job)
                additional.append(make_job("resnet50", "split_official", 0, True, cfg, False))
                for job in additional:
                    training_results.append(execute_job(job, "train", root, report["dataset"], args.device))
                jobs.extend(additional)
            report["training"] = training_results
            by_id = {r["id"]: r for r in training_results}
            for job in jobs:
                trained = by_id[job["id"]]
                if trained["status"] != "trained":
                    report["runs"].append(trained)
                    continue
                job["checkpoint"] = trained["checkpoint"]
                report["runs"].append(execute_job(job, "evaluate", root, report["dataset"], args.device))
        else:
            for job in jobs:
                report["runs"].append(execute_job(job, "evaluate", root, report["dataset"], args.device))
            report["limitations"].append("Benchmark mode evaluates configured screen checkpoints; required transfer/protocol training is not scheduled in this mode.")
        team = config.team_model
        if team.enabled:
            job = make_job(team.architecture, "split_grouped", 0, True, cfg, smoke)
            job.update(id="team-model", checkpoint=team.checkpoint)
            report["runs"].append(execute_job(job, "evaluate", root, report["dataset"], args.device))
        else:
            report["runs"].append({"id": "team-model", "status": "unavailable", "required": False,
                                   "error": "Future team checkpoint/download slot intentionally unset"})
        report["plan"]["requested_experiment_ids"] = [job["id"] for job in jobs]
        required = [r for r in report["runs"] if r.get("required", True)]
        completed = [r for r in required if r["status"] == "completed"]
        report["overall_status"] = overall_status(report["runs"])
        for arch, weights in report["initialization_weights"].items():
            if weights["status"] == "failed":
                report["limitations"].append(f"{arch} initialization cache: {weights['error']}")
        if args.mode == "train" and len(report["plan"].get("finalists", [])) != 2:
            report["overall_status"] = "partial" if completed else "no_evaluable_models"
        report["required_experiments_complete"] = args.mode == "train" and report["overall_status"] == "complete"
        report["scientific_project_complete"] = False
        report["limitations"].append("Scientific completion requires human review and interpretation of the generated results.")
    except Exception as error:
        traceback.print_exc()
        report["overall_status"] = "setup_failure"
        report["limitations"].append(f"Setup failure: {type(error).__name__}: {error}")
    package_report(report, root)
    print(f"{report['overall_status']}: results/final.json", flush=True)
    return int(report["overall_status"] != "complete")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("benchmark", "train", "smoke"), default="benchmark")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--setup-failed", action="store_true")
    parser.add_argument("--job", type=Path)
    args = parser.parse_args()
    configure_torch()
    if args.setup:
        Path("results").mkdir(exist_ok=True)
        with Path("results/setup.log").open("w") as stream:
            try:
                load_config()
                download_data()
            except Exception:
                traceback.print_exc(file=stream)
                raise
        return 0
    if args.job:
        return run_worker(args.job)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
