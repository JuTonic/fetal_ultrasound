"""Portable strict JSON, CSVs, figures and logs without raw data or weights."""
from __future__ import annotations

import csv
import json
from importlib.metadata import distributions
import statistics
import zipfile
from pathlib import Path


def overall_status(runs: list[dict]) -> str:
    required = [run for run in runs if run.get("required", True)]
    completed = [run for run in required if run["status"] == "completed"]
    if not completed:
        return "no_evaluable_models"
    if all(run["status"] == "completed" and run["cost"]["status"] == "completed" for run in required):
        return "complete"
    return "partial"


def compare_runs(report: dict) -> dict:
    completed = [r for r in report["runs"] if r["status"] == "completed" and not r.get("smoke")]
    screen = [r for r in completed if r["protocol"] == "split_grouped" and r["seed"] == 0 and r["pretrained"]]
    screen.sort(key=lambda r: (-r["quality"]["val"]["macro_f1"], r["arch"]))
    transfer = []
    finalists = report["plan"].get("finalists", [])
    seeds = report["plan"]["configuration"]["finalist_seeds"]
    for arch in finalists:
        summaries = {}
        paired = []
        for pretrained in (True, False):
            runs = [r for r in completed if r["arch"] == arch and r["protocol"] == "split_grouped" and r["pretrained"] == pretrained]
            values = [r["quality"]["test"]["macro_f1"] for r in runs]
            summaries["imagenet" if pretrained else "scratch"] = {
                "seeds": [r["seed"] for r in runs], "test_macro_f1": values,
                "mean": statistics.mean(values) if values else None,
                "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                "complete": set(r["seed"] for r in runs) == set(seeds),
            }
        for seed in seeds:
            pair = [r for r in completed if r["arch"] == arch and r["protocol"] == "split_grouped" and r["seed"] == seed]
            by_init = {r["pretrained"]: r for r in pair}
            if set(by_init) != {True, False}:
                continue
            # Pair by identity, never assume equal row order.
            identities = []
            for pretrained in (True, False):
                identities.append({p["image"]: (p["label"], p["patient_group"], p["protocol"])
                                   for p in by_init[pretrained]["predictions"] if p["partition"] == "test"})
            if identities[0] != identities[1]:
                paired.append({"seed": seed, "error": "Prediction image IDs/labels/groups/protocol differ"})
                continue
            paired.append({"seed": seed, "pretrained_minus_scratch":
                           by_init[True]["quality"]["test"]["macro_f1"] - by_init[False]["quality"]["test"]["macro_f1"]})
        transfer.append({"arch": arch, **summaries, "paired_deltas": paired})
    protocols = [{"protocol": r["protocol"], "test": r["quality"]["test"], "run_id": r["id"]}
                 for r in completed if r["arch"] == "resnet50" and r["pretrained"] and r["seed"] == 0]
    return {
        "architecture_screen": [{"arch": r["arch"], "validation_macro_f1": r["quality"]["val"]["macro_f1"],
                                 "test_macro_f1": r["quality"]["test"]["macro_f1"], "cost": r["cost"]} for r in screen],
        "transfer": transfer, "protocol_comparison": protocols,
    }


def package_report(report: dict, root: Path) -> None:
    report["comparisons"] = compare_runs(report)
    statuses = [f"{run['id']}: {run['status']}" for run in report["runs"] if run["status"] != "completed"]
    report["limitations"].extend(statuses)
    report["artifact_paths"] = {"run_directory": str(root), "archive": "results/final.zip"}
    temporary = root / "final.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(root / "final.json")
    rows = []
    for run in report["runs"]:
        row = {"id": run["id"], "status": run["status"], "error": run.get("error", "")}
        if run["status"] == "completed":
            row.update(arch=run["arch"], protocol=run["protocol"], seed=run["seed"],
                       pretrained=run["pretrained"], smoke=run["smoke"],
                       val_macro_f1=run["quality"]["val"]["macro_f1"],
                       test_macro_f1=run["quality"]["test"]["macro_f1"],
                       test_balanced_accuracy=run["quality"]["test"]["balanced_accuracy"],
                       test_accuracy=run["quality"]["test"]["accuracy"])
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["id", "status"]
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    fields = ["run_id", "image", "label", "prediction", "patient_group", "machine", "operator",
              "partition", "protocol", "seed", "checkpoint_sha256", "probabilities"]
    with (root / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for run in report["runs"]:
            for prediction in run.get("predictions", []):
                writer.writerow({"run_id": run["id"], **prediction,
                                 "probabilities": json.dumps(prediction["probabilities"])})
    lines = ["# Fetal ultrasound architecture benchmark", "", f"Status: **{report['overall_status']}**",
             f"Mode: {report['requested_mode']}; run: {report['run_id']}", "",
             "| Experiment | Status | Validation macro-F1 | Test macro-F1 |", "|---|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['id']} | {row['status']} | {row.get('val_macro_f1', '')} | {row.get('test_macro_f1', '')} |")
    lines.extend(["", "Selection uses validation only. Test evaluation follows the frozen selection.", "",
                  "## Interpretation and limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(["", "Brain sub-plane similarity, scarce Brain: other examples, conflicting labels, scanner/style cues and short training budgets are competing explanations for errors. Inspect the per-class metrics, confusion figures and sensitivity results before drawing conclusions.",
                  "", "Historical preprocessing provenance: docs/preprocessing.md and results/ablation/ on the preprocessing branch. These are not fresh benchmark results."])
    (root / "report.md").write_text("\n".join(lines) + "\n")
    (root / "config.json").write_text(json.dumps(report["plan"], indent=2))
    versions = sorted(f"{package.metadata['Name']}=={package.version}" for package in distributions())
    (root / "environment.txt").write_text("\n".join(versions) + "\n")
    output = Path("results")
    (output / "final.tmp").write_bytes((root / "final.json").read_bytes())
    (output / "final.tmp").replace(output / "final.json")
    archive = output / "final.zip.tmp"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != ".lock" and path.suffix not in (".pt", ".pth", ".tmp"):
                bundle.write(path, str(path.relative_to(root)))
        if (output / "setup.log").exists():
            bundle.write(output / "setup.log", "setup.log")
    archive.replace(output / "final.zip")
