"""Reporting checks without downloads, Docker or model execution."""
import json
import zipfile

from fetal.package import overall_status, package_report


def test_failed_model_followed_by_success_is_partial_and_exports_both(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "results" / "pipeline" / "test"
    root.mkdir(parents=True)
    failed = {"id": "failed-first", "status": "failed", "error": "deliberate test error"}
    success = {
        "id": "successful-second", "status": "completed", "arch": "resnet50",
        "protocol": "split_grouped", "pretrained": True, "seed": 0, "smoke": True,
        "quality": {part: {"macro_f1": 0.5, "balanced_accuracy": 0.5, "accuracy": 0.5}
                    for part in ("val", "test")},
        "cost": {"status": "completed"},
        "predictions": [{"image": "image-1", "label": 0, "prediction": 0,
                         "patient_group": 1, "machine": "Other", "operator": "Other",
                         "partition": "test", "protocol": "split_grouped", "seed": 0,
                         "checkpoint_sha256": "test", "probabilities": [1.0] + [0.0] * 8}],
    }
    runs = [failed, success]
    report = {
        "schema_version": 1, "run_id": "test", "requested_mode": "smoke",
        "overall_status": overall_status(runs), "runs": runs, "limitations": [],
        "plan": {"configuration": {"finalist_seeds": [0, 1]}},
    }
    package_report(report, root)
    result = json.loads((tmp_path / "results/final.json").read_text())
    assert result["overall_status"] == "partial"
    assert [r["status"] for r in result["runs"]] == ["failed", "completed"]
    assert result["runs"][1]["predictions"][0]["image"] == "image-1"
    assert result["comparisons"]["architecture_screen"] == []
    with zipfile.ZipFile(tmp_path / "results/final.zip") as bundle:
        assert {"final.json", "report.md", "predictions.csv", "summary.csv"} <= set(bundle.namelist())
        assert all(not name.startswith("/") for name in bundle.namelist())


def test_missing_checkpoints_and_failed_profiling_cannot_report_complete():
    assert overall_status([{"status": "unavailable"}]) == "no_evaluable_models"
    success = {"status": "completed", "cost": {"status": "completed"}}
    disabled = {"status": "unavailable", "required": False}
    assert overall_status([success, disabled]) == "complete"
    assert overall_status([{"status": "completed", "cost": {"status": "failed"}}]) == "partial"
