"""Run one distributed task inside the same image as the sequential pipeline."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
import torchvision

from fetal.download import validate_data
from fetal.files import write_json
from fetal.jobs import execute_job
from fetal.runtime import configure_torch, pick_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    configure_torch()
    request = json.loads(args.request.read_text())
    dataset = validate_data()
    if dataset != request["dataset"]:
        raise ValueError("Worker data differs from the coordinator's frozen dataset")
    device = pick_device(request["device"])
    environment = {
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "device": str(device), "torch": torch.__version__, "torchvision": torchvision.__version__,
        "precision": "fp32", "python": platform.python_version(),
    }
    root = args.request.parent.parent
    result = execute_job(request["job"], request["action"], root, dataset, request["device"])
    result["environment"] = environment
    write_json(args.request.parent / f"{request['action']}.result.json", result)
    return int(result["status"] not in ("trained", "completed"))


if __name__ == "__main__":
    raise SystemExit(main())
