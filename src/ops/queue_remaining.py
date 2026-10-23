"""Start the remaining architecture continuations when the active plan completes."""
import json
import os
import subprocess
import time
from pathlib import Path


plan = "580564cc8439644b8c81"
report = Path("results/distributed") / plan / "final.json"
controller = int(Path("results/distributed-continuation20.pid").read_text())
retries = 0

while True:
    if report.exists() and json.loads(report.read_text())["overall_status"] == "complete":
        break
    try:
        os.kill(controller, 0)
    except ProcessLookupError:
        laptop = subprocess.run([
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
            "-o", "ConnectionAttempts=1", "maibenben", "true",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        if laptop.returncode == 0:
            retries += 1
            if retries > 3:
                raise SystemExit("First continuation remains incomplete after three recovery attempts")
            print(f"Laptop reachable; resuming first continuation (attempt {retries})", flush=True)
            subprocess.call(["./distribute.sh", "--resume-plan", plan])
        else:
            print("Waiting for laptop to return before resuming first continuation", flush=True)
    time.sleep(30)

print("First continuation complete; starting ConvNeXt-Tiny and Swin-T", flush=True)
raise SystemExit(subprocess.call([
    "./distribute.sh", "--continue-from", plan,
    "--continue-models", "convnext_tiny,swin_t", "--target-epochs", "20",
]))
