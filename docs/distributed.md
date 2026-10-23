# Two-worker execution

`./distribute.sh --prepare` prepares the workers, runs the shared queue and collects
`results/final.json` plus `results/final.zip`. The lightweight SSH controller uses
local uv; all data processing, model training and evaluation run inside Docker.
Workers and the final profiling host are configured in `configs/pipeline.json`.

The current plan uses batch size 8, FP32 and six epochs on both GPUs. The initial
queue assigns Swin-T to the 3080 Ti and ResNet-50 to the 4060, followed by the other
screen models as workers become free. An independent official-split ResNet-50 fills
otherwise idle time. Once all four screens succeed, validation macro-F1 selects two
finalists and six additional paired-seed/initialization jobs join the queue. The two
matching ImageNet seed-0 screens are reused, for eleven total training jobs.

The PC favors longer jobs and the laptop shorter jobs. After completed jobs, training
durations update architecture cost estimates with an approximate two-to-one initial
worker speed normalization. This is a scheduling heuristic, not a promise of optimal
makespan. Epoch durations in each job's `learning_curve.csv` support runtime estimates.

Only one task runs per worker. Containers are detached so a controller/SSH interruption
does not kill training. The controller records active assignments in `scheduler.json`
and reattaches on restart. An explicitly failed model may retry on the other worker;
its latest recovery snapshot is transferred when available. If SSH fails and the
original container's state cannot be confirmed, the worker is left unavailable and
the uncertain job is not duplicated. Other independent work continues. Rerun after
connectivity returns to reattach. This is not distributed minibatch training.

Completed jobs are collected and checkpoint hashes verified. Successful model weights
and logs are copied locally; large optimizer snapshots stay on the worker unless a
job fails and needs recovery/failover. Keep both worker disks until the plan is complete.
Recovery records hardware transitions, and GPU execution is not promised bitwise equal
across machines. Existing completed training is skipped on matching reruns.

All final validation/test prediction and cost profiling runs on `kr54-pc`, after
training and finalist selection. This keeps latency hardware consistent. Individual
training results retain their worker/hardware identity. The final JSON embeds all
predictions and comparisons; the archive contains configs, figures, logs and CSVs.
Human scientific review remains necessary.

Preparation builds one image on the PC, transfers that exact image to the laptop,
downloads the original dataset and initialization weights once, then copies verified
caches and extracts on each worker. Both nodes must report identical dataset and image
identities before scheduling. Image transfer uses content-addressed OCI blobs, so small
code changes reuse the CUDA layers. Image/data copies use rsync partial-transfer recovery;
HTTP downloads use the shared retry/range implementation. Scientific source, resolved
configuration and frozen metadata identify a stable plan directory. Changing these
creates a new plan instead of silently reusing incompatible results.

On `kr54-pc`, `/srv/neuroml` is an ext4 filesystem backed by
`D:\NeuroML\storage.ext4` (64 GiB capacity). Docker data, containerd data and the project
are inside it. This keeps large workload files off C: while WSL stays running.
The mount is recorded in `/etc/fstab`; Docker/containerd require the mount before
starting. Original configuration files were backed up and old storage was preserved.
The laptop uses `/home/semyon/neuroml` in its normal WSL filesystem.

The initial deployment installed NVIDIA Container Toolkit on both hosts using
`src/ops/setup-gpu.sh`. `src/ops/setup-storage.sh` configured D: storage on the PC.
No Linux display driver was installed in WSL. Do not rerun storage migration while
training is active. The earlier full-WSL move scripts are not used by this setup.

To launch in the background from the coordinating workspace:

```bash
mkdir -p results
nohup ./distribute.sh --prepare >results/distributed-launch.log 2>&1 &
tail -f results/distributed-launch.log
```

After successful preparation, restart the controller with `./distribute.sh` to reuse
`preparation.json`. If interrupted during preparation, use `--prepare` again; cached
builds/downloads and partial copies are reused. Avoid launching a second controller
for the same plan; a lock prevents this. The coordinating machine must stay running
and retain SSH access for scheduling and collection. Detached active jobs can keep
running while the controller is offline.

After a controller-only fix, resume a prepared plan explicitly with
`./distribute.sh --resume-plan PLAN_ID`. This checks that the scientific source,
configuration, metadata and deployed image still match the saved plan. It records the
updated controller hash while retaining the original image and run directory. It refuses
scientific changes and cannot rebuild the image. The current deployment uses plan
`8c25c6b3b89b3a133e0f`; its background log is `results/distributed-launch.log`.

Extend selected finalists from six to twenty total epochs with:

```bash
./distribute.sh --continue-from 8c25c6b3b89b3a133e0f --target-epochs 20
```

This continues both finalists across ImageNet and scratch initialization, seeds 0 and
1. It restores each atomic epoch-six model, optimizer and scheduler snapshot, then
starts a lower-rate cosine phase for the remaining epochs. The continuation gets its
own run directory and records source checkpoint hashes; it retains the six-epoch
benchmark archive as `results/final-six-epoch.zip`.

The remaining architecture screens can then be continued from their epoch-six
optimizer snapshots, while retaining the completed finalist results:

```bash
./distribute.sh --continue-from 580564cc8439644b8c81 \
  --continue-models convnext_tiny,swin_t --target-epochs 20
```

`src/ops/queue_remaining.py` waits for the finalist continuation to finish and
then runs this command. Its output is in `results/queue-remaining.log`.
