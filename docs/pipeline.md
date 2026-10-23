# Docker pipeline

Application and preprocessing Python code lives in `src/fetal/`; tests stay in
`tests/`, JSON defaults in `configs/`, and frozen metadata in `data/processed/`.
Historical preprocessing results remain in `results/`. No compatible trained fetal
checkpoints are included. ImageNet initialization alone is not a fetal classifier.

## Running

The host needs Bash, Git and Docker. GPU execution also needs an NVIDIA driver and
NVIDIA Container Toolkit. Run from any directory:

```bash
./run.sh                              # benchmark compatible local fetal checkpoints
./run.sh --mode train                  # full configured experiment plan, CUDA
./run.sh --mode smoke --device cpu     # small random-initialization plumbing check
```

The default training plan schedules eleven runs: four architecture screens, the two
validation-selected finalists with ImageNet/scratch initialization across two seeds,
and a separate official-protocol ResNet-50. Matching screen runs are reused. Test
prediction follows training and selection. Six epochs is a short declared budget;
it does not establish convergence, particularly for scratch models.

All downloads and dataset execution happen inside Docker. The image uses uv with
`uv.lock`, torch 2.5.1 and torchvision 0.20.1 CUDA 12.4 wheels. CUDA libraries come
from those locked dependencies; the NVIDIA driver comes from the host. The entrypoint
executes the uv-installed virtual environment directly, so it does not resolve
dependencies or initialize a build cache at runtime. Raw data is
writable for setup and read-only for model jobs. Outputs and caches stay on the host.

## Configuration

`src/fetal/config/` follows the reference in `docs/config/`: frozen Pydantic records,
a settings object and one loader. It reads JSON rather than YAML. Precedence is
**process environment > `.env` > `configs/pipeline.json` > field defaults**.
Nested overrides use the `FETAL_` prefix and double underscores:

```bash
FETAL_TRAINING__BATCH_SIZE=8 ./run.sh --mode train
```

Or put settings in an untracked `.env`:

```dotenv
FETAL_TRAINING__BATCH_SIZE=8
FETAL_TRAINING__NUM_WORKERS=2
```

`run.sh` forwards exported `FETAL_*` variables and mounts `.env` when present.
Settings are validated before downloading data. The report and job requests record
the resolved configuration. FP32 is currently the only supported precision; batch
size is not silently reduced after an out-of-memory failure.

The future team model stays disabled with null checkpoint/download locations.
Enabling it requires a supported architecture and a compatible local checkpoint.
Automatic team-weight downloads are not implemented.

## Recovery

After an interruption, rerun the **same command and settings**. Matching code,
configuration, metadata, image, mode and hardware identify the same results directory.
Completed training and evaluation jobs are skipped after configuration/checkpoint
checks. Failed or interrupted jobs are retried on the next invocation; independent
jobs continue during the current invocation. Logs append, `progress.json` records
outcomes, and a lock prevents concurrent writes to the same experiment.

Each fully trained and validated epoch atomically saves `resume.pt`: latest model,
AdamW state, LR scheduler, learning curve, completed epoch and best validation model.
An interrupted write leaves the previous snapshot intact. Restart restores that
state and repeats at most the uncommitted epoch. Epoch seeds and fresh loader
workers preserve this boundary's randomness. Cross-device bitwise equality is not
guaranteed. Timing excludes work lost during interruptions.

`model.pt` exports the best validation epoch, with earlier epochs winning ties.
It can be rebuilt from the recovery snapshot if exports were interrupted. Both
checkpoints remain under `results/pipeline/<run_id>/<experiment_id>/`. Recovery
requires these files to remain on disk. Changed code/settings/data/image select a
new directory; earlier timestamp-based directories are not automatically imported.

Benchmark mode reads the four screen checkpoints from
`checkpoints/<experiment_id>.pt`, for example
`checkpoints/resnet50-split_grouped-imagenet-seed0.pt`. Copy exported training
checkpoints there for a separate benchmark invocation. Checkpoints must identify
classes, preprocessing, configuration, initialization, training groups and frozen
metadata/split/subset identities. Unknown training/test provenance is rejected.

## Data and outputs

Dataset and ImageNet-weight downloads share `src/fetal/transfer.py`. Each download
call makes at most **five attempts**, with 2/4/8/16-second backoff and a 60-second
connection/read timeout. HTTP 408/429 and server errors are retried; permanent errors
such as 404 fail immediately. Numeric `Retry-After` values are respected up to the
30-second backoff cap. Failed model-weight downloads do not prevent other models.

Partial bytes are kept in `.part` files under the mounted dataset/torch caches.
Retries and later invocations request `Range: bytes=<saved-size>-`. If the server
ignores Range, the file restarts from zero instead of appending duplicate bytes.
Invalid ranges, stale oversized partials and checksum mismatches trigger a fresh
transfer. Completed files are reused only after checksum verification: the dataset's
full published MD5 or torchvision's published SHA-256 filename prefix for weights.
Only verified files are renamed to their final cache names. No automatic resume is
possible if the remote server does not support byte ranges.

Setup verifies the original Zenodo archive's published MD5, extracts metadata/images,
and checks completeness before marking extraction finished. Later runs reuse the
cache and validate frozen checksums, labels, patients, machines, operators and official
test membership. Stored pixel hashes are not recomputed on every invocation.

All models share B0: grayscale, aspect-preserving 224-square letterbox, training-only
moderate augmentation, three channels and ImageNet normalization. There is no extra
center crop. Duplicates and label conflicts remain flagged; sensitivity metrics are
reported separately. Frozen metadata is reused, never silently regenerated.

`results/final.json` embeds prediction records, all nine probabilities, validation/test
metrics, confusion counts, machine slices, cost measurements, comparisons, provenance,
failures and limitations. `results/final.zip` adds CSVs, plots, configuration and logs
with relative paths. Neither artifact contains raw images, caches or checkpoints.
Top-level final files refer to the latest packaged invocation, including smoke;
smoke results are marked and excluded from scientific comparisons.

Model jobs run sequentially in separate processes. Missing checkpoints are
`unavailable`; job failures are `failed`; successful jobs are `completed`. Incomplete
plans exit nonzero after packaging as `partial` or `no_evaluable_models`. The disabled
team model is not required. Shared setup failures stop execution; Docker build/start
failures can prevent reporting. Sudden termination may leave the previous final
report in place until the next invocation packages current results.

Latency uses 20 warmups and 100 synchronized batch-1 FP32 forwards on resident input,
excluding loading/preprocessing. fvcore counts a multiply-add as one operation and
lists unsupported operations. Profiling failures preserve quality metrics but make
the plan partial. Compare latency only on the same hardware. Seed summaries use sample
standard deviation; scanner slices average over supported classes. Optional bootstrap,
Grad-CAM and finalist data-efficiency experiments are not implemented. Scientific
completion still requires human interpretation of the results.

## Development and verification

Python 3.11 is selected in `.python-version`; the lock targets Linux x86-64. Choose
one mutually exclusive torch backend when using uv locally:

```bash
uv sync --locked --extra cpu
uv run --locked --extra cpu pytest -q
# CUDA environment, when needed:
uv sync --locked --extra cuda
```

Local Python is for development/tests; `run.sh` keeps dataset execution in Docker.
`pyproject.toml` declares dependencies and the `src/` package; `uv.lock` locks CPU/CUDA
wheels and transitive packages. `requirements.txt` is no longer used.

The original local implementation check passed 89 tests, with 11 raw-image tests
skipped. The full Docker training and evaluation were subsequently run on CUDA
machines; the final compact run records are in [experiment tracking](tracking/README.md).
Large training checkpoints, raw images and full local result archives are excluded
from Git. A fresh run is required to regenerate those artifacts.

uv configuration follows its official [PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/)
and [Docker integration guide](https://docs.astral.sh/uv/guides/integration/docker/).
Historical scientific context remains in `docs/preprocessing.md` and the preserved
ablation results. Scanner balance is not unseen-scanner robustness; official/grouped
score differences also reflect cohort and training-size differences.
