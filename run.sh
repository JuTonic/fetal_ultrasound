#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mode=benchmark
device=cuda
while (($#)); do
    case "$1" in
        --mode) mode="${2:?Missing mode}"; shift 2 ;;
        --device) device="${2:?Missing device}"; shift 2 ;;
        --help|-h) echo 'Usage: ./run.sh [--mode benchmark|train|smoke] [--device cuda|cpu]'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
case "$mode" in benchmark|train|smoke) ;; *) echo "Invalid mode: $mode" >&2; exit 2 ;; esac
case "$device" in cuda|cpu) ;; *) echo "Invalid device: $device" >&2; exit 2 ;; esac
mkdir -p data/raw data/downloads checkpoints .cache results
image=neuroml-pipeline:local
docker build -t "$image" .
gpu=()
if [[ "$device" == cuda ]]; then gpu=(--gpus all); fi
commit=$(git rev-parse HEAD)
dirty=$(git status --porcelain)
image_id=$(docker image inspect --format '{{.Id}}' "$image")
# Data setup alone can write raw data. Every model process sees it read-only.
common=(--rm --user "$(id -u):$(id -g)" --shm-size 4g
    -e HOME=/tmp -e "PIPELINE_COMMIT=$commit" -e "PIPELINE_DIRTY=$dirty"
    -e "PIPELINE_IMAGE=$image_id"
    -v "$PWD/data/downloads:/downloads" -v "$PWD/.cache:/cache"
    -v "$PWD/results:/app/results" -v "$PWD/checkpoints:/app/checkpoints")
# Forward only application settings; dotenv is read by the validated config loader.
if [[ -f .env ]]; then common+=(-v "$PWD/.env:/app/.env:ro"); fi
for name in ${!FETAL_@}; do common+=(-e "$name"); done
if docker run "${common[@]}" -v "$PWD/data/raw:/app/data/raw" "$image" --setup; then
    docker run "${common[@]}" "${gpu[@]}" -v "$PWD/data/raw:/app/data/raw:ro" \
        "$image" --mode "$mode" --device "$device"
else
    docker run "${common[@]}" "$image" --setup-failed --mode "$mode"
    exit 1
fi
