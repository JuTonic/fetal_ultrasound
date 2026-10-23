#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec uv run --locked --extra cpu python -m fetal.distributed "$@"
