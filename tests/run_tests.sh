#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="nciplot-4.4"
ENV_FILE="$ROOT_DIR/environment.yml"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env '$ENV_NAME' not found. Creating from $ENV_FILE"
  conda env create -f "$ENV_FILE"
fi

cd "$ROOT_DIR"
conda run -n "$ENV_NAME" python -m pytest -q
