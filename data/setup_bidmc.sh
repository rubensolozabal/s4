#!/usr/bin/env bash
set -euo pipefail

# From anywhere inside the repo, find the repo base (git root).
# If you don't use git, replace this with: REPO_ROOT="$(pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# --- 1) Create folder structure ---
mkdir -p data/bidmc/{RR,HR,SpO2}

# --- 2) Download datasets into their folders ---
download() {
  local url="$1"
  local outdir="$2"
  local fname
  fname="$(basename "$url")"

  echo "Downloading $fname -> $outdir/"
  if command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "$outdir/$fname" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail --progress-bar -o "$outdir/$fname" "$url"
  else
    echo "ERROR: need wget or curl installed." >&2
    exit 1
  fi
}

# RR
download "https://zenodo.org/record/4001463/files/BIDMC32RR_TRAIN.ts" "data/bidmc/RR"
download "https://zenodo.org/record/4001463/files/BIDMC32RR_TEST.ts"  "data/bidmc/RR"

# HR
download "https://zenodo.org/record/4001456/files/BIDMC32HR_TRAIN.ts" "data/bidmc/HR"
download "https://zenodo.org/record/4001456/files/BIDMC32HR_TEST.ts"  "data/bidmc/HR"

# SpO2
download "https://zenodo.org/record/4001464/files/BIDMC32SpO2_TRAIN.ts" "data/bidmc/SpO2"
download "https://zenodo.org/record/4001464/files/BIDMC32SpO2_TEST.ts"  "data/bidmc/SpO2"

# --- 3) Copy processing scripts into data/bidmc ---
SRC_DIR="src/dataloaders/prepare/bidmc"
DEST_DIR="data/bidmc"

if [[ ! -f "$SRC_DIR/process_data.py" || ! -f "$SRC_DIR/data_loader.py" ]]; then
  echo "ERROR: Missing expected files in $SRC_DIR" >&2
  echo "Expected: process_data.py and data_loader.py" >&2
  exit 1
fi

cp "$SRC_DIR/process_data.py" "$SRC_DIR/data_loader.py" "$DEST_DIR/"

# --- 4) Run processing script ---
echo "Running preprocessing: (cd data/bidmc && python process_data.py)"
cd "$DEST_DIR"
python process_data.py

echo "Done."
