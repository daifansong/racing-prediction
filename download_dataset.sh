#!/usr/bin/env bash
# Download archive from Google Drive and extract to ./dataset
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEST_DIR="dataset"
ARCHIVE="dataset.tar.gz"
URL="https://drive.google.com/file/d/1LFsdkIXK4uvA-E1YuC5eSkuocJh5VjUV/view?usp=sharing"

if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown not found. Install with: python3 -m pip install gdown"
  exit 1
fi

echo "Downloading from Google Drive..."
echo "URL: $URL"
gdown --fuzzy "$URL" -O "$ARCHIVE"

echo "Extracting to ${DEST_DIR}/..."
rm -rf "$DEST_DIR"
tar -xzf "$ARCHIVE"

# Archive may contain a top-level dataset/ dir, or loose CSV files.
if [[ ! -d "$DEST_DIR" ]]; then
  mkdir -p "$DEST_DIR"
  for f in x_train.csv x_val.csv x_test.csv y_train.csv y_val.csv y_test.csv; do
    if [[ -f "$f" ]]; then
      mv "$f" "$DEST_DIR/"
    fi
  done
fi

echo "Done. Contents:"
ls -lh "$DEST_DIR"
