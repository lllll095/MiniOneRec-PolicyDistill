#!/bin/bash
set -euo pipefail

UPSTREAM="${1:-../MiniOneRec}"

if [ ! -d "$UPSTREAM" ]; then
  echo "ERROR: upstream MiniOneRec directory not found: $UPSTREAM"
  exit 1
fi

for f in LogitProcessor.py sasrec.py; do
  if [ ! -f "$UPSTREAM/$f" ]; then
    echo "ERROR: missing upstream file: $UPSTREAM/$f"
    exit 1
  fi
  cp "$UPSTREAM/$f" "src/$f"
  echo "copied: src/$f"
done

echo "Done. Keep original copyright/license headers."
