#!/bin/bash
set -euo pipefail

UPSTREAM="${1:-../MiniOneRec}"

if [ ! -d "$UPSTREAM" ]; then
  echo "ERROR: upstream MiniOneRec directory not found: $UPSTREAM"
  exit 1
fi

echo "Using upstream MiniOneRec: $UPSTREAM"

copy_required() {
  local src="$1"
  local dst="$2"
  if [ ! -f "$src" ]; then
    echo "ERROR: missing required upstream file: $src"
    exit 1
  fi
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
  echo "copied: $dst"
}

# Apache-2.0 license and upstream-derived implementation files.
copy_required "$UPSTREAM/LICENSE" "LICENSE"
copy_required "$UPSTREAM/data.py" "src/data.py"
copy_required "$UPSTREAM/evaluate.py" "src/evaluate.py"
copy_required "$UPSTREAM/calc.py" "src/calc.py"
copy_required "$UPSTREAM/minionerec_trainer.py" "src/minionerec_trainer.py"
copy_required "$UPSTREAM/rl.py" "src/grpo_train.py"
copy_required "$UPSTREAM/sft.py" "src/sft_train.py"
copy_required "$UPSTREAM/LogitProcessor.py" "src/LogitProcessor.py"
copy_required "$UPSTREAM/sasrec.py" "src/sasrec.py"
copy_required "$UPSTREAM/config/zero2_opt.yaml" "configs/zero2_opt.yaml"

chmod +x scripts/*.sh

echo
echo "Running syntax checks..."

python -m py_compile \
  src/policy_distill.py \
  diagnostics/analyze_reward_sparsity.py \
  diagnostics/summarize_results.py \
  scripts/check_repo.py

bash -n scripts/train_sft.sh
bash -n scripts/train_grpo.sh
bash -n scripts/train_kd.sh
bash -n scripts/train_opd.sh
bash -n scripts/evaluate.sh

echo
python diagnostics/summarize_results.py

echo
python scripts/check_repo.py

echo
echo "Repository finalization complete."
