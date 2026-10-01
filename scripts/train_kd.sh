#!/bin/bash
set -euo pipefail

STUDENT_MODEL="${STUDENT_MODEL:?Please set STUDENT_MODEL}"
TEACHER_MODEL="${TEACHER_MODEL:?Please set TEACHER_MODEL}"
DATA_ROOT="${DATA_ROOT:?Please set DATA_ROOT}"

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/minionerec_05b_gold_kd_rkl1649_tau1_seed42}"
MAX_STEPS="${MAX_STEPS:-1649}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-5}"
SEED="${SEED:-42}"

python -u -m src.policy_distill \
  --state_mode gold \
  --student "$STUDENT_MODEL" \
  --teacher "$TEACHER_MODEL" \
  --train_file "$DATA_ROOT/train/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  --index_file "$DATA_ROOT/index/Industrial_and_Scientific.index.json" \
  --item_file "$DATA_ROOT/index/Industrial_and_Scientific.item.json" \
  --max_steps "$MAX_STEPS" \
  --batch_size "$BATCH_SIZE" \
  --learning_rate "$LR" \
  --distill_temperature 1.0 \
  --seed "$SEED" \
  --output_dir "$OUTPUT_DIR" \
  --log_every 100 \
  --save_final
