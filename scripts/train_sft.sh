#!/bin/bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:?Please set BASE_MODEL}"
DATA_ROOT="${DATA_ROOT:?Please set DATA_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/minionerec_05b_sft_seed42}"
SEED="${SEED:-42}"

export NCCL_IB_DISABLE=1

torchrun --nproc_per_node 2 \
  src/sft_train.py \
  --base_model "$BASE_MODEL" \
  --batch_size 1024 \
  --micro_batch_size 16 \
  --num_epochs 10 \
  --learning_rate 3e-4 \
  --cutoff_len 512 \
  --train_file "$DATA_ROOT/train/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  --eval_file "$DATA_ROOT/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  --output_dir "$OUTPUT_DIR" \
  --wandb_project MiniOneRecPolicyDistill \
  --wandb_run_name minionerec_05b_sft_seed42 \
  --category Industrial_and_Scientific \
  --train_from_scratch False \
  --seed "$SEED" \
  --sid_index_path "$DATA_ROOT/index/Industrial_and_Scientific.index.json" \
  --item_meta_path "$DATA_ROOT/index/Industrial_and_Scientific.item.json" \
  --freeze_LLM False
