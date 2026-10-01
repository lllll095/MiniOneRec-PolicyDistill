#!/bin/bash
set -euo pipefail

MODEL="${MODEL:?Please set MODEL}"
DATA_ROOT="${DATA_ROOT:?Please set DATA_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/minionerec_05b_grpo_seed42}"
SEED="${SEED:-42}"
PORT="${PORT:-29503}"

export NCCL_IB_DISABLE=1

CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch \
  --config_file ./configs/zero2_opt.yaml \
  --num_processes 2 \
  --main_process_port "$PORT" \
  src/grpo_train.py \
  --model_path "$MODEL" \
  --train_batch_size 64 \
  --eval_batch_size 128 \
  --num_train_epochs 2 \
  --gradient_accumulation_steps 8 \
  --train_file "$DATA_ROOT/train/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  --eval_file "$DATA_ROOT/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  --info_file "$DATA_ROOT/info/Industrial_and_Scientific_5_2016-10-2018-11.txt" \
  --category Industrial_and_Scientific \
  --sample_train False \
  --eval_step 0.0999 \
  --reward_type ranking \
  --num_generations 16 \
  --mask_all_zero False \
  --dynamic_sampling False \
  --sync_ref_model True \
  --beam_search True \
  --test_during_training False \
  --temperature 1.0 \
  --learning_rate 1e-5 \
  --add_gt False \
  --beta 1e-3 \
  --dapo False \
  --gspo False \
  --seed "$SEED" \
  --output_dir "$OUTPUT_DIR" \
  --wandb_project MiniOneRecPolicyDistill \
  --wandb_run_name minionerec_05b_grpo_seed42 \
  --sid_index_path "$DATA_ROOT/index/Industrial_and_Scientific.index.json" \
  --item_meta_path "$DATA_ROOT/index/Industrial_and_Scientific.item.json"
