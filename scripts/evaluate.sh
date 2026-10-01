#!/bin/bash
set -euo pipefail

MODEL="${MODEL:?Please set MODEL}"
DATA_ROOT="${DATA_ROOT:?Please set DATA_ROOT}"
RESULT_JSON="${RESULT_JSON:-./results/eval_predictions.json}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BEAM="${BEAM:-50}"
GPU="${GPU:-0}"

TEST="$DATA_ROOT/test/Industrial_and_Scientific_5_2016-10-2018-11.csv"
TRAIN="$DATA_ROOT/train/Industrial_and_Scientific_5_2016-10-2018-11.csv"
INFO="$DATA_ROOT/info/Industrial_and_Scientific_5_2016-10-2018-11.txt"

mkdir -p "$(dirname "$RESULT_JSON")"

echo "============================================================"
echo "MiniOneRec constrained beam evaluation"
echo "Model : $MODEL"
echo "Test  : $TEST"
echo "Beam  : $BEAM"
echo "GPU   : $GPU"
echo "============================================================"

PYTHONPATH=src CUDA_VISIBLE_DEVICES="$GPU" \
python -u src/evaluate.py \
  --base_model "$MODEL" \
  --train_file "$TRAIN" \
  --info_file "$INFO" \
  --category Industrial_and_Scientific \
  --test_data_path "$TEST" \
  --result_json_data "$RESULT_JSON" \
  --batch_size "$BATCH_SIZE" \
  --length_penalty 0.0 \
  --max_new_tokens 256 \
  --num_beams "$BEAM"

echo
echo "============================================================"
echo "METRICS"
echo "============================================================"

PYTHONPATH=src python -u src/calc.py \
  --path "$RESULT_JSON" \
  --item_path "$INFO"
