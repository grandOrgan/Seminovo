#!/usr/bin/env bash
set -euo pipefail

: "${DATA_DIR:?Set DATA_DIR to a NovoBench dataset directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for checkpoints and logs}"

seminovo-train \
  --config "${CONFIG:-configs/seminovo.yaml}" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "${BATCH_SIZE:-32}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-512}" \
  --num-workers "${NUM_WORKERS:-16}" \
  --epochs "${EPOCHS:-30}" \
  "$@"
