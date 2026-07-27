#!/usr/bin/env bash
set -euo pipefail

: "${DATA_DIR:?Set DATA_DIR to a NovoBench dataset directory}"
: "${UNLABELED_DIR:?Set UNLABELED_DIR to a DarkSpec array store}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for checkpoints and logs}"

seminovo-train-semi \
  --config "${CONFIG:-configs/seminovo.yaml}" \
  --data-dir "$DATA_DIR" \
  --unlabeled-dir "$UNLABELED_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "${BATCH_SIZE:-512}" \
  --unlabeled-batch-size "${UNLABELED_BATCH_SIZE:-512}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-2048}" \
  --num-workers "${NUM_WORKERS:-16}" \
  --epochs "${EPOCHS:-6}" \
  --confidence-threshold "${CONFIDENCE_THRESHOLD:-0.9}" \
  --lambda-u "${LAMBDA_U:-1.0}" \
  "$@"
