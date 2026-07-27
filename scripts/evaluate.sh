#!/usr/bin/env bash
set -euo pipefail

: "${DATA_DIR:?Set DATA_DIR to a NovoBench dataset directory}"
: "${CHECKPOINT:?Set CHECKPOINT to a trained SemiNovo checkpoint}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for predictions and metrics}"

seminovo-evaluate \
  --config "${CONFIG:-configs/seminovo.yaml}" \
  --data-dir "$DATA_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "${BATCH_SIZE:-512}" \
  --num-workers "${NUM_WORKERS:-16}" \
  --beams "${BEAMS:-20}" \
  --decode-strategy "${DECODE_STRATEGY:-casanovo}" \
  "$@"
