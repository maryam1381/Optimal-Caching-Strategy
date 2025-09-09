#!/usr/bin/env bash
# experiments/run_train.sh
#
# Small wrapper to run the training script with sensible defaults and log output.
#
# Usage:
#   ./experiments/run_train.sh [SAVE_DIR]
#
# Example:
#   ./experiments/run_train.sh experiments/checkpoints/run01
#
# The script:
#  - creates the requested save directory (default experiments/checkpoints/run_default_<ts>)
#  - invokes main.py with a set of CLI options
#  - writes stdout/stderr to a timestamped .log file inside the save directory

set -euo pipefail

# Optional first argument: save directory
SAVE_DIR=${1:-experiments/checkpoints/run_default_$(date +%Y%m%d_%H%M%S)}
mkdir -p "${SAVE_DIR}"
LOGFILE="${SAVE_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

# You can tweak these hyperparameters below:
SEED=42
EPOCHS=100
DATASET_SIZE=1000
N_POOL=2000
BATCH_SIZE=16
L=200
LR=1e-3
GAMMA=0.05
S_CACHE=10
M=100
W=200
LAMBDA_TRUE=2.5

echo "Running training. Saving to ${SAVE_DIR}. Log: ${LOGFILE}"
echo "Python executable: $(which python3)"

python3 main.py \
    --save-dir "${SAVE_DIR}" \
    --seed ${SEED} \
    --epochs ${EPOCHS} \
    --dataset-size ${DATASET_SIZE} \
    --N-pool ${N_POOL} \
    --batch-size ${BATCH_SIZE} \
    --L ${L} \
    --lr ${LR} \
    --gamma ${GAMMA} \
    --S-cache ${S_CACHE} \
    --M ${M} \
    --W ${W} \
    --lambda-true ${LAMBDA_TRUE} \
    2>&1 | tee "${LOGFILE}"
