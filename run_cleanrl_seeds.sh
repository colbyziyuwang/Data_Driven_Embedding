#!/bin/bash

# Usage: ./run.sh <ENV_NAME> <ALGO_NAME>
# Example: ./run.sh LunarLander-v2 DQN

ENV_NAME="$1"
ALGO_NAME="$2"
EXP_NAME="clean_rl_dqn"
SEEDS=(1 2 3 4 5)

if [ -z "$ENV_NAME" ] || [ -z "$ALGO_NAME" ]; then
    echo "Error: Missing arguments."
    echo "Usage: ./run.sh <ENV_NAME> <ALGO_NAME>"
    exit 1
fi

LOG_DIR="logs_${ENV_NAME}_${ALGO_NAME}"
mkdir -p "$LOG_DIR"

for SEED in "${SEEDS[@]}"; do
    echo "Running $EXP_NAME on $ENV_NAME with algo $ALGO_NAME and seed $SEED"
    python clean_rl_dqn.py \
        --env-id "$ENV_NAME" \
        --algo_name "$ALGO_NAME" \
        --seed "$SEED" \
        > "$LOG_DIR/seed_${SEED}.log" 2>&1 &
done

echo "All jobs launched! Logs are saved under $LOG_DIR/"

