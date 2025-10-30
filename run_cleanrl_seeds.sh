#!/bin/bash
ENV_NAME="CartPole-v1"
EXP_NAME="clean_rl_dqn"
SEEDS=(1 2 3 4 5)

for SEED in "${SEEDS[@]}"; do
    echo "=============================="
    echo "Running $EXP_NAME on $ENV_NAME with seed $SEED"
    echo "=============================="

    python clean_rl_dqn.py \
        --env-id "$ENV_NAME" \
        --seed "$SEED"
done
