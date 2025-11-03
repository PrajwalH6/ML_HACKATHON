#!/bin/bash

set -e

echo "=== Hangman ML Agent - Full Pipeline ==="

# Step 1: Preprocess data
echo "[1/4] Preprocessing data..."
bash scripts/preprocess_data.sh

# Step 2: Train HMM
echo "[2/4] Training HMM emissions..."
bash scripts/train_hmm.sh

# Step 3: Train RL agent
echo "[3/4] Training RL agent..."
bash scripts/train_rl.sh

# Step 4: Evaluate
echo "[4/4] Running evaluation..."
bash scripts/evaluate.sh

echo "=== Pipeline Complete ==="
echo "Check reports/ for results"
