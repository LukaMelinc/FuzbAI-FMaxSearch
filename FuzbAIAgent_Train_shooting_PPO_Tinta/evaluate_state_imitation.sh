#!/usr/bin/env bash
set -euo pipefail

checkpoint_dir="trained_models/state_imitation"
expert_csv="imitation_data/threshold_expert.csv"

checkpoint="$(find "$checkpoint_dir" -maxdepth 1 -type f -name 'state_imitation_steps_*.pth' -printf '%f\n' \
    | sort -V \
    | tail -n 1)"

if [[ -z "$checkpoint" ]]; then
    echo "No state-imitation checkpoint found in $checkpoint_dir" >&2
    exit 1
fi

echo "Evaluating $checkpoint_dir/$checkpoint"
python3 FuzbAISim.py \
    --mode state_imitation \
    --inference \
    --expert-csv "$expert_csv" \
    --ball-x-threshold 605 \
    --checkpoint "$checkpoint_dir/$checkpoint"
