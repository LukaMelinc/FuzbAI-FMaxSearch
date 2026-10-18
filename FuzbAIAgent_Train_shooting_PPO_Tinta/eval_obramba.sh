#!/usr/bin/bash

set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

python3 evaluate_defense.py \
    --checkpoint '/home/tinta/Desktop/FuzbAI-FMaxSearch/FuzbAIAgent_Train_shooting_PPO_Tinta/Final_models/Defence/#6.pth' \
    --agent-mode single_rod_ppo \
    --rod-id 2 \
    --episodes 250 \
    --x1 200 \
    --x2 545 \
    --failure-x 150 \
    --output report/eval_obramba_linearno_v1.csv \
    --gui \
    --episode-seconds 1.5

