#!/usr/bin/bash

set -e

cd -- "$(dirname -- "${BASE_SOURCE[0]}")"

python3 evaluate_defense.py \
    --checkpoint '/home/tinta/Desktop/FuzbAI-FMaxSearch/FuzbAIAgent_Train_shooting_PPO_Tinta/Final_models/Defence/#6.pth' \
    --episodes 250 \
    --x1 215 \
    --x2 245 \
    --failure-x 150 \
    --output report/eval_obramba_linearno_v1.csv \
    --gui \

