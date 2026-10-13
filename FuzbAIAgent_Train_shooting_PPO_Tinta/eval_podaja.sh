#!/usr/bin/bash

set -e

cd -- "$(dirname -- "${BASE_SOURCE[0]}")"

python3 evaluate_passing.py \
    --checkpoint '/home/tinta/Desktop/FuzbAI-FMaxSearch/FuzbAIAgent_Train_shooting_PPO_Tinta/Final_models/Passing/PASSING_FINAL.pth' \
    --episodes 250 \
    --x1 800 \
    --x2 920 \
    --output report/eval_passing.csv \
    --overshoot-x 980 \
    --episode-seconds 5 \
    --failure-x 330 \
    --stop-speed 0.05 \
    --stop-duration 0.5 \
    --spawn-x1 0.660 \
    --spawn-x2 0.661 \
    --gui




