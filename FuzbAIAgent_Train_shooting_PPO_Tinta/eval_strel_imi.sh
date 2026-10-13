#!/usr/bin/bash
set -e  # Prekini skripto ob napaki

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

python3 evaluate_shooting.py \
    --checkpoint '/home/tinta/Desktop/FuzbAI-FMaxSearch/FuzbAIAgent_Train_shooting_PPO_Tinta/Final_models/Imitacija-strelNaGol/state_imitation_steps_823225.pth' \
    --episodes 250 \
    --episode-seconds 7 \
    --negative-speed 0.05 \
    --negative-duration 5 \
    --output report/eval_strel_imi.csv \
    --gui \
    --spawn-x1 0.950 \
    --spawn-x2 0.951
