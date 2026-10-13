#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
#/home/tinta/Desktop/FuzbAI-FMaxSearch/FuzbAIAgent_Train_shooting_PPO_Tinta/Final_models/Imitacija-strelNaGol/state_imitation_steps_823225.pth
#/home/tinta/Desktop/FuzbAI-FMaxSearch/FuzbAIAgent_Train_shooting_PPO_Tinta/Final_models/Imitacija+RLfinetune-strel/imitation_finetune_ppo_steps_892513.pth
python3 evaluate_shooting.py \
  --checkpoint "/home/tinta/Desktop/FuzbAI-FMaxSearch/FuzbAIAgent_Train_shooting_PPO_Tinta/Final_models/Imitacija+RLfinetune-strel/imitation_finetune_ppo_steps_892513.pth" \
  --episodes 250 \
  --negative-speed 0.5 \
  --negative-duration 0.3 \
  --failure-x 750 \
  --episode-seconds 5 \
  --output report/imitation_shooting_rl_Posttraining.csv