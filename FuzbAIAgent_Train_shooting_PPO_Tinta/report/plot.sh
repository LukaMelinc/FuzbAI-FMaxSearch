#!/usr/bin/bash

set -e

cd -- "$(dirname -- "${BASE_SOURCE[0]}")"

python3 plot_summary_bar_charts.py \
  --summary-file /home/tinta/Desktop/FuzbAI-FMaxSearch/FuzbAIAgent_Train_shooting_PPO_Tinta/report/eval_imi+rl_strel_s_podajo.json \
  --summary-file /home/tinta/Desktop/FuzbAI-FMaxSearch/FuzbAIAgent_Train_shooting_PPO_Tinta/report/eval_imi_strel_s-podajo.json\
  --output-dir report/figures \
  --title "Zabeležene kontaktne sile ob strelu an gol - povprečna, najmanjša, najmočnejša"