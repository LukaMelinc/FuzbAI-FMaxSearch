#!/usr/bin/bash
set -e  # Prekini skripto ob napaki

#cd --"$(dirname -- "${BASH_SOURCE[0]}")"   # Če je FuzbAISim.py v istem direktoriju kot shell skirpta
#cd --"$(dirname -- "${BASH_SOURCE[0]}")/.."   # Če je FuzbAISim.py en nivo pod shell skripto
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

python3 evaluate_shooting.py \
    --checkpoint 'Final_models/Shoting/#3-2.pth' \
    --episodes 250 \
    --episode-seconds 5 \
    --negative-speed 0.05 \
    --negative-duration 0.3 \
    --output report/eval_strel-sirina3.csv \
    --gui \
    --spawn-x1 0.969 \
    --spawn-x2 0.9691
