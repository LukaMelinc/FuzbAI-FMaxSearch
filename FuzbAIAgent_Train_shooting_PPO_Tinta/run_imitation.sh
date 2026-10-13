#!/usr/bin/bash

set -e

cd -- "$(dirname -- "${BASE_SOURCE[0]}")"

python3 FuzbAISim.py \
  --mode state_imitation \
  --inference \
  --checkpoint "Final_models/Imitacija-strelNaGol/state_imitation_steps_823225.pth"