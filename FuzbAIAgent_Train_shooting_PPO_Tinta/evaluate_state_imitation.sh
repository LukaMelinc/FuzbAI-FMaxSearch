#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
checkpoint="${1:-}"
if [[ -z "$checkpoint" ]]; then
    checkpoint="$(python3 - <<'PY'
from pathlib import Path
files = list(Path('trained_models/table_imitation').glob('state_imitation_steps_*.pth'))
if files:
    print(max(files, key=lambda p: int(p.stem.rsplit('_', 1)[1])))
PY
)"
fi
if [[ -z "$checkpoint" ]]; then
    echo 'No table-imitation checkpoint found. Pass its path as the first argument.' >&2
    exit 1
fi
python3 FuzbAISim.py --mode state_imitation --inference --checkpoint "$checkpoint"
