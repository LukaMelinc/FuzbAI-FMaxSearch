# Imitation shooting evaluation

Run from this directory:

```bash
python3 evaluate_shooting.py \
  --checkpoint "Final_models/Imitacija+RLfinetune-strel/imitation_finetune_ppo_steps_892513.pth" \
  --episodes 100 \
  --negative-speed 0.5 \
  --negative-duration 0.3 \
  --failure-x 750 \
  --episode-seconds 5 \
  --output report/imitation_shooting.csv
```

The thresholds above are examples: choose values appropriate to your experiment.
Add `--gui` to watch. Headless execution is the default. `--seed` defaults to 42.
Use the imitation-only checkpoint path to evaluate that model instead. Fine-tuned
checkpoints select `imitation_finetune_ppo` automatically; otherwise the default
is `state_imitation`. `--agent-mode hybrid_imitation_ppo` is also supported.
Calibration and recorded starting states come from the checkpoint. Each episode
samples a recorded start using the seeded random generator. Training is disabled
and policy actions are deterministic.

## Criteria, in priority order

1. `success`: the agent's score increases (red/right goal).
2. `missed-shot`: an opponent goal, or ball X velocity is strictly less than
   `-negative-speed` continuously for at least `negative-duration` seconds.
   Any sample at or above the velocity threshold clears this timer.
3. `agent-missed-ball`: ball X is strictly below `failure-x` without a goal.
4. `timeout`: the configured episode time expires without any of these outcomes.
5. `failure`: another simulator terminal event without a classified outcome.

The first matched criterion ends and resets the episode. A goal takes priority
when multiple criteria match the same sample. Duration uses simulation time,
not wall time. Criteria are sampled at the checkpoint's control interval, so
crossings/timers have that temporal resolution.

X uses the simulator's player-1 geometry coordinates in millimetres:
`ball_x = 1000 * world_x_metres - 115`. Increasing X points toward the agent's
target goal; "behind" means smaller X. Velocity uses m/s. Evaluation uses physical
coordinates without camera noise or parallax. Set `failure-x` below the desired
boundary; if a recorded start is already below it, that episode immediately
qualifies as `agent-missed-ball` at its first sample.

## Reports

The CSV contains one row per completed episode, its outcome/reason, duration,
final position/velocity, peak speed, and contact events. The companion
`.summary.json` records outcome counts, success rate, parameters, and whether the
requested number of episodes completed. Ctrl+C saves completed episodes; an
interrupted partial episode is omitted. Model files are not written.

`goal_kick_force_n` reports the peak simulated normal contact force in newtons
for the last detected kick by the controlled rod before a successful goal.
Simultaneous player/ball contact forces are summed at each physics tick; the
peak is retained between control observations, including during kick cooldown.
Continuing contact measurements update the last kick until the next kick event.
This measures contact force at the player, rather than ball speed at the goal.
It is a simulation measurement, not a calibrated real-world sensor reading.
Failed episodes and goals without a measured kick leave this CSV field empty.
The console prints the force for measured successful shots, and the summary's
`goal_kick_force_n` contains `count`, `mean`, `min`, and `max` for those shots
only (statistics are null when none were measured). Existing outcome criteria
and rewards are unchanged.

The observer/reset/report approach follows `evaluate_shooting.py` and
`evaluation_common.py` from the repository's `single_rod_training` branch:
https://github.com/LukaMelinc/FuzbAI-FMaxSearch/tree/single_rod_training/FuzbAIAgent_Train_shooting_PPO_Tinta

## Validation

```bash
python3 -m unittest discover -s tests -v
```
