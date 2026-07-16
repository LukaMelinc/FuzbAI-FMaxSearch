# State-only imitation sanity check

This experiment intentionally trains without scripted action labels. The expert
CSV contains the same 10-value observation used by `PPOAgent`. A transition
discriminator learns to separate expert `(state, next_state)` pairs from PPO
pairs, and its output becomes the PPO reward.

Run commands from `FuzbAIAgent_Train_shooting_PPO_Tinta` because the simulator
loads `geometry.json` and model assets relative to that directory.

## 1. Record scripted expert states

```bash
python3 FuzbAISim.py \
  --headless \
  --mode scripted_imitation \
  --episodes 200 \
  --ball-x-threshold 605 \
  --expert-csv imitation_data/threshold_expert.csv
```

The teacher rotates rod 4 in opposite directions on the two sides of the ball-x
threshold. The simulator randomizes ball state and records only observations.
Rows are grouped by episode so reset jumps are excluded from expert transitions.

## 2. Train state-only imitation PPO

```bash
python3 FuzbAISim.py \
  --headless \
  --mode state_imitation \
  --expert-csv imitation_data/threshold_expert.csv \
  --ball-x-threshold 605 \
  --steps-per-env 512 \
  --save-model-every 50
```

During training, expect both `[Discriminator]` and `[Training]` lines. A useful
discriminator should initially separate expert and agent transitions. Accuracy
remaining exactly 100% for a long time means PPO is not approaching the expert;
accuracy near 50% from the beginning can mean the discriminator is too weak.

Checkpoints are written under `trained_models/state_imitation` at the configured
episode interval. Stop after a checkpoint has been reported in the log.

For a quick pipeline test, use `--steps-per-env 32`. Use a larger buffer for an
actual learning run.

## 3. Evaluate a checkpoint

```bash
python3 FuzbAISim.py \
  --mode state_imitation \
  --inference \
  --expert-csv imitation_data/threshold_expert.csv \
  --checkpoint trained_models/state_imitation/state_imitation_steps_STEP.pth
```

Evaluation uses the actor mean deterministically. Verify behavior on ball spawns
from both sides of the threshold; discriminator loss alone is not a success
metric.

## Actor transfer to another PPO agent

The checkpoint contains a separate `actor_state_dict`. For an agent with the
same observation, action, and actor architecture:

```python
checkpoint = torch.load(path, map_location=ppo_agent.device)
ppo_agent.ac.actor.load_state_dict(checkpoint["actor_state_dict"])
```

Initialize the new critic separately when changing the reward or task. Do not
transfer the discriminator unless continuing the same imitation experiment.
