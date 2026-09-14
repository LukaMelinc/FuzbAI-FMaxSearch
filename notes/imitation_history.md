# Imitation with rod motion and observation history

`state_imitation` now uses 10 features per observation: the existing eight
features plus rod angular and lateral velocity. The actor and critic receive
the last five observations, oldest first, flattened into 50 values. At the
configured 0.05-second control interval this spans 0.20 seconds.

Both recordings and simulator observations use the same causal finite
differences. Angular displacement uses the shortest signed angle difference
to handle wraparound; this assumes less than pi radians of rotation between
samples. Angular velocity is expressed in radians/second divided by pi.
Lateral velocity is expressed in fractions of the rod's calibrated travel per
second. No additional clipping or smoothing is applied to either velocity.

At an episode start, both velocities are zero and the initial observation is
repeated five times. Subsequent observations shift the history by one frame.
Terminal observations finish the previous trajectory before history is cleared,
including during inference. Each continuous recording segment starts with zero
velocity; differences and transitions never bridge tracking gaps. Existing
25% start/end trimming is unchanged.

The discriminator still receives adjacent observations (20 values), not policy
histories or motor-action labels. A longer discriminator motion window is a
separate future experiment. PPO stores the exact 50-value history used for each
action, including at rollout buffer boundaries.

## Training

From the repository root:

```bash
cd FuzbAIAgent_Train_shooting_PPO_Tinta
python3 FuzbAISim.py \
  --mode state_imitation \
  --headless \
  --expert-data kick_imitation_smol \
  --recording-config recording_config.json \
  --steps-per-env 512 \
  --max-agent-steps 0 \
  --output-dir trained_models/table_imitation_history
```

Interrupt with Ctrl+C to stop and save. The first rollout bootstraps the
discriminator; policy updates begin with subsequent rollouts. A fresh run
limited to 300 observations still cannot train a policy with a 512-entry buffer.

## Inference and Compatibility

Use the actual checkpoint filename saved by the new run:

```bash
python3 FuzbAISim.py \
  --mode state_imitation \
  --inference \
  --checkpoint trained_models/table_imitation_history/state_imitation_steps_REPLACE_WITH_SAVED_STEP.pth
```

New checkpoints include feature names, velocity normalization/version, and
history length. Old eight-input imitation checkpoints cannot be resumed or
evaluated by this expanded architecture; start fresh training. Existing shooting
and passing observation formats are unchanged. Transferring the new imitation
actor to an eight-input task policy also requires matching that task policy's
input architecture and history processing first.

## Verification

```bash
python3 -m unittest discover -s tests -p 'test_imitation_history.py' -v
python3 -m unittest discover -s tests -p 'test_recording_trim.py' -v
```

These tests cover angle wraparound, both velocity directions, expert/live
feature agreement, history order and padding, PPO input storage, terminal
resets in training/inference, bootstrap and policy updates, and checkpoint
compatibility.
