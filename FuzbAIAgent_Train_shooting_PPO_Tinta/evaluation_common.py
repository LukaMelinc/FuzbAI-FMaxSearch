"""Shooting evaluation: physical X in mm, velocity in m/s, simulation time in s."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import time


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.rows = []
        self.completed = False
        self.active = False

    def begin_episode(self, t, score):
        self.start = float(t)
        self.previous_score = list(score)
        self.negative_since = None
        self.last_sample_time = None
        self.peak_speed = 0.0
        self.kicks = []
        self.last_kick_force = None
        self.active = not self.completed

    def observe(self, camera):
        if not self.active or self.completed:
            return False
        t = float(camera['simulation_time'])
        if t == self.last_sample_time:
            return False
        cd = camera['camData'][0]
        x, vx, vy = (float(cd[k]) for k in ('ball_x', 'ball_vx', 'ball_vy'))
        if not all(math.isfinite(v) for v in (t, x, vx, vy)):
            raise ValueError('Non-finite evaluation sample')
        if t < self.start or (self.last_sample_time is not None and t < self.last_sample_time):
            raise ValueError('Evaluation time moved backwards')
        self.last_sample_time = t
        score = list(camera['score'])
        goal = score[0] > self.previous_score[0]
        own_goal = score[1] > self.previous_score[1]
        self.previous_score = score
        speed = math.hypot(vx, vy)
        self.peak_speed = max(self.peak_speed, speed)
        if camera.get('ball_kicked', False):
            self.kicks.append(dict(time_s=t-self.start, speed_m_s=speed,
                                   force_n=camera.get('kick_normal_force')))
            self.last_kick_force = None
        force = camera.get('kick_normal_force')
        if force is not None:
            force = float(force)
            if not math.isfinite(force) or force < 0:
                raise ValueError('Kick force must be finite and non-negative')
            if self.kicks:
                # A contact can continue into subsequent control intervals even
                # when the simulator suppresses another event during cooldown.
                self.last_kick_force = max(self.last_kick_force or 0.0, force)
                self.kicks[-1]['force_n'] = self.last_kick_force
        if vx < -self.args.negative_speed:
            if self.negative_since is None:
                self.negative_since = t
        else:
            self.negative_since = None
        outcome = reason = None
        # A goal wins over failure criteria on the same sample.
        if goal:
            outcome, reason = 'success', 'goal'
        elif own_goal:
            outcome, reason = 'missed-shot', 'opponent_goal'
        elif self.negative_since is not None and t-self.negative_since + 1e-9 >= self.args.negative_duration:
            outcome, reason = 'missed-shot', 'negative_velocity'
        elif x < self.args.failure_x:
            outcome, reason = 'agent-missed-ball', 'behind_x_threshold'
        elif camera.get('terminated_by_timeout', False) or t-self.start + 1e-9 >= self.args.episode_seconds:
            outcome, reason = 'timeout', 'episode_timeout'
        elif any(camera.get(k, False) for k in ('end_episode', 'terminated_by_kick', 'terminated_by_x_threshold')):
            outcome, reason = 'failure', 'episode_end_without_success'
        if outcome is None:
            return False
        self.rows.append(dict(episode=len(self.rows)+1, outcome=outcome, reason=reason,
                              duration_s=t-self.start, final_x_mm=x, final_vx_m_s=vx,
                              kick_count=len(self.kicks), peak_ball_speed_m_s=self.peak_speed,
                              goal_kick_force_n=self.last_kick_force if goal else None,
                              kicks_json=json.dumps(self.kicks)))
        self.active = False
        self.completed = len(self.rows) >= self.args.episodes
        self.write()
        force_text = (f', kick force={self.last_kick_force:.3f} N'
                      if goal and self.last_kick_force is not None else '')
        print(f'[Evaluation] {len(self.rows)}/{self.args.episodes}: {outcome} ({reason}){force_text}')
        return True

    def write(self):
        path = Path(self.args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.rows:
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
        counts = Counter(row['outcome'] for row in self.rows)
        goal_forces = [row['goal_kick_force_n'] for row in self.rows
                       if row['goal_kick_force_n'] is not None]
        summary = dict(episodes=len(self.rows), requested_episodes=self.args.episodes,
                       completed=self.completed,
                       outcomes={key: counts[key] for key in
                                 ('success', 'missed-shot', 'agent-missed-ball', 'timeout', 'failure')},
                       success_rate=counts['success']/len(self.rows) if self.rows else 0,
                       goal_kick_force_n=dict(count=len(goal_forces),
                                             mean=sum(goal_forces)/len(goal_forces) if goal_forces else None,
                                             min=min(goal_forces) if goal_forces else None,
                                             max=max(goal_forces) if goal_forces else None),
                       parameters=vars(self.args))
        path.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Evaluate imitation shooting policies in deterministic inference')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--agent-mode', '--mode', dest='agent_mode',
                        choices=('state_imitation', 'hybrid_imitation_ppo', 'imitation_finetune_ppo'),
                        help='Defaults to checkpoint agent_type, or state_imitation')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--episode-seconds', type=float, default=5)
    parser.add_argument('--negative-speed', type=float, default=0.05, help='X: positive magnitude of negative vx, in m/s')
    parser.add_argument('--negative-duration', type=float, default=0.3, help='Y: continuous duration in simulation seconds')
    parser.add_argument('--failure-x', type=float, required=True, help='Ball behind this X (ball_x < threshold), in camera/geometry mm')
    parser.add_argument('--output', default='report/evaluation_imitation.csv')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gui', action='store_true')
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error('episodes must be positive')
    for name in ('episode_seconds', 'negative_speed', 'negative_duration', 'failure_x'):
        if not math.isfinite(getattr(args, name)):
            parser.error(f'{name} must be finite')
    if args.episode_seconds <= 0 or args.negative_speed < 0 or args.negative_duration <= 0:
        parser.error('Require positive episode-seconds and negative-duration, and negative-speed >= 0')
    if not Path(args.checkpoint).is_file():
        parser.error('Checkpoint does not exist')
    import random
    import numpy as np
    import torch
    import pybullet as p
    from recorded_expert import RecordingConfig
    from FuzbAISim import FuzbAISim
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    saved_mode = checkpoint.get('agent_type')
    args.agent_mode = args.agent_mode or saved_mode or 'state_imitation'
    if args.agent_mode not in ('state_imitation', 'hybrid_imitation_ppo', 'imitation_finetune_ppo'):
        parser.error('Unsupported checkpoint agent_type; specify --agent-mode')
    if saved_mode == 'imitation_finetune_ppo' and args.agent_mode != saved_mode:
        parser.error('Fine-tuned checkpoints require --agent-mode imitation_finetune_ppo')
    config = RecordingConfig(**checkpoint.get('recording_config', {}))
    kwargs = dict(recording_config=config, inference=True, training_enabeled=False)
    if args.agent_mode == 'imitation_finetune_ppo':
        kwargs['resume_checkpoint'] = args.checkpoint
    else:
        kwargs.update(model_save_path=args.checkpoint, load_model=True, expert_data=None,
                      history_length=checkpoint.get('history_length', 5))
    evaluator = Evaluator(args)
    sim = None
    try:
        sim = FuzbAISim(agent1_mode=args.agent_mode, agent1_kwargs=kwargs,
                       evaluator=evaluator, kick_observed_rod_id=6, render_gui=args.gui)
        sim.debug_print_red_kicks = False
        sim.run()
        while sim.isRunning:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print('[Evaluation] Interrupted; saving completed episodes')
    finally:
        if sim is not None:
            sim.stop()
            if getattr(sim, 'simThread', None) is not None:
                sim.simThread.join()
        evaluator.write()
        if p.isConnected():
            p.disconnect()
    if sim is not None and sim.run_error is not None:
        raise RuntimeError('Evaluation failed') from sim.run_error
