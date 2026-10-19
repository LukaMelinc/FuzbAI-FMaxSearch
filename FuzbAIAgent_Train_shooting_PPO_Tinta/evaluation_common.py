"""Inference observers; coordinates in mm, velocity m/s, simulation time seconds."""
import argparse
import csv
import json
import math
from pathlib import Path
import time


class Evaluator:
    def __init__(self, task, args):
        self.task, self.args = task, args
        self.rows = []
        self.active = False
        self.completed = False
        self.previous_score = None
        self.last_sample_time = None
        self.awaiting_reset = False
        self.last_terminal_time = None

    def observe(self, camera):
        if self.completed:
            return False
        t = float(camera['simulation_time'])
        if t == self.last_sample_time:
            return False
        self.last_sample_time = t
        terminal = any(camera.get(k, False) for k in (
            'end_episode', 'terminated_by_timeout', 'terminated_by_x_threshold', 'terminated_by_kick'))
        if self.awaiting_reset:
            if terminal:
                return False
            self.awaiting_reset = False
        cd = camera['camData'][0]
        x, vx, vy = (float(cd[k]) for k in ('ball_x', 'ball_vx', 'ball_vy'))
        if not all(math.isfinite(v) for v in (t, x, vx, vy)):
            raise ValueError('Non-finite evaluation sample')
        score = list(camera.get('score', [0, 0]))
        if not self.active:
            self.active = True
            self.start = t
            self.contact = False
            self.negative_since = self.stopped_since = None
            self.kicks = []
            self.peak_speed = 0.0
            if self.previous_score is None:
                self.previous_score = score
        goal = score[0] > self.previous_score[0]
        own_goal = score[1] > self.previous_score[1]
        self.previous_score = score
        self.last_x = x
        speed = math.hypot(vx, vy)
        self.peak_speed = max(self.peak_speed, speed)
        if camera.get('ball_kicked', False):
            self.contact = True
            self.kicks.append(dict(time_s=t-self.start,
                                   force_n=float(camera.get('kick_normal_force', 0)),
                                   speed_m_s=speed, vx_m_s=vx))
        a = self.args
        outcome = reason = None
        if self.task == 'shooting':
            self.negative_since = (t if self.negative_since is None else self.negative_since) if vx < -a.negative_speed else None
            if goal:
                outcome, reason = 'success', 'goal'
            elif own_goal:
                outcome, reason = 'failure', 'opponent_goal'
            elif self.negative_since is not None and t-self.negative_since + 1e-9 >= a.negative_duration:
                outcome, reason = 'failure', 'negative_velocity'
        elif self.task == 'defense':
            if x < a.failure_x:
                outcome = 'partial_success' if self.contact else 'failure'
                reason = 'escaped_after_contact' if self.contact else 'escaped_without_contact'
            elif terminal:
                in_band = a.x1 <= x <= a.x2
                outcome = 'success' if in_band else 'failure'
                reason = 'ended_in_band' if in_band else 'ended_outside_band'
        else:
            if x < a.failure_x:
                outcome, reason = 'failure', 'wrong_direction'
            elif x > a.overshoot_x:
                outcome, reason = 'failure', 'failed_to_stop'
            else:
                stopped = a.x1 <= x <= a.x2 and speed <= a.stop_speed
                self.stopped_since = (t if self.stopped_since is None else self.stopped_since) if stopped else None
                if self.stopped_since is not None and t-self.stopped_since + 1e-9 >= a.stop_duration:
                    outcome, reason = 'success', 'stopped_in_band'
        if outcome is None and terminal:
            outcome, reason = 'failure', 'episode_end_without_success'
        if outcome:
            self.finish(outcome, reason, t)
            return True
        return False

    def finish(self, outcome, reason, t):
        row = dict(episode=len(self.rows)+1, outcome=outcome, reason=reason,
                   duration_s=t-self.start, final_x_mm=self.last_x,
                   rod_contact=self.contact, kick_count=len(self.kicks),
                   max_kick_force_n=max((k['force_n'] for k in self.kicks), default=0),
                   max_kick_speed_m_s=max((k['speed_m_s'] for k in self.kicks), default=0),
                   peak_ball_speed_m_s=self.peak_speed, kicks_json=json.dumps(self.kicks))
        self.rows.append(row)
        self.active = False
        self.awaiting_reset = True
        self.completed = len(self.rows) >= self.args.episodes
        self.write()
        print(f"[Evaluation] {row['episode']}/{self.args.episodes}: {outcome} ({reason})")

    def finish_external(self):
        if self.active:
            self.finish('failure', 'agent_ended_without_success', self.last_sample_time)

    def write(self):
        path = Path(self.args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.rows:
            with path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
        counts = {key: sum(r['outcome'] == key for r in self.rows)
                  for key in ('success', 'partial_success', 'failure')}
        summary = dict(task=self.task, episodes=len(self.rows), **counts,
                       success_rate=counts['success']/len(self.rows) if self.rows else 0,
                       parameters=vars(self.args))
        path.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2))


def main(task):
    p = argparse.ArgumentParser(description=f'Evaluate {task} during deterministic inference')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--episode-seconds', type=float, default=5)
    p.add_argument('--episode-pause', type=float, default=0,
                   help='Wall-clock seconds to pause before resetting between episodes')
    p.add_argument('--output', default=f'report/evaluation_{task}.csv')
    p.add_argument('--gui', action='store_true')
    p.add_argument('--rod-id', type=int, default=4 if task == 'passing' else 6)
    p.add_argument('--observed-rod-id', type=int, default=7 if task == 'shooting' else 5)
    p.add_argument('--receiver-rod-id', type=int, default=6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--agent-kwargs', default='{}', help='Extra constructor kwargs as JSON')
    if task != 'passing':
        p.add_argument('--agent-mode', choices=('single_rod_ppo', 'two_rod_ppo'),
                       default='two_rod_ppo', help='Agent architecture used to train the checkpoint')
    p.add_argument('--spawn-x1', type=float, help='Ball spawn world X minimum in metres')
    p.add_argument('--spawn-x2', type=float, help='Ball spawn world X maximum in metres')
    p.add_argument('--spawn-speed', type=float, help='Initial ball speed in m/s')
    if task == 'shooting':
        p.add_argument('--negative-speed', type=float, default=0.05)
        p.add_argument('--negative-duration', type=float, default=0.3)
    else:
        p.add_argument('--x1', type=float, required=True)
        p.add_argument('--x2', type=float, required=True)
        p.add_argument('--failure-x', type=float, required=True)
        if task == 'passing':
            p.add_argument('--overshoot-x', type=float, required=True)
            p.add_argument('--stop-speed', type=float, default=0.05)
            p.add_argument('--stop-duration', type=float, default=0.2)
    a = p.parse_args()
    if not math.isfinite(a.episode_pause) or a.episode_pause < 0:
        p.error('episode-pause must be finite and nonnegative')
    if a.episodes <= 0 or a.episode_seconds <= 0 or not 1 <= a.rod_id <= 8:
        p.error('episodes and episode-seconds must be positive; rod-id must be 1..8')
    if task != 'shooting' and not a.failure_x <= a.x1 < a.x2:
        p.error('Require failure-x <= x1 < x2')
    if task == 'passing' and (a.overshoot_x < a.x2 or a.stop_speed < 0 or a.stop_duration < 0):
        p.error('Require overshoot-x >= x2 and nonnegative stopping thresholds')
    if task == 'shooting' and (a.negative_speed < 0 or a.negative_duration <= 0):
        p.error('Require negative-speed >= 0 and negative-duration > 0')
    if (a.spawn_x1 is None) != (a.spawn_x2 is None) or (a.spawn_x1 is not None and a.spawn_x1 >= a.spawn_x2):
        p.error('Provide both spawn-x1 and spawn-x2 with x1 < x2')
    if not Path(a.checkpoint).is_file():
        p.error('Checkpoint does not exist')
    try:
        kwargs = json.loads(a.agent_kwargs)
        if not isinstance(kwargs, dict):
            raise ValueError()
    except (ValueError, TypeError):
        p.error('agent-kwargs must be a JSON object')
    import random
    import numpy as np
    import torch
    from FuzbAISim import FuzbAISim
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if task == 'passing':
        kwargs.update(passer_rod_id=a.rod_id, receiver_rod_id=a.receiver_rod_id,
                      opponent_rod_id=a.observed_rod_id, load_backbone=False)
        mode = 'pass_ppo'
    elif a.agent_mode == 'single_rod_ppo':
        kwargs.update(controlled_rod_id=a.rod_id)
        mode = 'single_rod_ppo'
    else:
        kwargs.update(controlled_rod_id=a.rod_id, observed_rod_id=a.observed_rod_id,
                      training_task='shooting' if task == 'shooting' else 'defending')
        mode = 'two_rod_ppo'
    kwargs.update(model_save_path=a.checkpoint, load_model=True, inference=True,
                  training_enabeled=False, auto_train=False)
    evaluator = Evaluator(task, a)
    sim = FuzbAISim(agent1_mode=mode, agent1_kwargs=kwargs, evaluator=evaluator,
                   kick_observed_rod_id=a.rod_id, render_gui=a.gui)
    sim.max_num_steps = max(1, math.ceil(a.episode_seconds/sim.control_dt))
    sim.episode_pause_seconds = a.episode_pause
    sim.debug_print_red_kicks = False
    if a.spawn_x1 is not None:
        sim.ball_spawn_areas = {'evaluation': (a.spawn_x1, a.spawn_x2)}
    if a.spawn_speed is not None:
        sim.ball_spawn_speed_range = (a.spawn_speed, a.spawn_speed)
    sim.run()
    try:
        while sim.isRunning:
            if evaluator.completed:
                sim.stop()
            time.sleep(0.05)
    except KeyboardInterrupt:
        sim.stop()
    finally:
        sim.stop()
        sim.simThread.join()
        evaluator.write()
    if sim.run_error:
        raise RuntimeError('Evaluation failed') from sim.run_error
