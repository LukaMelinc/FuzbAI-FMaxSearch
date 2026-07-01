import argparse
import os
import queue
import time
import multiprocessing as mp

import numpy as np

from FuzbAISim import FuzbAISim
from memory.main import MultiEnvPPOBuffer
from strel_PPO import PassPPOAgent
from log_utils import setup_logging


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "trained_models", "#23B.pth")
POLICY_SYNC_PATH = "/tmp/fuzbai_parallel_policy.npz"


class PassPPOCollectorAgent(PassPPOAgent):
    """Worker-side agent: acts with PPO policy and emits rollout samples."""

    def __init__(self, worker_id, transition_queue, command_queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.worker_id = int(worker_id)
        self.transition_queue = transition_queue
        self.command_queue = command_queue
        self.paused = False

    def _poll_commands(self):
        while True:
            try:
                msg = self.command_queue.get_nowait()
            except queue.Empty:
                return

            msg_type = msg.get("type")
            if msg_type == "policy_state":
                self.set_policy_state(msg["policy_state"])
                self.paused = False
            elif msg_type == "policy_path":
                self.set_policy_state(load_policy_state(msg["policy_path"]))
                self.paused = False
            elif msg_type == "pause":
                self.paused = True
            elif msg_type == "stop":
                raise KeyboardInterrupt

    def _wait_if_paused(self):
        if not self.paused:
            return

        self.transition_queue.put({
            "type": "paused",
            "worker_id": self.worker_id,
        })
        while self.paused:
            msg = self.command_queue.get()
            msg_type = msg.get("type")
            if msg_type == "policy_state":
                self.set_policy_state(msg["policy_state"])
                self.paused = False
            elif msg_type == "policy_path":
                self.set_policy_state(load_policy_state(msg["policy_path"]))
                self.paused = False
            elif msg_type == "stop":
                raise KeyboardInterrupt

    def process_data(self, camera):
        self._poll_commands()
        self._wait_if_paused()

        self.total_steps += 1
        obs, bxy, vxy, passer, receiver, opponent = self.extract_observation(camera)

        end_episode = bool(camera.get("end_episode", False))
        terminated_by_kick = bool(camera.get("terminated_by_kick", False))
        terminated_by_x_threshold = bool(camera.get("terminated_by_x_threshold", False))
        ball_kicked = bool(camera.get("ball_kicked", False))

        reward = 0.0
        reward_breakdown = {}

        if self.last_obs is not None:
            reward, reward_breakdown = self.compute_pass_reward(
                bxy,
                vxy,
                receiver,
                passer,
                ball_kicked=ball_kicked,
                threshold_failure=terminated_by_x_threshold,
                episode_timeout=(
                    end_episode
                    and not terminated_by_kick
                    and not terminated_by_x_threshold
                    and not ball_kicked
                ),
            )
            self.ep_reward += float(reward)
            self.current_episode_step_rewards.append(float(reward))
            if ball_kicked:
                self.current_episode_ball_kicks += 1
                self.kick_latch = True
            if terminated_by_x_threshold:
                self.current_episode_x_threshold_terminations += 1

        done = bool(self.kick_latch or terminated_by_kick or terminated_by_x_threshold or end_episode)

        action, value, logp = self.compute_action(obs, deterministic=self.inference)
        if not np.all(np.isfinite(action)):
            print(f"[Worker {self.worker_id}] Nan or Inf detected in action: {action}")
            action = np.zeros_like(action)

        if self.last_obs is not None:
            transition_done = bool(done)
            self.transition_queue.put({
                "type": "transition",
                "worker_id": self.worker_id,
                "obs": self.last_obs,
                "act": self.last_action,
                "rew": float(reward),
                "val": float(self.last_val),
                "logp": float(self.last_logp),
                "done": transition_done,
                "next_val": 0.0 if transition_done else float(value),
                "episode_reward": float(self.ep_reward) if transition_done else None,
                "reward_breakdown": reward_breakdown,
                "ball_kicked": bool(ball_kicked),
                "terminated_by_x_threshold": bool(terminated_by_x_threshold),
            })

        if done:
            self.episode_count += 1
            self.episode_steps = 0
            self.ep_reward = 0.0
            self.last_obs = None
            self.last_action = None
            self.last_val = None
            self.last_logp = None
            self.prev_ball_x = None
            self.prev_ball_vxy = None
            self.kick_latch = False
            self.current_episode_ball_kicks = 0
            self.current_episode_x_threshold_terminations = 0
            self.current_episode_step_rewards = []
        else:
            self.episode_steps += 1
            self.prev_ball_x = bxy[0]
            self.prev_ball_vxy = vxy
            self.last_obs = obs
            self.last_action = action
            self.last_val = value
            self.last_logp = logp

        return self.scale_to_motor_commands(action)


def make_pass_agent(*, load_model, model_save_path, training_enabled, inference):
    return PassPPOAgent(
        passer_rod_id=4,
        receiver_rod_id=6,
        opponent_rod_id=5,
        model_save_path=model_save_path,
        load_model=load_model,
        load_backbone=not load_model,
        inference=inference,
        training_enabeled=training_enabled,
        action_std_override=(0.10, 0.10, 0.15, 0.15, 0.10, 0.10, 0.15, 0.15),
    )


def apply_stage2_freeze(agent):
    agent.freeze_layers(agent.ac, {
        "receiver_encoder",
        "receiver_translation_head",
        "receiver_rotation_head",
        "critic",
    })
    agent.rebuild_optimizer()


def save_policy_state(policy_state, path):
    arrays = {"log_std": policy_state["log_std"]}
    for key, value in policy_state["actor_critic_state_dict"].items():
        arrays[f"ac.{key}"] = value
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp_path, path)
    return path


def load_policy_state(path):
    with np.load(path) as data:
        return {
            "actor_critic_state_dict": {
                key[3:]: data[key].copy()
                for key in data.files
                if key.startswith("ac.")
            },
            "log_std": data["log_std"].copy(),
        }


def worker_main(worker_id, transition_queue, command_queue, initial_policy_path, args):
    os.chdir(SCRIPT_DIR)
    setup_logging()
    agent = PassPPOCollectorAgent(
        worker_id,
        transition_queue,
        command_queue,
        passer_rod_id=4,
        receiver_rod_id=6,
        opponent_rod_id=5,
        model_save_path=args.model_path,
        load_model=False,
        load_backbone=False,
        inference=False,
        training_enabeled=False,
        action_std_override=(0.10, 0.10, 0.15, 0.15, 0.10, 0.10, 0.15, 0.15),
    )
    agent.set_policy_state(load_policy_state(initial_policy_path))
    transition_queue.put({"type": "ready", "worker_id": worker_id})

    sim = FuzbAISim(
        gui=bool(args.gui_worker == worker_id),
        worker_id=worker_id,
        player1_agent=agent,
    )
    sim.run()
    try:
        while sim.isRunning:
            time.sleep(0.1)
    except KeyboardInterrupt:
        sim.stop()
        if sim.simThread is not None:
            sim.simThread.join(timeout=5.0)


def wait_for_workers_ready(transition_queue, num_workers):
    ready = set()
    while len(ready) < num_workers:
        msg = transition_queue.get()
        if msg.get("type") == "ready":
            ready.add(int(msg["worker_id"]))
            print(f"[ParallelTrainer] Worker {msg['worker_id']} ready")


def pause_workers(command_queues, transition_queue, num_workers, timeout_s=10.0):
    for command_queue in command_queues:
        command_queue.put({"type": "pause"})

    paused = set()
    deadline = time.time() + timeout_s
    discarded = 0
    while len(paused) < num_workers and time.time() < deadline:
        try:
            msg = transition_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if msg.get("type") == "paused":
            paused.add(int(msg["worker_id"]))
        elif msg.get("type") == "transition":
            discarded += 1

    while True:
        try:
            msg = transition_queue.get_nowait()
        except queue.Empty:
            break
        if msg.get("type") == "transition":
            discarded += 1

    if discarded:
        print(f"[ParallelTrainer] Discarded {discarded} in-flight samples during policy sync")
    if len(paused) < num_workers:
        print(f"[ParallelTrainer] Pause timeout: paused {len(paused)}/{num_workers} workers")


def broadcast_policy(command_queues, policy_path):
    for command_queue in command_queues:
        command_queue.put({
            "type": "policy_path",
            "policy_path": policy_path,
        })


def parse_args():
    parser = argparse.ArgumentParser(description="Train one PPO model from multiple FuzbAI simulators.")
    parser.add_argument("--num-sims", type=int, default=2)
    parser.add_argument("--steps-per-update", type=int, default=512)
    parser.add_argument("--max-updates", type=int, default=0, help="0 means run forever")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--gui-worker", type=int, default=-1, help="Worker id to show in GUI, or -1 for all headless")
    return parser.parse_args()


def main():
    os.chdir(SCRIPT_DIR)
    setup_logging()
    args = parse_args()
    num_workers = max(1, int(args.num_sims))

    trainer = make_pass_agent(
        load_model=True,
        model_save_path=args.model_path,
        training_enabled=True,
        inference=False,
    )
    apply_stage2_freeze(trainer)
    trainer.buf = MultiEnvPPOBuffer(
        trainer.obs_dim,
        trainer.act_dim,
        int(args.steps_per_update),
        num_workers,
        gamma=trainer.buf.gamma,
        lam=trainer.buf.lam,
    )

    transition_queue = mp.Queue(maxsize=max(1024, int(args.steps_per_update) * 4))
    command_queues = [mp.Queue() for _ in range(num_workers)]
    policy_path = save_policy_state(trainer.get_policy_state(), POLICY_SYNC_PATH)

    processes = []
    for worker_id in range(num_workers):
        proc = mp.Process(
            target=worker_main,
            args=(worker_id, transition_queue, command_queues[worker_id], policy_path, args),
            daemon=True,
        )
        proc.start()
        processes.append(proc)

    wait_for_workers_ready(transition_queue, num_workers)
    latest_next_values = {worker_id: 0.0 for worker_id in range(num_workers)}
    start_training_count = int(trainer.training_count)

    try:
        while args.max_updates <= 0 or (trainer.training_count - start_training_count) < args.max_updates:
            msg = transition_queue.get()
            if msg.get("type") != "transition":
                continue

            worker_id = int(msg["worker_id"])
            stored = trainer.buf.store(
                worker_id,
                msg["obs"],
                msg["act"],
                msg["rew"],
                msg["val"],
                msg["logp"],
            )
            if not stored:
                continue

            trainer.total_steps += 1
            latest_next_values[worker_id] = float(msg["next_val"])
            if msg["done"]:
                trainer.buf.finish_path(worker_id, last_val=0.0)
                latest_next_values[worker_id] = 0.0

            for key, value in msg.get("reward_breakdown", {}).items():
                trainer.update_reward_breakdown_sums[key] = (
                    trainer.update_reward_breakdown_sums.get(key, 0.0) + float(value)
                )
            if msg.get("ball_kicked", False):
                trainer.update_samples_with_kick += 1

            if trainer.buf.ptr >= trainer.buf.max_size:
                pause_workers(command_queues, transition_queue, num_workers)
                trainer.buf.finish_all_paths(latest_next_values)
                print(
                    f"[ParallelTrainer] Buffer full: {trainer.buf.ptr}/{trainer.buf.max_size}. "
                    "Training central PPO model..."
                )
                trainer.train_on_buffer()
                trainer.save_model()
                latest_next_values = {worker_id: 0.0 for worker_id in range(num_workers)}
                if args.max_updates > 0 and (trainer.training_count - start_training_count) >= args.max_updates:
                    break
                policy_path = save_policy_state(trainer.get_policy_state(), POLICY_SYNC_PATH)
                broadcast_policy(command_queues, policy_path)

    except KeyboardInterrupt:
        print("[ParallelTrainer] Stopping workers...")
    finally:
        for command_queue in command_queues:
            command_queue.put({"type": "stop"})
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.join(timeout=5.0)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
