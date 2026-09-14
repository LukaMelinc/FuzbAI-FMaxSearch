"""Motion features, causal policy history, and imitation checkpoint contracts."""
import csv
from datetime import datetime, timedelta
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] / 'FuzbAIAgent_Train_shooting_PPO_Tinta'
sys.path.insert(0, str(ROOT))
from recorded_expert import ExpertTransitionDataset, RecordingConfig, RodMotionObservation
from state_imitation import StateImitationPPOAgent


class ImitationHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.recording = Path(self.temp.name) / 'motion.csv'
        rows = []
        for i in range(12):
            row = dict(Timestamp=(datetime(2026, 1, 1) + timedelta(seconds=i * .05)).strftime('%d.%m.%Y %H:%M:%S.%f'),
                       Ball_x=520, Ball_y=350, Ball_vx=0, Ball_vy=0, Ball_size=34)
            for j in range(8):
                row[f'Rod{j}_p'] = .4 + i * .01
                row[f'Rod{j}_fi'] = 3.1 + i * .1
            rows.append(row)
        with self.recording.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=';')
            writer.writeheader()
            writer.writerows(rows)

    def agent(self, training=True, **kwargs):
        previous = Path.cwd()
        try:
            os.chdir(ROOT)
            return StateImitationPPOAgent(
                expert_data=self.recording if training else None,
                training_enabeled=training, inference=not training,
                hidden_size=16, steps_per_env=8, discriminator_updates=1,
                model_save_path=self.temp.name, **kwargs,
            )
        finally:
            os.chdir(previous)

    def camera(self, i, **flags):
        data = dict(ball_x=520, ball_y=350, ball_vx=0, ball_vy=0,
                    rod_position_calib=[.4 + i * .01] * 8,
                    rod_angle=[(3.1 + i * .1) * 32 / np.pi] * 8)
        return dict(camData=[data, data], **flags)

    def test_wrap_direction_lateral_speed_and_reset(self):
        motion = RodMotionObservation(.05)
        first = motion.update(np.zeros(8), .5, np.pi - .02)
        second = motion.update(np.zeros(8), .51, -np.pi + .03)
        np.testing.assert_array_equal(first[-2:], [0, 0])
        np.testing.assert_allclose(second[-2:], [1 / np.pi, .2], rtol=1e-6)
        reverse = motion.update(np.zeros(8), .50, np.pi - .02)
        np.testing.assert_allclose(reverse[-2:], [-1 / np.pi, -.2], rtol=1e-6)
        motion.reset()
        np.testing.assert_array_equal(motion.update(np.zeros(8), .1, 0)[-2:], [0, 0])

    def test_expert_and_live_features_agree(self):
        agent = self.agent()
        expert = agent.expert_dataset.episodes[0]
        self.assertEqual(expert.shape, (12, 10))
        self.assertEqual(agent.expert_dataset.transitions.shape, (11, 20))
        for i, expected in enumerate(expert):
            state, history = agent.policy_observation(self.camera(i))
            np.testing.assert_allclose(state, expected, atol=1e-6)
            self.assertEqual(history.shape, (50,))

    def test_history_order_padding_and_stored_policy_input(self):
        agent = self.agent()
        agent.process_data(self.camera(0))
        initial = agent.last_obs.copy()
        np.testing.assert_array_equal(initial.reshape(5, 10), np.tile(initial[-10:], (5, 1)))
        agent.process_data(self.camera(1))
        np.testing.assert_array_equal(agent.buf.obs_buf[0], initial)
        np.testing.assert_array_equal(agent.last_obs[:-10], initial[10:])
        self.assertEqual(agent.agent_transitions[0].shape, (20,))
        self.assertGreater(agent.last_obs[-1], 0)

    def test_recording_gap_restarts_velocity_estimation(self):
        with self.recording.open() as stream:
            reader = csv.DictReader(stream, delimiter=';')
            rows = list(reader)
            fields = reader.fieldnames
        rows[5]['Ball_size'] = '0'
        with self.recording.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter=';')
            writer.writeheader()
            writer.writerows(rows)
        dataset = self.agent().expert_dataset
        self.assertEqual(len(dataset.episodes), 2)
        for segment in dataset.episodes:
            np.testing.assert_array_equal(segment[0, -2:], [0, 0])
            np.testing.assert_allclose(segment[1, -2:], [2 / np.pi, .2], atol=1e-6)

    def test_terminal_clears_history_without_cross_episode_transition(self):
        for training in (True, False):
            with self.subTest(training=training):
                agent = self.agent(training)
                agent.process_data(self.camera(0))
                self.assertEqual(agent.process_data(self.camera(1, end_episode=True)), [])
                self.assertEqual(len(agent.observation_history), 0)
                count = agent.buf.ptr
                agent.process_data(self.camera(9))
                self.assertEqual(agent.buf.ptr, count)
                np.testing.assert_array_equal(agent.observation_history[-1][-2:], [0, 0])

    def test_bootstrap_then_policy_update_preserves_history(self):
        agent = self.agent()
        initial = [p.detach().clone() for p in agent.ac.actor.parameters()]
        for i in range(9):
            agent.process_data(self.camera(i))
        self.assertTrue(agent.discriminator_ready)
        self.assertEqual(agent.training_count, 0)
        self.assertEqual(len(agent.observation_history), 5)
        for i in range(9, 17):
            agent.process_data(self.camera(i))
        self.assertGreater(agent.training_count, 0)
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(initial, agent.ac.actor.parameters())))

    def test_checkpoint_roundtrip_and_legacy_rejection(self):
        agent = self.agent()
        path = Path(self.temp.name) / 'model.pth'
        agent.save_model(path)
        restored = self.agent(False)
        restored.load_model(path)
        state, history = agent.policy_observation(self.camera(0))
        np.testing.assert_array_equal(agent.compute_action(history, True)[0], restored.compute_action(history, True)[0])
        checkpoint = torch.load(path, weights_only=True)
        self.assertEqual(checkpoint['history_length'], 5)
        checkpoint.pop('motion_schema')
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, 'new training run'):
            restored.load_model(path)


if __name__ == '__main__':
    unittest.main()
