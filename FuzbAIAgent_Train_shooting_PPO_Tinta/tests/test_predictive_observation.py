import contextlib
import io
import unittest

import numpy as np
import torch

from actor_critic.main import ActorCriticNet
from recorded_expert import observation
from reward.single_bar_shoting import predictive_player_alignment_reward
from state_imitation import StateImitationPPOAgent
from strel_PPO import ImitationFineTunePPOAgent, PPOAgent


class PredictiveObservationTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        self.agent = PPOAgent.__new__(PPOAgent)
        self.agent.controlled_rod_id = 6
        self.agent.rods_by_id = {
            6: dict(position=830, travel=180, first_offset=55, spacing=208, players=3),
        }
        self.agent.last_obs = None
        self.agent.episode_steps = 0

    def camera(self, held=False, **changes):
        cd = dict(ball_x=1130, ball_y=300, ball_vx=-1, ball_vy=0.6,
                  rod_position_calib=[0.5] * 8, rod_angle=[0] * 8)
        cd.update(changes)
        camera = dict(camData=[cd, None])
        if held:
            camera['episode_samples'] = 10
        return camera

    def test_projected_branch_and_other_features(self):
        camera = self.camera()
        obs, ball_xy, velocity, angle, rod = self.agent.extract_observation(camera)
        # 300 ms to the rod: crossing Y = 300 + 600 * 0.3 = 480 mm.
        # Third player offset = 471 mm, desired translation = 9 / 180 = 0.05.
        self.assertAlmostEqual(rod['target_y'], 480)
        self.assertAlmostEqual(rod['target_pos'], 0.05)
        self.assertAlmostEqual(float(obs[5]), -0.45)
        self.assertEqual(obs.shape, (8,))
        self.assertEqual(obs.dtype, np.float32)
        previous = observation(1130, 300, -1, 0.6, 0.5, 0, rod['info'])
        np.testing.assert_array_equal(np.delete(obs, 5), np.delete(previous, 5))
        self.assertEqual(ball_xy, (1130, 300))
        self.assertEqual(velocity, (-1, 0.6))
        self.assertEqual(angle, 0)

    def test_zero_error_and_max_reward_at_projected_target(self):
        camera = self.camera(held=True, rod_position_calib=[0.05] * 8)
        obs, xy, velocity, _, rod = self.agent.extract_observation(camera)
        self.assertAlmostEqual(float(obs[5]), 0)
        reward = predictive_player_alignment_reward(
            ball_x=xy[0], ball_y=xy[1], ball_vx=velocity[0], ball_vy=velocity[1],
            rod_x=830, rod_pos_calib=rod['pos_calib'], rod_info=rod['info'],
            vx_threshold=0.05, t_max=3, reward_scale=1,
            **rod['projection_options'],
        )
        self.assertAlmostEqual(reward, 1)

    def test_current_target_before_midpoint_and_episode_reset(self):
        self.agent.extract_observation(self.camera(held=True))
        self.agent.last_obs = np.zeros(8)
        self.agent.episode_steps = 4
        obs, _, _, _, rod = self.agent.extract_observation(
            self.camera(held=True, ball_y=100, ball_vy=0))
        self.assertEqual(rod['target_y'], 100)
        self.assertAlmostEqual(float(obs[5]), -0.25)
        self.agent.episode_steps = 5
        obs, _, _, _, rod = self.agent.extract_observation(
            self.camera(held=True, ball_y=100, ball_vy=0))
        self.assertEqual(rod['target_y'], 100)
        self.assertAlmostEqual(float(obs[5]), -0.25)
        self.agent.last_obs = None
        self.agent.episode_steps = 0
        _, _, _, _, rod = self.agent.extract_observation(self.camera(held=True))
        self.assertEqual(rod['target_y'], 480)

    def test_invalid_projection_uses_existing_fallback(self):
        for changes in (dict(ball_vx=0), dict(ball_vx=1), dict(ball_vx=-0.06)):
            with self.subTest(changes=changes):
                obs, _, _, _, rod = self.agent.extract_observation(self.camera(**changes))
                self.assertEqual(rod['target_y'], 300)
                self.assertAlmostEqual(rod['target_pos'], 37 / 180)
                self.assertAlmostEqual(float(obs[5]), 37 / 180 - 0.5)

    def test_policy_network_receives_projected_error(self):
        self.agent.device = torch.device('cpu')
        self.agent.ac = ActorCriticNet(8, 4, hidden_size=8)
        self.agent.log_std = torch.zeros(4)
        self.agent.total_steps = 0
        self.agent.training_enabled = False
        self.agent.inference = True
        received = []
        hook = self.agent.ac.register_forward_pre_hook(
            lambda module, args: received.append(args[0].detach().clone()))
        self.addCleanup(hook.remove)
        self.agent.process_data(self.camera(held=True))
        self.assertEqual(len(received), 1)
        self.assertAlmostEqual(received[0][0, 5].item(), -0.45)

    def test_imitation_observation_contract_is_preserved(self):
        for agent_type in (StateImitationPPOAgent, ImitationFineTunePPOAgent):
            with self.subTest(agent_type=agent_type.__name__):
                agent = agent_type.__new__(agent_type)
                agent.controlled_rod_id = 6
                agent.rods_by_id = self.agent.rods_by_id
                obs, _, _, _, rod = agent.extract_observation(self.camera(held=True))
                expected = observation(1130, 300, -1, 0.6, 0.5, 0, rod['info'])
                np.testing.assert_array_equal(obs, expected)


if __name__ == '__main__':
    unittest.main()
