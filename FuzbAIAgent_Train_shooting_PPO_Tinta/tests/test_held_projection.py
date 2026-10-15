import unittest
from reward.single_bar_shoting import predictive_player_alignment_reward


class PerStepProjectionTests(unittest.TestCase):
    def call_reward(self, state, step, **changes):
        args = dict(ball_x=0, ball_y=100, ball_vx=1, ball_vy=0.1,
                    rod_x=1000, rod_pos_calib=0.5,
                    rod_info=dict(travel=400, first_offset=0, spacing=0, players=1),
                    reward_scale=1, projection_state=state,
                    episode_sample=step, episode_samples=10)
        args.update(changes)
        return predictive_player_alignment_reward(**args)

    def test_target_updates_before_and_after_midpoint(self):
        state = {}
        self.assertAlmostEqual(self.call_reward(state, 0), 1)
        self.call_reward(state, 4, ball_y=300)
        self.assertEqual(state['target_y'], 400)
        self.call_reward(state, 5, ball_y=300)
        self.assertEqual(state['target_y'], 400)
        self.assertAlmostEqual(self.call_reward(state, 9, ball_y=0), 0.75)
        self.assertEqual(state['target_y'], 100)
        self.call_reward(state, 0, ball_y=50)
        self.assertEqual(state['target_y'], 150)
        self.assertNotIn('refreshed', state)

    def test_invalid_projection_falls_back_and_next_sample_recovers(self):
        state = {}
        self.call_reward(state, 0, ball_vx=0)
        self.call_reward(state, 1, ball_y=300)
        self.assertEqual(state['target_y'], 400)
        self.call_reward(state, 5, ball_vx=-1, ball_y=250)
        self.assertEqual(state['target_y'], 250)

    def test_velocity_change_updates_immediately(self):
        state = {}
        self.call_reward(state, 0)
        self.assertAlmostEqual(self.call_reward(state, 1, ball_vy=-0.05), 0.625)
        self.assertEqual(state['target_y'], 50)

    def test_legacy_mode_updates_each_call(self):
        self.assertAlmostEqual(self.call_reward(None, 0), 1)
        self.assertAlmostEqual(self.call_reward(None, 1, ball_y=300), 0.5)


if __name__ == '__main__':
    unittest.main()
