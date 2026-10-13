import math
import unittest

from reward.single_bar_shoting import passer_calm_reward
from strel_PPO import PassPPOAgent


class PasserCalmTests(unittest.TestCase):
    def sample(self, state, time, angle=0.0, target=0.0, x=600.0, vx=1.0, kick=False):
        return passer_calm_reward(
            control_state=state, sample_time=time, rod_angle_radians=angle,
            rotation_target=target, ball_x=x, ball_vx=vx, passer_x=530.0,
            ball_kicked=kick)

    def test_requires_forward_contact_clearance_and_grace(self):
        state = {}
        self.sample(state, 0.0)
        self.assertEqual(sum(self.sample(state, 0.1, angle=1.0).values()), 0)
        self.sample(state, 0.2, x=540, kick=True)
        self.assertNotIn("active_after", state)
        self.assertEqual(sum(self.sample(state, 0.3, angle=2.0).values()), 0)
        self.assertEqual(sum(self.sample(state, 0.35, angle=3.0).values()), 0)
        terms = self.sample(state, 0.5, angle=4.0, target=1.0)
        self.assertLess(terms["passer_rotation_motion"], 0)
        self.assertLess(terms["passer_rotation_command_change"], 0)
        # The phase stays active even if the ball comes back.
        self.assertLess(sum(self.sample(state, 0.6, angle=5.0, x=500, vx=-1).values()), 0)
        self.assertEqual(sum(self.sample(state, 0.7, angle=5.0, target=0.0).values()), 0)

    def test_rate_integration_and_unwrapped_angles(self):
        totals = []
        for dt in (0.05, 0.1):
            state = {"active_after": 0.0}
            self.sample(state, 0.0)
            total = 0.0
            for i in range(1, round(1 / dt) + 1):
                time = i * dt
                total += sum(self.sample(state, time, angle=4 * math.pi * time).values())
            totals.append(total)
        self.assertAlmostEqual(totals[0], -0.8)
        self.assertAlmostEqual(totals[0], totals[1])

    def test_agent_settles_then_resets_state(self):
        agent = PassPPOAgent(hidden_size=16, load_backbone=False, steps_per_env=32)

        def camera(time, angle=0.0, kick=False):
            angles = [0.0] * 8
            angles[3] = angle
            return dict(simulation_time=time, ball_kicked=kick, camData=[dict(
                ball_x=600.0, ball_y=350.0, ball_vx=1.0, ball_vy=0.0,
                rod_position_calib=[0.5] * 8, rod_angle=angles), None])

        agent.process_data(camera(0.0))
        agent.process_data(camera(0.05, kick=True))
        agent.receiving_control_state["hold_paid"] = True
        agent.process_data(camera(0.1))
        self.assertFalse(agent.pass_completed)
        agent.process_data(camera(0.3, angle=16))
        self.assertLess(agent.update_reward_breakdown_sums["passer_rotation_motion"], 0)
        self.assertEqual(agent.process_data(camera(0.6, angle=32)), [])
        self.assertTrue(agent.pass_completed)
        self.assertEqual(agent.passer_calm_state, {})
        self.assertIsNone(agent.pass_received_time)
        self.assertIsNone(agent.last_obs)


if __name__ == "__main__":
    unittest.main()
