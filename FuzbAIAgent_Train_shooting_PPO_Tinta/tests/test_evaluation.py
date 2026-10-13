import unittest
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from evaluation_common import Evaluator


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.args = SimpleNamespace(episodes=3, output=self.temp.name+'/test.csv',
            negative_speed=0.05, negative_duration=0.3, x1=500, x2=700,
            failure_x=450, overshoot_x=750, stop_speed=0.05, stop_duration=0.2)

    def sample(self, t, x=600, vx=0, vy=0, **kw):
        return dict(simulation_time=t, camData=[dict(ball_x=x, ball_vx=vx, ball_vy=vy)], **kw)

    def test_shot_negative_timer_resets_and_goal_wins(self):
        e = Evaluator('shooting', self.args)
        e.observe(self.sample(0, vx=-1))
        e.observe(self.sample(.2, vx=1))
        e.observe(self.sample(.3, vx=-1))
        self.assertFalse(e.observe(self.sample(.5, vx=-1)))
        self.assertTrue(e.observe(self.sample(.6, vx=-1, score=[1, 0], end_episode=True)))
        self.assertEqual(e.rows[0]['outcome'], 'success')

    def test_defense_contact_and_reset(self):
        e = Evaluator('defense', self.args)
        e.observe(self.sample(0, ball_kicked=True))
        e.observe(self.sample(.1, x=400))
        self.assertEqual(e.rows[0]['outcome'], 'partial_success')
        e.observe(self.sample(.2))
        e.observe(self.sample(.3, x=400))
        self.assertEqual(e.rows[1]['outcome'], 'failure')

    def test_defense_requires_whole_episode_in_band(self):
        e = Evaluator('defense', self.args)
        e.observe(self.sample(0, x=720))
        e.observe(self.sample(1, end_episode=True))
        self.assertEqual(e.rows[0]['reason'], 'left_band')

    def test_pass_uses_total_speed_and_continuous_hold(self):
        e = Evaluator('passing', self.args)
        e.observe(self.sample(0, vy=1))
        e.observe(self.sample(.2))
        self.assertFalse(e.observe(self.sample(.3)))
        self.assertTrue(e.observe(self.sample(.4)))
        self.assertEqual(e.rows[0]['outcome'], 'success')

    def test_pass_thresholds_and_duplicate_terminal(self):
        e = Evaluator('passing', self.args)
        e.observe(self.sample(0, x=800, end_episode=True))
        e.observe(self.sample(.1, x=800, end_episode=True))
        self.assertEqual(len(e.rows), 1)
        self.assertEqual(e.rows[0]['reason'], 'failed_to_stop')
        e.observe(self.sample(.2, x=400))
        self.assertEqual(e.rows[1]['reason'], 'wrong_direction')


if __name__ == '__main__':
    unittest.main()
