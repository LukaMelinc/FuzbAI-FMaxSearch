import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from evaluation_common import Evaluator


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.args = SimpleNamespace(episodes=2, episode_seconds=5, negative_speed=.1,
                                    negative_duration=.3, failure_x=450,
                                    output=self.temp.name+'/results.csv')
        self.e = Evaluator(self.args)
        self.e.begin_episode(0, [0, 0])

    def sample(self, t, x=600, vx=0, score=None, **flags):
        return dict(simulation_time=t, score=score or [0, 0],
                    camData=[dict(ball_x=x, ball_vx=vx, ball_vy=0)], **flags)

    def test_goal_has_priority_over_both_failures_and_timeout(self):
        self.e.observe(self.sample(0, vx=-1))
        self.e.observe(self.sample(5, x=400, vx=-1, score=[1, 0], terminated_by_timeout=True))
        self.assertEqual(self.e.rows[0]['outcome'], 'success')

    def test_negative_velocity_must_be_continuous_and_strict(self):
        self.e.observe(self.sample(0, vx=-1))
        self.e.observe(self.sample(.2, vx=-.1))
        self.e.observe(self.sample(.3, vx=-1))
        self.assertFalse(self.e.observe(self.sample(.5, vx=-1)))
        self.assertTrue(self.e.observe(self.sample(.6, vx=-1)))
        self.assertEqual(self.e.rows[0]['outcome'], 'missed-shot')

    def test_position_threshold_and_duplicate_terminal(self):
        self.assertFalse(self.e.observe(self.sample(0, x=450)))
        self.assertTrue(self.e.observe(self.sample(.1, x=449)))
        self.e.observe(self.sample(.2, x=449, end_episode=True))
        self.assertEqual(len(self.e.rows), 1)
        self.assertEqual(self.e.rows[0]['outcome'], 'agent-missed-ball')

    def test_reset_clears_timer_and_preserves_score_baseline(self):
        self.e.observe(self.sample(0, vx=-1))
        self.e.observe(self.sample(.3, vx=-1))
        self.e.begin_episode(.3, [2, 0])
        self.assertFalse(self.e.observe(self.sample(.4, vx=-1, score=[2, 0])))
        self.e.observe(self.sample(.7, vx=-1, score=[2, 0]))
        self.assertTrue(self.e.completed)
        with Path(self.args.output).open() as f:
            self.assertEqual(len(list(csv.DictReader(f))), 2)
        summary = json.loads(Path(self.args.output).with_suffix('.summary.json').read_text())
        self.assertEqual(summary['outcomes']['missed-shot'], 2)

    def test_timeout_is_not_missed_ball(self):
        self.e.observe(self.sample(5, terminated_by_timeout=True))
        self.assertEqual(self.e.rows[0]['outcome'], 'timeout')

    def test_nonfinite_sample_rejected(self):
        with self.assertRaises(ValueError):
            self.e.observe(self.sample(0, vx=float('nan')))


if __name__ == '__main__':
    unittest.main()
