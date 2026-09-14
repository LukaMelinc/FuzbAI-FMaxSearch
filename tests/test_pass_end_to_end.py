"""Run from repository root: python3 -m unittest discover -s tests -v."""
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'FuzbAIAgent_Train_shooting_PPO_Tinta'
sys.path.insert(0, str(ROOT))
import numpy as np
from strel_PPO import PassPPOAgent


class PassingTests(unittest.TestCase):
    def setUp(self):
        actual = Path.cwd()
        try:
            os.chdir(ROOT)
            self.agent = PassPPOAgent(hidden_size=32, load_backbone=False, steps_per_env=32)
        finally:
            os.chdir(actual)
        a = self.agent
        a.compute_action = lambda obs, deterministic=False: (np.zeros(8), 0.5, 0.0)
        def extract(c):
            rod = {'info': a.rods_by_id[6], 'pos_calib': 0.5}
            y = rod['info']['first_offset'] + 0.5 * rod['info']['travel']
            return np.zeros(19), (c.get('x', 530), y), (c.get('vx', 0), 0), rod, rod, rod
        a.extract_observation = extract

    def step(self, t, **kw):
        return self.agent.process_data(dict(simulation_time=t, **kw))

    def test_kick_bonus_once_and_no_termination(self):
        self.step(0)
        self.step(.05, ball_kicked=True, vx=1)
        self.step(.1, ball_kicked=True, vx=1)
        self.assertEqual(self.agent.update_reward_breakdown_sums['forward_kick'], 2)
        self.assertEqual(self.agent.episode_count, 0)
        self.step(.15, ball_kicked=True, vx=-1)
        self.assertEqual(self.agent.update_reward_breakdown_sums['backward_kick'], -1)

    def test_contact_hold_completes_and_next_spawn_is_not_transition(self):
        self.step(0)
        self.step(.05, ball_kicked=True, vx=1)
        self.step(.1, x=830, receiver_contact=True)
        self.assertFalse(self.agent.pass_completed)
        commands = self.step(.4, x=830)
        self.assertTrue(self.agent.pass_completed)
        self.assertEqual(commands, [])
        self.assertIsNone(self.agent.last_obs)
        self.assertEqual(self.agent.update_reward_breakdown_sums['hold'], 5)
        count = self.agent.buf.ptr
        self.agent.pass_completed = False  # simulator consumes completion
        self.step(.45)
        self.assertEqual(self.agent.buf.ptr, count)

    def test_no_contact_cannot_complete(self):
        self.step(0, x=830)
        self.step(.1, x=830)
        self.step(.5, x=830)
        self.assertFalse(self.agent.pass_completed)

    def test_buffer_boundary_preserves_episode_and_reception(self):
        a = self.agent
        a.buf.max_size = 2
        a.train_on_buffer = lambda: a.buf.get()
        self.step(0)
        self.step(.05, x=830, vx=.5, receiver_contact=True)
        self.step(.1, x=830, vx=.4)
        self.assertEqual(a.buf.ptr, 0)
        self.assertEqual(a.episode_count, 0)
        self.assertTrue(a.receiving_control_state['contacted'])
        self.step(.15, x=830, vx=.3)
        self.assertEqual(a.buf.ptr, 1)

    def test_timeout_and_threshold_penalties(self):
        self.step(0)
        self.step(.05, end_episode=True, terminated_by_timeout=True)
        self.assertEqual(self.agent.update_reward_breakdown_sums['no_kick_timeout'], -1)
        self.assertEqual(self.agent.episode_count, 1)
        self.step(.1)
        self.step(.15, end_episode=True, terminated_by_x_threshold=True)
        self.assertEqual(self.agent.update_reward_breakdown_sums['failure'], -5)


class SimulatorTerminationTests(unittest.TestCase):
    def test_constructor_honors_checkpoint_interval_and_output(self):
        import tempfile
        from FuzbAISim import FuzbAISim
        actual = Path.cwd()
        with tempfile.TemporaryDirectory() as output:
            try:
                os.chdir(ROOT)
                sim = FuzbAISim(render_gui=False, agent1_mode="pass_ppo",
                    agent1_kwargs=dict(load_model=False, load_backbone=False,
                        hidden_size=32, save_model_every=2, model_save_path=output))
            finally:
                os.chdir(actual)
            self.assertEqual(sim.p1.save_model_every, 2)
            sim.p1.total_steps = 10
            sim.p1.finish_episode()
            self.assertEqual(list(Path(output).glob("*.pth")), [])
            sim.p1.total_steps = 20
            sim.p1.finish_episode()
            self.assertTrue((Path(output) / "pass_ppo_steps_20.pth").is_file())

    def test_passing_threshold_defers_reset_until_observation(self):
        from FuzbAISim import FuzbAISim
        sim = FuzbAISim.__new__(FuzbAISim)
        sim._x_threshold_terminated_latch = False
        sim.recorded_imitation = False
        sim.agent1_mode = "pass_ppo"
        sim.episode_end_ball_x_threshold_mm = 1
        sim.episode_end_ball_other_x_threshold_mm = 950
        sim.end_episode_latch = False
        sim._ball_x_mm_camera = lambda: 960
        def premature_reset(*args, **kwargs):
            raise AssertionError("Terminal state reset before agent consumes it")
        sim.ResetBallToLocation = premature_reset
        sim._update_x_threshold_termination()
        self.assertTrue(sim.end_episode_latch)
        self.assertTrue(sim._x_threshold_terminated_latch)


if __name__ == '__main__':
    unittest.main()
