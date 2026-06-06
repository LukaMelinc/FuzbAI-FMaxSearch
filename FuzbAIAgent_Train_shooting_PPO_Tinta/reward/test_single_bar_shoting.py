import unittest

from reward.single_bar_shoting import closest_player_alignment_reward, player_alignment_target


ROD = {
    "position": 830,
    "travel": 180,
    "players": 3,
    "first_offset": 55,
    "spacing": 208,
}


class PlayerAlignmentRewardTest(unittest.TestCase):
    def test_selects_player_by_ball_region(self):
        self.assertEqual(player_alignment_target(ball_y=100, rod_info=ROD)[0], 0)
        self.assertEqual(player_alignment_target(ball_y=350, rod_info=ROD)[0], 1)
        self.assertEqual(player_alignment_target(ball_y=600, rod_info=ROD)[0], 2)

    def test_reward_increases_toward_each_regions_target(self):
        for ball_y in (100, 350, 600):
            _, target, _ = player_alignment_target(ball_y=ball_y, rod_info=ROD)
            wrong_end = 1.0 if target < 0.5 else 0.0
            midpoint = (target + wrong_end) / 2.0
            target_reward = closest_player_alignment_reward(
                ball_x=830, ball_y=ball_y, rod_pos_calib=target, rod_info=ROD
            )
            midpoint_reward = closest_player_alignment_reward(
                ball_x=830, ball_y=ball_y, rod_pos_calib=midpoint, rod_info=ROD
            )
            wrong_reward = closest_player_alignment_reward(
                ball_x=830, ball_y=ball_y, rod_pos_calib=wrong_end, rod_info=ROD
            )
            self.assertGreater(target_reward, midpoint_reward)
            self.assertGreater(midpoint_reward, wrong_reward)

    def test_target_is_clamped_for_unreachable_edge(self):
        player, target, unavoidable_dist = player_alignment_target(ball_y=0, rod_info=ROD)
        self.assertEqual(player, 0)
        self.assertEqual(target, 0.0)
        self.assertEqual(unavoidable_dist, 55.0)


if __name__ == "__main__":
    unittest.main()
