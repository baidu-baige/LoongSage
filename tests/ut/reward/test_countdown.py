"""Unit tests for the countdown reward's equation check."""

import unittest

from omegaconf import OmegaConf

from coda.custom.countdown.reward import CountdownReward, evaluate_equation


def _answer_reward(equation, target, nums):
    reward = CountdownReward(OmegaConf.create({"format_reward": 0.0}))
    messages = [{"role": "user", "content": "q"},
                {"role": "assistant", "content": f"<answer>{equation}</answer>"}]
    return reward(messages, {"target": target, "nums": nums}, {}).final_reward


class TestCountdownEquation(unittest.TestCase):
    def test_plain_arithmetic_is_scored(self):
        self.assertEqual(_answer_reward("(7 - 2) * 3", 15, [7, 2, 3]), 1.0)
        self.assertEqual(_answer_reward("8 / 4 + 1", 3, [8, 4, 1]), 1.0)

    def test_floor_division_is_rejected(self):
        # 7 // 2 == 3, but no + - * / equation of 7 and 2 makes 3.
        self.assertIsNone(evaluate_equation("7 // 2"))
        self.assertEqual(_answer_reward("7 // 2", 3, [7, 2]), 0.0)

    def test_a_decimal_point_is_rejected(self):
        # "1.5" reads 1 and 5 as one number: 1.5 * 4 == 6 on [1, 5, 4], unreachable with + - * /.
        self.assertIsNone(evaluate_equation("1.5 * 4"))
        self.assertEqual(_answer_reward("1.5 * 4", 6, [1, 5, 4]), 0.0)

    def test_power_is_still_rejected(self):
        self.assertIsNone(evaluate_equation("2 ** 3"))

    def test_true_division_keeps_working(self):
        self.assertEqual(evaluate_equation("7 / 2"), 3.5)


if __name__ == "__main__":
    unittest.main()
