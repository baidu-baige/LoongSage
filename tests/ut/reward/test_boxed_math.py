"""Unit tests for the boxed_math reward."""

import os
import unittest

from omegaconf import OmegaConf

from coda.reward import create_reward_fn


def _reward():
    return create_reward_fn(OmegaConf.create({"name": "boxed_math"}))


def _score(response, label):
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": response}]
    return _reward()(messages, label, {})


class TestBoxedMathReward(unittest.TestCase):
    def test_correct(self):
        r = _score("reasoning... \\boxed{270}", "270")
        self.assertEqual(r.final_reward, 1.0)
        self.assertTrue(r.is_correct)

    def test_wrong_is_zero_not_negative(self):
        self.assertEqual(_score("\\boxed{12}", "270").final_reward, 0.0)

    def test_no_boxed_is_zero(self):
        self.assertEqual(_score("The answer is 270.", "270").final_reward, 0.0)

    def test_uses_last_boxed(self):
        self.assertEqual(_score("\\boxed{1} then \\boxed{270}", "270").final_reward, 1.0)

    def test_numeric_equivalence(self):
        self.assertEqual(_score("\\boxed{270.0}", "270").final_reward, 1.0)

    def test_latex_answer_with_nested_braces(self):
        self.assertEqual(_score("\\boxed{6\\sqrt{3}}", "6\\sqrt{3}").final_reward, 1.0)

    def test_dfrac_equals_frac(self):
        self.assertEqual(_score("\\boxed{\\dfrac{26}{3}}", "\\frac { 26 } { 3 }").final_reward, 1.0)
        self.assertEqual(_score("\\boxed{\\frac{26}{3}}", "\\dfrac{26}{3}").final_reward, 1.0)
        self.assertEqual(_score("\\boxed{4\\tfrac{4}{11}}", "4 \\frac { 4 } { 11 }").final_reward, 1.0)

    def test_dfrac_wrong_value_still_zero(self):
        self.assertEqual(_score("\\boxed{\\dfrac{26}{5}}", "\\frac { 26 } { 3 }").final_reward, 0.0)

    def test_symbolic_equivalence(self):
        for pred, label in (
            ("3\\sqrt{15}", "\\sqrt { 135 }"),
            ("10.5\\sqrt{2}", "\\frac { 21 \\sqrt 2 } { 2 }"),
            ("\\dfrac{157}{20}", "7.85"),
            ("10y(2y-1)", "20 y^2 - 10 y"),
            ("2\\sqrt{3}\\cdot 3", "6\\sqrt{3}"),
            ("\\frac{\\sqrt{2}+\\sqrt{3}}{\\sqrt{2}-\\sqrt{3}}", "-5-2\\sqrt{6}"),
        ):
            with self.subTest(pred=pred, label=label):
                self.assertEqual(_score(f"\\boxed{{{pred}}}", label).final_reward, 1.0)

    def test_symbolic_does_not_accept_approximations_or_different_values(self):
        for pred, label in (
            ("13.33", "\\frac { 40 } { 3 }"),
            ("31.4", "10 \\pi"),
            ("2\\sqrt{3}", "3\\sqrt{2}"),
            ("10y(2y-1)", "20 y^2 - 10 y + 1"),
            ("y", "x"),
        ):
            with self.subTest(pred=pred, label=label):
                self.assertEqual(_score(f"\\boxed{{{pred}}}", label).final_reward, 0.0)

    def test_symbolic_parser_rejects_untrusted_code_and_huge_powers(self):
        for pred in (
            "__import__('os').system('true')",
            "open('/tmp/boxed_math_should_not_exist','w')",
            "9^9^9",
            "9^{9^{9}}",
            "2**3**4",
            "1" * 500,
            "x" * 300,
        ):
            with self.subTest(pred=pred[:30]):
                self.assertEqual(_score(f"\\boxed{{{pred}}}", "7").final_reward, 0.0)
        self.assertFalse(os.path.exists("/tmp/boxed_math_should_not_exist"))

    def test_dict_label(self):
        self.assertEqual(_score("\\boxed{5}", {"answer": "5"}).final_reward, 1.0)

    def test_missing_label_is_invalid(self):
        r = _score("\\boxed{5}", None)
        self.assertEqual(r.final_reward, 0.0)
        self.assertFalse(r.is_valid)

    def test_no_assistant_message(self):
        r = _reward()([{"role": "user", "content": "q"}], "5", {})
        self.assertEqual(r.final_reward, 0.0)


if __name__ == "__main__":
    unittest.main()
