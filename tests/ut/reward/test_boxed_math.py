"""Unit tests for the boxed_math reward."""

import os
import time
import unittest

from omegaconf import OmegaConf

from coda.reward import create_reward_fn


def _reward():
    return create_reward_fn(OmegaConf.create({"name": "boxed_math", "group_rm": False}))


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

    def test_wrong_large_polynomial_is_rejected_without_simplify(self):
        # simplify() alone takes ~39s here; the random-point probe must reject it well under the timeout.
        start = time.perf_counter()
        r = _score("\\boxed{(a+b+c+d+e+f+g)^{12}-(a-b)^{12}}", "1")
        self.assertEqual(r.final_reward, 0.0)
        self.assertLess(time.perf_counter() - start, 3.0)

    def test_correct_large_polynomial_still_matches(self):
        pred = "(a+b+c+d+e+f+g)^{11}(a+b+c+d+e+f)+g(a+b+c+d+e+f+g)^{11}"
        self.assertEqual(_score(f"\\boxed{{{pred}}}", "(a+b+c+d+e+f+g)^{12}").final_reward, 1.0)

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

    def test_variable_a_is_not_stripped_as_article(self):
        """`a ` must not be removed when it is a variable: `3a + 2` used to normalize to `3+2`."""
        for pred, label in [
            ("3a + 2", "5"),
            ("a + b", "b"),
            ("2a + 1", "3"),
            ("a \\sin x", "\\sin x"),
            ("\\tan x", "\\t x"),
        ]:
            with self.subTest(pred=pred, label=label):
                self.assertEqual(_score(f"\\boxed{{{pred}}}", label).final_reward, 0.0)

    def test_variable_a_still_matches_equivalent_forms(self):
        self.assertEqual(_score("\\boxed{3a + 2}", "2 + 3a").final_reward, 1.0)
        self.assertEqual(_score("\\boxed{a + b}", "b + a").final_reward, 1.0)

    def test_english_article_is_still_ignored(self):
        self.assertEqual(_score("\\boxed{\\text{a triangle}}", "triangle").final_reward, 1.0)
        self.assertEqual(_score("\\boxed{\\text{an octagon}}", "octagon").final_reward, 1.0)

    def test_dict_label(self):
        self.assertEqual(_score("\\boxed{5}", {"answer": "5"}).final_reward, 1.0)

    def test_missing_label_is_invalid(self):
        r = _score("\\boxed{5}", None)
        self.assertEqual(r.final_reward, 0.0)
        self.assertFalse(r.is_valid)

    def test_no_assistant_message(self):
        r = _reward()([{"role": "user", "content": "q"}], "5", {})
        self.assertEqual(r.final_reward, 0.0)


class TestBoxedMathUnits(unittest.TestCase):
    """A numeric label should match an answer that only adds a trailing unit.

    Seen in geo3k rollouts: the value is right but the box carries a unit that
    ``normalize_final_answer`` has no entry for (e.g. ``ft``), so the reward was 0.
    The relaxation applies only when the label is a plain number.
    """

    def test_trailing_unit_is_ignored(self):
        for pred, label in (
            ("16.76\\text{ ft}", "16.76"),
            ("16.76 \\text{ ft}", "16.76"),
            ("100\\text{ ft}^2", "100"),
            ("100\\text{ ft}^{2}", "100"),
            ("100 \\text{ square feet}", "100"),
            ("8\\mathrm{ft}", "8"),
            ("8\\text{ in}", "8"),
            ("12\\text{ cm}", "12"),  # already handled by normalize_final_answer
            ("30^\\circ", "30"),  # already handled by normalize_final_answer
        ):
            with self.subTest(pred=pred, label=label):
                self.assertEqual(_score(f"\\boxed{{{pred}}}", label).final_reward, 1.0)

    def test_unit_with_wrong_number_is_still_zero(self):
        for pred, label in (
            ("16.7\\text{ ft}", "16.76"),
            ("50\\text{ ft}", "5"),
            ("100\\text{ ft}^2", "10"),
        ):
            with self.subTest(pred=pred, label=label):
                self.assertEqual(_score(f"\\boxed{{{pred}}}", label).final_reward, 0.0)

    def test_variable_or_expression_is_not_a_unit(self):
        # Only a `\text{..}` / `\mathrm{..}` suffix is a unit; a bare letter or radical changes the value.
        for pred, label in (
            ("5x", "5"),
            ("5\\sqrt{2}", "5"),
            ("5\\pi", "5"),
            ("\\frac{5}{2}", "5"),
        ):
            with self.subTest(pred=pred, label=label):
                self.assertEqual(_score(f"\\boxed{{{pred}}}", label).final_reward, 0.0)

    def test_non_numeric_label_is_not_relaxed(self):
        self.assertEqual(_score("\\boxed{6\\sqrt{3}\\text{ ft}}", "6\\sqrt{3}").final_reward, 0.0)


class TestBoxedMathVariableApprox(unittest.TestCase):
    """`x \\approx N` is correct when N is exactly the numeric label.

    Seen in geo3k rollouts: the model boxes ``x\\approx 5.7`` for the label ``5.7``. The value is
    right, but the ``x \\approx`` prefix made the strings differ. Only an exact match of the number
    is accepted; a different rounding, an exact-vs-approximate form, or extra text stays wrong.
    """

    def test_variable_approx_with_exact_number_is_correct(self):
        for pred, label in (
            ("x\\approx 5.7", "5.7"),
            ("x \\approx 53.2", "53.2"),
            ("x \\approx 3.2", "3.2"),
            ("b \\approx 4.6", "4.6"),
            ("x\\approx 53.2 \\text{ units}", "53.2"),
        ):
            with self.subTest(pred=pred, label=label):
                self.assertEqual(_score(f"\\boxed{{{pred}}}", label).final_reward, 1.0)

    def test_different_rounding_is_still_zero(self):
        for pred, label in (
            ("x \\approx 53.17", "53.2"),
            ("b \\approx 4.62", "4.6"),
            ("x\\approx 5.4", "5.42"),
            ("x \\approx 38.12", "38.1"),
        ):
            with self.subTest(pred=pred, label=label):
                self.assertEqual(_score(f"\\boxed{{{pred}}}", label).final_reward, 0.0)

    def test_approximation_of_an_exact_label_is_still_zero(self):
        for pred, label in (
            ("x\\approx 27.71", "16 \\sqrt { 3 }"),
            ("x\\approx 7.5", "\\frac { 15 } { 2 }"),
        ):
            with self.subTest(pred=pred, label=label):
                self.assertEqual(_score(f"\\boxed{{{pred}}}", label).final_reward, 0.0)

    def test_extra_text_after_the_number_is_still_zero(self):
        for pred, label in (
            ("x \\approx 53.2 \\text{ (or } \\frac{32}{\\cos 53^\\circ}\\text{ exactly)}", "53.2"),
            ("x\\approx 5.4\\text{ (more precisely }5.42\\text{)}", "5.42"),
            ("x\\approx 5.42 \\text{ (or about }5.4\\text{)}", "5.42"),
        ):
            with self.subTest(pred=pred, label=label):
                self.assertEqual(_score(f"\\boxed{{{pred}}}", label).final_reward, 0.0)

    def test_two_alternative_answers_after_equals_is_still_zero(self):
        for pred, label in (
            ("x=\\frac{91}{8}\\ \\text{(or }11.375\\text{)}", "11.375"),
            ("x=\\frac{40}{3}\\text{ or }13\\frac{1}{3}", "\\frac { 40 } { 3 }"),
        ):
            with self.subTest(pred=pred, label=label):
                self.assertEqual(_score(f"\\boxed{{{pred}}}", label).final_reward, 0.0)


if __name__ == "__main__":
    unittest.main()
