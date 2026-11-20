"""Unit tests for the BrowseComp-Plus (bcp) reward."""

import json
import unittest

from omegaconf import OmegaConf

from coda.reward import create_reward_fn
from coda.reward.functions.bcp import _outcome_reward


def _finish(answer):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_finish",
            "type": "function",
            "function": {"name": "finish", "arguments": json.dumps({"answer": answer})},
        }],
    }


class TestBcpOutcomeReward(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(_outcome_reward("Georgia Hirst", "Georgia Hirst"), 1.0)

    def test_match_after_normalisation(self):
        self.assertEqual(_outcome_reward("  georgia hirst. ", "Georgia Hirst"), 1.0)
        self.assertEqual(_outcome_reward("The Dial", "Dial"), 1.0)
        self.assertEqual(_outcome_reward("1,944", "1944"), 1.0)

    def test_empty_prediction_is_zero(self):
        self.assertEqual(_outcome_reward(None, "Boston"), 0.0)
        self.assertEqual(_outcome_reward("", "Boston"), 0.0)

    def test_answer_fragment_is_not_credited(self):
        self.assertEqual(_outcome_reward("Hirst", "Georgia Hirst"), 0.0)
        self.assertEqual(_outcome_reward("University", "Queen Arwa University"), 0.0)

    def test_numeric_superset_is_not_credited(self):
        self.assertEqual(_outcome_reward("2004", "200"), 0.0)
        self.assertEqual(_outcome_reward("21", "2"), 0.0)

    def test_list_of_guesses_is_not_credited(self):
        self.assertEqual(_outcome_reward("Boston, Chicago, New York, Paris", "Boston"), 0.0)

    def test_short_guess_inside_answer_is_not_credited(self):
        self.assertEqual(_outcome_reward("1", "1942"), 0.0)
        self.assertEqual(_outcome_reward("no", "Nobel Prize"), 0.0)


class TestBcpReward(unittest.TestCase):
    def setUp(self):
        self.reward_fn = create_reward_fn(OmegaConf.create({"name": "bcp"}))

    def test_fragment_answer_gets_process_reward_only(self):
        messages = [{"role": "user", "content": "q"}, _finish("Hirst")]
        reward = self.reward_fn(messages, {"ground_truth": "Georgia Hirst"}, {})
        self.assertAlmostEqual(reward.final_reward, 0.2)
        self.assertFalse(reward.is_correct)

    def test_correct_answer(self):
        messages = [{"role": "user", "content": "q"}, _finish("Georgia Hirst")]
        reward = self.reward_fn(messages, {"ground_truth": "Georgia Hirst"}, {})
        self.assertAlmostEqual(reward.final_reward, 1.2)
        self.assertTrue(reward.is_correct)


if __name__ == "__main__":
    unittest.main()
