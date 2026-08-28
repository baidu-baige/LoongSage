"""Unit tests for coda/data_factory/dataset.py."""

import base64
import hashlib
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from coda.data_factory.data_pre_processor import (
    BCP_CANARY,
    BCP_SOURCE_COLUMNS,
    bcp_pre_process,
    gsm8k_pre_process,
    r2e_gym_pre_process,
)
from coda.data_factory.dataset import (
    Dataset,
    _build_messages,
    _parse_generalized_path,
    read_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# _parse_generalized_path
# ---------------------------------------------------------------------------

class TestParseGeneralizedPath(unittest.TestCase):

    def test_plain_path_no_slice(self):
        path, sl = _parse_generalized_path("data.jsonl")
        self.assertEqual(path, "data.jsonl")
        self.assertIsNone(sl)

    def test_slice_both_bounds(self):
        path, sl = _parse_generalized_path("data.jsonl@[10:50]")
        self.assertEqual(path, "data.jsonl")
        self.assertEqual(sl, slice(10, 50))

    def test_slice_open_end(self):
        path, sl = _parse_generalized_path("data.jsonl@[5:]")
        self.assertEqual(path, "data.jsonl")
        self.assertEqual(sl, slice(5, None))

    def test_slice_open_start(self):
        path, sl = _parse_generalized_path("data.jsonl@[:20]")
        self.assertEqual(path, "data.jsonl")
        self.assertEqual(sl, slice(None, 20))

    def test_slice_both_open(self):
        path, sl = _parse_generalized_path("data.jsonl@[:]")
        self.assertEqual(path, "data.jsonl")
        self.assertEqual(sl, slice(None, None))

    def test_slice_negative_bounds(self):
        path, sl = _parse_generalized_path("data.jsonl@[-10:-1]")
        self.assertEqual(path, "data.jsonl")
        self.assertEqual(sl, slice(-10, -1))

    def test_path_with_directory_and_slice(self):
        path, sl = _parse_generalized_path("/some/dir/data.jsonl@[0:100]")
        self.assertEqual(path, "/some/dir/data.jsonl")
        self.assertEqual(sl, slice(0, 100))

    def test_no_slice_parquet(self):
        path, sl = _parse_generalized_path("/tmp/file.parquet")
        self.assertEqual(path, "/tmp/file.parquet")
        self.assertIsNone(sl)


# ---------------------------------------------------------------------------
# gsm8k_pre_process
# ---------------------------------------------------------------------------

class TestGsm8kPreProcess(unittest.TestCase):

    HINT = "#### <your_answer>"

    def test_string_prompt_gets_format_hint(self):
        data = gsm8k_pre_process({"question": "What is 1+1?"}, "question")
        self.assertEqual(
            data["question"],
            "What is 1+1?\nProvide the final answer in the format:\n#### <your_answer>",
        )

    def test_only_user_role_modified(self):
        data = gsm8k_pre_process(
            {
                "prompt": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Solve it."},
                ]
            },
            "prompt",
        )
        self.assertEqual(data["prompt"][0]["content"], "You are helpful.")
        self.assertIn(self.HINT, data["prompt"][1]["content"])

    def test_multiple_user_messages_all_modified(self):
        data = gsm8k_pre_process(
            {
                "prompt": [
                    {"role": "user", "content": "Q1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "Q2"},
                ]
            },
            "prompt",
        )
        self.assertIn(self.HINT, data["prompt"][0]["content"])
        self.assertEqual(data["prompt"][1]["content"], "A1")
        self.assertIn(self.HINT, data["prompt"][2]["content"])

    def test_missing_prompt_key_is_noop(self):
        data = gsm8k_pre_process({"other": "x"}, "question")
        self.assertEqual(data, {"other": "x"})

    def test_returns_same_dict_object(self):
        data = {"question": "Q"}
        self.assertIs(gsm8k_pre_process(data, "question"), data)

    def test_empty_message_list_returns_empty(self):
        data = gsm8k_pre_process({"prompt": []}, "prompt")
        self.assertEqual(data["prompt"], [])


# ---------------------------------------------------------------------------
# bcp_pre_process
# ---------------------------------------------------------------------------

class TestBCPPreProcess(unittest.TestCase):

    @staticmethod
    def _obfuscate(value):
        plain = value.encode("utf-8")
        key = hashlib.sha256(BCP_CANARY.encode("utf-8")).digest()
        full_key = key * (len(plain) // len(key)) + key[: len(plain) % len(key)]
        return base64.b64encode(
            bytes(a ^ b for a, b in zip(plain, full_key, strict=False))
        ).decode("utf-8")

    def _raw(self, query="Who won?", answer="Alice", query_id="q-1"):
        return {
            "query": self._obfuscate(query),
            "answer": self._obfuscate(answer),
            "query_id": query_id,
            "unused": "large unused value",
        }

    def test_decrypts_query_and_answer(self):
        data = bcp_pre_process(self._raw(), "prompt")
        self.assertEqual(data["prompt"], "Who won?")
        self.assertEqual(data["answer"], "Alice")
        self.assertEqual(data["metadata"], {"query_id": "q-1"})

    def test_plaintext_fields_are_accepted(self):
        data = bcp_pre_process(
            {"query": "Question", "answer": "Answer", "query_id": "q"}, "prompt"
        )
        self.assertEqual(data["prompt"], "Question")
        self.assertEqual(data["answer"], "Answer")

    def test_missing_query_or_answer_raises_value_error(self):
        with self.assertRaises(ValueError):
            bcp_pre_process(
                {"query": "", "answer": "Answer", "query_id": "q"}, "prompt"
            )
        with self.assertRaises(ValueError):
            bcp_pre_process(
                {"query": "Question", "answer": "", "query_id": "q"}, "prompt"
            )

    def test_prepared_record_is_unchanged(self):
        data = {
            "prompt": [{"role": "user", "content": "Question"}],
            "answer": "Answer",
            "metadata": {"query_id": "q"},
        }
        self.assertIs(bcp_pre_process(data, "prompt"), data)
        self.assertEqual(data["prompt"][0]["content"], "Question")

    def test_declares_only_required_source_columns(self):
        self.assertEqual(bcp_pre_process.source_columns, BCP_SOURCE_COLUMNS)

    def test_dataset_reads_sharded_parquet_directory(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmpdir:
            pq.write_table(
                pa.Table.from_pylist([self._raw(query_id="q-1")]),
                os.path.join(tmpdir, "part-1.parquet"),
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [self._raw(query="Where?", answer="Paris", query_id="q-2")]
                ),
                os.path.join(tmpdir, "part-2.parquet"),
            )
            dataset = Dataset(
                tmpdir,
                max_length=None,
                prompt_key="prompt",
                label_key="answer",
                metadata_key="metadata",
                data_pre_processor=bcp_pre_process,
            )

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0].prompt, [{"role": "user", "content": "Who won?"}])
        self.assertEqual(dataset[0].label, {"value": "Alice"})
        self.assertEqual(dataset[0].metadata, {"query_id": "q-1"})
        self.assertEqual(dataset[1].prompt, [{"role": "user", "content": "Where?"}])
        self.assertEqual(dataset[1].label, {"value": "Paris"})


# ---------------------------------------------------------------------------
# r2e_gym_pre_process
# ---------------------------------------------------------------------------

class TestR2EGymPreProcess(unittest.TestCase):

    RAW = {
        "repo_name": "orange3",
        "commit_hash": "2d9617bd0cb1f0ba61771258410ab8fae8e7e24d",
        "prompt": "You are an expert software engineer tasked with creating GitHub issues...",
        "problem_statement": "[ISSUE]\nContext migration fails.",
        "docker_image": "namanjain12/orange3_final:2d9617bd",
        "expected_output_json": '{"TestContextHandler.test_close_context": "PASSED"}',
    }

    def _process(self, **overrides):
        return r2e_gym_pre_process({**self.RAW, **overrides}, "prompt")

    def test_prompt_taken_from_problem_statement(self):
        self.assertEqual(self._process()["prompt"], self.RAW["problem_statement"])

    def test_metadata_carries_agent_and_reward_fields(self):
        metadata = self._process()["metadata"]
        self.assertEqual(metadata["docker_image"], self.RAW["docker_image"])
        self.assertEqual(metadata["expected_output_json"], self.RAW["expected_output_json"])
        self.assertEqual(metadata["repo_path"], "/testbed")
        self.assertEqual(metadata["instance_id"], f"orange3@{self.RAW['commit_hash']}")

    def test_missing_required_field_raises_value_error(self):
        for key in ("repo_name", "commit_hash", "problem_statement", "docker_image", "expected_output_json"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self._process(**{key: None})

    def test_blank_required_field_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._process(docker_image="   ")

    def test_dataset_end_to_end_with_pre_processor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "raw.jsonl")
            _write_jsonl(path, [self.RAW])
            ds = Dataset(path, max_length=None, prompt_key="prompt", data_pre_processor=r2e_gym_pre_process)
            self.assertEqual(ds[0].prompt, [{"role": "user", "content": self.RAW["problem_statement"]}])
            self.assertEqual(ds[0].metadata["docker_image"], self.RAW["docker_image"])


# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------

class TestBuildMessages(unittest.TestCase):

    def test_string_wrapped_in_user_message(self):
        result = _build_messages({"text": "Hello world"}, "text")
        self.assertEqual(result, [{"role": "user", "content": "Hello world"}])

    def test_list_returned_as_is(self):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]
        result = _build_messages({"text": msgs}, "text")
        self.assertIs(result, msgs)

    def test_missing_key_returns_none(self):
        result = _build_messages({"other": "val"}, "text")
        self.assertIsNone(result)

    def test_none_value_returns_none(self):
        result = _build_messages({"text": None}, "text")
        self.assertIsNone(result)

    def test_custom_prompt_key(self):
        result = _build_messages({"prompt": "hello"}, "prompt")
        self.assertEqual(result, [{"role": "user", "content": "hello"}])

    def test_empty_list_returned_as_is(self):
        result = _build_messages({"text": []}, "text")
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

class TestReadFile(unittest.TestCase):

    def setUp(self):
        self.tmpdir = self.enterContext(tempfile.TemporaryDirectory())

    def _path(self, name):
        return os.path.join(self.tmpdir, name)

    def test_read_all_rows(self):
        path = self._path("data.jsonl")
        records = [{"id": i} for i in range(5)]
        _write_jsonl(path, records)
        self.assertEqual(list(read_file(path)), records)

    def test_read_with_slice(self):
        path = self._path("data.jsonl")
        records = [{"id": i} for i in range(10)]
        _write_jsonl(path, records)
        self.assertEqual(list(read_file(f"{path}@[2:5]")), records[2:5])

    def test_read_open_start_slice(self):
        path = self._path("data.jsonl")
        records = [{"id": i} for i in range(6)]
        _write_jsonl(path, records)
        self.assertEqual(list(read_file(f"{path}@[:3]")), records[:3])

    def test_read_open_end_slice(self):
        path = self._path("data.jsonl")
        records = [{"id": i} for i in range(6)]
        _write_jsonl(path, records)
        self.assertEqual(list(read_file(f"{path}@[4:]")), records[4:])

    def test_skips_empty_lines(self):
        path = self._path("empty_lines.jsonl")
        with open(path, "w") as f:
            f.write('{"a": 1}\n\n{"b": 2}\n')
        self.assertEqual(list(read_file(path)), [{"a": 1}, {"b": 2}])

    def test_skips_invalid_json_lines(self):
        path = self._path("bad.jsonl")
        with open(path, "w") as f:
            f.write('{"good": 1}\nnot-json\n{"good": 2}\n')
        self.assertEqual(list(read_file(path)), [{"good": 1}, {"good": 2}])

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            list(read_file("/nonexistent/path.jsonl"))

    def test_file_not_found_with_slice(self):
        with self.assertRaises(FileNotFoundError):
            list(read_file("/nonexistent/path.jsonl@[0:10]"))

    def test_unsupported_format_raises_value_error(self):
        path = self._path("data.csv")
        open(path, "w").close()
        with self.assertRaises(ValueError):
            list(read_file(path))

    def test_parquet_without_pyarrow_raises_import_error(self):
        path = self._path("data.parquet")
        open(path, "w").close()
        with patch("coda.data_factory.dataset.pq", None):
            with self.assertRaises(ImportError):
                list(read_file(path))

    # ── directory input ──────────────────────────────────────────────────

    def test_directory_concatenates_shards_in_sorted_order(self):
        os.mkdir(self._path("data"))
        _write_jsonl(self._path("data/train-00001-of-00002.jsonl"), [{"id": 2}])
        _write_jsonl(self._path("data/train-00000-of-00002.jsonl"), [{"id": 0}, {"id": 1}])
        self.assertEqual(
            list(read_file(self._path("data"))),
            [{"id": 0}, {"id": 1}, {"id": 2}],
        )

    def test_directory_slice_applies_to_concatenated_stream(self):
        os.mkdir(self._path("data"))
        _write_jsonl(self._path("data/a.jsonl"), [{"id": i} for i in range(3)])
        _write_jsonl(self._path("data/b.jsonl"), [{"id": i} for i in range(3, 6)])
        self.assertEqual(
            list(read_file(f"{self._path('data')}@[2:4]")),
            [{"id": 2}, {"id": 3}],
        )

    def test_directory_ignores_non_dataset_files_and_hidden_dirs(self):
        _write_jsonl(self._path("train.jsonl"), [{"id": 0}])
        open(self._path("README.md"), "w").close()
        os.mkdir(self._path(".cache"))
        _write_jsonl(self._path(".cache/stale.jsonl"), [{"id": 99}])
        self.assertEqual(list(read_file(self.tmpdir)), [{"id": 0}])

    def test_empty_directory_raises_value_error(self):
        os.mkdir(self._path("empty"))
        with self.assertRaises(ValueError):
            list(read_file(self._path("empty")))

    # ── column projection ────────────────────────────────────────────────

    def _write_parquet(self, name, rows):
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = self._path(name)
        pq.write_table(pa.Table.from_pylist(rows), path)
        return path

    def test_parquet_projection_reads_only_requested_columns(self):
        path = self._write_parquet("p.parquet", [{"prompt": "q", "big": "x" * 100, "keep": "1"}])
        self.assertEqual(list(read_file(path, columns=["prompt", "keep"])), [{"prompt": "q", "keep": "1"}])

    def test_parquet_projection_ignores_unknown_columns(self):
        path = self._write_parquet("p.parquet", [{"prompt": "q", "big": "x"}])
        self.assertEqual(list(read_file(path, columns=["prompt", "absent"])), [{"prompt": "q"}])

    def test_parquet_projection_without_any_match_reads_all_columns(self):
        path = self._write_parquet("p.parquet", [{"prompt": "q", "big": "x"}])
        self.assertEqual(list(read_file(path, columns=["absent"])), [{"prompt": "q", "big": "x"}])

    def test_projection_ignored_for_jsonl(self):
        path = self._path("data.jsonl")
        _write_jsonl(path, [{"prompt": "q", "big": "x"}])
        self.assertEqual(list(read_file(path, columns=["prompt"])), [{"prompt": "q", "big": "x"}])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TestDataset(unittest.TestCase):

    def setUp(self):
        self.tmpdir = self.enterContext(tempfile.TemporaryDirectory())

    def _make_jsonl(self, records, name="ds.jsonl"):
        path = os.path.join(self.tmpdir, name)
        _write_jsonl(path, records)
        return path

    # ── loading ──────────────────────────────────────────────────────────

    def test_string_prompts_wrapped(self):
        path = self._make_jsonl([{"text": f"p{i}"} for i in range(3)])
        ds = Dataset(path, max_length=None)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0].prompt, [{"role": "user", "content": "p0"}])

    def test_list_prompts_kept_as_is(self):
        msgs = [{"role": "user", "content": "hi"}]
        path = self._make_jsonl([{"text": msgs}])
        ds = Dataset(path, max_length=None)
        self.assertEqual(ds[0].prompt, msgs)

    def test_label_key(self):
        path = self._make_jsonl([{"text": "q", "answer": "42"}])
        ds = Dataset(path, max_length=None, label_key="answer")
        self.assertEqual(ds[0].label, {"value": "42"})

    def test_label_is_none_by_default(self):
        path = self._make_jsonl([{"text": "q", "answer": "42"}])
        ds = Dataset(path, max_length=None)
        self.assertIsNone(ds[0].label)

    def test_metadata_loaded(self):
        path = self._make_jsonl([{"text": "q", "metadata": {"src": "web"}}])
        ds = Dataset(path, max_length=None)
        self.assertEqual(ds[0].metadata["src"], "web")

    def test_missing_metadata_defaults_to_empty_dict(self):
        path = self._make_jsonl([{"text": "q"}])
        ds = Dataset(path, max_length=None)
        self.assertEqual(ds[0].metadata, {})

    def test_custom_prompt_key(self):
        path = self._make_jsonl([{"prompt": "hello"}])
        ds = Dataset(path, max_length=None, prompt_key="prompt")
        self.assertEqual(ds[0].prompt, [{"role": "user", "content": "hello"}])

    def test_slice_path(self):
        records = [{"text": f"r{i}"} for i in range(10)]
        path = self._make_jsonl(records)
        ds = Dataset(f"{path}@[3:7]", max_length=None)
        self.assertEqual(len(ds), 4)
        self.assertEqual(ds[0].prompt[0]["content"], "r3")

    def test_empty_file(self):
        path = self._make_jsonl([])
        ds = Dataset(path, max_length=None)
        self.assertEqual(len(ds), 0)

    def test_data_pre_processor_applied_before_build_messages(self):
        path = self._make_jsonl([{"text": "What is 2+2?"}])
        ds = Dataset(path, max_length=None, data_pre_processor=gsm8k_pre_process)
        content = ds[0].prompt[0]["content"]
        self.assertIn("#### <your_answer>", content)
        self.assertTrue(content.startswith("What is 2+2?"))

    def test_data_pre_processor_receives_prompt_key(self):
        path = self._make_jsonl([{"question": "What is 2+2?"}])
        seen = {}

        def pre_process(data, prompt_key):
            seen["prompt_key"] = prompt_key
            data[prompt_key] = data[prompt_key].upper()
            return data

        ds = Dataset(path, max_length=None, prompt_key="question", data_pre_processor=pre_process)
        self.assertEqual(seen["prompt_key"], "question")
        self.assertEqual(ds[0].prompt[0]["content"], "WHAT IS 2+2?")

    def test_label_already_dict_not_wrapped(self):
        path = self._make_jsonl([{"text": "q", "answer": {"value": "42", "extra": 1}}])
        ds = Dataset(path, max_length=None, label_key="answer")
        self.assertEqual(ds[0].label, {"value": "42", "extra": 1})

    # ── column projection ────────────────────────────────────────────────

    def _make_parquet(self, rows, name="ds.parquet"):
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = os.path.join(self.tmpdir, name)
        pq.write_table(pa.Table.from_pylist(rows), path)
        return path

    PROJECTION_ROW = {"text": "raw", "problem_statement": "issue", "unused": "blob"}

    def _capturing_pre_process(self, source_columns=None):
        seen = []

        def pre_process(data, prompt_key):
            seen.append(sorted(data))
            data[prompt_key] = data["problem_statement"]
            return data

        if source_columns is not None:
            pre_process.source_columns = source_columns

        return pre_process, seen

    def test_declared_source_columns_restrict_parquet_read(self):
        path = self._make_parquet([self.PROJECTION_ROW])
        pre_process, seen = self._capturing_pre_process(source_columns=("problem_statement",))
        ds = Dataset(path, max_length=None, data_pre_processor=pre_process)
        self.assertEqual(seen, [["problem_statement", "text"]])
        self.assertEqual(ds[0].prompt[0]["content"], "issue")

    def test_undeclared_source_columns_read_everything(self):
        path = self._make_parquet([self.PROJECTION_ROW])
        pre_process, seen = self._capturing_pre_process()
        Dataset(path, max_length=None, data_pre_processor=pre_process)
        self.assertEqual(seen, [["problem_statement", "text", "unused"]])

    def test_without_pre_processor_only_key_columns_are_read(self):
        path = self._make_parquet([{"text": "q", "answer": "42", "unused": "blob"}])
        ds = Dataset(path, max_length=None, label_key="answer")
        self.assertEqual(ds[0].prompt[0]["content"], "q")
        self.assertEqual(ds[0].label, {"value": "42"})

    def test_no_pre_processor_leaves_prompt_unchanged(self):
        path = self._make_jsonl([{"text": "q"}])
        ds = Dataset(path, max_length=None)
        self.assertEqual(ds[0].prompt[0]["content"], "q")

    # ── max_length filtering ─────────────────────────────────────────────

    def test_max_length_drops_long_string_prompts(self):
        path = self._make_jsonl([{"text": "short"}, {"text": "a" * 1000}])
        ds = Dataset(path, max_length=10)
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0].prompt[0]["content"], "short")

    def test_max_length_none_keeps_all(self):
        path = self._make_jsonl([{"text": "x" * 500}])
        ds = Dataset(path, max_length=None)
        self.assertEqual(len(ds), 1)

    def test_max_length_boundary_inclusive(self):
        path = self._make_jsonl([{"text": "12345"}])
        self.assertEqual(len(Dataset(path, max_length=5)), 1)

    def test_max_length_boundary_exceeded(self):
        path = self._make_jsonl([{"text": "123456"}])
        self.assertEqual(len(Dataset(path, max_length=5)), 0)

    def test_max_length_list_prompt_sums_content(self):
        # content lengths: len("abc") + len("def") = 6
        msgs = [{"role": "user", "content": "abc"}, {"role": "assistant", "content": "def"}]
        path = self._make_jsonl([{"text": msgs}])
        self.assertEqual(len(Dataset(path, max_length=6)), 1)
        self.assertEqual(len(Dataset(path, max_length=5)), 0)

    # ── shuffle ──────────────────────────────────────────────────────────

    def test_shuffle_changes_order(self):
        records = [{"text": str(i)} for i in range(20)]
        path = self._make_jsonl(records)
        ds = Dataset(path, max_length=None, seed=0)
        before = [t.prompt[0]["content"] for t in ds.prompts]
        ds.shuffle(1)
        self.assertNotEqual(before, [t.prompt[0]["content"] for t in ds.prompts])

    def test_shuffle_same_epoch_is_idempotent(self):
        records = [{"text": str(i)} for i in range(10)]
        path = self._make_jsonl(records)
        ds = Dataset(path, max_length=None, seed=42)
        ds.shuffle(3)
        order_first = [t.prompt[0]["content"] for t in ds.prompts]
        ds.shuffle(3)
        self.assertEqual(order_first, [t.prompt[0]["content"] for t in ds.prompts])

    def test_shuffle_different_epochs_differ(self):
        records = [{"text": str(i)} for i in range(20)]
        path = self._make_jsonl(records)
        ds = Dataset(path, max_length=None, seed=7)
        ds.shuffle(0)
        order0 = [t.prompt[0]["content"] for t in ds.prompts]
        ds.shuffle(1)
        self.assertNotEqual(order0, [t.prompt[0]["content"] for t in ds.prompts])

    def test_shuffle_does_not_mutate_origin_prompts(self):
        records = [{"text": str(i)} for i in range(10)]
        path = self._make_jsonl(records)
        ds = Dataset(path, max_length=None, seed=1)
        original_copy = list(ds.origin_prompts)
        ds.shuffle(5)
        self.assertEqual(ds.origin_prompts, original_copy)

    def test_epoch_id_starts_at_minus_one(self):
        path = self._make_jsonl([{"text": "x"}])
        ds = Dataset(path, max_length=None)
        self.assertEqual(ds.epoch_id, -1)

    def test_epoch_id_updated_after_shuffle(self):
        path = self._make_jsonl([{"text": "x"}])
        ds = Dataset(path, max_length=None)
        ds.shuffle(2)
        self.assertEqual(ds.epoch_id, 2)

    def test_shuffle_reproducible_with_same_seed(self):
        records = [{"text": str(i)} for i in range(15)]
        path = self._make_jsonl(records)
        ds1 = Dataset(path, max_length=None, seed=99)
        ds1.shuffle(0)
        ds2 = Dataset(path, max_length=None, seed=99)
        ds2.shuffle(0)
        self.assertEqual(
            [t.prompt[0]["content"] for t in ds1.prompts],
            [t.prompt[0]["content"] for t in ds2.prompts],
        )

    # ── __getitem__ / __len__ ────────────────────────────────────────────

    def test_getitem_returns_correct_prompt(self):
        records = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        path = self._make_jsonl(records)
        ds = Dataset(path, max_length=None)
        self.assertEqual(ds[1].prompt, [{"role": "user", "content": "b"}])

    def test_getitem_negative_index(self):
        records = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        path = self._make_jsonl(records)
        ds = Dataset(path, max_length=None)
        self.assertEqual(ds[-1].prompt, [{"role": "user", "content": "c"}])

    def test_len(self):
        records = [{"text": f"r{i}"} for i in range(7)]
        path = self._make_jsonl(records)
        ds = Dataset(path, max_length=None)
        self.assertEqual(len(ds), 7)


# ---------------------------------------------------------------------------
# Dataset.filter_long_prompt (unit-level, using an empty-file Dataset)
# ---------------------------------------------------------------------------

class TestFilterLongPrompt(unittest.TestCase):

    def setUp(self):
        self.tmpdir = self.enterContext(tempfile.TemporaryDirectory())
        path = os.path.join(self.tmpdir, "empty.jsonl")
        open(path, "w").close()
        self.ds = Dataset(path, max_length=None)

    def _make_prompts(self, prompts):
        from coda.agentflow.trajectory_store import Trajectory
        return [Trajectory(trajectory_id=f"t{i}", prompt_id="p0", prompt=p) for i, p in enumerate(prompts)]

    def test_none_max_length_returns_original(self):
        prompts = self._make_prompts(["abc", "def"])
        result = self.ds.filter_long_prompt(prompts, None)
        self.assertIs(result, prompts)

    def test_empty_list_returns_empty(self):
        result = self.ds.filter_long_prompt([], max_length=10)
        self.assertEqual(result, [])

    def test_string_prompt_filtered(self):
        prompts = self._make_prompts(["hi", "x" * 100])
        result = self.ds.filter_long_prompt(prompts, max_length=10)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].prompt, "hi")

    def test_list_prompt_content_summed(self):
        msgs = [{"role": "user", "content": "abc"}, {"role": "assistant", "content": "def"}]
        prompts = self._make_prompts([msgs])   # total = 6
        self.assertEqual(len(self.ds.filter_long_prompt(prompts, max_length=6)), 1)
        self.assertEqual(len(self.ds.filter_long_prompt(prompts, max_length=5)), 0)

    def test_boundary_inclusive(self):
        prompts = self._make_prompts(["exact"])   # length = 5
        self.assertEqual(len(self.ds.filter_long_prompt(prompts, max_length=5)), 1)
        self.assertEqual(len(self.ds.filter_long_prompt(prompts, max_length=4)), 0)


if __name__ == "__main__":
    unittest.main()
