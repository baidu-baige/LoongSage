"""Unit tests for coda/utils/checkpoint_utils.py.

Scope: ``resolve_dist_ckpt_dir``, the resolution of the read-only weight-source
paths (``ref_dist_ckpt_path``, ``opd.teachers[].dist_ckpt_path``). These name one
concrete ``dist_ckpt`` dir. Unlike the student's ``checkpoint_path`` they are
purely inputs -- there is no "first run, nothing saved yet" case -- so a path
that is set but unusable must raise instead of silently falling back to the HF
weights, which would train against the wrong model without any error.

The student's tracker-file functions are covered here too, since they keep the
old base-dir semantics and the two must not be conflated.
"""

import os

import pytest

from coda.utils.checkpoint_utils import (
    DIST_CKPT_CONFIG_FILE,
    find_latest_ckpt_path,
    get_ckpt_dir,
    get_data_source_dir,
    get_hf_dir,
    get_tracker_file,
    resolve_dist_ckpt_dir,
)

KEY = "ref_dist_ckpt_path"


def make_dist_ckpt(parent, name="dist_ckpt"):
    """Create a directory that passes the dist-checkpoint predicate."""
    d = parent / name
    d.mkdir(parents=True)
    (d / DIST_CKPT_CONFIG_FILE).write_text('{"sharded_backend": "torch_dist"}')
    return d


def make_base_dir(parent, step=7):
    """Create an old-style checkpoint base dir (tracker file + train_step_N)."""
    base = parent / "run"
    make_dist_ckpt(base / f"train_step_{step}")
    (base / "latest_checkpointed_iteration.txt").write_text(str(step))
    return base


class TestResolveDistCkptDirUnset:
    """An unset path is not an error -- the caller falls back to its HF source."""

    @pytest.mark.parametrize("value", [None, ""])
    def test_unset_returns_none(self, value):
        assert resolve_dist_ckpt_dir(value, KEY) is None


class TestResolveDistCkptDirAccepts:
    def test_returns_the_configured_dir(self, tmp_path):
        d = make_dist_ckpt(tmp_path)
        assert resolve_dist_ckpt_dir(str(d), KEY) == str(d)

    def test_normalizes_trailing_slash(self, tmp_path):
        """Normalizing keeps teacher_lm_head's _by_source dedup from splitting."""
        d = make_dist_ckpt(tmp_path)
        assert resolve_dist_ckpt_dir(str(d) + "/", KEY) == str(d)
        assert resolve_dist_ckpt_dir(str(d) + "//.", KEY) == str(d)


class TestResolveDistCkptDirRejects:
    def test_nonexistent_path(self, tmp_path):
        with pytest.raises(ValueError, match="is not a directory"):
            resolve_dist_ckpt_dir(str(tmp_path / "nope"), KEY)

    def test_file_instead_of_dir(self, tmp_path):
        f = tmp_path / "ckpt"
        f.write_text("")
        with pytest.raises(ValueError, match="is not a directory"):
            resolve_dist_ckpt_dir(str(f), KEY)

    def test_dir_without_metadata_json(self, tmp_path):
        """A real dir holding no dist checkpoint is still a misconfiguration."""
        with pytest.raises(ValueError, match=DIST_CKPT_CONFIG_FILE):
            resolve_dist_ckpt_dir(str(tmp_path), KEY)

    def test_old_base_dir_layout(self, tmp_path):
        """The pre-rename value (a base dir) is rejected, not silently scanned."""
        base = make_base_dir(tmp_path)
        with pytest.raises(ValueError, match=DIST_CKPT_CONFIG_FILE):
            resolve_dist_ckpt_dir(str(base), KEY)

    def test_error_names_the_offending_key(self, tmp_path):
        key = "opd.teachers[1].dist_ckpt_path"
        with pytest.raises(ValueError, match=r"opd\.teachers\[1\]\.dist_ckpt_path"):
            resolve_dist_ckpt_dir(str(tmp_path / "nope"), key)

    def test_error_shows_the_expected_layer(self, tmp_path):
        """Pointing one level too high is the likely mistake, so say what to write."""
        with pytest.raises(ValueError, match=r"train_step_100/dist_ckpt"):
            resolve_dist_ckpt_dir(str(tmp_path), KEY)


class TestStudentPathsUnchanged:
    """The student's base-dir semantics must survive the ref/teacher change."""

    def test_find_latest_reads_tracker_file(self, tmp_path):
        base = make_base_dir(tmp_path, step=7)
        assert find_latest_ckpt_path(str(base)) == get_ckpt_dir(str(base), 7)

    def test_find_latest_returns_none_on_first_run(self, tmp_path):
        """No tracker file means "nothing saved yet", which is not an error."""
        assert find_latest_ckpt_path(str(tmp_path)) is None
        assert find_latest_ckpt_path(str(tmp_path / "nope")) is None

    def test_find_latest_returns_none_when_tracker_dangles(self, tmp_path):
        base = tmp_path / "run"
        base.mkdir()
        (base / "latest_checkpointed_iteration.txt").write_text("99")
        assert find_latest_ckpt_path(str(base)) is None

    def test_step_dir_builders(self):
        assert get_ckpt_dir("/b", 5) == os.path.join("/b", "train_step_5", "dist_ckpt")
        assert get_hf_dir("/b", 5) == os.path.join("/b", "train_step_5", "hf_model")
        assert get_data_source_dir("/b", 5) == os.path.join("/b", "train_step_5", "data_source")
        assert get_tracker_file("/b") == "/b/latest_checkpointed_iteration.txt"
