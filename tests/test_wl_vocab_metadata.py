"""
Tests for wl_vocab_cache's vocab-metadata plumbing (sage/domains/gym_taxi/utils/
wl_vocab_cache.py): is_wl_override_configured(), get_wl_vocab_metadata(),
validate_wl_vocab_metadata(), and configure_wl_vocab_override()'s graph_convention
check - plus the SAME validate_wl_vocab_metadata check as run from
WLPlanFeedbackPolicy._load_vocab (sage/agent/wl_plan_feedback_policy.py), the other of
the two independent places a vocab file gets loaded for a real run.

Every test that calls configure_wl_vocab_override() resets it in tearDown - this module
mutates process-global state (see wl_vocab_cache.py's own docstring), and the full test
suite runs every file in one process, so a leaked override would silently affect
unrelated tests running after this file.

Run from the repo root with:
    python -m pytest tests/test_wl_vocab_metadata.py -v
"""
import json
import tempfile
import os
import unittest

from sage.domains.gym_taxi.utils.wl_vocab_cache import (
    configure_wl_vocab_override,
    reset_wl_vocab_override,
    is_wl_override_configured,
    get_wl_vocab_metadata,
    validate_wl_vocab_metadata,
)
from sage.agent.wl_plan_feedback_policy import _load_vocab


def write_vocab(path, graph_convention=None, num_iterations=None):
    """A minimal, structurally-valid vocab file (one OOV entry - the smallest thing
    _load_vocab_from_path/_load_vocab can actually decode), with or without metadata."""
    payload = {"vocab_size": 1, "entries": [{"signature": {"kind": "oov"}, "id": 0}]}
    if graph_convention is not None:
        payload["graph_convention"] = graph_convention
    if num_iterations is not None:
        payload["num_iterations"] = num_iterations
    with open(path, "w") as f:
        json.dump(payload, f)


class TempVocabDir(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmpdir.name

    def tearDown(self):
        self._tmpdir.cleanup()
        reset_wl_vocab_override()

    def path(self, name):
        return os.path.join(self.tmpdir, name)


class TestIsOverrideConfigured(TempVocabDir):
    def test_false_by_default(self):
        self.assertFalse(is_wl_override_configured())

    def test_true_after_configure(self):
        p = self.path("v.json")
        write_vocab(p, graph_convention="atom", num_iterations=2)
        configure_wl_vocab_override(p, 2, graph_convention="atom")
        self.assertTrue(is_wl_override_configured())

    def test_false_after_reset(self):
        p = self.path("v.json")
        write_vocab(p, graph_convention="atom", num_iterations=2)
        configure_wl_vocab_override(p, 2, graph_convention="atom")
        reset_wl_vocab_override()
        self.assertFalse(is_wl_override_configured())


class TestGetWlVocabMetadata(TempVocabDir):
    def test_none_for_default_oracle_sage_vocab(self):
        # No override configured - falls back to WL_VOCAB_PATH, an old vocab file with
        # no recorded metadata (predates this feature).
        self.assertIsNone(get_wl_vocab_metadata())

    def test_returns_recorded_metadata_for_override(self):
        p = self.path("v.json")
        write_vocab(p, graph_convention="atom", num_iterations=2)
        configure_wl_vocab_override(p, 2, graph_convention="atom")
        self.assertEqual(get_wl_vocab_metadata(), {"graph_convention": "atom", "num_iterations": 2})

    def test_none_for_metadata_less_override(self):
        p = self.path("v.json")
        write_vocab(p)  # no metadata
        configure_wl_vocab_override(p, 1)  # graph_convention=None -> allowed even metadata-less
        self.assertIsNone(get_wl_vocab_metadata())


class TestValidateWlVocabMetadataDirect(unittest.TestCase):
    """Direct tests of the shared validator, independent of any file I/O."""

    def test_none_metadata_passes_for_non_atom_expectation(self):
        validate_wl_vocab_metadata(None, "oracle_sage", 1, "test")
        validate_wl_vocab_metadata(None, None, 1, "test")  # no expected convention at all

    def test_none_metadata_raises_for_atom_expectation(self):
        with self.assertRaises(ValueError):
            validate_wl_vocab_metadata(None, "atom", 2, "test")

    def test_matching_metadata_passes(self):
        validate_wl_vocab_metadata({"graph_convention": "atom", "num_iterations": 2}, "atom", 2, "test")

    def test_convention_mismatch_raises(self):
        with self.assertRaises(ValueError):
            validate_wl_vocab_metadata({"graph_convention": "vilg", "num_iterations": 2}, "atom", 2, "test")

    def test_num_iterations_mismatch_raises(self):
        with self.assertRaises(ValueError):
            validate_wl_vocab_metadata({"graph_convention": "atom", "num_iterations": 1}, "atom", 2, "test")

    def test_convention_check_skipped_when_expected_is_none(self):
        # metadata says "vilg", caller didn't ask for a specific convention -> only L is checked
        validate_wl_vocab_metadata({"graph_convention": "vilg", "num_iterations": 2}, None, 2, "test")


class TestConfigureWlVocabOverrideGuards(TempVocabDir):
    """configure_wl_vocab_override() (env_to_graph/Planner.plan's vocab-loading path)."""

    def test_atom_metadata_less_vocab_raises(self):
        p = self.path("v.json")
        write_vocab(p)  # no metadata at all
        with self.assertRaises(ValueError):
            configure_wl_vocab_override(p, 2, graph_convention="atom")
        self.assertFalse(is_wl_override_configured())  # rejected override never took effect

    def test_atom_wrong_recorded_convention_raises(self):
        p = self.path("v.json")
        write_vocab(p, graph_convention="vilg", num_iterations=2)
        with self.assertRaises(ValueError):
            configure_wl_vocab_override(p, 2, graph_convention="atom")
        self.assertFalse(is_wl_override_configured())

    def test_atom_wrong_num_iterations_raises(self):
        p = self.path("v.json")
        write_vocab(p, graph_convention="atom", num_iterations=1)
        with self.assertRaises(ValueError):
            configure_wl_vocab_override(p, 2, graph_convention="atom")
        self.assertFalse(is_wl_override_configured())

    def test_atom_matching_metadata_succeeds(self):
        p = self.path("v.json")
        write_vocab(p, graph_convention="atom", num_iterations=2)
        configure_wl_vocab_override(p, 2, graph_convention="atom")
        self.assertTrue(is_wl_override_configured())

    def test_oracle_sage_metadata_less_vocab_still_works(self):
        """Old vocab files without metadata (every oracle_sage/vilg vocab shipped before
        this feature existed) must keep loading - existing tests/workflows stay green."""
        p = self.path("v.json")
        write_vocab(p)  # no metadata
        configure_wl_vocab_override(p, 1, graph_convention="oracle_sage")
        self.assertTrue(is_wl_override_configured())

    def test_no_graph_convention_argument_skips_convention_check_even_with_metadata(self):
        """Backward-compatible call shape (graph_convention omitted, the pre-existing
        2-arg form) still only checks L, never convention, even once metadata exists."""
        p = self.path("v.json")
        write_vocab(p, graph_convention="vilg", num_iterations=2)
        configure_wl_vocab_override(p, 2)  # no graph_convention kwarg
        self.assertTrue(is_wl_override_configured())


class TestLoadVocabGuards(TempVocabDir):
    """WLPlanFeedbackPolicy._load_vocab (the policy's own, independent vocab-loading
    path) - same validate_wl_vocab_metadata rules, applied the same way."""

    def test_atom_metadata_less_vocab_raises(self):
        p = self.path("v.json")
        write_vocab(p)
        with self.assertRaises(ValueError):
            _load_vocab(p, graph_convention="atom", num_iterations=2)

    def test_atom_wrong_recorded_convention_raises(self):
        p = self.path("v.json")
        write_vocab(p, graph_convention="oracle_sage", num_iterations=2)
        with self.assertRaises(ValueError):
            _load_vocab(p, graph_convention="atom", num_iterations=2)

    def test_atom_wrong_num_iterations_raises(self):
        p = self.path("v.json")
        write_vocab(p, graph_convention="atom", num_iterations=3)
        with self.assertRaises(ValueError):
            _load_vocab(p, graph_convention="atom", num_iterations=2)

    def test_atom_matching_metadata_loads(self):
        p = self.path("v.json")
        write_vocab(p, graph_convention="atom", num_iterations=2)
        vocab = _load_vocab(p, graph_convention="atom", num_iterations=2)
        self.assertEqual(len(vocab), 1)

    def test_default_call_shape_tolerates_metadata_less_vocab(self):
        """_load_vocab(path) with no graph_convention/num_iterations - the shape every
        existing WLPlanFeedbackPolicy construction in tests/test_wl_plan_feedback_policy.py
        uses - must keep working against the real, metadata-less default vocab file."""
        from sage.domains.gym_taxi.utils.wl_vocab_cache import WL_VOCAB_PATH
        vocab = _load_vocab(str(WL_VOCAB_PATH))
        self.assertGreater(len(vocab), 0)


if __name__ == "__main__":
    unittest.main()
