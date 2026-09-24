"""
Tests for WL colours wired into the atom convention's DECODER side (attach_wl,
sage/domains/gym_taxi/utils/representations.py, called from json_to_atom_graph and from
the atom planner's _atoms_to_projection, sage/domains/gym_taxi/simulator/planner.py).

Every test that configures the WL vocab override does so via a small, temp-dir-built
vocab (build_wl_vocab.sample_graphs/save_vocab with graph_convention="atom") and resets
the override in tearDown - configure_wl_vocab_override mutates process-global state (see
wl_vocab_cache.py's own docstring), and the full suite runs every test file in one
process.

Run from the repo root with:
    python -m pytest tests/test_wl_atom_wiring.py -v
"""
import copy
import os
import tempfile
import unittest

import numpy as np
import torch as th
from torch_geometric.data import Data

# Importing build_wl_vocab (used directly below, not just for its import side effect)
# applies its numpy/gym compatibility shims - needed to construct envs at all in this
# sandbox's drifted gym/numpy. See build_wl_vocab.py's own docstring for why.
from sage.domains.utils.build_wl_vocab import sample_graphs, save_vocab

from sage.domains.gym_taxi.simulator.taxi_world import TaxiWorldSimulator
from sage.domains.gym_taxi.simulator.planner import Planner
from sage.domains.gym_taxi.utils.config import PREDICTABLE5, CITY
from sage.domains.gym_taxi.utils.representations import (
    env_to_atoms,
    env_to_atom_graph,
    env_to_atom_json,
    json_to_atom_graph,
)
from sage.domains.gym_taxi.utils.wl_vocab_cache import (
    configure_wl_vocab_override,
    reset_wl_vocab_override,
    get_wl_vocab,
    get_wl_num_iterations,
)
from sage.domains.utils.wl_colours import freeze_vocab, wl_colours


def build_atom_vocab(path, seed, episodes, steps_per_episode, num_iterations, scenario="predictable5"):
    vocab = {}
    sample_graphs(
        vocab, episodes=episodes, steps_per_episode=steps_per_episode,
        num_iterations=num_iterations, seed=seed, scenario=scenario,
        graph_convention="atom", log_every=10 ** 9,
    )
    freeze_vocab(vocab)
    save_vocab(vocab, path, graph_convention="atom", num_iterations=num_iterations)
    return path


def make_sim(seed, **scenario):
    return TaxiWorldSimulator(np.random.RandomState(seed), planning=True, graph_convention="atom", **scenario)


def direct_wl(sim):
    node_feats, edge_feats, edge_index, mask, global_feats = env_to_atom_graph(sim)
    x = th.as_tensor(node_feats, dtype=th.float)
    edge_index_t = th.as_tensor(edge_index, dtype=th.long)
    edge_attr_t = th.as_tensor(edge_feats, dtype=th.float)
    return wl_colours(
        x, edge_index_t, edge_attr_t,
        num_iterations=get_wl_num_iterations(), vocab=get_wl_vocab(), frozen=True,
        graph_convention="atom",
    )


class TestDecoderMatchesDirectSameCorpusVocab(unittest.TestCase):
    """A vocab built from the SAME seed/scenario as the graph under test - decoder path
    (env -> atom JSON -> json_to_atom_graph) must equal the direct path
    (env_to_atom_graph -> wl_colours), id-for-id and histogram-for-histogram."""

    L = 1
    SEED = 4001

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.vocab_path = os.path.join(self._tmpdir.name, "vocab.json")
        build_atom_vocab(self.vocab_path, seed=self.SEED, episodes=3, steps_per_episode=30, num_iterations=self.L)
        configure_wl_vocab_override(self.vocab_path, self.L, graph_convention="atom")

    def tearDown(self):
        reset_wl_vocab_override()
        self._tmpdir.cleanup()

    def test_json_roundtrip_matches_direct_recomputation(self):
        sim = make_sim(self.SEED, **PREDICTABLE5)
        expected_colours, expected_hist = direct_wl(sim)

        js = env_to_atom_json(sim)
        batch = json_to_atom_graph([[js]])

        self.assertTrue(hasattr(batch, "wl_colours"))
        self.assertTrue(hasattr(batch, "wl_histogram"))
        self.assertEqual(batch.wl_colours.dtype, th.long)
        self.assertEqual(batch.wl_histogram.dtype, th.float)
        n_nodes = len(env_to_atoms(sim))
        self.assertEqual(tuple(batch.wl_colours.shape), (n_nodes,))
        self.assertEqual(tuple(batch.wl_histogram.shape), (1, len(expected_hist)))
        self.assertEqual(batch.wl_colours.tolist(), expected_colours.tolist())
        self.assertEqual(batch.wl_histogram.squeeze(0).tolist(), expected_hist.tolist())


class TestDecoderMatchesDirectDifferentCorpusVocabWithOov(unittest.TestCase):
    """A vocab built from a tiny, disjoint corpus (different seed AND scenario) - OOV
    should fire for most/all of the test graph's signatures. Decoder-vs-direct
    equivalence must still hold exactly under OOV resolution (both paths resolve the
    SAME unseen signatures to the SAME OOV id, deterministically)."""

    L = 2
    BUILD_SEED = 1

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.vocab_path = os.path.join(self._tmpdir.name, "vocab.json")
        # deliberately tiny + a different scenario than the test graph below
        build_atom_vocab(self.vocab_path, seed=self.BUILD_SEED, episodes=1, steps_per_episode=3, num_iterations=self.L, scenario="predictable5")
        configure_wl_vocab_override(self.vocab_path, self.L, graph_convention="atom")

    def tearDown(self):
        reset_wl_vocab_override()
        self._tmpdir.cleanup()

    def test_matches_under_oov(self):
        sim = make_sim(98765, **CITY)  # far larger/structurally different from the tiny build corpus

        expected_colours, expected_hist = direct_wl(sim)

        js = env_to_atom_json(sim)
        batch = json_to_atom_graph([[js]])

        self.assertEqual(batch.wl_colours.tolist(), expected_colours.tolist())
        self.assertEqual(batch.wl_histogram.squeeze(0).tolist(), expected_hist.tolist())

        # confirm OOV actually fired (test is meaningful, not vacuous) - OOV always gets
        # the last id in a frozen vocab (see freeze_vocab)
        oov_id = len(get_wl_vocab()) - 1
        self.assertIn(oov_id, batch.wl_colours.tolist())


class TestNoWlFieldsWithoutOverride(unittest.TestCase):
    """The atom GNN baseline (no --wl-vocab-path) must carry no WL fields at all -
    neither env_to_atom_graph (which never computes WL, regardless of override state)
    nor json_to_atom_graph when no override is configured (attach_wl's no-op path)."""

    def test_env_to_atom_graph_is_a_plain_5_tuple(self):
        sim = make_sim(1, **PREDICTABLE5)
        result = env_to_atom_graph(sim)
        self.assertEqual(len(result), 5)

    def test_json_to_atom_graph_has_no_wl_attrs_when_override_not_configured(self):
        reset_wl_vocab_override()  # guard against test-order leakage from another file
        sim = make_sim(1, **PREDICTABLE5)
        js = env_to_atom_json(sim)
        batch = json_to_atom_graph([[js]])
        self.assertFalse(hasattr(batch, "wl_colours"))
        self.assertFalse(hasattr(batch, "wl_histogram"))


class TestPlannerProjectionMatchesExecutedEnv(unittest.TestCase):
    """The projection's WL histogram must equal the WL histogram of the REAL env after
    actually executing the plan's actions on a copy of it - not just internal
    self-consistency of the planner's own re-encoding."""

    L = 1
    SEED = 4002

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.vocab_path = os.path.join(self._tmpdir.name, "vocab.json")
        build_atom_vocab(self.vocab_path, seed=self.SEED, episodes=3, steps_per_episode=30, num_iterations=self.L)
        configure_wl_vocab_override(self.vocab_path, self.L, graph_convention="atom")

    def tearDown(self):
        reset_wl_vocab_override()
        self._tmpdir.cleanup()

    def _to_data(self, sim):
        node_feats, edge_feats, edge_index, mask, global_feats = env_to_atom_graph(sim)
        d = Data(
            x=th.as_tensor(node_feats, dtype=th.float32),
            edge_index=th.as_tensor(edge_index, dtype=th.long),
            edge_attr=th.as_tensor(edge_feats, dtype=th.float32),
        )
        d.mask = th.as_tensor(mask, dtype=th.bool)
        d.global_features = th.as_tensor(global_feats, dtype=th.float32).unsqueeze(0)
        return d

    def test_projection_histogram_matches_post_execution_env(self):
        # passenger_creation_probability=0: a mid-plan spawn would make the executed
        # state diverge from the projection through no fault of the planner/WL wiring.
        sim = make_sim(self.SEED, **{**PREDICTABLE5, "passenger_creation_probability": 0})
        pid = next(iter(sim.passengers))
        planner = Planner(graph_convention="atom")

        for goal in [pid, sim.taxi.location, 0]:
            with self.subTest(goal=goal):
                graph = self._to_data(sim)
                projection, actions = planner.plan(graph, goal)

                executed = copy.deepcopy(sim)
                for a in actions:
                    executed.act(int(a))

                expected_colours, expected_hist = direct_wl(executed)

                self.assertTrue(hasattr(projection, "wl_colours"))
                self.assertTrue(hasattr(projection, "wl_histogram"))
                self.assertEqual(projection.wl_colours.tolist(), expected_colours.tolist())
                self.assertEqual(projection.wl_histogram.squeeze(0).tolist(), expected_hist.tolist())


if __name__ == "__main__":
    unittest.main()
