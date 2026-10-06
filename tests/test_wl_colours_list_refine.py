"""
Regression test for wl_colours/refine's list-based implementation (colour ids collected in
a Python list, one tensor built at the end, instead of one tensor write per node).

The reference below is a verbatim copy of the previous per-node implementation. The new
code must return identical colours (values, dtype, shape) and histograms, and in growing
mode leave the vocab with identical contents AND insertion order (ids are assigned in
first-seen order, which is what a saved vocab file records), for every graph_convention,
in frozen and growing mode, at several depths L - on real city graphs (live states and
planner projections) and on hand-built edge cases.

Run from the repo root with:
    python -m pytest tests/test_wl_colours_list_refine.py -v
"""
import unittest
from copy import deepcopy

import numpy as np
import torch as th

# --- numpy/gym compat shim (same pattern as tests/test_atom_wiring.py) ---
import gym.utils.seeding as _seeding

if not hasattr(np.random.Generator, "randint"):
    class _RandintCompatGenerator:
        def __init__(self, generator):
            self._generator = generator

        def randint(self, low, high=None):
            if hasattr(self._generator, "randint"):
                return self._generator.randint(low, high) if high is not None else self._generator.randint(low)
            return self._generator.integers(low, high)

        def __getattr__(self, name):
            return getattr(self._generator, name)

    _original_np_random = _seeding.np_random

    def _np_random_with_randint(seed=None):
        generator, seed = _original_np_random(seed)
        return _RandintCompatGenerator(generator), seed

    _seeding.np_random = _np_random_with_randint

from torch_geometric.data import Data

from sage.domains.gym_taxi import REWARDS
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.simulator.planner import Planner
from sage.domains.gym_taxi.utils.representations import env_to_graph, env_to_vilg_graph, env_to_atom_graph
from sage.domains.gym_taxi.utils.wl_vocab_cache import WL_VOCAB_PATH, _load_vocab_from_path
from sage.domains.utils import wl_colours as W

VOCABS = {
    "oracle_sage": WL_VOCAB_PATH,
    "vilg": WL_VOCAB_PATH.parent / "wl_vocab_taxi_city_vilg_L2.json",
    "atom": WL_VOCAB_PATH.parent / "wl_vocab_taxi_city_atom_L1_full.json",
}
BUILDERS = {"oracle_sage": env_to_graph, "vilg": env_to_vilg_graph, "atom": env_to_atom_graph}


# --- reference: the previous implementation, verbatim apart from names ---
def reference_refine(node_colours, edge_index, edge_labels, vocab, frozen=False):
    W._check_frozen_vocab(vocab, frozen, "refine")
    n = node_colours.shape[0]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    node_colours_list = node_colours.tolist()
    edge_labels_list = edge_labels.tolist()
    neighbours = [[] for _ in range(n)]
    for e, s in enumerate(src):
        d = dst[e]
        neighbours[s].append((node_colours_list[d], edge_labels_list[e]))
    new_colours = th.empty(n, dtype=th.long, device=node_colours.device)
    for v in range(n):
        signature = (node_colours_list[v], tuple(sorted(neighbours[v])))
        new_colours[v] = W._resolve(signature, vocab, frozen)
    return new_colours


def reference_wl_colours(x, edge_index, edge_attr, num_iterations=5, vocab=None, frozen=False,
                         graph_convention="oracle_sage"):
    if vocab is None:
        vocab = {}
    W._check_frozen_vocab(vocab, frozen, "wl_colours")
    if graph_convention == "vilg":
        type_colours = W.initial_colours_vilg(x).tolist()
    elif graph_convention == "atom":
        type_colours = W.initial_colours_atom(x).tolist()
    else:
        type_colours = W.initial_colours(x).tolist()
    colours = th.empty(x.shape[0], dtype=th.long, device=x.device)
    for v, type_id in enumerate(type_colours):
        colours[v] = W._resolve(("init", type_id), vocab, frozen)
    if graph_convention == "vilg":
        labels = W.edge_labels_vilg(edge_attr)
    elif graph_convention == "atom":
        labels = W.edge_labels_atom(edge_attr)
    else:
        labels = W.edge_labels(edge_attr)
    for _ in range(num_iterations):
        colours = reference_refine(colours, edge_index, labels, vocab, frozen=frozen)
    histogram = th.bincount(colours, minlength=len(vocab)).float()
    return colours, histogram


def city_graphs(convention, seed, steps=150, sample_every=15, goals=3):
    """Live states along a random-walk episode plus planner projections from them."""
    env = GraphTaxiEnv(representation="graph", scenario="city", mask=False,
                       rewards=REWARDS["v1"], graph_convention=convention)
    env.seed(seed)
    env.reset()
    planner = Planner(graph_convention=convention)
    rng = np.random.RandomState(seed)
    graphs = []
    for t in range(steps):
        sim = env.sim
        if t % sample_every == 0:
            nf, ef, ei, mask, gf = BUILDERS[convention](sim)[:5]
            d = Data(x=th.as_tensor(nf, dtype=th.float32), edge_index=th.as_tensor(ei, dtype=th.long),
                     edge_attr=th.as_tensor(ef, dtype=th.float32))
            d.mask = th.as_tensor(mask, dtype=th.bool)
            d.global_features = th.as_tensor(gf, dtype=th.float32).unsqueeze(0)
            graphs.append((d.x, d.edge_index, d.edge_attr))
            targets = np.nonzero(mask)[0]
            for g in rng.choice(targets, size=goals, replace=False):
                proj, _ = planner.plan(deepcopy(d), int(g))
                graphs.append((proj.x, proj.edge_index, proj.edge_attr))
        # one random legal step: stay, or move to an adjacent location
        loc = sim.taxi.location
        nbrs = [n for n in sim.graph.successors(loc) if sim.graph.nodes[n]["attr"] == [1, 0, 0]]
        env.step(int(rng.choice(nbrs + [loc])))
    return graphs


class _Base(unittest.TestCase):
    def assert_same(self, new, ref):
        (c1, h1), (c0, h0) = new, ref
        self.assertEqual((c1.dtype, h1.dtype, c1.shape, h1.shape), (c0.dtype, h0.dtype, c0.shape, h0.shape))
        self.assertTrue(th.equal(c1, c0))
        self.assertTrue(th.equal(h1, h0))


class TestRealGraphs(_Base):
    @classmethod
    def setUpClass(cls):
        cls.graphs = {c: city_graphs(c, seed) for c, seed in (("oracle_sage", 0), ("vilg", 300), ("atom", 600))}

    def test_frozen_mode(self):
        for conv, graphs in self.graphs.items():
            vocab = _load_vocab_from_path(str(VOCABS[conv]))
            for L in (0, 1, 2, 3):
                with self.subTest(conv=conv, L=L):
                    for x, ei, ea in graphs:
                        self.assert_same(
                            W.wl_colours(x, ei, ea, num_iterations=L, vocab=vocab, frozen=True, graph_convention=conv),
                            reference_wl_colours(x, ei, ea, num_iterations=L, vocab=vocab, frozen=True, graph_convention=conv),
                        )

    def test_growing_mode_vocab_contents_and_order(self):
        for conv, graphs in self.graphs.items():
            for L in (1, 2, 3):
                with self.subTest(conv=conv, L=L):
                    v_new, v_ref = {}, {}
                    for x, ei, ea in graphs:
                        self.assert_same(
                            W.wl_colours(x, ei, ea, num_iterations=L, vocab=v_new, graph_convention=conv),
                            reference_wl_colours(x, ei, ea, num_iterations=L, vocab=v_ref, graph_convention=conv),
                        )
                    self.assertEqual(list(v_new.items()), list(v_ref.items()))
                    # then freeze both and keep going in frozen mode on the same graphs
                    W.freeze_vocab(v_new)
                    W.freeze_vocab(v_ref)
                    self.assertEqual(list(v_new.items()), list(v_ref.items()))
                    for x, ei, ea in graphs[::3]:
                        self.assert_same(
                            W.wl_colours(x, ei, ea, num_iterations=L, vocab=v_new, frozen=True, graph_convention=conv),
                            reference_wl_colours(x, ei, ea, num_iterations=L, vocab=v_ref, frozen=True, graph_convention=conv),
                        )

    def test_refine_directly(self):
        x, ei, ea = self.graphs["vilg"][0]
        labels = W.edge_labels_vilg(ea)
        start = W.initial_colours_vilg(x)
        v_new, v_ref = {}, {}
        self.assertTrue(th.equal(W.refine(start, ei, labels, v_new), reference_refine(start, ei, labels, v_ref)))
        self.assertEqual(list(v_new.items()), list(v_ref.items()))


class TestEdgeCases(_Base):
    def test_empty_graph(self):
        """Version-independent: torch 1.7.1 (RCP) raises inside initial_colours' th.argmax
        on an empty tensor, newer torch returns an empty result. Either way the new code
        must behave exactly like the reference: both raise the same exception type, or
        both return identical results. (Real Taxi graphs are never empty.)"""
        x = th.zeros((0, 3))
        ei = th.zeros((2, 0), dtype=th.long)
        ea = th.zeros((0, 4))

        def outcome(fn, vocab, frozen):
            try:
                return "ok", fn(x, ei, ea, num_iterations=2, vocab=dict(vocab), frozen=frozen)
            except Exception as exc:  # noqa: BLE001 - the exception type is what is compared
                return "raised", type(exc)

        for frozen in (False, True):
            with self.subTest(frozen=frozen):
                vocab = {W.OOV_SIGNATURE: 0} if frozen else {}
                new_kind, new = outcome(W.wl_colours, vocab, frozen)
                ref_kind, ref = outcome(reference_wl_colours, vocab, frozen)
                self.assertEqual(new_kind, ref_kind)
                if new_kind == "raised":
                    self.assertIs(new, ref)
                else:
                    self.assert_same(new, ref)

    def test_isolated_nodes_and_oov(self):
        # oracle_sage layout: 3 nodes, one road edge pair, node 2 isolated
        x = th.tensor([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        ei = th.tensor([[0, 1], [1, 0]])
        ea = th.tensor([[1., 0., 0., 1.], [1., 0., 0., -1.]])
        vocab = {("init", 0): 0, W.OOV_SIGNATURE: 1}  # init type 1 and every refinement unseen
        self.assert_same(
            W.wl_colours(x, ei, ea, num_iterations=2, vocab=vocab, frozen=True),
            reference_wl_colours(x, ei, ea, num_iterations=2, vocab=vocab, frozen=True),
        )

    def test_frozen_without_oov_still_raises(self):
        x = th.tensor([[1., 0., 0.]])
        with self.assertRaises(ValueError):
            W.wl_colours(x, th.zeros((2, 0), dtype=th.long), th.zeros((0, 4)), vocab={}, frozen=True)
        with self.assertRaises(ValueError):
            W.refine(th.zeros(1, dtype=th.long), th.zeros((2, 0), dtype=th.long), th.zeros(0, dtype=th.long), {}, frozen=True)


if __name__ == "__main__":
    unittest.main()
