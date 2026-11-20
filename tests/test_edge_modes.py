"""
W4 (paper-edges): edge-type variants MGr / EGr (sage/agent/edge_modes.py, graph_net.TypedGraphNet).

Checked here, on real states of both domains and all three representations:
- the type tables (counts and names) and the refused combinations;
- Gr -> typed edges loses nothing: every Gr edge gets at least one type, and the Gr graph
  can be rebuilt exactly from the MGr one (labels and Object direction included);
- MGr equals Gr where it is skipped (Object-Atom; standard Object up to the label code), and
  otherwise the parallel edges carry exactly the Gr label bits;
- EGr: each type's messages go only through that type's message function, and with one
  type EGr is exactly a plain message function without edge labels (GraphNet, edge_size 0).

Run from the repo root (CPU):
    PYTHONPATH=. python -m pytest tests/test_edge_modes.py -v
"""
import random
import unittest
from collections import defaultdict

import gym as gym_module
import torch as th
from torch import nn

import tests.test_ternary_planner  # noqa: F401  installs the gym-0.26 randint shim
from sage.agent.edge_modes import (EDGE_MODES, edge_type_spec, expand_typed_edges, split_by_type, to_mgr,
                                   type_mask, validate_edge_mode)
from sage.agent.graph_net import GraphNet, TypedGraphNet
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.simulator.ternary_scripted_policy import scripted_policy
from sage.domains.gym_taxi.utils.compat import spec_kwargs

STD_ENV_ID = "city-taxi-unmasked-v1"
TERNARY_ENV_ID = "city-taxi-ternary-unmasked-v1"
CONVENTIONS = ("oracle_sage", "vilg", "atom")


def _std_action(sim, rng):
    loc = sim.taxi.location
    moves = [n for n in sim.graph[loc] if list(sim.graph.nodes[n]["attr"]) == [1, 0, 0] or n in sim.passengers]
    return rng.choice(moves + [0])


_STATES = {}


def real_graphs(ternary, convention, steps=600, every=150, seed=0):
    """Gr graphs (torch Data) of real states: scripted play (ternary) or random legal moves
    (standard), one every `every` steps. Cached per (domain, convention)."""
    key = (ternary, convention)
    if key not in _STATES:
        env_id = TERNARY_ENV_ID if ternary else STD_ENV_ID
        env = GraphTaxiEnv(**dict(spec_kwargs(gym_module.spec(env_id)), graph_convention=convention))
        env.seed(seed)
        obs = env.reset()
        rng = random.Random(seed)
        graphs = []
        for t in range(steps + 1):
            if t % every == 0:
                graphs.append(env.observation_space.converter([[obs]]))
            action = scripted_policy(env.sim, rng) if ternary else _std_action(env.sim, rng)
            obs, _r, done, _info = env.step(action)
            if done:
                obs = env.reset()
        _STATES[key] = graphs
    return _STATES[key]


def rebuild_gr(typed_edge_index, edge_type, spec, width):
    """Inverse of the Gr -> typed-edges map: OR the types' columns per (src, dst); Object dir
    is -1 if any reverse type is on the pair, else +1."""
    rows = defaultdict(lambda: [0.0] * width)
    for (u, v), t in zip(typed_edge_index.t().tolist(), edge_type.tolist()):
        row = rows[(u, v)]
        etype = spec.types[t]
        row[etype.column] = 1.0
        if spec.dir_column is not None:
            if etype.direction == -1 or row[spec.dir_column] == -1.0:
                row[spec.dir_column] = -1.0
            else:
                row[spec.dir_column] = 1.0
    return rows


class TestTypeTables(unittest.TestCase):
    def test_counts_and_names(self):
        expected = {
            ("oracle_sage", False): ("adjacent", "in>", "in<", "destination>", "destination<"),
            ("oracle_sage", True): ("road", "at>", "at<", "picked_up_at>", "picked_up_at<",
                                    "req12", "req21", "req13", "req31", "req23", "req32"),
            ("vilg", False): ("pos1", "pos2"),
            ("vilg", True): ("pos1", "pos2", "pos3"),
            ("atom", False): ("(1,1)", "(1,2)", "(2,1)", "(2,2)"),
            ("atom", True): tuple(f"({i},{j})" for i in (1, 2, 3) for j in (1, 2, 3)),
        }
        for (convention, ternary), names in expected.items():
            spec = edge_type_spec(convention, ternary, "egr")
            self.assertEqual(spec.names, names)
            self.assertEqual(spec.num_types, len(names))

    def test_refused_combinations(self):
        self.assertEqual(EDGE_MODES, ("gr", "mgr", "egr"))
        with self.assertRaises(ValueError):
            validate_edge_mode("multi")
        for convention, ternary in (("vilg", False), ("vilg", True), ("oracle_sage", False)):
            with self.assertRaises(ValueError):
                edge_type_spec(convention, ternary, "mgr")
            edge_type_spec(convention, ternary, "egr")  # EGr is run for every representation
        for convention, ternary in (("oracle_sage", True), ("atom", False), ("atom", True)):
            edge_type_spec(convention, ternary, "mgr")
        validate_edge_mode("gr", "vilg", False)  # gr is always allowed


class TestGrToTypedEdges(unittest.TestCase):
    def test_lossless_on_real_states(self):
        """Every Gr edge has at least one type, the typed edges number exactly the type bits,
        and the Gr graph is rebuilt exactly from them."""
        for ternary in (False, True):
            for convention in CONVENTIONS:
                spec = edge_type_spec(convention, ternary, "egr")
                for g in real_graphs(ternary, convention):
                    with self.subTest(ternary=ternary, convention=convention):
                        mask = type_mask(g.edge_attr, spec)
                        self.assertTrue(bool(mask.any(1).all()), "a Gr edge with no type")
                        typed_edge_index, edge_type = expand_typed_edges(g.edge_index, g.edge_attr, spec)
                        self.assertEqual(edge_type.shape[0], int(mask.sum()))
                        rows = rebuild_gr(typed_edge_index, edge_type, spec, g.edge_attr.shape[1])
                        gr = {(u, v): row for (u, v), row in zip(g.edge_index.t().tolist(), g.edge_attr.tolist())}
                        self.assertEqual(set(rows), set(gr))
                        for pair, row in gr.items():
                            self.assertEqual(rows[pair], row, f"pair {pair}")

    def test_order_is_by_gr_edge_then_type(self):
        g = real_graphs(True, "atom")[-1]
        spec = edge_type_spec("atom", True, "mgr")
        mask = type_mask(g.edge_attr, spec)
        gr_edge, edge_type = mask.nonzero(as_tuple=True)
        key = gr_edge * spec.num_types + edge_type
        self.assertTrue(bool((key[1:] > key[:-1]).all()))
        typed_edge_index, edge_type2 = expand_typed_edges(g.edge_index, g.edge_attr, spec)
        self.assertTrue(th.equal(typed_edge_index, g.edge_index[:, gr_edge]))
        self.assertTrue(th.equal(edge_type2, edge_type))


class TestMGr(unittest.TestCase):
    def test_object_atom_mgr_is_gr(self):
        """Why Object-Atom MGr is skipped: an object appears at most once per atom, so every
        Gr label is already one-hot and the 'MGr' graph is the Gr graph itself."""
        for ternary in (False, True):
            spec = edge_type_spec("vilg", ternary, "egr")
            for g in real_graphs(ternary, "vilg"):
                edge_index, labels = to_mgr(g.edge_index, g.edge_attr, spec)
                self.assertTrue(th.equal(edge_index, g.edge_index))
                self.assertTrue(th.equal(labels, g.edge_attr))

    def test_standard_object_mgr_has_the_gr_edges(self):
        """Why standard Object MGr is skipped: no pair has two predicates, so MGr has exactly
        the Gr edges; only the label code changes ([pred, dir] -> one-hot over 5 types)."""
        spec = edge_type_spec("oracle_sage", False, "egr")
        for g in real_graphs(False, "oracle_sage"):
            edge_index, labels = to_mgr(g.edge_index, g.edge_attr, spec)
            self.assertTrue(th.equal(edge_index, g.edge_index))
            self.assertTrue(bool((labels.sum(1) == 1).all()))

    def test_parallel_edges_carry_exactly_the_gr_bits(self):
        """Object (ternary) and Atom: for each Gr edge, its MGr copies are one-hot and their
        types are exactly the Gr edge's set label bits (Object: with its direction)."""
        for ternary, convention in ((True, "oracle_sage"), (False, "atom"), (True, "atom")):
            spec = edge_type_spec(convention, ternary, "mgr")
            for g in real_graphs(ternary, convention):
                with self.subTest(ternary=ternary, convention=convention):
                    edge_index, labels = to_mgr(g.edge_index, g.edge_attr, spec)
                    self.assertTrue(bool((labels.sum(1) == 1).all()))
                    self.assertEqual(labels.dtype, g.edge_attr.dtype)
                    got = defaultdict(set)
                    for (u, v), t in zip(edge_index.t().tolist(), labels.argmax(1).tolist()):
                        got[(u, v)].add(spec.types[t].column)
                    width = g.edge_attr.shape[1] - (1 if spec.dir_column is not None else 0)
                    for (u, v), row in zip(g.edge_index.t().tolist(), g.edge_attr.tolist()):
                        self.assertEqual(got[(u, v)], {c for c in range(width) if row[c] > 0})
                    self.assertGreaterEqual(edge_index.shape[1], g.edge_index.shape[1])
            # where Gr really merges types, MGr has more edges
            self.assertGreater(edge_index.shape[1], g.edge_index.shape[1])


class TestEGrLayer(unittest.TestCase):
    def _graph(self, n=7, seed=0):
        gen = th.Generator().manual_seed(seed)
        x = th.randn(n, 32, generator=gen)
        xg = th.randn(1, 32, generator=gen)
        batch = th.zeros(n, dtype=th.long)
        # node 6 receives nothing
        src = th.tensor([0, 1, 2, 3, 4, 5, 0, 2])
        dst = th.tensor([1, 2, 3, 4, 5, 0, 3, 1])
        return x, xg, batch, src, dst

    def test_one_type_is_a_plain_message_function(self):
        x, xg, batch, src, dst = self._graph()
        th.manual_seed(0)
        typed = TypedGraphNet(32, 1, 32, 32, 32, nn.LeakyReLU)
        plain = GraphNet(32, 0, 32, 32, 32, nn.LeakyReLU)
        plain.f_mess.load_state_dict(typed.f_mess[0].state_dict())
        plain.f_agg.load_state_dict(typed.f_agg.state_dict())
        out_typed = typed(x, [(src, dst)], xg, batch)
        out_plain = plain(x, th.zeros(src.shape[0], 0), th.stack([src, dst]), xg, batch)
        self.assertTrue(th.allclose(out_typed, out_plain, atol=1e-6))

    def test_messages_go_only_through_their_own_type(self):
        x, xg, batch, src, dst = self._graph()
        types = th.tensor([0, 1, 2, 0, 1, 2, 1, 0])
        typed_edges = [(src[types == t], dst[types == t]) for t in range(3)]
        layer = TypedGraphNet(32, 3, 32, 32, 32, nn.LeakyReLU)
        seen = {}
        for t, f in enumerate(layer.f_mess):
            f.register_forward_hook(lambda _m, inp, _out, t=t: seen.__setitem__(t, inp[0].detach().clone()))
        out = layer(x, typed_edges, xg, batch)
        for t in range(3):
            self.assertTrue(th.equal(seen[t], x[src[types == t]]), f"type {t}")
        # changing type 1's function changes only the nodes that receive type-1 edges
        with th.no_grad():
            layer.f_mess[1][0].weight.add_(1.0)
        out2 = layer(x, typed_edges, xg, batch)
        changed = set((out2 - out).abs().sum(1).nonzero().flatten().tolist())
        self.assertEqual(changed, set(dst[types == 1].tolist()))

    def test_split_by_type_matches_the_type_mask(self):
        for ternary in (False, True):
            for convention in CONVENTIONS:
                spec = edge_type_spec(convention, ternary, "egr")
                g = real_graphs(ternary, convention)[-1]
                mask = type_mask(g.edge_attr, spec)
                parts = split_by_type(g.edge_index, g.edge_attr, spec)
                self.assertEqual(len(parts), spec.num_types)
                for t, (s, d) in enumerate(parts):
                    self.assertTrue(th.equal(th.stack([s, d]), g.edge_index[:, mask[:, t]]))


if __name__ == "__main__":
    unittest.main()
