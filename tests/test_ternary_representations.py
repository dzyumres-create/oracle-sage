"""
Tests for sage/domains/gym_taxi/utils/ternary_representations.py: facts_to_object_graph,
facts_to_vilg_graph, facts_to_atom_graph, facts_to_json/json_to_facts.

Run from the repo root with:
    PYTHONPATH=. python -m pytest tests/test_ternary_representations.py -v
"""
import random as pyrandom
import time
import unittest

import networkx as nx
import numpy as np
import torch as th

from sage.domains.gym_taxi.simulator.taxi_world import TaxiWorldSimulator
from sage.domains.gym_taxi.simulator.ternary_taxi_world import TernaryTaxiWorldSimulator
from sage.domains.gym_taxi.utils.config import CITY, CITY_TERNARY
from sage.domains.gym_taxi.utils.representations import (
    ATOM_PREDICATES as OLD_ATOM_PREDICATES,
    env_to_atoms,
    atoms_to_graph,
)
from sage.domains.gym_taxi.utils.ternary_representations import (
    ATOM_MAX_ARITY,
    ATOM_PREDICATES_TERNARY,
    OBJECT_EDGE_LABELS,
    TERNARY_GRAPH_CONVENTIONS,
    TERNARY_PREDICATES,
    VILG_PREDICATE_ORDER,
    VILG_NONGOAL_STATUS,
    _atoms_to_graph_generic,
    _validate_facts,
    facts_to_atom_graph,
    facts_to_json,
    facts_to_object_graph,
    facts_to_vilg_graph,
    json_to_facts,
)
from analysis.indistinguishability import (
    PART2_CONVENTIONS,
    PART2_ENCODERS,
    _build_part2_graphs,
    colour_multiset_hash,
    ext_edge_labels_atom,
    ext_edge_labels_object_atom,
    ext_initial_colours_atom,
    wl_colours_per_level,
)
from analysis.ternary_collision_counter import build_flipped_facts, facts_to_ext_atoms
from tests.test_ternary_world import crossed_state, legal_actions, scripted_policy


PLANNING_META = {"time": 0, "timeout": 2000, "planning": True}


def make_hand_checked_sim():
    """
    size=2, no walls, RandomState(0): taxi at location 1 (same seed/config as
    TestHandCheckedPair -- see tests/test_ternary_world.py). Requests are chosen
    deliberately so the object encoding exercises BOTH a location pair that is
    simultaneously a road AND a request pair ((1,2) and (3,4) are real road edges
    AND the (origin, destination) pair of one of p's own requests each), and the
    dir-priority conflict (a waiting passenger's own location is also one of their
    own request origins, so in(p, l) and request(p, l, d) label the SAME directed
    pair (l, p) with different opinions about `dir` -- see
    _add_object_edge_label's docstring):

        p (pid 5): requests (1 -> 2), (3 -> 4); starts at origin 1 (== taxi start)
        q (pid 6): requests (1 -> 4), (3 -> 2); starts at origin 3

    Road edges (verified from the running code, RandomState(0), size=2, no walls --
    same maze test_atom_encoding.py's TestHandCheckedGrid documents for the old
    domain under the same seed/config): (1,2),(2,1),(1,3),(3,1),(2,4),(4,2),(3,4),(4,3).
    """
    return TernaryTaxiWorldSimulator(
        np.random.RandomState(0),
        size=2,
        random_walls=False,
        delivery_limit=1,
        concurrent_passengers=2,
        requests_per_passenger=2,
        passenger_creation_probability=0,
        observation_fn=lambda s: None,
        initial_pair=(((1, 2), (3, 4)), 1, ((1, 4), (3, 2)), 3),
    )


def _edges_as_dict(edge_index, edge_feats):
    return {
        (int(edge_index[0, k]), int(edge_index[1, k])): [float(v) for v in edge_feats[k]]
        for k in range(edge_index.shape[1])
    }


# ==========================================================================================
# Validation (item 2): malformed fact lists must raise, never be silently accepted
# ==========================================================================================

class TestValidation(unittest.TestCase):
    def test_well_formed_facts_pass(self):
        facts = [("taxi", (0,)), ("location", (1,)), ("passenger", (2,)), ("adjacent", (1, 1))]
        self.assertEqual(_validate_facts(facts), 3)

    def test_shuffled_type_atom_order_raises(self):
        facts = [("location", (1,)), ("taxi", (0,)), ("passenger", (2,))]
        with self.assertRaises(ValueError):
            _validate_facts(facts)

    def test_gapped_type_atom_ids_raises(self):
        facts = [("taxi", (0,)), ("location", (1,)), ("location", (3,))]
        with self.assertRaises(ValueError):
            _validate_facts(facts)

    def test_type_atom_after_relational_fact_raises(self):
        facts = [("taxi", (0,)), ("location", (1,)), ("adjacent", (1, 1)), ("passenger", (2,))]
        with self.assertRaises(ValueError):
            _validate_facts(facts)

    def test_unrecognised_predicate_raises(self):
        facts = [("taxi", (0,)), ("location", (1,)), ("destination", (0, 1))]
        with self.assertRaises(ValueError):
            _validate_facts(facts)

    def test_relational_fact_referencing_unknown_object_id_raises(self):
        facts = [("taxi", (0,)), ("location", (1,)), ("adjacent", (1, 5))]
        with self.assertRaises(ValueError):
            _validate_facts(facts)

    def test_wrong_arity_relational_fact_raises(self):
        facts = [("taxi", (0,)), ("location", (1,)), ("adjacent", (1,))]
        with self.assertRaises(ValueError):
            _validate_facts(facts)

    def test_wrong_arity_type_atom_raises(self):
        facts = [("taxi", (0, 1)), ("location", (1,))]
        with self.assertRaises(ValueError):
            _validate_facts(facts)

    def test_each_converter_raises_on_malformed_facts(self):
        bad = [("location", (1,)), ("taxi", (0,))]  # shuffled
        for converter in (facts_to_object_graph, facts_to_vilg_graph, facts_to_atom_graph):
            with self.assertRaises(ValueError):
                converter(bad, PLANNING_META)


# ==========================================================================================
# (a) Hand-checkable
# ==========================================================================================

class TestHandCheckedConverters(unittest.TestCase):
    def setUp(self):
        self.sim = make_hand_checked_sim()
        self.facts = self.sim.facts()
        self.assertEqual(len(self.facts), 22)  # 7 type atoms + 8 adjacent + 3 in + 4*3 request

    def test_object_graph_node_features(self):
        nf, _ef, _ei, _mask, _gf = facts_to_object_graph(self.facts, PLANNING_META)
        self.assertEqual(nf.shape, (7, 3))
        np.testing.assert_array_equal(nf[0], [0, 1, 0])  # taxi
        np.testing.assert_array_equal(nf[1], [1, 0, 0])  # location(1)
        np.testing.assert_array_equal(nf[5], [0, 0, 1])  # passenger(5)
        np.testing.assert_array_equal(nf[6], [0, 0, 1])  # passenger(6)

    def test_object_graph_edge_count_and_named_labels(self):
        """Full hand derivation (see the module docstring / conversation record): 30
        unique directed edges. Named edges below cover every DISTINCT label pattern
        that occurs: a pure road edge, a road+request merge (both directions), a pure
        `at` tether, the `at`+`request` dir-priority-conflict merge (both directions),
        and pure single-request edges."""
        _nf, ef, ei, _mask, _gf = facts_to_object_graph(self.facts, PLANNING_META)
        self.assertEqual(ei.shape[1], 30)
        edges = _edges_as_dict(ei, ef)

        def label_vector(**bits):
            row = [0] * len(OBJECT_EDGE_LABELS)
            row[OBJECT_EDGE_LABELS.index("dir")] = bits.pop("dir")
            for name, value in bits.items():
                row[OBJECT_EDGE_LABELS.index(name)] = value
            return row

        expected = {
            (0, 1): label_vector(at=1, dir=1),
            (1, 0): label_vector(at=1, dir=-1),
            (1, 3): label_vector(road=1, dir=1),
            (3, 1): label_vector(road=1, dir=1),
            (2, 4): label_vector(road=1, dir=1),
            (4, 2): label_vector(road=1, dir=1),
            (1, 2): label_vector(road=1, req23=1, dir=1),
            (2, 1): label_vector(road=1, req32=1, dir=1),
            (3, 4): label_vector(road=1, req23=1, dir=1),
            (4, 3): label_vector(road=1, req32=1, dir=1),
            (5, 1): label_vector(at=1, req12=1, dir=1),
            (1, 5): label_vector(at=1, req21=1, dir=-1),  # dir-priority conflict, in wins
            (6, 3): label_vector(at=1, req12=1, dir=1),
            (3, 6): label_vector(at=1, req21=1, dir=-1),  # dir-priority conflict, in wins
            (5, 2): label_vector(req13=1, dir=1),
            (2, 5): label_vector(req31=1, dir=1),
            (6, 4): label_vector(req13=1, dir=1),
            (4, 6): label_vector(req31=1, dir=1),
            (1, 4): label_vector(req23=1, dir=1),  # pure request, (1,4) is not a road
            (4, 1): label_vector(req32=1, dir=1),
            (3, 2): label_vector(req23=1, dir=1),  # pure request, (3,2) is not a road
            (2, 3): label_vector(req32=1, dir=1),
        }
        for pair, expected_row in expected.items():
            self.assertIn(pair, edges, f"missing edge {pair}")
            self.assertEqual(edges[pair], [float(v) for v in expected_row], f"edge {pair} label mismatch")

    def test_object_graph_dims_match_ternary_graph_conventions(self):
        nf, ef, _ei, _mask, gf = facts_to_object_graph(self.facts, PLANNING_META)
        self.assertEqual(nf.shape[1], TERNARY_GRAPH_CONVENTIONS["oracle_sage"][0])
        self.assertEqual(ef.shape[1], TERNARY_GRAPH_CONVENTIONS["oracle_sage"][1])
        self.assertEqual(gf.shape, (32,))

    def test_object_graph_planning_false_mask(self):
        meta = {"time": 0, "timeout": 2000, "planning": False}
        _nf, _ef, _ei, mask, _gf = facts_to_object_graph(self.facts, meta)
        # taxi at location 1: mask True on {1} (self), {2,3} (road-adjacent), {0}
        # (the taxi itself, via in(0,1)), {5} (passenger 5, waiting at 1, via in(5,1)).
        expected = np.zeros(7, dtype=bool)
        for i in (0, 1, 2, 3, 5):
            expected[i] = True
        np.testing.assert_array_equal(mask, expected)

    def test_vilg_graph_shapes_and_named_rows(self):
        nf, ef, ei, mask, _gf = facts_to_vilg_graph(self.facts, PLANNING_META)
        self.assertEqual(nf.shape, (22, TERNARY_GRAPH_CONVENTIONS["vilg"][0]))
        self.assertEqual(ef.shape[1], TERNARY_GRAPH_CONVENTIONS["vilg"][1])
        # 8 adjacent + 3 in + 4 request, each mirrored (2 * arity edges per fact):
        # 8*2*2 + 3*2*2 + 4*2*3 = 32 + 12 + 24 = 68
        self.assertEqual(ei.shape[1], 68)
        np.testing.assert_array_equal(mask, [i < 7 for i in range(22)])

        # row 7 is the first relational fact: adjacent(1, 2)
        self.assertEqual(self.facts[7], ("adjacent", (1, 2)))
        expected_row = [0, 0, 0] + [1 if p == "adjacent" else 0 for p in VILG_PREDICATE_ORDER] + list(VILG_NONGOAL_STATUS)
        np.testing.assert_array_equal(nf[7], expected_row)

        # a request proposition: facts[18] is request(5, 1, 2) (first request fact --
        # 7 type atoms + 8 adjacent + 3 in come before it)
        self.assertEqual(self.facts[18], ("request", (5, 1, 2)))
        expected_row = [0, 0, 0] + [1 if p == "request" else 0 for p in VILG_PREDICATE_ORDER] + list(VILG_NONGOAL_STATUS)
        np.testing.assert_array_equal(nf[18], expected_row)

    def test_vilg_edges_mirrored_for_a_request_proposition(self):
        _nf, ef, ei, _mask, _gf = facts_to_vilg_graph(self.facts, PLANNING_META)
        # facts[18] = request(5, 1, 2) -> proposition node 18, arguments (5, 1, 2) at
        # positions 1, 2, 3 respectively. Every (18, obj) / (obj, 18) pair must carry
        # the SAME position one-hot in both directions.
        edges = _edges_as_dict(ei, ef)
        for position, obj in zip((1, 2, 3), (5, 1, 2)):
            expected = [0.0, 0.0, 0.0]
            expected[position - 1] = 1.0
            self.assertEqual(edges[(18, obj)], expected)
            self.assertEqual(edges[(obj, 18)], expected)

    def test_atom_graph_shapes_and_named_rows(self):
        nf, ef, ei, mask, _gf = facts_to_atom_graph(self.facts, PLANNING_META)
        self.assertEqual(nf.shape, (22, TERNARY_GRAPH_CONVENTIONS["atom"][0]))
        self.assertEqual(ef.shape[1], TERNARY_GRAPH_CONVENTIONS["atom"][1])
        np.testing.assert_array_equal(mask, [i < 7 for i in range(22)])
        # row0 = taxi(0)
        expected = [0] * len(ATOM_PREDICATES_TERNARY)
        expected[ATOM_PREDICATES_TERNARY.index("taxi")] = 1
        np.testing.assert_array_equal(nf[0], expected)
        # row7 = adjacent(1,2)
        expected = [0] * len(ATOM_PREDICATES_TERNARY)
        expected[ATOM_PREDICATES_TERNARY.index("adjacent")] = 1
        np.testing.assert_array_equal(nf[7], expected)

    def test_atom_planning_false_raises(self):
        meta = {"time": 0, "timeout": 2000, "planning": False}
        with self.assertRaises(NotImplementedError):
            facts_to_atom_graph(self.facts, meta)


# ==========================================================================================
# Shared real-state sampling helpers, for (b)-(g)
# ==========================================================================================

def _sample_real_states(seeds=(0, 1, 2, 3, 4), sample_every=5, per_seed_target=110, random_prob=0.2):
    """Real facts() snapshots from scripted-policy play on CITY_TERNARY -- reuses
    scripted_policy from tests/test_ternary_world.py unmodified, exactly as Part 1's
    coverage sweep and Part 2's collision counter both already do."""
    states = []
    for seed in seeds:
        sim = TernaryTaxiWorldSimulator(np.random.RandomState(seed), observation_fn=lambda s: None, **CITY_TERNARY)
        policy_rng = pyrandom.Random(seed + 4000)
        step = 0
        collected = 0
        while collected < per_seed_target:
            if step % sample_every == 0:
                states.append((seed, step, sim.facts()))
                collected += 1
            sim.act(scripted_policy(sim, policy_rng, random_prob=random_prob))
            step += 1
            if sim.done:
                sim = TernaryTaxiWorldSimulator(
                    np.random.RandomState(seed * 100003 + step), observation_fn=lambda s: None, **CITY_TERNARY
                )
    return states


def _sample_real_crossed_fact_pairs(seeds=(0, 1, 2, 3, 4), sample_every=5, per_seed_target=12, random_prob=0.2):
    """Real crossed states (taxi carrying p, buddy q still present) paired with their
    flipped counterfactual, via build_flipped_facts (analysis/ternary_collision_counter.py,
    unmodified) -- the exact same State-A/State-B construction the collision counter
    uses, on the RAW production facts (not the analysis's ext-atom adaptation)."""
    pairs = []
    for seed in seeds:
        sim = TernaryTaxiWorldSimulator(np.random.RandomState(seed), observation_fn=lambda s: None, **CITY_TERNARY)
        policy_rng = pyrandom.Random(seed + 6000)
        step = 0
        collected = 0
        while collected < per_seed_target:
            if step % sample_every == 0 and crossed_state(sim):
                _p, _q, flipped_facts = build_flipped_facts(sim)
                pairs.append((sim.facts(), flipped_facts))
                collected += 1
            sim.act(scripted_policy(sim, policy_rng, random_prob=random_prob))
            step += 1
            if sim.done:
                sim = TernaryTaxiWorldSimulator(
                    np.random.RandomState(seed * 100003 + step), observation_fn=lambda s: None, **CITY_TERNARY
                )
    return pairs


# ==========================================================================================
# (b) Structural properties over 500+ real states
# ==========================================================================================

class TestStructuralProperties(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.states = _sample_real_states(per_seed_target=110)  # 5 seeds * 110 = 550
        assert len(cls.states) >= 500

    def test_shapes_match_ternary_graph_conventions(self):
        for convention, converter in (
            ("oracle_sage", facts_to_object_graph),
            ("vilg", facts_to_vilg_graph),
            ("atom", facts_to_atom_graph),
        ):
            node_dim, edge_dim = TERNARY_GRAPH_CONVENTIONS[convention]
            for _seed, _step, facts in self.states[::20]:  # subsample for speed
                nf, ef, ei, _mask, gf = converter(facts, PLANNING_META)
                self.assertEqual(nf.shape[1], node_dim, convention)
                self.assertEqual(ef.shape[1], edge_dim, convention)
                self.assertEqual(ei.shape[0], 2)
                self.assertEqual(gf.shape, (32,))

    def test_features_are_zero_one_except_dir(self):
        for _seed, _step, facts in self.states[::20]:
            nf, ef, _ei, _mask, _gf = facts_to_object_graph(facts, PLANNING_META)
            self.assertTrue(np.isin(nf, [0, 1]).all())
            non_dir = ef[:, :-1]
            dir_col = ef[:, -1]
            self.assertTrue(np.isin(non_dir, [0, 1]).all())
            self.assertTrue(np.isin(dir_col, [1, -1]).all())

            nf, ef, _ei, _mask, _gf = facts_to_vilg_graph(facts, PLANNING_META)
            self.assertTrue(np.isin(nf, [0, 1]).all())
            self.assertTrue(np.isin(ef, [0, 1]).all())

            nf, ef, _ei, _mask, _gf = facts_to_atom_graph(facts, PLANNING_META)
            self.assertTrue(np.isin(nf, [0, 1]).all())
            self.assertTrue(np.isin(ef, [0, 1]).all())

    def test_vilg_edges_exactly_mirrored(self):
        for _seed, _step, facts in self.states[::40]:
            _nf, ef, ei, _mask, _gf = facts_to_vilg_graph(facts, PLANNING_META)
            edges = _edges_as_dict(ei, ef)
            for (u, v), label in edges.items():
                self.assertIn((v, u), edges, f"({u},{v}) has no mirrored ({v},{u})")
                self.assertEqual(edges[(v, u)], label, f"({u},{v}) and ({v},{u}) labels differ")

    def test_atom_labels_transposed_on_reverse_edges(self):
        n_labels = ATOM_MAX_ARITY * ATOM_MAX_ARITY
        labels = [(i, j) for i in range(1, ATOM_MAX_ARITY + 1) for j in range(1, ATOM_MAX_ARITY + 1)]
        transpose_bit = {bit: labels.index((j, i)) for bit, (i, j) in enumerate(labels)}
        for _seed, _step, facts in self.states[::40]:
            _nf, ef, ei, _mask, _gf = facts_to_atom_graph(facts, PLANNING_META)
            edges = _edges_as_dict(ei, ef)
            for (u, v), row in edges.items():
                reverse = edges[(v, u)]
                for bit in range(n_labels):
                    if row[bit] == 1.0:
                        self.assertEqual(reverse[transpose_bit[bit]], 1.0, f"edge ({u},{v}) bit {bit} has no transposed reverse")

    def test_object_every_forward_label_has_reverse_counterpart(self):
        for _seed, _step, facts in self.states[::40]:
            _nf, ef, ei, _mask, _gf = facts_to_object_graph(facts, PLANNING_META)
            edges = _edges_as_dict(ei, ef)
            for (u, v), row in edges.items():
                self.assertIn((v, u), edges, f"({u},{v}) has no reverse ({v},{u})")

    def test_row_k_is_object_k_in_every_convention(self):
        for _seed, _step, facts in self.states[::20]:
            n_obj = _validate_facts(facts)
            for i, (predicate, args) in enumerate(facts[:n_obj]):
                self.assertEqual(args[0], i)

            for converter in (facts_to_object_graph, facts_to_vilg_graph, facts_to_atom_graph):
                nf, _ef, _ei, mask, _gf = converter(facts, PLANNING_META)
                self.assertGreaterEqual(nf.shape[0], n_obj)
                # mask must be True on exactly the type-atom rows for planning=True
                self.assertTrue(mask[:n_obj].all())
                if nf.shape[0] > n_obj:
                    self.assertFalse(mask[n_obj:].any())

    def test_mask_true_only_where_the_convention_says(self):
        for _seed, _step, facts in self.states[::40]:
            n_obj = _validate_facts(facts)
            _nf, _ef, _ei, mask_o, _gf = facts_to_object_graph(facts, PLANNING_META)
            self.assertTrue(mask_o.all())  # planning=True -> every object selectable

            _nf, _ef, _ei, mask_v, _gf = facts_to_vilg_graph(facts, PLANNING_META)
            self.assertTrue(mask_v[:n_obj].all())
            self.assertFalse(mask_v[n_obj:].any())

            _nf, _ef, _ei, mask_a, _gf = facts_to_atom_graph(facts, PLANNING_META)
            self.assertTrue(mask_a[:n_obj].all())
            self.assertFalse(mask_a[n_obj:].any())


# ==========================================================================================
# WL machinery for the PRODUCTION atom/vilg tensors (object needs none -- see (c)/(d))
# ==========================================================================================

def _production_initial_colours_vilg(x):
    """Disjoint-block decode for the PRODUCTION 10-dim vilg node layout (object
    one-hot(3) + predicate-one-hot(4)/nongoal-status(3) block) -- mirrors
    ext_initial_colours_object_atom's own disjoint-block reasoning (that one is
    hardcoded to the ANALYSIS's 5-dim layout, so it isn't reusable here as-is)."""
    x = th.as_tensor(x, dtype=th.float32)
    is_object = x[:, 0:3].sum(dim=1) > 0
    obj_type = x[:, 0:3].argmax(dim=1)
    prop_type = 3 + x[:, 3:10].argmax(dim=1)
    return th.where(is_object, obj_type, prop_type).long()


def _as_tensors(node_feats, edge_index, edge_feats):
    """ext_initial_colours_*/ext_edge_labels_* (analysis/indistinguishability.py)
    assume torch tensor input, with no internal cast (see that module's own _t3
    helper, which does the same cast before ever calling wl_colours_per_level) --
    facts_to_*_graph returns plain numpy per this module's own contract (item 3), so
    every caller of wl_colours_per_level on production output must cast first."""
    return (
        th.as_tensor(node_feats, dtype=th.float32),
        th.as_tensor(edge_index, dtype=th.long),
        th.as_tensor(edge_feats, dtype=th.float32),
    )


def _wl_collision_rates(pairs, converter, init_fn, edge_fn, max_l=5):
    """Fraction of (facts_a, facts_b) pairs whose WL colour-multiset hash MATCHES at
    each L=1..max_l, for one PRODUCTION converter -- the direct analogue of
    analysis/ternary_collision_counter.py's flip_pair_collision_rates, but for the
    production tensors instead of the analysis's own ext encoders."""
    vocab = {}
    matches = {level: 0 for level in range(1, max_l + 1)}
    for facts_a, facts_b in pairs:
        nf_a, ef_a, ei_a, _m, _g = converter(facts_a, PLANNING_META)
        nf_b, ef_b, ei_b, _m, _g = converter(facts_b, PLANNING_META)
        nf_a, ei_a, ef_a = _as_tensors(nf_a, ei_a, ef_a)
        nf_b, ei_b, ef_b = _as_tensors(nf_b, ei_b, ef_b)
        levels_a = wl_colours_per_level(nf_a, ei_a, ef_a, init_fn, edge_fn, vocab, max_l)
        levels_b = wl_colours_per_level(nf_b, ei_b, ef_b, init_fn, edge_fn, vocab, max_l)
        for level in range(1, max_l + 1):
            if colour_multiset_hash(levels_a[level]) == colour_multiset_hash(levels_b[level]):
                matches[level] += 1
    n = len(pairs)
    return {level: (matches[level] / n if n else float("nan")) for level in range(1, max_l + 1)}


def _ext_collision_rates(pairs, convention, init_fn, edge_fn, max_l=5):
    """Same computation, but adapting facts through facts_to_ext_atoms and using the
    ANALYSIS's own encoder for `convention` -- for the (d) cross-check."""
    vocab = {}
    matches = {level: 0 for level in range(1, max_l + 1)}
    for facts_a, facts_b in pairs:
        graphs_a = _build_part2_graphs(facts_to_ext_atoms(facts_a))[convention]
        graphs_b = _build_part2_graphs(facts_to_ext_atoms(facts_b))[convention]
        levels_a = wl_colours_per_level(*graphs_a, init_fn, edge_fn, vocab, max_l)
        levels_b = wl_colours_per_level(*graphs_b, init_fn, edge_fn, vocab, max_l)
        for level in range(1, max_l + 1):
            if colour_multiset_hash(levels_a[level]) == colour_multiset_hash(levels_b[level]):
                matches[level] += 1
    n = len(pairs)
    return {level: (matches[level] / n if n else float("nan")) for level in range(1, max_l + 1)}


# ==========================================================================================
# (c) Premise test: the experiment's premise, on the PRODUCTION converters
# ==========================================================================================

class TestPremiseOnProductionConverters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = _sample_real_crossed_fact_pairs(per_seed_target=12)  # 5 seeds * 12 = 60
        assert len(cls.pairs) >= 50, f"only {len(cls.pairs)} crossed pairs sampled, need >= 50"

    def test_object_tensors_exactly_identical(self):
        for facts_a, facts_b in self.pairs:
            nf_a, ef_a, ei_a, mask_a, gf_a = facts_to_object_graph(facts_a, PLANNING_META)
            nf_b, ef_b, ei_b, mask_b, gf_b = facts_to_object_graph(facts_b, PLANNING_META)
            np.testing.assert_array_equal(nf_a, nf_b)
            np.testing.assert_array_equal(ef_a, ef_b)
            np.testing.assert_array_equal(ei_a, ei_b)
            np.testing.assert_array_equal(mask_a, mask_b)
            np.testing.assert_array_equal(gf_a, gf_b)

    def test_atom_and_vilg_differ_at_L5_for_every_pair(self):
        atom_rates = _wl_collision_rates(self.pairs, facts_to_atom_graph, ext_initial_colours_atom, ext_edge_labels_atom)
        vilg_rates = _wl_collision_rates(
            self.pairs, facts_to_vilg_graph, _production_initial_colours_vilg, ext_edge_labels_object_atom
        )
        self.__class__._atom_rates = atom_rates
        self.__class__._vilg_rates = vilg_rates
        self.assertEqual(atom_rates[5], 0.0, "atom still collides for some pair at L=5")
        self.assertEqual(vilg_rates[5], 0.0, "vilg still collides for some pair at L=5")

    def test_report_atom_and_vilg_collision_rates(self):
        atom_rates = getattr(self.__class__, "_atom_rates", None) or _wl_collision_rates(
            self.pairs, facts_to_atom_graph, ext_initial_colours_atom, ext_edge_labels_atom
        )
        vilg_rates = getattr(self.__class__, "_vilg_rates", None) or _wl_collision_rates(
            self.pairs, facts_to_vilg_graph, _production_initial_colours_vilg, ext_edge_labels_object_atom
        )
        print(f"\n[PRODUCTION] n={len(self.pairs)} crossed-state/flipped pairs")
        print(f"{'convention':<12}" + "".join(f"L={l:<8}" for l in range(1, 6)))
        print(f"{'atom':<12}" + "".join(f"{atom_rates[l]:<10.0%}" for l in range(1, 6)))
        print(f"{'vilg':<12}" + "".join(f"{vilg_rates[l]:<10.0%}" for l in range(1, 6)))


# ==========================================================================================
# (d) Cross-check against the independent analysis encoders
# ==========================================================================================

class TestCrossCheckAgainstAnalysisEncoders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = _sample_real_crossed_fact_pairs(per_seed_target=12)

    def test_object_matches_analysis_100_percent(self):
        n = len(self.pairs)
        identical = 0
        for facts_a, facts_b in self.pairs:
            nf_a, ef_a, ei_a, _m, _g = facts_to_object_graph(facts_a, PLANNING_META)
            nf_b, ef_b, ei_b, _m, _g = facts_to_object_graph(facts_b, PLANNING_META)
            if np.array_equal(nf_a, nf_b) and np.array_equal(ef_a, ef_b) and np.array_equal(ei_a, ei_b):
                identical += 1
        production_rate = identical / n
        self.assertEqual(production_rate, 1.0)
        # the analysis's own object encoder, on the SAME pairs, adapted via facts_to_ext_atoms
        analysis_rates = _ext_collision_rates(
            self.pairs, "object", *PART2_ENCODERS["object"][1:]
        )
        for level in range(1, 6):
            self.assertEqual(analysis_rates[level], 1.0)
            self.assertEqual(production_rate, analysis_rates[level])

    def test_atom_and_vilg_rates_at_most_analysis_rates(self):
        production_atom = _wl_collision_rates(self.pairs, facts_to_atom_graph, ext_initial_colours_atom, ext_edge_labels_atom)
        production_vilg = _wl_collision_rates(
            self.pairs, facts_to_vilg_graph, _production_initial_colours_vilg, ext_edge_labels_object_atom
        )
        analysis_atom = _ext_collision_rates(self.pairs, "atom", *PART2_ENCODERS["atom"][1:])
        analysis_vilg = _ext_collision_rates(self.pairs, "object_atom", *PART2_ENCODERS["object_atom"][1:])

        print(f"\n[PRODUCTION vs ANALYSIS] n={len(self.pairs)} crossed-state/flipped pairs")
        print(f"{'':<20}" + "".join(f"L={l:<8}" for l in range(1, 6)))
        print(f"{'atom (production)':<20}" + "".join(f"{production_atom[l]:<10.0%}" for l in range(1, 6)))
        print(f"{'atom (analysis)':<20}" + "".join(f"{analysis_atom[l]:<10.0%}" for l in range(1, 6)))
        print(f"{'vilg (production)':<20}" + "".join(f"{production_vilg[l]:<10.0%}" for l in range(1, 6)))
        print(f"{'vilg (analysis)':<20}" + "".join(f"{analysis_vilg[l]:<10.0%}" for l in range(1, 6)))

        for level in range(1, 6):
            self.assertLessEqual(production_atom[level], analysis_atom[level] + 1e-9, f"atom L={level}")
            self.assertLessEqual(production_vilg[level], analysis_vilg[level] + 1e-9, f"vilg L={level}")


# ==========================================================================================
# (e) Regression: the generic atom builder reproduces cell5's atoms_to_graph at arity 2
# ==========================================================================================

class TestGenericAtomBuilderRegressionAgainstCell5(unittest.TestCase):
    def test_matches_cell5_atoms_to_graph_exactly(self):
        sim = TaxiWorldSimulator(np.random.RandomState(0), planning=True, graph_convention="atom", **CITY)
        rng = pyrandom.Random(1)
        for _ in range(50):  # move the taxi around, pick up/drop off, for a richer state
            location = sim.taxi.location
            candidates = [c for c in sim.graph.successors(location)]
            candidates.append(0)
            sim.act(rng.choice(candidates))

        old_atoms = env_to_atoms(sim)
        expected_nf, expected_ef, expected_ei = atoms_to_graph(old_atoms)
        actual_nf, actual_ef, actual_ei = _atoms_to_graph_generic(old_atoms, OLD_ATOM_PREDICATES, 2)

        np.testing.assert_array_equal(actual_nf, expected_nf)
        np.testing.assert_array_equal(actual_ef, expected_ef)
        np.testing.assert_array_equal(actual_ei, expected_ei)


# ==========================================================================================
# (f) JSON round-trip
# ==========================================================================================

class TestJSONRoundTrip(unittest.TestCase):
    def test_round_trip_is_exact_on_hand_checked_state(self):
        sim = make_hand_checked_sim()
        facts = sim.facts()
        meta = {"time": 3, "timeout": 2000, "planning": True}
        decoded_facts, decoded_meta = json_to_facts(facts_to_json(facts, meta))
        self.assertEqual(decoded_facts, facts)
        self.assertEqual(decoded_meta, meta)

    def test_round_trip_is_exact_over_real_states(self):
        for _seed, _step, facts in _sample_real_states(per_seed_target=20):
            meta = {"time": 7, "timeout": 2000, "planning": True}
            decoded_facts, decoded_meta = json_to_facts(facts_to_json(facts, meta))
            self.assertEqual(decoded_facts, facts)
            self.assertEqual(decoded_meta, meta)

    def test_max_json_length_over_5_seeds_full_episodes(self):
        max_length = 0
        max_seed = None
        for seed in CITY_TERNARY_SEEDS:
            sim = TernaryTaxiWorldSimulator(np.random.RandomState(seed), observation_fn=lambda s: None, **CITY_TERNARY)
            policy_rng = pyrandom.Random(seed + 8000)
            for step in range(2000):
                facts = sim.facts()
                meta = {"time": sim.time, "timeout": sim.timeout, "planning": True}
                length = len(facts_to_json(facts, meta))
                if length > max_length:
                    max_length = length
                    max_seed = seed
                sim.act(scripted_policy(sim, policy_rng, random_prob=0.2))
                if sim.done:
                    break
        print(f"\nMax facts_to_json length over {len(CITY_TERNARY_SEEDS)} seeds x up to 2000 steps: "
              f"{max_length} chars (seed {max_seed}), vs 250000 width")
        self.assertLess(max_length, 250000)


CITY_TERNARY_SEEDS = (0, 300, 600, 900, 1200)


# ==========================================================================================
# (g) Per-convention node/edge counts and conversion time on real city states
# ==========================================================================================

class TestMeasurements(unittest.TestCase):
    def test_measurements_over_real_city_states(self):
        states = _sample_real_states(per_seed_target=40)  # 5 seeds * 40 = 200
        results = {}
        for convention, converter in (
            ("oracle_sage", facts_to_object_graph),
            ("vilg", facts_to_vilg_graph),
            ("atom", facts_to_atom_graph),
        ):
            n_nodes, n_edges, times_ms = [], [], []
            for _seed, _step, facts in states:
                t0 = time.time()
                nf, _ef, ei, _mask, _gf = converter(facts, PLANNING_META)
                times_ms.append((time.time() - t0) * 1000)
                n_nodes.append(nf.shape[0])
                n_edges.append(ei.shape[1])
            results[convention] = {
                "mean_nodes": np.mean(n_nodes), "max_nodes": np.max(n_nodes),
                "mean_edges": np.mean(n_edges), "max_edges": np.max(n_edges),
                "mean_ms": np.mean(times_ms), "max_ms": np.max(times_ms),
            }

        print(f"\nPer-convention node/edge counts and conversion time, {len(states)} real city states "
              f"(Cell 5's atom reference: 1,448 atoms / 12,600 edges / 3 ms):")
        for convention, r in results.items():
            print(
                f"  {convention:<12} nodes mean={r['mean_nodes']:.0f} max={r['max_nodes']:.0f}  "
                f"edges mean={r['mean_edges']:.0f} max={r['max_edges']:.0f}  "
                f"time mean={r['mean_ms']:.2f}ms max={r['max_ms']:.2f}ms"
            )


if __name__ == "__main__":
    unittest.main()
