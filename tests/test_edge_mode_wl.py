"""
analysis/edge_mode_wl.py (A5), against cases whose answer is known without the script:

  - hand-built graphs where only Gr separates two nodes (a 2-type Gr edge vs two 1-type
    edges), and where only the multiset bag separates them (1 vs 2 equal neighbours);
  - mgr and egr give the same partition, multiset refines set, gr refines mgr -- on the
    hand graphs and on real states, through hierarchy_checks;
  - gr-multiset is the thesis WL: same partition as indistinguishability.collision_table
    (standard) and as ternary_collision_production.pair_metrics (ternary hand cases C /
    C_FLIP, whose answers tests/test_ternary_collision_production.py derives by hand);
  - standard Object and Object-Atom have no Gr edge with two types, Atom does.

Run from the repo root with:
    PYTHONPATH=. python -m pytest tests/test_edge_mode_wl.py -v
"""
import unittest

import torch as th

from analysis import edge_mode_wl as a5
from analysis import indistinguishability as ind
from analysis import ternary_collision_production as tcp
from tests.test_ternary_collision_production import EXPECTED_CARRIED_SAME_AS_FLIP, META
from tests.test_wl_ternary import CASE_C, CASE_C_FLIP


def atom_graph(n, edges):
    """A standard Atom-encoding graph: n `adjacent` atoms (one-hot column 3), edges
    (src, dst, bits) with bits over ATOM_LABELS [(1,1), (1,2), (2,1), (2,2)]."""
    x = th.zeros(n, 6)
    x[:, 3] = 1
    edge_index = th.tensor([[s for s, _d, _b in edges], [d for _s, d, _b in edges]], dtype=th.long)
    edge_attr = th.tensor([b for _s, _d, b in edges], dtype=th.float)
    return a5.Prepared(x, edge_index, edge_attr, "standard", "atom")


def colours_at(graph, variant, level=1):
    return a5.wl_levels(graph, variant, {}, level)[level]


class TestHandBuilt(unittest.TestCase):
    def test_only_gr_separates_a_two_type_edge(self):
        # node 0 -> 2 with types (1,2) and (2,1) on ONE Gr edge; node 1 -> 3 (1,2), 1 -> 4 (2,1)
        g = atom_graph(5, [(0, 2, [0, 1, 1, 0]), (1, 3, [0, 1, 0, 0]), (1, 4, [0, 0, 1, 0])])
        self.assertEqual((g.gr_edges, g.typed_edges), (3, 4))
        for variant in a5.VARIANTS:
            with self.subTest(variant=variant):
                c = colours_at(g, variant)
                self.assertEqual(c[0] != c[1], variant[0] == "gr")

    def test_only_the_multiset_counts_equal_neighbours(self):
        # node 0 -> 2 (1,1); node 1 -> 3 (1,1) and 1 -> 4 (1,1): 1 vs 2 equal neighbours
        g = atom_graph(5, [(0, 2, [1, 0, 0, 0]), (1, 3, [1, 0, 0, 0]), (1, 4, [1, 0, 0, 0])])
        for variant in a5.VARIANTS:
            with self.subTest(variant=variant):
                c = colours_at(g, variant)
                self.assertEqual(c[0] != c[1], variant[1] == "multiset")

    def test_egr_keeps_types_apart_like_mgr(self):
        # node 0 -> 2 (1,1), 0 -> 3 (2,2); node 1 -> 4 (1,2), 1 -> 5 (2,1): same colours, other types
        g = atom_graph(6, [(0, 2, [1, 0, 0, 0]), (0, 3, [0, 0, 0, 1]), (1, 4, [0, 1, 0, 0]), (1, 5, [0, 0, 1, 0])])
        for variant in a5.VARIANTS:
            with self.subTest(variant=variant):
                c = colours_at(g, variant)
                self.assertNotEqual(c[0], c[1])

    def test_refines_helper(self):
        self.assertTrue(a5._refines(["a", "a", "b"], ["x", "x", "y"]))
        self.assertTrue(a5._refines(["a", "b", "c"], ["x", "x", "y"]))
        self.assertFalse(a5._refines(["a", "a"], ["x", "y"]))
        self.assertFalse(a5._same_partition(["a", "b"], ["x", "x"]))

    def test_hierarchy_checks_catch_a_violation(self):
        ok = {v: {1: ["h1", "h2"]} for v in a5.VARIANTS}
        self.assertEqual(a5.hierarchy_checks(ok, [1]), [])
        bad = dict(ok)
        bad[("egr", "set")] = {1: ["h1", "h1"]}
        self.assertTrue(any("mgr-set and egr-set differ" in f for f in a5.hierarchy_checks(bad, [1])))


class TestAgainstTheThesisCode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.states = ind.generate_part1_states(seeds=(0, 300), target_total=40, verbose=False)
        seen = {}
        for s in cls.states:
            seen.setdefault(s["atoms"], s)
        cls.distinct = list(seen.values())

    def test_standard_gr_multiset_is_collision_table_and_hierarchy_holds(self):
        for convention in ind.PART1_CONVENTIONS:
            with self.subTest(convention=convention):
                init_fn, edge_fn = a5.DECODERS[("standard", convention)]
                thesis = ind.collision_table(self.states, convention, init_fn, edge_fn, max_l=3)
                graphs = [a5.Prepared(*s["graphs"][convention], "standard", convention) for s in self.distinct]
                hashes = {}
                for variant in a5.VARIANTS:
                    vocab = {}
                    levels = [a5.wl_levels(g, variant, vocab, 3) for g in graphs]
                    hashes[variant] = {L: [a5.histogram_key(lv[L]) for lv in levels] for L in (1, 2, 3)}
                for L in (1, 2, 3):
                    mine = a5.count_pairs(self.distinct, hashes[("gr", "multiset")][L])
                    for key in ("distinct_hashes", "collisions", "bad_collisions"):
                        self.assertEqual(mine[key], thesis[L][key])
                self.assertEqual(a5.hierarchy_checks(hashes, (1, 2, 3)), [])

    def test_only_standard_atom_has_two_type_edges(self):
        for convention in ind.PART1_CONVENTIONS:
            graphs = [a5.Prepared(*s["graphs"][convention], "standard", convention) for s in self.distinct]
            with self.subTest(convention=convention):
                multi = sum(g.typed_edges != g.gr_edges for g in graphs)
                if convention == "atom":
                    self.assertEqual(multi, len(graphs))
                else:
                    self.assertEqual(multi, 0)


class TestTernaryHandCases(unittest.TestCase):
    def test_gr_multiset_is_pair_metrics_and_mgr_egr_agree(self):
        for convention in tcp.CONVENTIONS:
            data_a, data_b = tcp.encode(CASE_C, META, convention), tcp.encode(CASE_C_FLIP, META, convention)
            ga = a5.Prepared(data_a.x, data_a.edge_index, data_a.edge_attr, "ternary", convention)
            gb = a5.Prepared(data_b.x, data_b.edge_index, data_b.edge_attr, "ternary", convention)
            for level, expected in EXPECTED_CARRIED_SAME_AS_FLIP[convention].items():
                m1, m2, _ = tcp.pair_metrics(data_a, data_b, 5, 5, convention, level, {})
                for variant in a5.VARIANTS:
                    vocab = {}
                    ca = a5.wl_levels(ga, variant, vocab, level)[level]
                    cb = a5.wl_levels(gb, variant, vocab, level)[level]
                    with self.subTest(convention=convention, L=level, variant=variant):
                        if variant == ("gr", "multiset"):
                            self.assertEqual(ca[5] == cb[5], expected)
                            self.assertEqual((sorted(ca) == sorted(cb), ca[5] == cb[5]), (m1, m2))
                        if variant[1] == "multiset":  # hand derivation holds per type too
                            self.assertEqual(ca[5] == cb[5], expected)


if __name__ == "__main__":
    unittest.main()
