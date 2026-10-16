"""
wl_colours on the ternary domain (domain="ternary"), against ground truth that does not
reuse the code under test.

  a. Hand-built fact sets, with node partitions at L=1 and L=2 derived by hand (the
     derivations are in the comments next to each expected value), run through the
     production path: facts -> facts_to_json -> the policy-side ternary decoder
     (json_to_ternary_graph_*) -> wl_colours. Also the whole-graph histogram of each state
     against its counterfactual with the two buddies' request tables exchanged.
  b. An independent reference WL, written here with no sage imports, over a labelled
     DiGraph built DIRECTLY from the facts using each encoding's definition; node
     PARTITIONS (not colour ids) are compared at L = 0 (initial colouring), 1, 2, 3 on
     real simulator states
     from several seeds, including spawn_pair / initial_pair states and crossed states.
     Where networkx has weisfeiler_lehman_subgraph_hashes, it is a second reference.
  c. Standard-domain regression: wl_colours equals the fixture recorded at ebbf25b
     (analysis/make_wl_standard_fixture.py) for all three conventions.
  d. Width guards: a ternary graph decoded as standard raises, and vice versa.
  e. CUDA-gated: identical outputs for CPU and CUDA tensors.

Direction convention, checked against refine() in sage/domains/utils/wl_colours.py: a
node's new colour is (own colour, sorted multiset of (neighbour colour, edge label)) over
its OUTGOING edges (edge_index[0] == node), each edge carrying its own label. The
reference WL below and networkx (G.neighbors on a DiGraph = successors) use the same one.

Run from the repo root with:
    PYTHONPATH=. python -m pytest tests/test_wl_ternary.py -v
"""
import gzip
import hashlib
import json
import os
import random as pyrandom
import unittest

import networkx as nx
import numpy as np
import torch as th

import sage.domains.gym_taxi  # noqa: F401  (registers the env ids)
from sage.domains.gym_taxi.utils import wl_vocab_cache
from sage.domains.gym_taxi.utils.ternary_representations import (
    TERNARY_GRAPH_CONVENTION_CONVERTERS,
    facts_to_json,
)
from sage.domains.utils.wl_colours import wl_colours
from tests.test_ternary_planner import make_ternary_sim
from sage.domains.gym_taxi.simulator.ternary_scripted_policy import scripted_policy
from tests.test_ternary_world import crossed_state

CONVENTIONS = ("oracle_sage", "vilg", "atom")
FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "wl_standard_ebbf25b.json.gz")


# ==========================================================================================
# Production path and partition helpers
# ==========================================================================================

def ternary_tensors(facts, convention):
    """facts -> the compact JSON the env emits -> the policy-side decoder, i.e. exactly the
    tensors a policy batch carries."""
    meta = {"time": 0, "timeout": 100, "planning": True}
    batch = TERNARY_GRAPH_CONVENTION_CONVERTERS[convention]([[facts_to_json(facts, meta)]])
    return batch.x, batch.edge_index, batch.edge_attr


def partition(labels):
    """Node partition induced by a per-node label list: a set of frozensets of node ids."""
    groups = {}
    for node, label in enumerate(labels):
        groups.setdefault(label, set()).add(node)
    return {frozenset(g) for g in groups.values()}


def as_partition(groups):
    return {frozenset(g) for g in groups}


def production_colours(facts, convention, num_iterations, vocab=None):
    if vocab is None:
        vocab = {}
    colours, histogram = wl_colours(
        *ternary_tensors(facts, convention), num_iterations=num_iterations, vocab=vocab,
        frozen=False, graph_convention=convention, domain="ternary",
    )
    return colours.tolist(), histogram.tolist()


def same_histogram(h1, h2):
    """Histograms from one shared growing vocab can differ in length (the vocab grew
    between the calls); equal means equal counts for every id, missing ids counting 0."""
    n = max(len(h1), len(h2))
    return list(h1) + [0.0] * (n - len(h1)) == list(h2) + [0.0] * (n - len(h2))


# ==========================================================================================
# a. Hand-built fact sets
#
# Objects in every case: taxi 0, locations 1..4 (1..3 in case R), passengers 5 = p and
# 6 = q, buddies (q's requests are p's with the destinations rotated, rule 2):
#   p: request(5, 1, 3), request(5, 2, 4)        q: request(6, 1, 4), request(6, 2, 3)
# Row order in every encoding: objects 0..n_obj-1 first, then (vilg) one node per
# relational fact / (atom) one node per fact, in list order.
# ==========================================================================================

TYPE_ATOMS_4 = [("taxi", (0,)), ("location", (1,)), ("location", (2,)), ("location", (3,)),
                ("location", (4,)), ("passenger", (5,)), ("passenger", (6,))]

# W: both buddies waiting at origin 1, taxi at location 3.
CASE_W = TYPE_ATOMS_4 + [
    ("in", (0, 3)), ("in", (5, 1)), ("in", (6, 1)),
    ("request", (5, 1, 3)), ("request", (5, 2, 4)), ("request", (6, 1, 4)), ("request", (6, 2, 3)),
]
# W with p's and q's request tables exchanged: the same graph with p and q relabelled.
CASE_W_FLIP = TYPE_ATOMS_4 + [
    ("in", (0, 3)), ("in", (5, 1)), ("in", (6, 1)),
    ("request", (5, 1, 4)), ("request", (5, 2, 3)), ("request", (6, 1, 3)), ("request", (6, 2, 4)),
]
# C: crossed. p boarded at origin 1 and is in the taxi; q still waits at 1; the taxi is at
# location 3, which is p's true destination (origin 1 -> request(5, 1, 3)).
CASE_C = TYPE_ATOMS_4 + [
    ("in", (0, 3)), ("in", (5, 0)), ("in", (6, 1)),
    ("request", (5, 1, 3)), ("request", (5, 2, 4)), ("request", (6, 1, 4)), ("request", (6, 2, 3)),
    ("picked_up_at", (5, 1)),
]
# C with the request tables exchanged: p's true destination becomes 4.
CASE_C_FLIP = TYPE_ATOMS_4 + [
    ("in", (0, 3)), ("in", (5, 0)), ("in", (6, 1)),
    ("request", (5, 1, 4)), ("request", (5, 2, 3)), ("request", (6, 1, 3)), ("request", (6, 2, 4)),
    ("picked_up_at", (5, 1)),
]
# R: roads only. Path 1 - 2 - 3 (both directions), taxi at the middle location 2.
CASE_R = [("taxi", (0,)), ("location", (1,)), ("location", (2,)), ("location", (3,)),
          ("adjacent", (1, 2)), ("adjacent", (2, 1)), ("adjacent", (2, 3)), ("adjacent", (3, 2)),
          ("in", (0, 2))]

# Rows, case W/W_FLIP:  vilg/atom 7 in(0,3)  8 in(5,1)  9 in(6,1)  10 R1  11 R2  12 R3  13 R4
#                       (R1..R4 = the four requests in list order)
# Rows, case C/C_FLIP:  as W, with 8 in(5,0) and 14 picked_up_at(5,1)
# Rows, case R:         vilg/atom 4 adj(1,2)  5 adj(2,1)  6 adj(2,3)  7 adj(3,2)  8 in(0,2)
EXPECTED = {
    # ---------------------------------------------------------------- object, case W
    # Edges (all facts give both directions): in(x,y): x->y AT, y->x AT with dir -1;
    # request(p,o,d): p->o req12, o->p req21, p->d req13, d->p req31, o->d req23, d->o req32.
    # L=1, own colour + multiset of (neighbour's type, merged edge label) over out-edges:
    #   0: (loc,AT)                                  unique
    #   1: (pass,AT+req21,dir-)x2, (loc,req23)x2     2: (pass,req21)x2, (loc,req23)x2
    #   3: (taxi,AT,dir-), (pass,req31)x2, (loc,req32)x2      4: (pass,req31)x2, (loc,req32)x2
    #   5: (loc,AT+req12) [to 1], (loc,req12) [2], (loc,req13) [3], (loc,req13) [4]
    #   6: (loc,AT+req12) [1], (loc,req12) [2], (loc,req13) [4], (loc,req13) [3]
    # 5 and 6 have the SAME out-neighbours (1,2,3,4) with the SAME labels, so no L can
    # ever split them: their signatures are built from identical neighbour colours.
    ("W", "oracle_sage", 1): [{0}, {1}, {2}, {3}, {4}, {5, 6}],
    ("W", "oracle_sage", 2): [{0}, {1}, {2}, {3}, {4}, {5, 6}],
    # ---------------------------------------------------------------- vilg, case W
    # Mirrored object<->proposition edges labelled by argument position; a proposition's
    # initial colour is its predicate (status is the constant nongoal for every one).
    # L=1: a node sees only its neighbours' TYPES and the positions:
    #   1: (in,2)x2, (req,2)x2   2: (req,2)x2   3: (in,2), (req,3)x2   4: (req,3)x2  -> all differ
    #   5 and 6: (in,1), (req,1)x2  -> same
    #   7 in(0,3): (taxi,1),(loc,2)    8, 9 in(p|q,1): (pass,1),(loc,2) -> 8 = 9
    #   10..13 requests: (pass,1),(loc,2),(loc,3) -> all four the same
    ("W", "vilg", 1): [{0}, {1}, {2}, {3}, {4}, {5, 6}, {7}, {8, 9}, {10, 11, 12, 13}],
    # L=2: requests now see WHICH locations (all distinct at L=1) at positions 2 and 3:
    #   R1 (P,1,3), R2 (P,2,4), R3 (P,1,4), R4 (P,2,3) -> all four differ.
    #   5 and 6 still see only L=1 colours: {in8/9, two requests} with the requests still
    #   one class at L=1 -> 5 = 6. 8 and 9 see (passenger class, location 1) -> 8 = 9.
    # (L=3: 5 sees {R1, R2}, 6 sees {R3, R4}, distinct at L=2 -> the buddies split at L=3.)
    ("W", "vilg", 2): [{0}, {1}, {2}, {3}, {4}, {5, 6}, {7}, {8, 9}, {10}, {11}, {12}, {13}],
    # ---------------------------------------------------------------- atom, case W
    # Every fact is a node; a->b is labelled with every (i, j) such that argument i of a
    # equals argument j of b. L=1 (neighbour predicate + label set):
    #   passengers 5, 6: (in,{11}), (req,{11})x2               -> same
    #   in(5,1) [8]: (pass,{11}), (loc,{21}), (in,{22}) [in(6,1)], (req,{11,22}) [R1],
    #                (req,{11}) [R2], (req,{22}) [R3]
    #   in(6,1) [9]: (pass,{11}), (loc,{21}), (in,{22}), (req,{11,22}) [R3], (req,{11}) [R4],
    #                (req,{22}) [R1]                             -> 8 = 9
    #   R1 (5,1,3): pass11, loc21, loc31, in{11,22}, in32 [in(0,3)], in22 [in(6,1)],
    #               req11 [R2], req22 [R3], req33 [R4]
    #   R2 (5,2,4): pass11, loc21, loc31, in11, req11, req33 [R3], req22 [R4]
    #   R3 (6,1,4): pass11, loc21, loc31, in{11,22}, in22 [in(5,1)], req11, req22, req33
    #   R4 (6,2,3): pass11, loc21, loc31, in11, in32 [in(0,3)], req11, req22, req33
    #   -> the four requests already differ at L=1 (the request atom sees, in one hop, the
    #      in-atoms that share its origin or destination); locations as in vilg.
    ("W", "atom", 1): [{0}, {1}, {2}, {3}, {4}, {5, 6}, {7}, {8, 9}, {10}, {11}, {12}, {13}],
    # L=2: 5 sees {R1, R2}, 6 sees {R3, R4}, all distinct at L=1 -> the buddies split at
    # L=2; 8 and 9 see different requests through their {11,22} edges -> split too.
    ("W", "atom", 2): [{i} for i in range(14)],
    # ---------------------------------------------------------------- object, case C
    #   0: (loc,AT) [to 3], (pass,AT,dir-) [to 5, reverse of in(5,0)]
    #   1: (pass,PUA+req21,dir-) [5], (pass,AT+req21,dir-) [6], (loc,req23)x2
    #   5: (taxi,AT), (loc,PUA+req12) [1], (loc,req12) [2], (loc,req13)x2
    #   6: (loc,AT+req12) [1], (loc,req12) [2], (loc,req13)x2   -> 5 != 6 already at L=1
    #   2, 3, 4 differ as in W (3 has the taxi's reverse AT edge).
    ("C", "oracle_sage", 1): [{i} for i in range(7)],
    ("C", "oracle_sage", 2): [{i} for i in range(7)],
    # ---------------------------------------------------------------- vilg, case C
    # Rows: 7 in(0,3), 8 in(5,0), 9 in(6,1), 10..13 R1..R4, 14 picked_up_at(5,1).
    # L=1: 5: (in,1), (req,1)x2, (pua,1)   6: (in,1), (req,1)x2   -> differ (p has boarded)
    #   3: (in,2) [in(0,3)], (req,3)x2   4: (req,3)x2   1: (in,2), (req,2)x2, (pua,2)
    #   8 in(5,0): (pass,1),(taxi,2)   9 in(6,1): (pass,1),(loc,2)   7: (taxi,1),(loc,2)
    #   14 pua: own colour differs from every in-node; requests still one class.
    ("C", "vilg", 1): [{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10, 11, 12, 13}, {14}],
    # L=2: requests see (5|6, which origin, which destination), all distinct at L=1.
    ("C", "vilg", 2): [{i} for i in range(15)],
    # ---------------------------------------------------------------- atom, case C
    # L=1: R1 (5,1,3) sees pua(5,1) through {11,22} AND in(0,3) through {32};
    #      R2 sees pua through {11} only; R3 sees pua through {22} and in(6,1) through
    #      {11,22}; R4 sees in(0,3) through {32} and no pua. Passengers: 5 has a pua edge,
    #      6 does not. Locations and in-atoms as before -> every node distinct at L=1.
    ("C", "atom", 1): [{i} for i in range(15)],
    ("C", "atom", 2): [{i} for i in range(15)],
    # ---------------------------------------------------------------- case R (roads)
    # The swap 1 <-> 3 (with adj(1,2) <-> adj(3,2), adj(2,1) <-> adj(2,3)) is an
    # automorphism fixing the taxi, so no L can split 1 from 3.
    # object: 1, 3: (loc,road) ; 2: (loc,road)x2, (taxi,AT,dir-) ; 0: (loc,AT).
    ("R", "oracle_sage", 1): [{0}, {2}, {1, 3}],
    ("R", "oracle_sage", 2): [{0}, {2}, {1, 3}],
    # vilg L=1: every adjacency proposition sees (loc,1),(loc,2) -> one class;
    #   1, 3: (adj,1),(adj,2)   2: (adj,1)x2,(adj,2)x2,(in,2)   8 in(0,2): (taxi,1),(loc,2)
    ("R", "vilg", 1): [{0}, {2}, {1, 3}, {4, 5, 6, 7}, {8}],
    # vilg L=2: adj(1,2) and adj(3,2) run end -> middle (position 2 is location 2),
    #   adj(2,1) and adj(2,3) run middle -> end.
    ("R", "vilg", 2): [{0}, {2}, {1, 3}, {4, 7}, {5, 6}, {8}],
    # atom L=1: adj(1,2) [4]: loc{11}, loc{21}, adj(2,1){12,21}, adj(2,3){21}, adj(3,2){22},
    #   in(0,2){22}; adj(2,1) [5]: loc{11}, loc{21}, adj(1,2){12,21}, adj(2,3){11},
    #   adj(3,2){12}, in(0,2){12} -> the two directions differ already at L=1 (one hop
    #   sees in(0,2)'s argument 2 at position 2 vs position 1); 7 mirrors 4, 6 mirrors 5.
    ("R", "atom", 1): [{0}, {2}, {1, 3}, {4, 7}, {5, 6}, {8}],
    ("R", "atom", 2): [{0}, {2}, {1, 3}, {4, 7}, {5, 6}, {8}],
}

CASES = {"W": CASE_W, "C": CASE_C, "R": CASE_R}

# L at which the two buddies (rows 5 and 6) first get different colours, derived by hand
# above (None: never, checked up to L=5).
EXPECTED_BUDDY_SPLIT = {
    ("W", "oracle_sage"): None, ("W", "vilg"): 3, ("W", "atom"): 2,
    ("C", "oracle_sage"): 1, ("C", "vilg"): 1, ("C", "atom"): 1,
}

# Whole-graph histogram of a state vs its counterfactual with the buddies' request
# tables exchanged (one shared growing vocab, so equal ids mean equal signatures).
# True = identical histograms at that L.
#   W vs W_FLIP: the same graph with p and q relabelled -> identical at every L.
#   C vs C_FLIP, object: every request contributes only pairwise edges, and the exchange
#     leaves the edge set unchanged (p and q each still touch 1, 2, 3 and 4 with the same
#     labels; the four origin->destination pairs all exist either way) -> identical
#     graphs, identical histograms at every L.
#   vilg: L=1 sees only predicate types -> identical; at L=2 the request nodes see which
#     origin/destination they join (5-1-3, 5-2-4, 6-1-4, 6-2-3 vs 5-1-4, 5-2-3, 6-1-3,
#     6-2-4) -> different.
#   atom: at L=1 p's matching request atom (origin 1) sees in(0,3) through label {32}
#     only when its destination is the taxi's location 3 -> different already at L=1.
EXPECTED_SAME_HISTOGRAM_AS_FLIP = {
    ("W", "oracle_sage"): {1: True, 2: True, 3: True},
    ("W", "vilg"): {1: True, 2: True, 3: True},
    ("W", "atom"): {1: True, 2: True, 3: True},
    ("C", "oracle_sage"): {1: True, 2: True, 3: True},
    ("C", "vilg"): {1: True, 2: False, 3: False},
    ("C", "atom"): {1: False, 2: False, 3: False},
}
FLIPS = {"W": CASE_W_FLIP, "C": CASE_C_FLIP}


class TestHandBuiltCases(unittest.TestCase):
    def test_partitions_match_hand_derivation(self):
        for (case, convention, level), expected in sorted(EXPECTED.items()):
            with self.subTest(case=case, convention=convention, L=level):
                colours, _ = production_colours(CASES[case], convention, level)
                self.assertEqual(partition(colours), as_partition(expected))

    def test_buddy_split_level(self):
        for (case, convention), split_at in sorted(EXPECTED_BUDDY_SPLIT.items()):
            with self.subTest(case=case, convention=convention):
                first_split = None
                for level in range(1, 6):
                    colours, _ = production_colours(CASES[case], convention, level)
                    if colours[5] != colours[6]:
                        first_split = level
                        break
                self.assertEqual(first_split, split_at)

    def test_histogram_against_flipped_counterfactual(self):
        for (case, convention), per_level in sorted(EXPECTED_SAME_HISTOGRAM_AS_FLIP.items()):
            for level, same in per_level.items():
                with self.subTest(case=case, convention=convention, L=level):
                    vocab = {}
                    _, h_state = production_colours(CASES[case], convention, level, vocab)
                    _, h_flip = production_colours(FLIPS[case], convention, level, vocab)
                    self.assertEqual(same_histogram(h_state, h_flip), same)

    def test_object_crossed_flip_is_the_same_graph(self):
        """The object encoding of C and of C_FLIP are identical tensors -- the reason for
        the 100% object collision."""
        a = ternary_tensors(CASE_C, "oracle_sage")
        b = ternary_tensors(CASE_C_FLIP, "oracle_sage")
        for ta, tb in zip(a, b):
            self.assertTrue(th.equal(ta, tb))


# ==========================================================================================
# b. Independent reference WL (no sage imports below this line's helpers)
# ==========================================================================================

def _objects(facts):
    """Type atoms form the prefix; object k is row k (facts()'s contract)."""
    n_obj = 0
    while n_obj < len(facts) and facts[n_obj][0] in ("taxi", "location", "passenger"):
        n_obj += 1
    return n_obj


def reference_object_graph(facts):
    """Object encoding: one node per object, labelled by its type. Every relational fact
    connects each ordered pair of its arguments (a_i -> a_j, i != j) with the label
    (predicate, i, j); labels on the same directed pair are merged into a set.
    The production converter gives adjacent no positions (the 'road' bit, both ways);
    for symmetric road facts (adjacent(a,b) and adjacent(b,a) both present -- asserted
    here, and always true of the simulator's undirected maze) the two definitions
    induce the same labels on every road pair."""
    n_obj = _objects(facts)
    roads = {args for pred, args in facts[n_obj:] if pred == "adjacent"}
    assert all((b, a) in roads for a, b in roads), "reference assumes symmetric roads"
    labels = [facts[k][0] for k in range(n_obj)]
    edges = {}
    for pred, args in facts[n_obj:]:
        for i, a in enumerate(args, start=1):
            for j, b in enumerate(args, start=1):
                if i != j:
                    edges.setdefault((a, b), set()).add((pred, i, j))
    return labels, {pair: frozenset(lab) for pair, lab in edges.items()}


def reference_vilg_graph(facts):
    """Object-atom (vILG): one node per object (labelled by type) plus one node per
    relational fact (labelled by its predicate); each fact node is joined to its k-th
    argument in both directions, with the label k."""
    n_obj = _objects(facts)
    labels = [facts[k][0] for k in range(n_obj)]
    edges = {}
    for pred, args in facts[n_obj:]:
        node = len(labels)
        labels.append("fact:" + pred)
        for k, obj in enumerate(args, start=1):
            edges.setdefault((node, obj), set()).add(k)
            edges.setdefault((obj, node), set()).add(k)
    return labels, {pair: frozenset(lab) for pair, lab in edges.items()}


def reference_atom_graph(facts):
    """Atom encoding (Horcik et al., Def. 2): one node per fact, labelled by predicate;
    a -> b (a != b) labelled with every (i, j) such that argument i of a equals argument
    j of b."""
    labels = [pred for pred, _args in facts]
    by_object = {}
    for node, (_pred, args) in enumerate(facts):
        for i, obj in enumerate(args, start=1):
            by_object.setdefault(obj, []).append((node, i))
    edges = {}
    for occurrences in by_object.values():
        for a, i in occurrences:
            for b, j in occurrences:
                if a != b:
                    edges.setdefault((a, b), set()).add((i, j))
    return labels, {pair: frozenset(lab) for pair, lab in edges.items()}


REFERENCE_BUILDERS = {
    "oracle_sage": reference_object_graph,
    "vilg": reference_vilg_graph,
    "atom": reference_atom_graph,
}


def reference_wl_partitions(labels, edges, max_level):
    """Plain WL: colour_{t+1}(v) = (colour_t(v), sorted multiset of (colour_t(w),
    label(v->w)) over v's OUT-edges). Signatures are compressed to small ints per level
    with a fresh dict (only partitions are compared). Returns {L: partition} for
    L = 0 (the initial colouring: node labels alone) .. max_level."""
    out = [[] for _ in labels]
    for (a, b), lab in edges.items():
        out[a].append((b, tuple(sorted(lab))))
    colours = list(labels)
    result = {0: partition(colours)}
    for level in range(1, max_level + 1):
        signatures = [
            (colours[v], tuple(sorted((repr(colours[w]), lab) for w, lab in out[v])))
            for v in range(len(labels))
        ]
        ids = {}
        colours = [ids.setdefault(sig, len(ids)) for sig in signatures]
        result[level] = partition(colours)
    return result


def networkx_partitions(labels, edges, max_level):
    """weisfeiler_lehman_subgraph_hashes on a DiGraph (G.neighbors = successors). Every
    initial node label and every edge label is a fixed-width string, so networkx's
    unseparated string concatenation cannot make two different neighbourhoods equal."""
    node_codes = {lab: "N%03d" % k for k, lab in enumerate(sorted(set(map(repr, labels))))}
    edge_codes = {lab: "E%04d" % k for k, lab in enumerate(sorted({repr(tuple(sorted(l))) for l in edges.values()}))}
    g = nx.DiGraph()
    for v, lab in enumerate(labels):
        g.add_node(v, label=node_codes[repr(lab)])
    for (a, b), lab in edges.items():
        g.add_edge(a, b, label=edge_codes[repr(tuple(sorted(lab)))])
    hashes = nx.weisfeiler_lehman_subgraph_hashes(g, edge_attr="label", node_attr="label", iterations=max_level)
    return {level: partition([hashes[v][level - 1] for v in range(len(labels))]) for level in range(1, max_level + 1)}


def real_states():
    """Real simulator states: scripted play on full city mazes from several seeds
    (sampled at fixed steps, plus the first crossed state of each run), and two
    hand-specified buddy pairs via initial_pair / spawn_pair, sampled before and after
    a boarding. Returns a list of (name, facts)."""
    states = []
    for seed in (0, 1, 2):
        sim = make_ternary_sim(seed)
        rng = pyrandom.Random(seed)
        took_crossed = False
        for step in range(400):
            if step in (0, 150, 399) or (not took_crossed and crossed_state(sim)):
                took_crossed = took_crossed or crossed_state(sim)
                states.append(("seed%d_step%d%s" % (seed, step, "_crossed" if crossed_state(sim) else ""), sim.facts()))
            sim.act(scripted_policy(sim, rng))
            if sim.done:
                break
    pair = (((5, 50), (9, 77)), 5, ((5, 77), (9, 50)), 9)
    for seed in (3, 4):
        sim = make_ternary_sim(seed, spawning=False, initial_pair=pair)
        states.append(("initial_pair_seed%d" % seed, sim.facts()))
        rng = pyrandom.Random(seed)
        for _ in range(300):
            sim.act(scripted_policy(sim, rng, random_prob=0.0))
            if sim.taxi.passenger is not None:
                states.append(("initial_pair_seed%d_boarded" % seed, sim.facts()))
                break
    sim = make_ternary_sim(5, spawning=False)
    sim.spawn_pair(((20, 300), (40, 250)), 20, ((20, 250), (40, 300)), 40)
    states.append(("spawn_pair_seed5", sim.facts()))
    return states


class TestAgainstReferenceWL(unittest.TestCase):
    MAX_LEVEL = 3

    @classmethod
    def setUpClass(cls):
        cls.states = real_states()

    def test_states_cover_crossed_and_boarded_cases(self):
        names = [name for name, _facts in self.states]
        self.assertTrue(any("_crossed" in n for n in names), names)
        self.assertTrue(any("boarded" in n for n in names), names)
        self.assertTrue(any(n.startswith("spawn_pair") for n in names), names)

    def test_partitions_equal_reference(self):
        for name, facts in self.states:
            for convention in CONVENTIONS:
                labels, edges = REFERENCE_BUILDERS[convention](facts)
                expected = reference_wl_partitions(labels, edges, self.MAX_LEVEL)
                # L=0 checks the initial colouring on its own: a decoder that merges two
                # predicates can be invisible at L >= 1, where edge structure re-separates them
                for level in range(0, self.MAX_LEVEL + 1):
                    with self.subTest(state=name, convention=convention, L=level):
                        colours, _ = production_colours(facts, convention, level)
                        self.assertEqual(len(colours), len(labels))
                        self.assertEqual(partition(colours), expected[level])

    def test_partitions_equal_networkx(self):
        if not hasattr(nx, "weisfeiler_lehman_subgraph_hashes"):
            self.skipTest("networkx %s has no weisfeiler_lehman_subgraph_hashes (added in 2.7); "
                          "the plain reference WL test still runs" % nx.__version__)
        for name, facts in self.states[:4]:
            for convention in CONVENTIONS:
                labels, edges = REFERENCE_BUILDERS[convention](facts)
                expected = networkx_partitions(labels, edges, self.MAX_LEVEL)
                for level in range(1, self.MAX_LEVEL + 1):
                    with self.subTest(state=name, convention=convention, L=level):
                        colours, _ = production_colours(facts, convention, level)
                        self.assertEqual(partition(colours), expected[level])

    def test_comparison_is_not_vacuous(self):
        """Control: the partitions compared above are fine enough to catch a decoder that
        loses edge information. Degrading the reference by giving every edge the same label
        (argument positions / pair labels / direction all dropped) changes the L=1
        partition on a real state with requests, for every convention, while the intact
        reference still matches production."""
        with_requests = [(n, f) for n, f in self.states if any(p == "request" for p, _a in f)]
        self.assertTrue(with_requests)
        name, facts = with_requests[0]
        for convention in CONVENTIONS:
            with self.subTest(state=name, convention=convention):
                labels, edges = REFERENCE_BUILDERS[convention](facts)
                flat = {pair: frozenset({"edge"}) for pair in edges}
                colours, _ = production_colours(facts, convention, 1)
                self.assertEqual(partition(colours), reference_wl_partitions(labels, edges, 1)[1])
                self.assertNotEqual(partition(colours), reference_wl_partitions(labels, flat, 1)[1])


# ==========================================================================================
# c. Standard-domain regression against the ebbf25b fixture
# ==========================================================================================

def _load_fixture():
    with gzip.open(FIXTURE_PATH, "rt") as f:
        return json.load(f)


def _fixture_tensors(record):
    return (th.as_tensor(record["x"], dtype=th.float), th.as_tensor(record["edge_index"], dtype=th.long),
            th.as_tensor(record["edge_attr"], dtype=th.float))


class TestStandardDomainUnchanged(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = _load_fixture()

    def test_fixture_was_recorded_at_ebbf25b(self):
        self.assertTrue(self.fixture["generated_at_commit"].startswith("ebbf25b"))

    def test_growing_mode_matches_fixture(self):
        for convention in CONVENTIONS:
            data = self.fixture["conventions"][convention]
            for level, expected in sorted(data["growing"].items()):
                with self.subTest(convention=convention, L=level):
                    vocab = {}
                    for k, record in enumerate(data["graphs"]):
                        colours, histogram = wl_colours(*_fixture_tensors(record), num_iterations=int(level),
                                                        vocab=vocab, frozen=False, graph_convention=convention)
                        self.assertEqual(colours.tolist(), expected["colours"][k])
                        self.assertEqual([int(v) for v in histogram.tolist()], expected["histograms"][k])
                    self.assertEqual([repr(s) for s in vocab], expected["vocab_order"])

    def test_frozen_production_vocab_matches_fixture(self):
        for convention in CONVENTIONS:
            data = self.fixture["conventions"][convention]
            frozen = data["frozen"]
            path = wl_vocab_cache.WL_VOCAB_PATH.parent / frozen["vocab_file"]
            vocab = wl_vocab_cache._load_vocab_from_path(str(path))
            with self.subTest(convention=convention):
                for k, record in enumerate(data["graphs"]):
                    colours, histogram = wl_colours(*_fixture_tensors(record), num_iterations=frozen["L"],
                                                    vocab=vocab, frozen=True, graph_convention=convention)
                    self.assertEqual(colours.tolist(), frozen["colours"][k])
                    self.assertEqual([int(v) for v in histogram.tolist()], frozen["histograms"][k])

    def test_explicit_standard_domain_is_the_default(self):
        record = self.fixture["conventions"]["vilg"]["graphs"][0]
        a = wl_colours(*_fixture_tensors(record), num_iterations=2, vocab={}, graph_convention="vilg")
        b = wl_colours(*_fixture_tensors(record), num_iterations=2, vocab={}, graph_convention="vilg",
                       domain="standard")
        self.assertTrue(th.equal(a[0], b[0]) and th.equal(a[1], b[1]))


# ==========================================================================================
# d. Width and value guards
# ==========================================================================================

class TestGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = _load_fixture()

    def test_ternary_graph_with_standard_decoder_raises(self):
        for convention in CONVENTIONS:
            with self.subTest(convention=convention):
                with self.assertRaises(ValueError):
                    wl_colours(*ternary_tensors(CASE_C, convention), num_iterations=1, vocab={},
                               graph_convention=convention, domain="standard")

    def test_standard_graph_with_ternary_decoder_raises(self):
        for convention in CONVENTIONS:
            record = self.fixture["conventions"][convention]["graphs"][0]
            with self.subTest(convention=convention):
                with self.assertRaises(ValueError):
                    wl_colours(*_fixture_tensors(record), num_iterations=1, vocab={},
                               graph_convention=convention, domain="ternary")

    def test_width_table_matches_the_envs_own_dimensions(self):
        """wl_colours' FEATURE_WIDTHS must be the (node, edge) dims GraphTaxiEnv registers for
        each domain -- taxi_env.GRAPH_CONVENTIONS and TERNARY_GRAPH_CONVENTIONS."""
        from sage.domains.gym_taxi.envs.taxi_env import GRAPH_CONVENTIONS
        from sage.domains.gym_taxi.utils.ternary_representations import TERNARY_GRAPH_CONVENTIONS
        from sage.domains.utils.wl_colours import FEATURE_WIDTHS
        self.assertEqual(FEATURE_WIDTHS["standard"], dict(GRAPH_CONVENTIONS))
        self.assertEqual(FEATURE_WIDTHS["ternary"], dict(TERNARY_GRAPH_CONVENTIONS))

    def test_unknown_domain_raises(self):
        with self.assertRaises(ValueError):
            wl_colours(*ternary_tensors(CASE_C, "atom"), num_iterations=1, vocab={},
                       graph_convention="atom", domain="binary")

    def test_invalid_values_raise(self):
        x, edge_index, edge_attr = ternary_tensors(CASE_C, "oracle_sage")
        bad = edge_attr.clone()
        bad[0, 9] = 0.0  # direction must be +1 or -1
        with self.assertRaises(ValueError):
            wl_colours(x, edge_index, bad, num_iterations=1, vocab={}, graph_convention="oracle_sage", domain="ternary")
        bad = edge_attr.clone()
        bad[0, 0:9] = 0.0  # an object edge always carries at least one label bit
        with self.assertRaises(ValueError):
            wl_colours(x, edge_index, bad, num_iterations=1, vocab={}, graph_convention="oracle_sage", domain="ternary")

        x, edge_index, edge_attr = ternary_tensors(CASE_C, "vilg")
        bad = x.clone()
        bad[10, 3] = 1.0  # a request proposition row with a second predicate bit
        with self.assertRaises(ValueError):
            wl_colours(bad, edge_index, edge_attr, num_iterations=1, vocab={}, graph_convention="vilg", domain="ternary")
        bad = x.clone()
        bad[1, 7] = 1.0  # an object row with a status bit
        with self.assertRaises(ValueError):
            wl_colours(bad, edge_index, edge_attr, num_iterations=1, vocab={}, graph_convention="vilg", domain="ternary")
        bad = edge_attr.clone()
        bad[0, 1] = 1.0  # two argument positions on one edge
        with self.assertRaises(ValueError):
            wl_colours(x, edge_index, bad, num_iterations=1, vocab={}, graph_convention="vilg", domain="ternary")

        x, edge_index, edge_attr = ternary_tensors(CASE_C, "atom")
        bad = x.clone()
        bad[0, 6] = 1.0  # two predicates on one atom
        with self.assertRaises(ValueError):
            wl_colours(bad, edge_index, edge_attr, num_iterations=1, vocab={}, graph_convention="atom", domain="ternary")
        bad = edge_attr.clone()
        bad[0, :] = 0.0  # an atom edge with no label
        with self.assertRaises(ValueError):
            wl_colours(x, edge_index, bad, num_iterations=1, vocab={}, graph_convention="atom", domain="ternary")


# ==========================================================================================
# e. CPU vs CUDA
# ==========================================================================================

@unittest.skipUnless(th.cuda.is_available(), "requires CUDA - run on RCP to exercise this")
class TestCudaMatchesCpu(unittest.TestCase):
    def test_identical_outputs(self):
        sim = make_ternary_sim(0)
        rng = pyrandom.Random(0)
        for _ in range(100):
            sim.act(scripted_policy(sim, rng))
        for facts in (CASE_C, sim.facts()):
            for convention in CONVENTIONS:
                for level in (1, 2):
                    with self.subTest(convention=convention, L=level):
                        x, edge_index, edge_attr = ternary_tensors(facts, convention)
                        cpu = wl_colours(x, edge_index, edge_attr, num_iterations=level, vocab={},
                                         graph_convention=convention, domain="ternary")
                        gpu = wl_colours(x.cuda(), edge_index.cuda(), edge_attr.cuda(), num_iterations=level,
                                         vocab={}, graph_convention=convention, domain="ternary")
                        self.assertEqual(gpu[0].device.type, "cuda")
                        self.assertTrue(th.equal(cpu[0], gpu[0].cpu()))
                        self.assertTrue(th.equal(cpu[1], gpu[1].cpu()))


if __name__ == "__main__":
    unittest.main()
