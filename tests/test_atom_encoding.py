"""
Tests for the atom-encoding graph convention (Horcik et al., AAAI-25, Def. 2):
env_to_atoms / atoms_to_graph / graph_to_atoms / env_to_atom_graph
(sage/domains/gym_taxi/utils/representations.py).

This branch (cell5-atom-gnn) is based on cell2-vilg-gnn, NOT cell4-wl-vilg: there is no
WL wiring here, vILG's edges are one-directional (no mirrored pair), and TaxiWorldSimulator
does not have the stale-edge fix (attempt_move/attempt_pickup only remove the forward
tether edge, leaving a stale reverse edge behind in env.graph). The atom encoding is
unaffected by the missing stale-edge fix by construction: env_to_atoms only ever reads
attr[-1] == 1 (forward) edges, exactly like env_to_vilg_graph, so a stale reverse edge
(always attr[-1] == -1) can never produce a spurious atom -- this is checked explicitly
below (TestGroundTruth.test_after_several_moves_stale_reverse_edges_produce_no_atoms).

Run from the repo root with:
    python -m pytest tests/test_atom_encoding.py -v
"""
import random
import time
import unittest

import numpy as np

from sage.domains.gym_taxi.simulator.taxi_world import TaxiWorldSimulator, Passenger
from sage.domains.gym_taxi.utils.config import CITY
from sage.domains.gym_taxi.utils.representations import (
    ATOM_PREDICATES,
    ATOM_LABELS,
    env_to_atoms,
    atoms_to_graph,
    graph_to_atoms,
    env_to_atom_graph,
)
from sage.domains.utils.representations import graph_to_json

CITY_SEEDS = [0, 300, 600, 900, 1200]

_ATTR_TO_PREDICATE = {(1, 0, 0): "location", (0, 1, 0): "taxi", (0, 0, 1): "passenger"}


def make_sim(seed, **scenario):
    """The DEFAULT (non-"vilg") simulator, per this task's spec: dropoff deletes the
    passenger node (TaxiWorldSimulator.attempt_dropoff's else branch)."""
    return TaxiWorldSimulator(np.random.RandomState(seed), planning=True, graph_convention="oracle_sage", **scenario)


def make_grid_sim(seed):
    """A tiny, fully hand-checkable 2x2 no-walls grid: 4 locations + 1 taxi + 1 passenger."""
    return make_sim(seed, size=2, random_walls=False)


def make_city_sim(seed):
    return make_sim(seed, **CITY)


def relocate_passenger_to(sim, pid, new_location):
    """Moves pid's location tether to new_location, mirroring add_passenger's own edge
    construction -- lets a test force a deterministic pickup."""
    old_location = sim.passengers[pid].location
    destination = sim.passengers[pid].destination
    sim.graph.remove_edge(pid, old_location)
    sim.graph.remove_edge(old_location, pid)
    sim.graph.add_edge(pid, new_location, attr=[0, 1, 0, 1])
    sim.graph.add_edge(new_location, pid, attr=[0, 1, 0, -1])
    sim.passengers[pid] = Passenger(new_location, destination)


def relocate_destination_to(sim, pid, new_destination):
    """Same idea, for the destination(pid, loc) tether -- lets a test force a
    deterministic (successful) dropoff."""
    old_destination = sim.passengers[pid].destination
    location = sim.passengers[pid].location
    sim.graph.remove_edge(pid, old_destination)
    sim.graph.remove_edge(old_destination, pid)
    sim.graph.add_edge(pid, new_destination, attr=[0, 0, 1, 1])
    sim.graph.add_edge(new_destination, pid, attr=[0, 0, 1, -1])
    sim.passengers[pid] = Passenger(location, new_destination)


def move_taxi_randomly(sim, rng, n_moves):
    """Calls attempt_move directly (bypassing .act()/.step(), so no passenger spawning
    or RNG consumption from sim.random) n_moves times, along real road edges. This branch
    lacks the stale-edge fix, so each move leaves a stale reverse tether edge behind in
    env.graph -- exactly the condition TestGroundTruth's post-move check targets."""
    for _ in range(n_moves):
        start = sim.taxi.location
        candidates = [
            c for c in sim.graph.successors(start)
            if sim.graph.edges[(start, c)]["attr"] == [1, 0, 0, 1]
        ]
        if not candidates:
            break
        target = rng.choice(candidates)
        sim.attempt_move(target)


def expected_in_destination_atoms(sim):
    """Ground truth for in/destination atoms, read directly off env.taxi/env.passengers
    (NOT off env.graph or env_to_atoms) -- ints throughout, so comparisons against
    env_to_atoms' output (which may carry np.int64 args drawn from np.random.RandomState.choice)
    still work since int == np.int64(int) is True and set membership uses ==/hash, which
    agree for equal numeric values."""
    expected = {("in", (0, int(sim.taxi.location)))}
    for pid, passenger in sim.passengers.items():
        expected.add(("in", (int(pid), int(passenger.location))))
        expected.add(("destination", (int(pid), int(passenger.destination))))
    return expected


def actual_in_destination_atoms(sim):
    atoms = env_to_atoms(sim)
    return {
        (pred, tuple(int(a) for a in args))
        for pred, args in atoms
        if pred in ("in", "destination")
    }


def brute_force_edges(atoms):
    """
    O(n^2) reference for atoms_to_graph, built directly from `atoms` (trusted -- see
    TestGroundTruth for env_to_atoms' own correctness) and INDEPENDENT of the
    scipy.sparse incidence-matmul code atoms_to_graph actually uses: dense pairwise
    argument-equality comparison via numpy broadcasting, not sparse matrices.

    :return: (edge_labels, instance_count)
        edge_labels: {(a, b): [bit11, bit12, bit21, bit22]}, deduplicated
        instance_count: total (a, b, i, j) matches found, BEFORE merging distinct labels
            for the same (a, b) into one row (see TestPreDedupInstanceCount)
    """
    n = len(atoms)
    arg1 = np.full(n, -1, dtype=np.int64)
    arg2 = np.full(n, -1, dtype=np.int64)
    for idx, (_pred, args) in enumerate(atoms):
        if len(args) >= 1:
            arg1[idx] = args[0]
        if len(args) >= 2:
            arg2[idx] = args[1]
    arg = {1: arg1, 2: arg2}

    not_self = ~np.eye(n, dtype=bool)
    label_matrices = []
    instance_count = 0
    for (i, j) in ATOM_LABELS:
        ai, aj = arg[i], arg[j]
        eq = (ai[:, None] == aj[None, :]) & (ai[:, None] != -1) & (aj[None, :] != -1) & not_self
        label_matrices.append(eq)
        instance_count += int(eq.sum())

    any_label = label_matrices[0] | label_matrices[1] | label_matrices[2] | label_matrices[3]
    rows, cols = np.nonzero(any_label)
    edge_labels = {
        (int(a), int(b)): [bool(m[a, b]) for m in label_matrices]
        for a, b in zip(rows.tolist(), cols.tolist())
    }
    return edge_labels, instance_count


def production_edge_labels(edge_index, edge_attr):
    labels = {}
    for k in range(edge_index.shape[1]):
        a, b = int(edge_index[0, k]), int(edge_index[1, k])
        labels[(a, b)] = [bool(v) for v in edge_attr[k]]
    return labels


class TestHandCheckedGrid(unittest.TestCase):
    """
    2x2 no-walls grid, seed=0: 4 locations + 1 taxi + 1 passenger = 6 objects, hence 6
    type atoms (rows 0-5, one per object, row k = object k):
        row 0: taxi(0)
        row 1: location(1)   row 2: location(2)   row 3: location(3)   row 4: location(4)
        row 5: passenger(5)
    Concretely, deterministic under RandomState(0): taxi at location 1; passenger at
    location 3, destination location 1. env_to_atoms then appends one atom per
    forward-direction (attr[-1]==1) env.graph edge, sorted by (u, v):
        row  6: in(0, 1)            -- taxi tethered to location 1
        row  7: adjacent(1, 2)      row  8: adjacent(1, 3)
        row  9: adjacent(2, 1)      row 10: adjacent(2, 4)
        row 11: adjacent(3, 1)      row 12: adjacent(3, 4)
        row 13: adjacent(4, 2)      row 14: adjacent(4, 3)
        row 15: destination(5, 1)   -- passenger's destination is location 1
        row 16: in(5, 3)            -- passenger currently at location 3
    17 atoms total (verified against the running code, not just derived by hand).

    Per-argument-value incidence (argument value : [(atom row, position), ...], now
    including EVERY atom's own arguments -- type atoms contribute (self, position 1)
    just like any arity-1 atom, per "type atoms are ordinary atoms, not a special case"):
        value 0 (taxi):     atoms referencing 0: row0@pos1 (itself), row6@pos1 (in's subject)
                             -> 2 entries -> 2*1  =  2 ordered pairs
        value 1 (loc1):     row1@pos1 (itself), row6@pos2, row7@pos1, row8@pos1, row9@pos2,
                             row11@pos2, row15@pos2
                             -> 7 entries -> 7*6  = 42 ordered pairs
        value 2 (loc2):     row2@pos1 (itself), row7@pos2, row9@pos1, row10@pos1, row13@pos2
                             -> 5 entries -> 5*4  = 20 ordered pairs
        value 3 (loc3):     row3@pos1 (itself), row8@pos2, row11@pos1, row12@pos1, row14@pos2,
                             row16@pos2
                             -> 6 entries -> 6*5  = 30 ordered pairs
        value 4 (loc4):     row4@pos1 (itself), row10@pos2, row12@pos2, row13@pos1, row14@pos1
                             -> 5 entries -> 5*4  = 20 ordered pairs
        value 5 (passenger):row5@pos1 (itself), row15@pos1, row16@pos1
                             -> 3 entries -> 3*2  =  6 ordered pairs
    Total pre-dedup (a, b, i, j) instances = 2+42+20+30+20+6 = 120. A handful of ordered
    pairs pick up a SECOND label from a different shared argument value (e.g. two
    locations that are both road-adjacent to each other AND both referenced by the same
    in/destination atom), collapsing 120 instances into 112 deduplicated directed edges.
    (These are the exact same 17/120/112 figures as the earlier, now-discarded
    cell4-based attempt -- unsurprising, since cell4's mirrored vILG edges and 9-dim
    features never entered this scheme; only the row order differs, since env_to_atoms
    sorts edges by (u, v) instead of raw graph-iteration order.)
    """

    def setUp(self):
        self.sim = make_grid_sim(0)
        self.atoms = env_to_atoms(self.sim)

    def test_atom_list_matches_hand_derivation(self):
        expected = [
            ("taxi", (0,)), ("location", (1,)), ("location", (2,)), ("location", (3,)),
            ("location", (4,)), ("passenger", (5,)),
            ("in", (0, 1)),
            ("adjacent", (1, 2)), ("adjacent", (1, 3)), ("adjacent", (2, 1)), ("adjacent", (2, 4)),
            ("adjacent", (3, 1)), ("adjacent", (3, 4)), ("adjacent", (4, 2)), ("adjacent", (4, 3)),
            ("destination", (5, 1)), ("in", (5, 3)),
        ]
        actual = [(pred, tuple(int(a) for a in args)) for pred, args in self.atoms]
        self.assertEqual(actual, expected)

    def test_edge_counts_match_hand_derivation(self):
        _nf, ef, ei = atoms_to_graph(self.atoms)
        self.assertEqual(ei.shape[1], 112)
        self.assertEqual(int(ef.sum()), 120)


class TestBruteForceEdgesMatchProduction(unittest.TestCase):
    """Brute-force (dense, independent of scipy.sparse) vs atoms_to_graph, exact match,
    across the 2x2 hand-checked case and all 5 required city seeds."""

    def test_edges_and_labels_match_exactly(self):
        cases = [("grid_seed0", make_grid_sim(0))] + [(f"city_seed{s}", make_city_sim(s)) for s in CITY_SEEDS]
        for name, sim in cases:
            with self.subTest(case=name):
                atoms = env_to_atoms(sim)
                expected_labels, _instance_count = brute_force_edges(atoms)

                _nf, ef, ei = atoms_to_graph(atoms)
                actual_labels = production_edge_labels(ei, ef)

                self.assertEqual(actual_labels, expected_labels)


class TestPreDedupInstanceCount(unittest.TestCase):
    """The pre-dedup (i, j) instance count (total (a, b, i, j) matches, before merging
    distinct labels for the same (a, b) into one multi-hot row) must equal the brute-force
    count exactly -- this is what atoms_to_graph's edge_feats.sum() reports, since no two
    atoms can share the same argument pair via two different values (each atom has each
    argument slot exactly once), so every match contributes exactly one bit."""

    def test_instance_count_matches_brute_force(self):
        cases = [("grid_seed0", make_grid_sim(0))] + [(f"city_seed{s}", make_city_sim(s)) for s in CITY_SEEDS]
        for name, sim in cases:
            with self.subTest(case=name):
                atoms = env_to_atoms(sim)
                _expected_labels, brute_instance_count = brute_force_edges(atoms)

                _nf, ef, _ei = atoms_to_graph(atoms)
                production_instance_count = int(ef.sum())

                self.assertEqual(production_instance_count, brute_instance_count)


class TestSymmetryAndNoSelfLoops(unittest.TestCase):
    def test_every_edge_has_a_reverse_with_transposed_label_and_no_self_loops(self):
        cases = [("grid_seed0", make_grid_sim(0))] + [(f"city_seed{s}", make_city_sim(s)) for s in CITY_SEEDS]
        transpose = [0, 2, 1, 3]  # (1,1)<->(1,1), (1,2)<->(2,1), (2,1)<->(1,2), (2,2)<->(2,2)
        for name, sim in cases:
            with self.subTest(case=name):
                atoms = env_to_atoms(sim)
                _nf, ef, ei = atoms_to_graph(atoms)
                labels = production_edge_labels(ei, ef)

                for (a, b) in labels:
                    self.assertNotEqual(a, b, f"self-loop found at {a}")

                for (a, b), bits in labels.items():
                    self.assertIn((b, a), labels, f"{(a, b)} has no reverse edge")
                    self.assertEqual(labels[(b, a)], [bits[k] for k in transpose])


class TestTypeAtomCountAndOrder(unittest.TestCase):
    """Number of type atoms == number of objects; rows 0..n_obj-1 are type atoms matching
    the env node attrs exactly."""

    def test_type_atoms_match_object_nodes(self):
        cases = [("grid_seed0", make_grid_sim(0))] + [(f"city_seed{s}", make_city_sim(s)) for s in CITY_SEEDS]
        for name, sim in cases:
            with self.subTest(case=name):
                n_obj = len(sim.graph.nodes)
                atoms = env_to_atoms(sim)

                type_atoms = [a for a in atoms if a[0] in ("location", "taxi", "passenger")]
                self.assertEqual(len(type_atoms), n_obj)

                for k in range(n_obj):
                    predicate, args = atoms[k]
                    self.assertEqual(args, (k,))
                    expected_predicate = _ATTR_TO_PREDICATE[tuple(sim.graph.nodes[k]["attr"])]
                    self.assertEqual(predicate, expected_predicate)

                for k in range(n_obj, len(atoms)):
                    self.assertNotIn(atoms[k][0], ("location", "taxi", "passenger"))


class TestGroundTruth(unittest.TestCase):
    """in/destination atoms must equal what env.taxi/env.passengers imply, checked
    directly against env_to_atoms' output (not against atoms_to_graph or any derived
    structure) across reset / pickup / several moves / dropoff, using the DEFAULT
    (non-"vilg") simulator."""

    def test_after_reset(self):
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed):
                sim = make_city_sim(seed)
                self.assertEqual(actual_in_destination_atoms(sim), expected_in_destination_atoms(sim))

    def test_after_pickup(self):
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed):
                sim = make_city_sim(seed)
                pid = next(iter(sim.passengers))
                relocate_passenger_to(sim, pid, sim.taxi.location)
                sim.attempt_pickup(pid)
                self.assertEqual(sim.taxi.passenger, pid)
                self.assertEqual(actual_in_destination_atoms(sim), expected_in_destination_atoms(sim))

    def test_after_several_moves_stale_reverse_edges_produce_no_atoms(self):
        """This branch (cell2-vilg-gnn base) lacks the stale-edge fix: attempt_move only
        removes the FORWARD tether edge (0 -> old_location), leaving a stale REVERSE edge
        (old_location -> 0) behind in env.graph permanently. env_to_atoms filters to
        attr[-1] == 1 only (same as env_to_vilg_graph), and the surviving stale edge is
        always the attr[-1] == -1 one, so it must never surface as a spurious atom here --
        this is the concrete regression this test guards."""
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed):
                sim = make_city_sim(seed)
                rng = random.Random(seed)
                move_taxi_randomly(sim, rng, n_moves=8)

                # confirm the stale-edge bug actually fired (test is meaningful, not vacuous)
                stale_found = any(
                    sim.graph.has_edge(loc, 0) and not sim.graph.has_edge(0, loc)
                    for loc in range(1, len(sim.graph.nodes))
                    if loc != sim.taxi.location
                )
                self.assertTrue(stale_found, "expected at least one stale reverse tether edge after 8 moves")

                self.assertEqual(actual_in_destination_atoms(sim), expected_in_destination_atoms(sim))

    def test_after_dropoff_default_simulator_deletes_passenger(self):
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed):
                sim = make_city_sim(seed)
                pid = next(iter(sim.passengers))
                relocate_passenger_to(sim, pid, sim.taxi.location)
                sim.attempt_pickup(pid)
                relocate_destination_to(sim, pid, sim.taxi.location)
                sim.attempt_dropoff(0)

                self.assertIsNone(sim.taxi.passenger)
                self.assertNotIn(pid, sim.passengers)
                self.assertNotIn(pid, dict(sim.graph.nodes))  # default (non-vilg): node is deleted

                self.assertEqual(actual_in_destination_atoms(sim), expected_in_destination_atoms(sim))


class TestRoundTrip(unittest.TestCase):
    """graph_to_atoms(atoms_to_graph(atoms)) == atoms, on the same states as TestGroundTruth."""

    def _assert_round_trip(self, sim):
        atoms = env_to_atoms(sim)
        nf, ef, ei = atoms_to_graph(atoms)
        atoms2 = graph_to_atoms(nf, ei, ef)
        self.assertEqual(atoms2, atoms)

    def test_after_reset(self):
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed):
                self._assert_round_trip(make_city_sim(seed))

    def test_after_pickup(self):
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed):
                sim = make_city_sim(seed)
                pid = next(iter(sim.passengers))
                relocate_passenger_to(sim, pid, sim.taxi.location)
                sim.attempt_pickup(pid)
                self._assert_round_trip(sim)

    def test_after_several_moves(self):
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed):
                sim = make_city_sim(seed)
                rng = random.Random(seed)
                move_taxi_randomly(sim, rng, n_moves=8)
                self._assert_round_trip(sim)

    def test_after_dropoff(self):
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed):
                sim = make_city_sim(seed)
                pid = next(iter(sim.passengers))
                relocate_passenger_to(sim, pid, sim.taxi.location)
                sim.attempt_pickup(pid)
                relocate_destination_to(sim, pid, sim.taxi.location)
                sim.attempt_dropoff(0)
                self._assert_round_trip(sim)


class TestEnvToAtomGraph(unittest.TestCase):
    def test_mask_true_exactly_on_type_atom_rows(self):
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed):
                sim = make_city_sim(seed)
                n_obj = len(sim.graph.nodes)
                nf, ef, ei, mask, gf = env_to_atom_graph(sim)

                self.assertTrue(mask[:n_obj].all())
                self.assertFalse(mask[n_obj:].any())
                self.assertEqual(int(mask.sum()), n_obj)

    def test_matches_atoms_to_graph_and_global_feats(self):
        sim = make_city_sim(0)
        atoms = env_to_atoms(sim)
        nf_expected, ef_expected, ei_expected = atoms_to_graph(atoms)

        nf, ef, ei, mask, gf = env_to_atom_graph(sim)
        np.testing.assert_array_equal(nf, nf_expected)
        np.testing.assert_array_equal(ef, ef_expected)
        np.testing.assert_array_equal(ei, ei_expected)

        expected_time_left = (sim.timeout - sim.time) / sim.timeout
        self.assertAlmostEqual(gf[0], expected_time_left)
        self.assertTrue((gf[1:] == 0).all())

    def test_planning_false_raises_not_implemented(self):
        sim = make_city_sim(0)
        sim.planning = False
        with self.assertRaises(NotImplementedError):
            env_to_atom_graph(sim)


class TestMeasurements(unittest.TestCase):
    """Not correctness tests -- prints per-seed measurements over a 300-step episode
    under DEFAULT (delete-on-delivery) semantics, for the report: max nodes, max directed
    edges, ms/call, and max JSON length vs JsonGraph's U250000 limit."""

    def test_measurements_over_episode(self):
        results = []
        for seed in CITY_SEEDS:
            np.random.seed(seed)
            sim = make_city_sim(seed)

            nf, ef, ei, mask, gf = env_to_atom_graph(sim)
            t0 = time.perf_counter()
            for _ in range(20):
                env_to_atom_graph(sim)
            per_call_ms = (time.perf_counter() - t0) / 20 * 1000

            max_nodes = 0
            max_edges = 0
            max_json_len = 0
            for _ in range(300):
                nf, ef, ei, mask, gf = env_to_atom_graph(sim)
                max_nodes = max(max_nodes, nf.shape[0])
                max_edges = max(max_edges, ei.shape[1])
                js = graph_to_json(nf, ef, ei, mask, gf)
                max_json_len = max(max_json_len, len(js))

                action = _sample_action(sim)
                sim.act(action)

            results.append((seed, max_nodes, max_edges, per_call_ms, max_json_len))

        print("\nseed | max_nodes | max_edges | ms/call | max_json_len (limit=250000)")
        for seed, max_nodes, max_edges, per_call_ms, max_json_len in results:
            print(f"{seed:5d} | {max_nodes:9d} | {max_edges:9d} | {per_call_ms:7.3f} | {max_json_len}")


def _sample_action(sim):
    taxi_location = sim.taxi.location
    candidates = {
        c for c in sim.graph.successors(taxi_location)
        if sim.graph.nodes[c]["attr"] != [0, 0, 1] or c in sim.passengers
    }
    candidates.add(taxi_location)
    candidates.add(0)
    return int(np.random.choice(list(candidates)))


if __name__ == "__main__":
    unittest.main()
