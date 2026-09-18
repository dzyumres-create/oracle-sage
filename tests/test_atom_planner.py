"""
Tests for Step 3: Planner.plan's "atom" branch (plan_atom, graph_to_state_atom,
_move_taxi_atoms, _deliver_atoms, _atoms_to_projection -- planner.py).

Design under test: decode -> plan -> re-encode. plan_atom never edits any tensor in
place; it decodes the input graph to a flat atom list (graph_to_atoms), computes the new
logical state as a plain Python transformation of that list, and re-encodes from scratch
via atoms_to_graph -- the same function env_to_atom_graph uses for a live env, so a
projected graph and a live env's graph are built by identical code.

Base branch: cell2-vilg-gnn -- no stale-edge fix, no optimizer-timing fix, no WL wiring.

Run from the repo root with:
    python -m pytest tests/test_atom_planner.py -v
"""
import copy
import unittest

import numpy as np
import torch as th
from torch_geometric.data import Data

from sage.domains.gym_taxi.simulator.taxi_world import TaxiWorldSimulator, Passenger
from sage.domains.gym_taxi.utils.config import CITY
from sage.domains.gym_taxi.utils.representations import env_to_atoms, env_to_atom_graph, graph_to_atoms, env_to_graph
from sage.domains.gym_taxi.simulator.planner import Planner, graph_to_state_atom

CITY_SEEDS = [0, 300, 600, 900, 1200]

_TYPE_PREDICATES = {"location", "taxi", "passenger"}


def make_sim(seed, **scenario):
    """passenger_creation_probability=0, per the corrected spec: otherwise a passenger
    can spawn mid-execution and the executed state can never equal the projection.
    Overrides (not adds to) any probability already in `scenario` (e.g. CITY's own 0.1)."""
    scenario = dict(scenario, passenger_creation_probability=0)
    return TaxiWorldSimulator(
        np.random.RandomState(seed), planning=True, graph_convention="atom", **scenario,
    )


def make_city_sim(seed):
    return make_sim(seed, **CITY)


def make_grid_sim(seed):
    return make_sim(seed, size=2, random_walls=False)


def to_data(node_feats, edge_feats, edge_index, mask, global_feats):
    g = Data(
        x=th.as_tensor(node_feats, dtype=th.float32),
        edge_index=th.as_tensor(edge_index, dtype=th.long),
        edge_attr=th.as_tensor(edge_feats, dtype=th.float32),
    )
    g.mask = th.as_tensor(mask, dtype=th.bool)
    g.global_features = th.as_tensor(global_feats, dtype=th.float32).unsqueeze(0)
    return g


def make_atom_graph(sim):
    return to_data(*env_to_atom_graph(sim))


def make_os_graph(sim):
    return to_data(*env_to_graph(sim))


def clone_data(g):
    g2 = Data(x=g.x.clone(), edge_index=g.edge_index.clone(), edge_attr=g.edge_attr.clone())
    g2.mask = g.mask.clone()
    g2.global_features = g.global_features.clone()
    return g2


def data_equal(a, b):
    return (
        th.equal(a.x, b.x) and th.equal(a.edge_index, b.edge_index)
        and th.equal(a.edge_attr, b.edge_attr) and th.equal(a.mask, b.mask)
        and th.equal(a.global_features, b.global_features)
    )


def edges_by_identity(x, edge_index, edge_attr):
    """
    Re-expresses a graph's edges in terms of stable ATOM IDENTITY (predicate, args --
    object ids, not row positions) instead of raw row indices, and returns (atoms,
    edge_set). Since every atom's (predicate, args) uniquely identifies it (no two
    distinct atoms in a valid Taxi state share a (predicate, args) pair -- e.g. two
    different `adjacent` atoms always differ in their (u, v) argument pair), this is a
    genuinely row-order-independent fingerprint: two graphs describe the identical
    x/edge_index/edge_attr content "up to row order" iff their edges_by_identity()
    outputs match exactly.
    """
    x = np.asarray(x)
    edge_index = np.asarray(edge_index)
    edge_attr = np.asarray(edge_attr)
    atoms = graph_to_atoms(x, edge_index, edge_attr)
    edges = set()
    for k in range(edge_index.shape[1]):
        a, b = int(edge_index[0, k]), int(edge_index[1, k])
        edges.add((atoms[a], atoms[b], tuple(round(float(v)) for v in edge_attr[k])))
    return atoms, edges


def n_obj_of(atoms):
    return sum(1 for predicate, _args in atoms if predicate in _TYPE_PREDICATES)


def execute(sim_copy, actions):
    """Runs the real env forward through `actions` via sim.act() (bypassing GraphTaxiEnv,
    matching how the planner's caller would actually apply a plan) -- the "executed"
    ground truth to compare a projection against."""
    for a in actions:
        sim_copy.act(int(a))
    return sim_copy


class TestRoadGraphFromAdjacentAtoms(unittest.TestCase):
    """graph_to_state_atom's state.graph must be an undirected nx.Graph built from
    `adjacent` atoms, matching graph_to_networkx's own nx.Graph(...) semantics."""

    def test_road_graph_is_undirected_and_matches_adjacent_atoms(self):
        import networkx as nx
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed):
                sim = make_city_sim(seed)
                graph = make_atom_graph(sim)
                state, atoms = graph_to_state_atom(graph)

                self.assertIsInstance(state.graph, nx.Graph)
                self.assertNotIsInstance(state.graph, nx.DiGraph)

                expected_edges = {
                    frozenset(args) for predicate, args in atoms if predicate == "adjacent"
                }
                actual_edges = {frozenset(e) for e in state.graph.edges}
                self.assertEqual(actual_edges, expected_edges)


class TestInputGraphNeverMutated(unittest.TestCase):
    """plan_atom must return a fresh Data; the input graph must be byte-identical before
    and after every call, across all three goal-type branches."""

    def test_input_unchanged_across_goal_types(self):
        planner = Planner(graph_convention="atom")
        for seed in CITY_SEEDS:
            sim = make_city_sim(seed)
            pid = next(iter(sim.passengers))
            for goal_name, goal in [("passenger", pid), ("location", 7), ("taxi_node", 0)]:
                with self.subTest(seed=seed, goal=goal_name):
                    graph = make_atom_graph(sim)
                    before = clone_data(graph)
                    projection, actions = planner.plan(graph, goal)
                    self.assertTrue(data_equal(graph, before), "input graph was mutated by plan_atom")
                    self.assertIsNot(projection, graph, "projection must be a fresh Data, not the input")


class TestGlobalFeaturesMatchIncrementTimer(unittest.TestCase):
    """projection.global_features[0,0] == input.global_features[0,0] - len(actions)/2000,
    matching increment_timer exactly. NOT compared against the executed env's own clock."""

    def test_time_decrement_matches_action_count(self):
        planner = Planner(graph_convention="atom")
        for seed in CITY_SEEDS:
            sim = make_city_sim(seed)
            pid = next(iter(sim.passengers))
            for goal in [pid, 7, 0]:
                with self.subTest(seed=seed, goal=goal):
                    graph = make_atom_graph(sim)
                    time_before = graph.global_features[0, 0].item()
                    projection, actions = planner.plan(graph, goal)
                    expected = time_before - len(actions) / 2000
                    self.assertAlmostEqual(projection.global_features[0, 0].item(), expected, places=5)
                    # every OTHER global feature column is untouched
                    self.assertTrue(th.equal(projection.global_features[0, 1:], graph.global_features[0, 1:]))


class TestProjectionMatchesExecutedState(unittest.TestCase):
    """The central correctness property: decoding+executing `actions` on a real copy of
    the sim must produce a state whose atoms/edges are identical (up to row order for
    propositions; exact row-for-row for type atoms) to what plan_atom projected."""

    def _assert_projection_matches_execution(self, sim, goal):
        planner = Planner(graph_convention="atom")
        graph = make_atom_graph(sim)
        projection, actions = planner.plan(graph, goal)

        executed = execute(copy.deepcopy(sim), actions)
        ground_truth_atoms = env_to_atoms(executed)
        ground_truth_x, ground_truth_ef, ground_truth_ei, _mask, _gf = env_to_atom_graph(executed)

        proj_atoms, proj_edges = edges_by_identity(projection.x.numpy(), projection.edge_index.numpy(), projection.edge_attr.numpy())
        gt_atoms, gt_edges = edges_by_identity(ground_truth_x, ground_truth_ei, ground_truth_ef)

        # item 2: atoms as sets (order-independent -- covers propositions)
        self.assertEqual(set(proj_atoms), set(gt_atoms))
        # item 2/4: x/edge_index/edge_attr equal up to row order, expressed via stable
        # atom-identity edges rather than raw row indices
        self.assertEqual(proj_edges, gt_edges)
        # item 4: type-atom row k is object k in BOTH -- exact row-for-row match, no
        # permutation allowed for this prefix
        n_obj = n_obj_of(proj_atoms)
        self.assertEqual(n_obj_of(gt_atoms), n_obj)
        self.assertEqual(proj_atoms[:n_obj], gt_atoms[:n_obj])

        return projection, actions, executed

    def test_deliver_current_scenario(self):
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed):
                sim = make_city_sim(seed)
                pid = next(iter(sim.passengers))
                self._assert_projection_matches_execution(sim, pid)

    def test_move_scenario(self):
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed):
                sim = make_city_sim(seed)
                # a real road-adjacent target, so `actions` is a genuine 1-hop move
                target = next(
                    c for c in sim.graph.successors(sim.taxi.location)
                    if sim.graph.edges[(sim.taxi.location, c)]["attr"] == [1, 0, 0, 1]
                )
                self._assert_projection_matches_execution(sim, target)

    def test_noop_dropoff_scenario(self):
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed):
                sim = make_city_sim(seed)
                self._assert_projection_matches_execution(sim, 0)  # taxi's own node, nothing carried

    def test_grid_hand_checkable_delivery(self):
        # small, fully inspectable scenario as a sanity anchor alongside the city-scale sweep
        sim = make_grid_sim(0)
        pid = next(iter(sim.passengers))
        self._assert_projection_matches_execution(sim, pid)


class TestDeliveryRenumberingNotHighestNumbered(unittest.TestCase):
    """Several concurrent passengers, delivering one that is NOT the highest-numbered
    object -- confirms the projection's renumbering matches TaxiWorldSimulator's own
    resort_passengers (which shifts every id above the removed one down by one), not
    just the easy "remove the last row" case oracle_sage's own tensor surgery relies on."""

    def test_deliver_middle_passenger(self):
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed):
                sim = make_city_sim(seed)
                # add two more passengers so there are 3 total, ids strictly increasing
                sim.add_passenger()
                sim.add_passenger()
                pids = sorted(sim.passengers.keys())
                self.assertEqual(len(pids), 3)
                middle_pid = pids[1]
                self.assertNotEqual(middle_pid, max(pids))  # the one we deliver is NOT the highest id
                delivered_destination = sim.passengers[middle_pid].destination  # identity survives renumbering

                planner = Planner(graph_convention="atom")
                graph = make_atom_graph(sim)
                projection, actions = planner.plan(graph, middle_pid)

                executed = execute(copy.deepcopy(sim), actions)
                # sanity: the delivered passenger is actually gone from the real env too.
                # NOTE: the *number* middle_pid can be REASSIGNED to a different, still-
                # surviving passenger by resort_passengers' renumbering (every id above
                # the removed one shifts down by one) -- so identity must be checked via
                # the delivered passenger's own destination, not by literal absence of
                # the id itself.
                self.assertEqual(len(executed.passengers), len(sim.passengers) - 1)
                self.assertNotIn(delivered_destination, [p.destination for p in executed.passengers.values()])

                ground_truth_atoms = env_to_atoms(executed)
                proj_atoms = graph_to_atoms(projection.x.numpy(), projection.edge_index.numpy(), projection.edge_attr.numpy())

                n_obj = n_obj_of(proj_atoms)
                self.assertEqual(n_obj_of(ground_truth_atoms), n_obj)
                # exact row-for-row match over type atoms: row k is object k in both,
                # after the SAME contiguous renumbering resort_passengers performs
                self.assertEqual(proj_atoms[:n_obj], ground_truth_atoms[:n_obj])
                self.assertEqual(set(proj_atoms), set(ground_truth_atoms))

                # the delivered passenger's identity must not survive anywhere -- checked
                # via its destination value, not the number middle_pid (which
                # resort_passengers-style renumbering can legitimately reassign to a
                # different, still-surviving passenger -- see the note above)
                surviving_destinations = [args[1] for pred, args in proj_atoms if pred == "destination"]
                self.assertNotIn(delivered_destination, surviving_destinations)


class TestMatchesOracleSageSemantics(unittest.TestCase):
    """Item 5: exact behavioural parity with oracle_sage for the same logical state."""

    def _fresh_graphs(self, sim):
        # freshly rebuilt from the SAME (unmutated-by-planning) sim every time -- oracle_sage's
        # planner mutates its own graph object in place, so each comparison needs its own copy
        return make_atom_graph(sim), make_os_graph(sim)

    def test_deliver_passenger_projection_never_shows_intermediate_aboard_state(self):
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed):
                sim = make_city_sim(seed)
                pid = next(iter(sim.passengers))
                self.assertNotEqual(sim.passengers[pid].location, sim.taxi.node)  # genuinely a 2-leg case

                atom_graph, _ = self._fresh_graphs(sim)
                planner = Planner(graph_convention="atom")
                projection, actions = planner.plan(atom_graph, pid)

                proj_atoms = graph_to_atoms(projection.x.numpy(), projection.edge_index.numpy(), projection.edge_attr.numpy())
                # the delivered passenger's `in` fact pointing at the taxi (the
                # "intermediate aboard" state) must never appear -- in fact NO atom
                # referencing this passenger at all should survive
                for predicate, args in proj_atoms:
                    self.assertNotIn(pid, args, f"projection still references delivered passenger {pid} via {(predicate, args)}")
                # actions DOES carry the full move+pickup+move+dropoff sequence
                self.assertIn(pid, actions)

    def test_noop_dropoff_unchanged_graph_and_actions(self):
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed):
                sim = make_city_sim(seed)
                atom_graph, _ = self._fresh_graphs(sim)
                atoms_before = env_to_atoms(sim)

                planner = Planner(graph_convention="atom")
                projection, actions = planner.plan(atom_graph, 0)  # taxi's own node

                self.assertEqual(actions, [sim.taxi.location])
                proj_atoms = graph_to_atoms(projection.x.numpy(), projection.edge_index.numpy(), projection.edge_attr.numpy())
                self.assertEqual(proj_atoms, atoms_before)  # literally unchanged, exact order too

    def test_move_only_taxi_in_fact_changes_node_count_unchanged(self):
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed):
                sim = make_city_sim(seed)
                target = next(
                    c for c in sim.graph.successors(sim.taxi.location)
                    if sim.graph.edges[(sim.taxi.location, c)]["attr"] == [1, 0, 0, 1]
                )
                atom_graph, _ = self._fresh_graphs(sim)
                atoms_before = env_to_atoms(sim)

                planner = Planner(graph_convention="atom")
                projection, actions = planner.plan(atom_graph, target)

                self.assertEqual(projection.x.shape[0], atom_graph.x.shape[0])  # node count unchanged
                proj_atoms = graph_to_atoms(projection.x.numpy(), projection.edge_index.numpy(), projection.edge_attr.numpy())

                diffs = [
                    (before, after) for before, after in zip(atoms_before, proj_atoms) if before != after
                ]
                self.assertEqual(len(diffs), 1, f"expected exactly one changed atom (the taxi's `in`), got {diffs}")
                (pred_before, args_before), (pred_after, args_after) = diffs[0]
                self.assertEqual(pred_before, "in")
                self.assertEqual(pred_after, "in")
                self.assertEqual(args_before[0], sim.taxi.node)
                self.assertEqual(args_after, (sim.taxi.node, target))

    def test_actions_identical_to_oracle_sage_for_same_logical_state(self):
        for seed in CITY_SEEDS:
            sim = make_city_sim(seed)
            pid = next(iter(sim.passengers))
            target = next(
                c for c in sim.graph.successors(sim.taxi.location)
                if sim.graph.edges[(sim.taxi.location, c)]["attr"] == [1, 0, 0, 1]
            )
            atom_planner = Planner(graph_convention="atom")
            os_planner = Planner(graph_convention="oracle_sage")

            for goal_name, goal in [("deliver", pid), ("move", target), ("noop_dropoff", 0)]:
                with self.subTest(seed=seed, goal=goal_name):
                    # fresh graphs each time: oracle_sage's planner mutates its graph
                    # object in place, so a stale reused graph would silently compare
                    # against an already-advanced state, not the original
                    atom_graph, os_graph = self._fresh_graphs(sim)
                    _, atom_actions = atom_planner.plan(atom_graph, goal)
                    _, os_actions = os_planner.plan(os_graph, goal)
                    self.assertEqual(list(atom_actions), list(os_actions))


if __name__ == "__main__":
    unittest.main()
