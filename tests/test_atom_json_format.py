"""
Tests for the atom convention's new compact JSON wire format (env_to_atom_json /
atoms_to_json / json_to_atom_graph, sage/domains/gym_taxi/utils/representations.py and
taxi_env.py's GRAPH_CONVENTION_CONVERTERS), replacing the old format that serialised the
full expanded node_feats/edge_feats/edge_index every step.

Motivation (py-spy profile on RCP, 120s/4798 samples of a real "atom" training run):
~79% of the main training thread was in _get_state_json -> env_to_atom_json on every
single env step (~55% in graph_to_json/json.dumps, ~19% in env_to_atom_graph/
atoms_to_graph's scipy.sparse matmul) -- including on intermediate plan steps whose
observations collect_rollouts immediately discards. The new format serialises ONLY the
flat atom list (predicate index + integer args) plus global_feats; x/edge_index/
edge_attr/mask are reconstructed lazily, only when a converter actually decodes a
JSON string, via atoms_to_graph -- the exact same function env_to_atom_graph itself
already used.

This file verifies the NEW format's decoded output is byte-identical to the OLD
format's, using the OLD path (env_to_atom_graph + the shared graph_to_json/json_to_graph)
as the reference -- those functions are unmodified, so this is a genuine
"current code vs before this change" comparison, not a comparison against a frozen
snapshot.

Run from the repo root with:
    python -m pytest tests/test_atom_json_format.py -v
"""
import copy
import unittest

import numpy as np
import torch as th

from sage.domains.gym_taxi.simulator.taxi_world import TaxiWorldSimulator, Passenger
from sage.domains.gym_taxi.utils.config import CITY
from sage.domains.gym_taxi.utils.representations import (
    env_to_atom_graph,
    env_to_atom_json,
    json_to_atom_graph,
)
from sage.domains.utils.representations import graph_to_json, json_to_graph
from sage.domains.gym_taxi.simulator.planner import Planner

CITY_SEEDS = [0, 300, 600, 900, 1200]


def make_sim(seed, **scenario):
    scenario = dict(scenario, passenger_creation_probability=0)
    return TaxiWorldSimulator(
        np.random.RandomState(seed), planning=True, graph_convention="atom", **scenario,
    )


def make_city_sim(seed):
    return make_sim(seed, **CITY)


def old_path_batch(sim):
    """The pre-existing (unmodified) expanded-format path: env_to_atom_graph ->
    graph_to_json -> json_to_graph. This IS "the current code before this change" --
    none of these three functions were touched by the new compact-format work."""
    js = graph_to_json(*env_to_atom_graph(sim))
    return json_to_graph([(js,)])


def new_path_batch(sim):
    """The new compact-format path: env_to_atom_json -> json_to_atom_graph."""
    js = env_to_atom_json(sim)
    return json_to_atom_graph([(js,)])


def assert_batches_identical(test_case, old_batch, new_batch):
    test_case.assertEqual(new_batch.x.dtype, old_batch.x.dtype)
    test_case.assertEqual(new_batch.edge_index.dtype, old_batch.edge_index.dtype)
    test_case.assertEqual(new_batch.edge_attr.dtype, old_batch.edge_attr.dtype)
    test_case.assertEqual(new_batch.mask.dtype, old_batch.mask.dtype)
    test_case.assertEqual(new_batch.global_features.dtype, old_batch.global_features.dtype)
    test_case.assertEqual(new_batch.x.device, old_batch.x.device)

    test_case.assertTrue(th.equal(new_batch.x, old_batch.x))
    test_case.assertTrue(th.equal(new_batch.edge_index, old_batch.edge_index))
    test_case.assertTrue(th.equal(new_batch.edge_attr, old_batch.edge_attr))
    test_case.assertTrue(th.equal(new_batch.mask, old_batch.mask))
    test_case.assertTrue(th.allclose(new_batch.global_features, old_batch.global_features))


def relocate_passenger_to(sim, pid, new_location):
    old_location = sim.passengers[pid].location
    destination = sim.passengers[pid].destination
    sim.graph.remove_edge(pid, old_location)
    sim.graph.remove_edge(old_location, pid)
    sim.graph.add_edge(pid, new_location, attr=[0, 1, 0, 1])
    sim.graph.add_edge(new_location, pid, attr=[0, 1, 0, -1])
    sim.passengers[pid] = Passenger(new_location, destination)


def relocate_destination_to(sim, pid, new_destination):
    old_destination = sim.passengers[pid].destination
    location = sim.passengers[pid].location
    sim.graph.remove_edge(pid, old_destination)
    sim.graph.remove_edge(old_destination, pid)
    sim.graph.add_edge(pid, new_destination, attr=[0, 0, 1, 1])
    sim.graph.add_edge(new_destination, pid, attr=[0, 0, 1, -1])
    sim.passengers[pid] = Passenger(location, new_destination)


class TestNewFormatMatchesOldFormatExactly(unittest.TestCase):
    """Byte-identical Data at reset, after pickup, after several moves, and after
    dropoff -- the checkpoints specified for verification."""

    def test_at_reset(self):
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed, checkpoint="reset"):
                sim = make_city_sim(seed)
                assert_batches_identical(self, old_path_batch(sim), new_path_batch(sim))

    def test_after_pickup(self):
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed, checkpoint="after_pickup"):
                sim = make_city_sim(seed)
                pid = next(iter(sim.passengers))
                relocate_passenger_to(sim, pid, sim.taxi.location)
                sim.attempt_pickup(pid)
                assert_batches_identical(self, old_path_batch(sim), new_path_batch(sim))

    def test_after_several_moves(self):
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed, checkpoint="after_moves"):
                sim = make_city_sim(seed)
                for _ in range(8):
                    start = sim.taxi.location
                    candidates = [
                        c for c in sim.graph.successors(start)
                        if sim.graph.edges[(start, c)]["attr"] == [1, 0, 0, 1]
                    ]
                    if not candidates:
                        break
                    sim.attempt_move(candidates[0])
                assert_batches_identical(self, old_path_batch(sim), new_path_batch(sim))

    def test_after_dropoff(self):
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed, checkpoint="after_dropoff"):
                sim = make_city_sim(seed)
                pid = next(iter(sim.passengers))
                relocate_passenger_to(sim, pid, sim.taxi.location)
                sim.attempt_pickup(pid)
                relocate_destination_to(sim, pid, sim.taxi.location)
                sim.attempt_dropoff(0)
                assert_batches_identical(self, old_path_batch(sim), new_path_batch(sim))


class TestPlannerInputUnaffectedByFormatChange(unittest.TestCase):
    """The planner receives a decoded Data, not a JSON string -- since the new format's
    decoded output is byte-identical (proven above), plan_atom's behaviour must be too.
    Confirms this directly: plans built from old-path and new-path decoded graphs (for
    the same underlying sim state) are identical."""

    def test_same_plan_from_either_decoded_graph(self):
        planner = Planner(graph_convention="atom")
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed):
                sim = make_city_sim(seed)
                pid = next(iter(sim.passengers))

                old_graph = old_path_batch(sim).to_data_list()[0]
                new_graph = new_path_batch(sim).to_data_list()[0]

                old_projection, old_actions = planner.plan(old_graph, pid)
                new_projection, new_actions = planner.plan(new_graph, pid)

                self.assertEqual(old_actions, new_actions)
                self.assertTrue(th.equal(old_projection.x, new_projection.x))
                self.assertTrue(th.equal(old_projection.edge_index, new_projection.edge_index))
                self.assertTrue(th.equal(old_projection.edge_attr, new_projection.edge_attr))


class TestCompactFormatIsActuallySmaller(unittest.TestCase):
    """A regression guard against silently reverting to a verbose format: the new
    encoding must be substantially smaller than the old one it replaced."""

    def test_new_json_much_smaller_than_old(self):
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed):
                sim = make_city_sim(seed)
                old_len = len(graph_to_json(*env_to_atom_graph(sim)))
                new_len = len(env_to_atom_json(sim))
                self.assertLess(new_len * 5, old_len)  # at least ~5x smaller


if __name__ == "__main__":
    unittest.main()
