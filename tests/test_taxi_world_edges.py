"""
Regression tests for a real bug in TaxiWorldSimulator (taxi_world.py):
`attempt_move` and `attempt_pickup` each used to tear down a tether edge
pair by calling `remove_edge` for only ONE direction, leaving a permanent
stale reverse edge in `self.graph` (a genuine `networkx.DiGraph`, so
neither direction is ever removed implicitly by the other). This was
confirmed to accumulate over an episode (thousands of stale tether edges
in a 300-step sample) and to cause a real WL-colour collision: a
location's colour failed to update after the taxi left it, because the
stale edge back to the taxi's old node was still present in that
location's outgoing-edge neighbourhood.

These tests check for the general CLASS of bug (any one-directional
`remove_edge` on a tether pair), not just today's two call sites, so a
future regression of the same shape would be caught here too.

Run from the repo root with:
    python -m unittest tests.test_taxi_world_edges -v
"""
import unittest

# Importing build_wl_vocab (not otherwise used here) applies its numpy/gym
# compatibility shims as an import side effect - needed to construct a
# "city" (random_walls=True) GraphTaxiEnv at all in this sandbox's drifted
# gym/numpy. See build_wl_vocab.py's own docstring for why. The production
# code under test does NOT depend on this - only this test harness does.
import sage.domains.utils.build_wl_vocab as build_wl_vocab
from sage.domains.utils.build_wl_vocab import GraphTaxiEnv, MASK, REWARDS_VARIANT, sample_action

from sage.domains.gym_taxi.simulator.taxi_world import Passenger


def make_city_env():
    env = GraphTaxiEnv(representation="graph", scenario="city", mask=MASK, rewards=REWARDS_VARIANT)
    env.reset()
    return env


def find_move_target(sim):
    """A location adjacent to the taxi's current location via a genuine road edge."""
    start = sim.taxi.location
    for candidate in sim.graph.successors(start):
        if candidate != start and sim.graph.edges[(start, candidate)]["attr"] == [1, 0, 0, 1]:
            return candidate
    raise AssertionError("taxi has no road-adjacent move target in this sampled env")


def relocate_passenger_to(sim, pid, new_location):
    """
    Moves a passenger's graph edges (and bookkeeping) to `new_location`,
    mirroring add_passenger's own edge-construction pattern exactly, so the
    graph stays internally consistent (passenger.location always matches
    its actual pid<->location tether edges) ahead of a deterministic pickup.
    """
    old_location = sim.passengers[pid].location
    destination = sim.passengers[pid].destination
    sim.graph.remove_edge(pid, old_location)
    sim.graph.remove_edge(old_location, pid)
    sim.graph.add_edge(pid, new_location, attr=[0, 1, 0, 1])
    sim.graph.add_edge(new_location, pid, attr=[0, 1, 0, -1])
    sim.passengers[pid] = Passenger(new_location, destination)


def one_direction_only_pairs(graph):
    """
    Groups directed edges of a DiGraph by edge type (attr index 0=road,
    1=tether, 2=destination) into unordered node pairs, and returns the
    pairs that have an edge in only ONE direction per type - i.e. exactly
    the invariant refine() (wl_colours.py) silently assumes always holds.

    :return: {"road": set_of_pairs, "tether": set_of_pairs, "destination": set_of_pairs}
    """
    type_names = {0: "road", 1: "tether", 2: "destination"}
    seen = {name: set() for name in type_names.values()}
    both = {name: set() for name in type_names.values()}
    for a, b, data in graph.edges(data=True):
        attr = data["attr"]
        for idx, name in type_names.items():
            if attr[idx] == 1:
                pair = frozenset((a, b))
                if pair in seen[name]:
                    both[name].add(pair)
                else:
                    seen[name].add(pair)
    return {name: seen[name] - both[name] for name in type_names.values()}


class TestAttemptMoveNoStaleEdges(unittest.TestCase):
    def test_move_removes_both_directions_of_old_tether(self):
        env = make_city_env()
        sim = env.sim
        old_loc = sim.taxi.location
        target = find_move_target(sim)

        sim.attempt_move(target)

        self.assertEqual(sim.taxi.location, target)
        self.assertFalse(sim.graph.has_edge(0, old_loc), "stale forward tether 0->old_loc survived the move")
        self.assertFalse(sim.graph.has_edge(old_loc, 0), "stale reverse tether old_loc->0 survived the move")
        # sanity: the new tether pair at the destination genuinely exists
        self.assertTrue(sim.graph.has_edge(0, target))
        self.assertTrue(sim.graph.has_edge(target, 0))


class TestAttemptPickupNoStaleEdges(unittest.TestCase):
    def test_pickup_removes_both_directions_of_old_tether(self):
        env = make_city_env()
        sim = env.sim
        pid = next(iter(sim.passengers))
        loc = sim.taxi.location
        relocate_passenger_to(sim, pid, loc)

        sim.attempt_pickup(pid)

        self.assertEqual(sim.taxi.passenger, pid)
        self.assertFalse(sim.graph.has_edge(pid, loc), "stale forward tether pid->loc survived the pickup")
        self.assertFalse(sim.graph.has_edge(loc, pid), "stale reverse tether loc->pid survived the pickup")
        # sanity: the new pid<->taxi tether pair genuinely exists
        self.assertTrue(sim.graph.has_edge(pid, 0))
        self.assertTrue(sim.graph.has_edge(0, pid))


class TestExtendedSamplingNoStaleEdgesAnywhere(unittest.TestCase):
    def test_no_one_direction_only_edges_across_full_episodes(self):
        """
        Re-runs the exact extended sampling check that originally caught
        2,497 one-direction-only tether connections across 300 sampled
        snapshots (5 resets x 60 random-action steps, scenario="city") -
        this must now report zero for every edge type.
        """
        import numpy as np

        np.random.seed(0)
        totals = {"road": 0, "tether": 0, "destination": 0}
        snapshots_checked = 0

        for _ in range(5):
            env = make_city_env()
            sim = env.sim
            for _ in range(60):
                bad = one_direction_only_pairs(sim.graph)
                for name, pairs in bad.items():
                    totals[name] += len(pairs)
                snapshots_checked += 1

                action = sample_action(sim)
                _, _, done, _ = env.step(action)
                if done:
                    env.reset()

        self.assertEqual(snapshots_checked, 300)
        self.assertEqual(totals, {"road": 0, "tether": 0, "destination": 0})


if __name__ == "__main__":
    unittest.main()
