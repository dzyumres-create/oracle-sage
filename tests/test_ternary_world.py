"""
Tests for TernaryTaxiWorldSimulator (sage/domains/gym_taxi/simulator/ternary_taxi_world.py).

This is a brand-new, independent simulator -- these tests never touch
TaxiWorldSimulator's own state, except in TestMatchesOldDomainRNG, which exists
specifically to confirm the two simulators' RNG streams line up through maze
generation and taxi placement.

Run from the repo root with:
    python -m pytest tests/test_ternary_world.py -v
"""
import random as pyrandom
import unittest

import networkx as nx
import numpy as np

from sage.domains.gym_taxi.simulator.taxi_world import TaxiWorldSimulator
from sage.domains.gym_taxi.simulator.ternary_taxi_world import (
    TernaryTaxiWorldSimulator,
    TernaryPassenger,
    RELATION_PREDICATES,
)

TYPE_PREDICATES = ("taxi", "location", "passenger")
VALID_PREDICATES = set(TYPE_PREDICATES) | set(RELATION_PREDICATES)
from sage.domains.gym_taxi.utils.config import CITY, CITY_TERNARY


def legal_actions(sim):
    """Every action guaranteed not to raise KeyError from the current state: a dropoff
    attempt, a pickup attempt on any live passenger, staying put, or moving along any
    real road edge out of the taxi's current location."""
    actions = [sim.taxi.node, sim.taxi.location]
    actions.extend(sim.passengers.keys())
    actions.extend(sim.roads.successors(sim.taxi.location))
    return actions


def _road_graph_from_facts(facts):
    """Builds a plain nx.DiGraph of locations from facts() alone -- deliberately not
    reading sim.roads directly, so this exercises exactly what a real converter/planner
    downstream would have to work with."""
    graph = nx.DiGraph()
    for predicate, args in facts:
        if predicate == "location":
            graph.add_node(args[0])
    for predicate, args in facts:
        if predicate == "adjacent":
            graph.add_edge(args[0], args[1])
    return graph


def _taxi_location_from_facts(facts):
    for predicate, args in facts:
        if predicate == "in" and args[0] == 0:
            return args[1]
    raise ValueError("facts() has no in(0, l) fact for the taxi")


def _carried_passenger_from_facts(facts):
    """Returns the carried passenger's id, or None -- in(p, 0) with p != 0."""
    for predicate, args in facts:
        if predicate == "in" and args[0] != 0 and args[1] == 0:
            return args[0]
    return None


def _waiting_passengers_from_facts(facts):
    """{pid: location} for every passenger currently at a location (not carried)."""
    return {args[0]: args[1] for predicate, args in facts if predicate == "in" and args[0] != 0 and args[1] != 0}


def destination_via_join(facts, pid):
    """
    The only legitimate way to find where a carried passenger is going: join their
    picked_up_at fact against their own request facts on origin == picked_up_at.
    Returns None if pid hasn't been picked up (no picked_up_at fact).
    """
    picked_up_at = None
    for predicate, args in facts:
        if predicate == "picked_up_at" and args[0] == pid:
            picked_up_at = args[1]
            break
    if picked_up_at is None:
        return None
    for predicate, args in facts:
        if predicate == "request" and args[0] == pid and args[1] == picked_up_at:
            return args[2]
    raise ValueError(f"no request({pid}, {picked_up_at}, ?) fact -- picked_up_at with no matching request")


def scripted_policy(sim, rng, random_prob=0.2):
    """
    A simple, reusable scripted policy for exercising realistic play (not just
    uniform-random wandering): plan to the nearest waiting passenger, pick them up,
    plan to their destination via the facts' picked_up_at/request join, drop off --
    mixed with `random_prob` uniform-random legal actions, so the play distribution
    isn't perfectly deterministic and still covers edge cases (failed pickups/dropoffs,
    idle moves) a pure planner would never produce.

    Reused by the Part 2 collision-counter's state sampler, so any change here changes
    what states that counter sees -- keep this in sync with facts()'s contract, not
    with sim internals (this deliberately reads only sim.facts(), sim.taxi.node and
    sim.passengers.keys() -- the id-membership check needed to issue a pickup/dropoff
    action, not to plan one).

    :param sim: a TernaryTaxiWorldSimulator
    :param rng: a random.Random instance (NOT sim.random -- policy randomness must
        stay independent of the simulator's own RNG stream, or determinism tests that
        replay a fixed action sequence would silently start depending on this
        function's internals)
    :param random_prob: probability of ignoring the plan and taking a uniform-random
        legal action instead
    :return: an action (node id) for sim.act()/sim._apply()
    """
    if rng.random() < random_prob:
        return rng.choice(legal_actions(sim))

    facts = sim.facts()
    taxi_location = _taxi_location_from_facts(facts)
    carried_pid = _carried_passenger_from_facts(facts)

    if carried_pid is not None:
        destination = destination_via_join(facts, carried_pid)
        if taxi_location == destination:
            return sim.taxi.node  # attempt dropoff
        road_graph = _road_graph_from_facts(facts)
        path = nx.shortest_path(road_graph, taxi_location, destination)
        return path[1]

    waiting = _waiting_passengers_from_facts(facts)
    if not waiting:
        return rng.choice(legal_actions(sim))

    road_graph = _road_graph_from_facts(facts)
    lengths = nx.single_source_shortest_path_length(road_graph, taxi_location)
    reachable = {pid: loc for pid, loc in waiting.items() if loc in lengths}
    if not reachable:
        return rng.choice(legal_actions(sim))
    nearest_pid = min(reachable, key=lambda pid: lengths[reachable[pid]])
    nearest_loc = reachable[nearest_pid]
    if taxi_location == nearest_loc:
        return nearest_pid  # attempt pickup
    path = nx.shortest_path(road_graph, taxi_location, nearest_loc)
    return path[1]


def instrument(sim):
    """
    Monkeypatches this sim INSTANCE (not the class) to count events a bare reward
    signal can't distinguish (pickup success and dropoff success both often carry the
    same "base"/"failed-action" reward under city-style rewards). Returns a dict of
    counters that fill in as the sim is stepped via sim.act()/sim._apply().

    Only touches this one instance's bound methods; TernaryTaxiWorldSimulator itself
    is not modified.
    """
    counters = {
        "pair_spawns": 0,
        "successful_pickups": 0,
        "successful_dropoffs": 0,
        "failed_dropoffs": 0,
        "renumberings": 0,
    }

    original_pickup = sim._attempt_pickup

    def _attempt_pickup(pid):
        # A pickup only actually happens on the None -> pid transition -- checking
        # sim.taxi.passenger == pid alone would also match a no-op re-attempt on the
        # passenger already being carried (their id stays in legal_actions() the
        # whole time they're carried, so a random policy re-issues it often).
        was_empty = sim.taxi.passenger is None
        reward = original_pickup(pid)
        if was_empty and sim.taxi.passenger == pid:
            counters["successful_pickups"] += 1
        return reward

    sim._attempt_pickup = _attempt_pickup

    original_dropoff = sim._attempt_dropoff

    def _attempt_dropoff():
        pid_before = sim.taxi.passenger
        reward = original_dropoff()
        if pid_before is not None:
            if sim.taxi.passenger is None:
                counters["successful_dropoffs"] += 1
            else:
                counters["failed_dropoffs"] += 1
        return reward

    sim._attempt_dropoff = _attempt_dropoff

    original_renumber = sim._renumber_passengers

    def _renumber_passengers():
        counters["renumberings"] += 1
        return original_renumber()

    sim._renumber_passengers = _renumber_passengers

    original_spawn_pair = sim.spawn_pair

    def spawn_pair(*args, **kwargs):
        counters["pair_spawns"] += 1
        return original_spawn_pair(*args, **kwargs)

    sim.spawn_pair = spawn_pair

    return counters


def crossed_state(sim):
    """True iff the taxi is currently carrying a passenger whose buddy is still a
    live (undelivered) passenger -- the state the whole ternary-domain extension is
    about, per the extended-domain handoff's crossed-pair generator rules."""
    pid = sim.taxi.passenger
    if pid is None or pid not in sim.passengers:
        return False
    buddy = sim.passengers[pid].buddy
    return buddy is not None and buddy in sim.passengers


def assert_invariants(sim):
    n_loc = sim.size * sim.size
    passenger_ids = sorted(sim.passengers.keys())
    assert passenger_ids == list(range(n_loc + 1, n_loc + 1 + len(passenger_ids))), (
        f"passenger ids not contiguous from {n_loc + 1}: {passenger_ids}"
    )
    assert len(sim.passengers) <= sim.concurrent_passengers, (
        f"{len(sim.passengers)} passengers exceeds cap {sim.concurrent_passengers}"
    )

    carried = [pid for pid, p in sim.passengers.items() if p.location is None]
    assert len(carried) <= 1, f"more than one passenger marked carried: {carried}"
    assert (sim.taxi.passenger is not None) == (len(carried) == 1), (
        f"taxi.passenger={sim.taxi.passenger} inconsistent with carried={carried}"
    )
    if carried:
        assert carried[0] == sim.taxi.passenger

    for pid, passenger in sim.passengers.items():
        origins = [o for o, _d in passenger.requests]
        assert len(set(origins)) == sim.requests_per_passenger, (
            f"passenger {pid} requests do not have {sim.requests_per_passenger} "
            f"distinct origins: {passenger.requests}"
        )
        if passenger.location is not None:
            assert passenger.location in origins, (
                f"waiting passenger {pid} at {passenger.location}, "
                f"not one of its own origins {origins}"
            )
        if passenger.picked_up_at is not None:
            matches = [d for o, d in passenger.requests if o == passenger.picked_up_at]
            assert len(matches) == 1, (
                f"passenger {pid} picked_up_at={passenger.picked_up_at} matches "
                f"{len(matches)} of its own requests, not exactly 1"
            )
        if passenger.buddy is not None and passenger.buddy in sim.passengers:
            buddy = sim.passengers[passenger.buddy]
            own_map = dict(passenger.requests)
            buddy_map = dict(buddy.requests)
            assert set(own_map) == set(buddy_map), (
                f"passenger {pid} and buddy {passenger.buddy} do not share origins: "
                f"{passenger.requests} vs {buddy.requests}"
            )
            for origin, destination in own_map.items():
                assert buddy_map[origin] != destination, (
                    f"passenger {pid} and buddy {passenger.buddy} share destination "
                    f"{destination} for origin {origin}"
                )

    facts = sim.facts()
    assert not any(predicate == "destination" for predicate, _args in facts), (
        "facts() produced a 'destination' fact"
    )
    bad_predicates = {predicate for predicate, _args in facts if predicate not in VALID_PREDICATES}
    assert not bad_predicates, (
        f"facts() produced predicate(s) outside the 7 valid ones {sorted(VALID_PREDICATES)}: "
        f"{bad_predicates} -- e.g. 'buddy' leaking out of TernaryPassenger as a fact"
    )
    assert not any("destination" in field for field in TernaryPassenger._fields), (
        f"TernaryPassenger has a field named/containing 'destination': {TernaryPassenger._fields}"
    )

    n_type_atoms = 1 + n_loc + len(sim.passengers)
    type_atoms = facts[:n_type_atoms]
    expected_type_predicates = ["taxi"] + ["location"] * n_loc + ["passenger"] * len(sim.passengers)
    assert [predicate for predicate, _args in type_atoms] == expected_type_predicates, (
        "type atoms are not [taxi, locations..., passengers...] in that block order"
    )
    expected_ids = [0] + list(range(1, n_loc + 1)) + passenger_ids
    assert [args[0] for _predicate, args in type_atoms] == expected_ids, (
        "type atoms are not in ascending object-id order (row k must be object k)"
    )
    for _predicate, args in facts:
        for arg in args:
            assert type(arg) is int, f"non-plain-int fact arg {arg!r} ({type(arg)})"


class TestHandCheckedPair(unittest.TestCase):
    """
    size=2, no walls, RandomState(0): the very first RNG draw (generate_road_network
    consumes none when random_walls=False; add_taxi's random.choice(range(1,5)) is
    the first) is identical to TaxiWorldSimulator's under the same seed/size/
    random_walls -- tests/test_atom_encoding.py's TestHandCheckedGrid documents this
    same seed placing the taxi at location 1 for the old domain, and
    TestMatchesOldDomainRNG below checks the general claim. The pair itself is
    injected via initial_pair, bypassing the RNG entirely, so every passenger/request
    value below is chosen by this test, not derived from a draw:

        p (pid 5): requests (1 -> 4), (3 -> 2); starts at origin 1 (== taxi start)
        q (pid 6): requests (1 -> 2), (3 -> 4); starts at origin 3

    Same origins {1, 3} for both, destinations rotated by one -- a genuine crossed
    pair per rule 2, not a relabelling: p's own requests already disagree with q's on
    every destination.
    """

    def setUp(self):
        self.requests_p = ((1, 4), (3, 2))
        self.requests_q = ((1, 2), (3, 4))
        self.sim = TernaryTaxiWorldSimulator(
            np.random.RandomState(0),
            size=2,
            random_walls=False,
            delivery_limit=1,
            concurrent_passengers=2,
            requests_per_passenger=2,
            passenger_creation_probability=0,
            observation_fn=lambda s: None,
            initial_pair=(self.requests_p, 1, self.requests_q, 3),
        )

    def test_taxi_start_matches_old_domain_under_same_seed(self):
        self.assertEqual(self.sim.taxi.location, 1)

    def test_initial_passengers_and_ids(self):
        self.assertEqual(sorted(self.sim.passengers.keys()), [5, 6])
        self.assertEqual(self.sim.passengers[5], TernaryPassenger(1, self.requests_p, None, 6))
        self.assertEqual(self.sim.passengers[6], TernaryPassenger(3, self.requests_q, None, 5))

    def test_initial_facts_match_hand_derivation(self):
        adjacent = [("adjacent", edge) for edge in sorted(self.sim.roads.edges())]
        # Type atoms first, in ascending object-id order: taxi (0), then locations
        # (1..4), then passengers (5, 6) -- NOT grouped alphabetically by predicate
        # name, and NOT the passenger/location/taxi order this test used before row-0-
        # is-the-taxi became a hard requirement (graph_policy.py's mask[:,0]=True).
        expected = (
            [("taxi", (0,))]
            + [("location", (l,)) for l in (1, 2, 3, 4)]
            + [("passenger", (5,)), ("passenger", (6,))]
            + adjacent
            + [("in", (0, 1)), ("in", (5, 1)), ("in", (6, 3))]
            + [
                ("request", (5, 1, 4)),
                ("request", (5, 3, 2)),
                ("request", (6, 1, 2)),
                ("request", (6, 3, 4)),
            ]
        )
        self.assertEqual(self.sim.facts(), expected)
        self.assertNotIn("picked_up_at", [predicate for predicate, _args in self.sim.facts()])
        self.assertNotIn("destination", [predicate for predicate, _args in self.sim.facts()])

    def test_type_atoms_come_first_in_ascending_object_id_order(self):
        facts = self.sim.facts()
        type_atoms = facts[:7]  # 1 taxi + 4 locations + 2 passengers
        self.assertEqual(
            type_atoms,
            [
                ("taxi", (0,)),
                ("location", (1,)),
                ("location", (2,)),
                ("location", (3,)),
                ("location", (4,)),
                ("passenger", (5,)),
                ("passenger", (6,)),
            ],
        )
        # object id == position in this block, exactly the "row k is object k"
        # contract a future atom-encoding converter needs.
        self.assertEqual([args[0] for _predicate, args in type_atoms], list(range(7)))

    def test_all_fact_args_are_plain_ints_not_numpy_or_str(self):
        for _predicate, args in self.sim.facts():
            for arg in args:
                self.assertIs(type(arg), int, f"non-plain-int arg {arg!r} ({type(arg)}) in facts()")

    def test_pickup_sets_picked_up_at_and_clears_location(self):
        reward = self.sim._apply(5)
        self.assertEqual(reward, self.sim.rewards["base"])
        self.assertEqual(self.sim.taxi.passenger, 5)
        self.assertEqual(self.sim.passengers[5], TernaryPassenger(None, self.requests_p, 1, 6))
        facts = self.sim.facts()
        self.assertIn(("picked_up_at", (5, 1)), facts)
        self.assertIn(("in", (5, 0)), facts)
        self.assertNotIn(("in", (5, 1)), facts)

    def test_dropoff_at_wrong_location_fails_and_retains_passenger(self):
        self.sim._apply(5)  # pickup, taxi still at location 1
        reward = self.sim._apply(0)  # dropoff attempt: destination resolves to 4, taxi at 1
        self.assertEqual(reward, self.sim.rewards["failed-action"])
        self.assertEqual(self.sim.taxi.passenger, 5)
        self.assertIn(5, self.sim.passengers)
        self.assertEqual(self.sim.passengers[5].picked_up_at, 1)

    def test_full_episode_pickup_move_dropoff_renumbers_and_clears_buddy(self):
        self.sim._apply(5)  # pickup p at its origin, location 1
        self.sim._apply(0)  # failed dropoff attempt at location 1

        path = nx.shortest_path(self.sim.roads, source=1, target=4)
        for node in path[1:]:
            reward = self.sim._apply(node)
            self.assertEqual(reward, self.sim.rewards["base"])
            self.assertEqual(self.sim.taxi.location, node)

        reward = self.sim._apply(0)  # p's request (1 -> 4) resolves; taxi is at 4
        self.assertEqual(reward, self.sim.rewards["drop-off"])
        self.assertIsNone(self.sim.taxi.passenger)

        # only q survives, renumbered down to size*size + 1 == 5, buddy cleared
        self.assertEqual(sorted(self.sim.passengers.keys()), [5])
        self.assertEqual(self.sim.passengers[5], TernaryPassenger(3, self.requests_q, None, None))

        facts = self.sim.facts()
        predicates = [predicate for predicate, _args in facts]
        self.assertNotIn("destination", predicates)
        self.assertIn(("passenger", (5,)), facts)
        self.assertIn(("request", (5, 1, 2)), facts)
        self.assertIn(("request", (5, 3, 4)), facts)
        self.assertIn(("in", (5, 3)), facts)
        self.assertIn(("in", (0, 4)), facts)
        self.assertFalse(any(predicate == "picked_up_at" for predicate, _args in facts))

    def test_act_updates_delivery_limit_and_done(self):
        self.sim.act(5)
        self.sim.act(0)  # failed dropoff
        path = nx.shortest_path(self.sim.roads, source=1, target=4)
        for node in path[1:]:
            self.sim.act(node)
        self.assertEqual(self.sim.delivery_limit, 1)
        self.assertFalse(self.sim.done)
        self.sim.act(0)  # successful dropoff, delivery_limit was 1
        self.assertEqual(self.sim.delivery_limit, 0)
        self.assertTrue(self.sim.done)

    def test_move_to_non_adjacent_location_raises_keyerror(self):
        current = self.sim.taxi.location
        non_adjacent = [
            l for l in (1, 2, 3, 4) if l != current and not self.sim.roads.has_edge(current, l)
        ]
        self.assertTrue(non_adjacent, "expected at least one non-adjacent location in a 2x2 grid")
        with self.assertRaises(KeyError):
            self.sim._apply(non_adjacent[0])

    def test_unrecognised_action_raises_keyerror(self):
        with self.assertRaises(KeyError):
            self.sim._apply(9999)


class TestConstructorAndActContracts(unittest.TestCase):
    def test_concurrent_passengers_below_two_raises(self):
        with self.assertRaises(AssertionError):
            TernaryTaxiWorldSimulator(
                np.random.RandomState(0), size=2, random_walls=False, concurrent_passengers=1
            )

    def test_act_without_observation_fn_raises_not_implemented(self):
        sim = TernaryTaxiWorldSimulator(
            np.random.RandomState(0), size=2, random_walls=False, concurrent_passengers=2
        )
        with self.assertRaises(NotImplementedError):
            sim.act(sim.taxi.node)


def _run_sweep(seed, policy, n_steps=2000):
    """Runs n_steps of `policy` on a fresh CITY_TERNARY sim, checking invariants after
    every step and counting coverage events. `policy(sim, rng) -> action`."""
    sim = TernaryTaxiWorldSimulator(
        np.random.RandomState(seed), observation_fn=lambda s: None, **CITY_TERNARY
    )
    policy_rng = pyrandom.Random(seed + 1000)
    counters = instrument(sim)
    counters["crossed_steps"] = 0

    assert_invariants(sim)
    for _step in range(n_steps):
        action = policy(sim, policy_rng)
        sim.act(action)
        assert_invariants(sim)
        if crossed_state(sim):
            counters["crossed_steps"] += 1

    return counters


class TestInvariantsOverRandomPlay(unittest.TestCase):
    """(b): at least 2,000 random steps, several seeds, every invariant checked after
    every single step -- not just at the end. Also reports per-seed coverage (pair
    spawns, pickups, dropoffs, renumberings, crossed-state steps) under pure random
    play, run with `pytest -s` to see the printed report."""

    def test_invariants_hold_over_2000_steps_several_seeds(self):
        for seed in (0, 1, 2):
            counters = _run_sweep(seed, lambda sim, rng: rng.choice(legal_actions(sim)))
            print(f"[random policy] seed={seed} {counters}")


class TestInvariantsOverScriptedPlay(unittest.TestCase):
    """Same invariant sweep, but driven by scripted_policy (plan-pickup-plan-dropoff,
    80% of the time) instead of pure random wandering, so passengers actually get
    delivered and crossed states (carried passenger's buddy still present) actually
    occur within the step budget -- see the Part 1 report for why pure random play
    alone wasn't enough."""

    def test_invariants_hold_over_2000_scripted_steps_several_seeds(self):
        for seed in (0, 1, 2):
            counters = _run_sweep(seed, lambda sim, rng: scripted_policy(sim, rng, random_prob=0.2))
            print(f"[scripted policy] seed={seed} {counters}")


class TestDeterminism(unittest.TestCase):
    """(c): the same seed gives the same fact sequence under a fixed action
    sequence. The action sequence is recorded once, then replayed against two
    independently-constructed simulators (never reusing the recording instance),
    so equality actually exercises reproducibility rather than tautology."""

    def test_same_seed_same_fact_sequence_for_fixed_actions(self):
        seed = 7
        recorder = TernaryTaxiWorldSimulator(
            np.random.RandomState(seed), observation_fn=lambda s: None, **CITY_TERNARY
        )
        policy_rng = pyrandom.Random(123)
        actions = []
        for _ in range(300):
            action = policy_rng.choice(legal_actions(recorder))
            actions.append(action)
            recorder.act(action)

        def replay():
            sim = TernaryTaxiWorldSimulator(
                np.random.RandomState(seed), observation_fn=lambda s: None, **CITY_TERNARY
            )
            fact_sequence = [sim.facts()]
            for action in actions:
                sim.act(action)
                fact_sequence.append(sim.facts())
            return fact_sequence

        self.assertEqual(replay(), replay())


class TestMatchesOldDomainRNG(unittest.TestCase):
    """(d): the same seed gives the same maze and taxi start as TaxiWorldSimulator --
    both call generate_city_maze/relabel then draw the taxi location the same way,
    with no RNG draws in between, so this must hold for every seed, not just one."""

    def test_same_seed_gives_same_maze_and_taxi_start(self):
        for seed in (0, 1, 2, 3):
            old_sim = TaxiWorldSimulator(np.random.RandomState(seed), planning=True, **CITY)
            new_sim = TernaryTaxiWorldSimulator(np.random.RandomState(seed), **CITY_TERNARY)

            self.assertEqual(old_sim.taxi.location, new_sim.taxi.location)

            old_locations = {
                n for n, data in old_sim.graph.nodes(data=True) if data["attr"] == [1, 0, 0]
            }
            self.assertEqual(old_locations, set(new_sim.roads.nodes()))

            old_roads = {
                (u, v)
                for u, v, data in old_sim.graph.edges(data=True)
                if tuple(data["attr"][:3]) == (1, 0, 0) and data["attr"][-1] == 1
            }
            self.assertEqual(old_roads, set(new_sim.roads.edges()))


if __name__ == "__main__":
    unittest.main()
