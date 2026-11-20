"""
F2 (paper-unified): the ternary planner's projection is truthful when the taxi is occupied.

Before F2, plan_on_facts planned a goal on a WAITING passenger g as a full delivery of g
(drive to g -> pick up -> drive to g's destination -> drop off) and projected g as
delivered, even with another passenger a aboard. In the simulator the pickup fails
(_attempt_pickup: the taxi is occupied), and the final drop-off acts on a. The projection
was therefore false whenever a was aboard.

F2 keeps the action sequence exactly as it was and changes only the projection, which now
follows the simulator: g stays waiting where it was, the taxi ends at g's destination, and
a is delivered there if and only if that is a's own destination; otherwise a stays aboard.

Every test executes the plan in a copy of the real simulator (spawning off, so no new
passengers appear mid-plan) and compares the real facts with the projected ones.

Run from the repo root (CPU):
    PYTHONPATH=. python -m pytest tests/test_ternary_planner_occupied.py -v -s
"""
import copy
import random as pyrandom
import unittest

import networkx as nx
import numpy as np

from tests.test_ternary_planner import (
    CONVENTIONS, ENCODERS, execute_with_rewards, make_graph, make_ternary_sim,
)
from sage.domains.gym_taxi.simulator.planner import Planner
from sage.domains.gym_taxi.simulator.ternary_planner import plan_on_facts
from sage.domains.gym_taxi.simulator.ternary_scripted_policy import scripted_policy

TAXI = 0


def executed_copy(sim, actions):
    """Runs `actions` on a deep copy of sim with spawning off; (copy, rewards)."""
    sim_copy = copy.deepcopy(sim)
    sim_copy.pair_creation_probability = 0
    sim_copy.passenger_creation_probability = 0
    return execute_with_rewards(sim_copy, actions)


def old_actions(facts, goal):
    """The action sequence the pre-F2 planner produced for a waiting-passenger goal --
    F2 must not change it."""
    roads = nx.Graph()
    taxi_loc, waiting, requests = None, {}, {}
    for pred, args in facts:
        if pred == "adjacent":
            roads.add_edge(*args)
        elif pred == "in" and args[0] == TAXI:
            taxi_loc = args[1]
        elif pred == "in" and args[1] != TAXI:
            waiting[args[0]] = args[1]
        elif pred == "request":
            requests.setdefault(args[0], []).append((args[1], args[2]))
    origin = waiting[goal]
    (destination,) = [d for o, d in requests[goal] if o == origin]
    move1 = nx.shortest_path(roads, taxi_loc, origin)[1:]
    move2 = nx.shortest_path(roads, origin, destination)[1:]
    return move1, move2


def pick_up(sim, pid):
    """Drives the taxi to pid and picks them up (real sim steps, no spawning)."""
    for node in nx.shortest_path(sim.roads, sim.taxi.location, sim.passengers[pid].location)[1:]:
        sim._apply(node)
    sim._apply(pid)
    assert sim.taxi.passenger == pid


def occupied_sims(seeds=range(6), per_seed=12, sample_every=3, random_prob=0.2):
    """Deep copies of live sims from scripted play, at moments when a passenger is aboard
    and at least one other passenger is waiting."""
    out = []
    for seed in seeds:
        sim = make_ternary_sim(seed)
        rng = pyrandom.Random(seed + 7000)
        step, got = 0, 0
        while got < per_seed and step < 5000:
            waiting = [p for p, q in sim.passengers.items() if q.location is not None]
            if step % sample_every == 0 and sim.taxi.passenger is not None and waiting:
                out.append(copy.deepcopy(sim))
                got += 1
            sim.act(scripted_policy(sim, rng, random_prob=random_prob))
            step += 1
            if sim.done:
                sim = make_ternary_sim(seed * 100003 + step)
    return out


def check_occupied_steps(test, sim, goal, actions):
    """Step by step in a copy of the simulator: the pickup of the goal passenger fails
    (taxi occupied); the passenger aboard stays aboard until the final drop-off; the goal
    passenger never moves and is never picked up. Ids don't change before the drop-off
    (only a delivery renumbers passengers)."""
    aboard = sim.taxi.passenger
    pickup_index = actions.index(goal)  # the plan's only action on a passenger id
    origin = sim.passengers[goal].location
    step_sim = copy.deepcopy(sim)
    step_sim.pair_creation_probability = 0
    step_sim.passenger_creation_probability = 0
    for i, action in enumerate(actions[:-1]):  # everything before the drop-off
        reward = step_sim.act(int(action))[1]
        if i == pickup_index:
            test.assertEqual(action, goal)
            test.assertEqual(reward, sim.rewards["failed-action"], "pickup must fail while occupied")
        test.assertEqual(step_sim.taxi.passenger, aboard, f"step {i}: the passenger aboard must stay aboard")
        test.assertEqual(step_sim.passengers[goal].location, origin, f"step {i}: the goal passenger must stay waiting")
        test.assertIsNone(step_sim.passengers[goal].picked_up_at)


def all_goals(sim):
    """The taxi, the taxi's location, a neighbouring location, every passenger."""
    neighbour = next(iter(sim.roads.successors(sim.taxi.location)))
    return [TAXI, sim.taxi.location, neighbour] + sorted(sim.passengers)


class TestPlanOnFactsMatchesExecution(unittest.TestCase):
    """plan_on_facts on the TRUE facts: the projected facts equal the facts of the real
    simulator after running the plan, for every kind of goal, occupied or not."""

    @classmethod
    def setUpClass(cls):
        cls.sims = occupied_sims()
        assert len(cls.sims) >= 50

    def test_every_goal_on_occupied_states(self):
        n_wait_goals = n_delivered_aboard = 0
        for sim in self.sims:
            facts = sim.facts()
            for goal in all_goals(sim):
                with self.subTest(goal=goal):
                    actions, projected = plan_on_facts(facts, goal)
                    executed, rewards = executed_copy(sim, actions)
                    self.assertEqual(projected, executed.facts())
                    if goal in sim.passengers and goal != sim.taxi.passenger:
                        n_wait_goals += 1
                        check_occupied_steps(self, sim, goal, actions)
                        if rewards[-1] == sim.rewards["drop-off"]:
                            n_delivered_aboard += 1
        print(f"\n[occupied] {n_wait_goals} waiting-passenger goals with a passenger aboard; "
              f"{n_delivered_aboard} of them deliver the passenger aboard")
        self.assertGreater(n_wait_goals, 0)

    def test_unoccupied_states_unchanged(self):
        """Empty taxi: every goal still matches execution (the pre-F2 behaviour)."""
        for seed in range(5):
            sim = make_ternary_sim(seed, spawning=False)
            self.assertIsNone(sim.taxi.passenger)
            facts = sim.facts()
            for goal in all_goals(sim):
                with self.subTest(seed=seed, goal=goal):
                    actions, projected = plan_on_facts(facts, goal)
                    executed, _ = executed_copy(sim, actions)
                    self.assertEqual(projected, executed.facts())


class TestOccupiedWaitingGoal(unittest.TestCase):
    """Hand-built buddy pair (k = 2): p is picked up at o0 (destination d0); q waits at o1,
    and q's request at o1 is (o1, d0) -- the same destination as p's. So the plan for q
    ends with a successful drop-off of p. A second case puts q at o0 (destination d1), so
    the drop-off fails and p stays aboard."""

    def _sim(self, q_at):
        """q_at: "o1" (q's destination is d0, p's own) or "o0" (q's destination is d1)."""
        sim = make_ternary_sim(0, spawning=False)
        sim.passengers.clear()
        locs = sorted(sim.roads.nodes)
        o0, o1, d0, d1 = locs[10], locs[150], locs[300], locs[390]
        q_location = {"o0": o0, "o1": o1}[q_at]
        p, q = sim.spawn_pair(((o0, d0), (o1, d1)), o0, ((o0, d1), (o1, d0)), q_location)
        pick_up(sim, p)
        return sim, p, q, (o0, o1, d0, d1)

    def _check(self, sim, goal):
        facts = sim.facts()
        actions, projected = plan_on_facts(facts, goal)
        move1, move2 = old_actions(facts, goal)
        self.assertEqual(actions, move1 + [goal] + move2 + [TAXI], "F2 must keep the action sequence")
        executed, rewards = executed_copy(sim, actions)
        self.assertEqual(projected, executed.facts())
        self.assertEqual(actions.index(goal), len(move1))
        check_occupied_steps(self, sim, goal, actions)
        return projected, executed, rewards

    def test_aboard_passenger_delivered_at_shared_destination(self):
        sim, p, q, (o0, o1, d0, d1) = self._sim("o1")
        projected, executed, rewards = self._check(sim, q)
        self.assertEqual(rewards[-1], sim.rewards["drop-off"])
        self.assertIsNone(executed.taxi.passenger)
        # p is gone; q (renumbered down to p's old id) still waits at o1, never picked up
        self.assertEqual(len(executed.passengers), 1)
        (only,) = executed.passengers.values()
        self.assertEqual((only.location, only.picked_up_at), (o1, None))
        self.assertIn(("in", (0, d0)), projected)

    def test_aboard_passenger_stays_aboard_when_destination_differs(self):
        sim, p, q, (o0, o1, d0, d1) = self._sim("o0")
        projected, executed, rewards = self._check(sim, q)
        self.assertEqual(rewards[-1], sim.rewards["failed-action"])
        self.assertEqual(executed.taxi.passenger, p)
        self.assertIn(("in", (p, 0)), projected)
        self.assertIn(("in", (q, o0)), projected)
        self.assertIn(("in", (0, d1)), projected)
        self.assertNotIn(("picked_up_at", (q, o0)), projected)


class TestRealPathAllConventions(unittest.TestCase):
    """Planner.plan (the path training takes) on the exact decoders (vilg, atom): the
    projection's tensors equal the encoding of the real facts after execution. Object
    (oracle_sage) decodes a hypothesis, so it is checked only on states where the decoder
    is exact (the decoded facts equal the true facts)."""

    @classmethod
    def setUpClass(cls):
        cls.sims = occupied_sims(seeds=range(3), per_seed=8)

    def test_projection_tensors_match_execution(self):
        from sage.domains.gym_taxi.simulator.ternary_planner import decode_object
        n_object_checked = 0
        for convention in CONVENTIONS:
            planner = Planner(graph_convention=convention, ternary=True)
            for sim in self.sims:
                graph = make_graph(sim, convention)
                if convention == "oracle_sage":
                    if decode_object(graph) != sim.facts():
                        continue
                    n_object_checked += 1
                for goal in sorted(sim.passengers):
                    with self.subTest(convention=convention, goal=goal):
                        projection, actions = planner.plan(graph, goal)
                        executed, _ = executed_copy(sim, actions)
                        meta = {"time": executed.time, "timeout": executed.timeout, "planning": executed.planning}
                        nf, ef, ei, mask, _gf = ENCODERS[convention](executed.facts(), meta)
                        np.testing.assert_array_equal(projection.x.numpy(), nf.astype(np.float32))
                        np.testing.assert_array_equal(projection.edge_attr.numpy(), ef.astype(np.float32))
                        np.testing.assert_array_equal(projection.edge_index.numpy(), ei.astype(np.int64))
                        np.testing.assert_array_equal(projection.mask.numpy(), mask)
        self.assertGreater(n_object_checked, 0)


if __name__ == "__main__":
    unittest.main()
