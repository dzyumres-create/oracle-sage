"""
Tests for the L-selection diagnostic tools (Step 4):
- build_wl_vocab.py's greedy-policy/planner-data helpers (road_graph, next_hop,
  greedy_action, to_planner_data)
- wl_depth_sweep.py (run_depth's checkpoint history, collect_policy_corpus,
  measure_held_out_oov)
- wl_collision_check.py (classify_goal, state_key, wl_to_stable_partition, _hist_equal,
  select_candidates)

These are diagnostic scripts, not production/policy code - coverage here is aimed at
catching regressions and confirming each building block does what its docstring
claims, not exhaustive branch coverage of every CLI flag combination.

Run from the repo root with:
    python -m pytest tests/test_wl_diagnostics.py -v
"""
import unittest

import numpy as np
import torch as th
import networkx as nx

import sage.domains.utils.build_wl_vocab as build_wl_vocab
from sage.domains.utils.build_wl_vocab import (
    GraphTaxiEnv, MASK, REWARDS_VARIANT, road_graph, next_hop, greedy_action, to_planner_data,
)
from sage.domains.gym_taxi.utils.config import PREDICTABLE5
from sage.domains.gym_taxi.simulator.taxi_world import TaxiWorldSimulator

from sage.domains.utils.wl_depth_sweep import (
    collect_graph_corpus, collect_policy_corpus, run_depth, measure_held_out_oov,
)
from sage.domains.utils.wl_colours import freeze_vocab, OOV_SIGNATURE

from sage.domains.utils.wl_collision_check import (
    classify_goal, state_key, decode_state, wl_to_stable_partition, _hist_equal, select_candidates,
)
from sage.domains.gym_taxi.simulator.planner import Planner, graph_to_networkx


def make_sim(seed, **scenario):
    return TaxiWorldSimulator(np.random.RandomState(seed), planning=True, graph_convention="oracle_sage", **scenario)


class TestRoadGraphAndGreedyPolicy(unittest.TestCase):
    def test_road_graph_is_undirected_and_matches_location_count(self):
        sim = make_sim(1, **PREDICTABLE5)
        G = road_graph(sim)
        self.assertFalse(G.is_directed())
        n_locations = sum(1 for _n, d in sim.graph.nodes(data=True) if d["attr"] == [1, 0, 0])
        self.assertEqual(G.number_of_nodes(), n_locations)

    def test_greedy_action_is_always_a_legal_action(self):
        """A legal action is: the taxi's own node (dropoff), a current passenger's
        node (pickup), or a node directly road-adjacent to the taxi (move) -
        env.step() must not raise on whatever greedy_action returns."""
        env = GraphTaxiEnv(representation="graph", scenario="predictable5", mask=MASK, rewards=REWARDS_VARIANT, graph_convention="oracle_sage")
        env.seed(3)
        env.reset()
        G = road_graph(env.sim)
        for _ in range(60):
            a = greedy_action(env.sim, G)
            road_neighbours = set(G.neighbors(env.sim.taxi.location)) if env.sim.taxi.location in G else set()
            legal = (a == 0) or (a in env.sim.passengers) or (a in road_neighbours) or (a == env.sim.taxi.location)
            self.assertTrue(legal, f"greedy_action returned illegal action {a}")
            _, _, done, _ = env.step(a)  # must not raise
            if done:
                env.reset()
                G = road_graph(env.sim)

    def test_greedy_action_eventually_delivers(self):
        """Positive control: the greedy policy actually completes pickups/deliveries
        within a bounded step budget (unlike pure random action sampling, which
        rarely does on a size-20 maze) - this is the whole reason it exists."""
        env = GraphTaxiEnv(representation="graph", scenario="predictable5", mask=MASK, rewards=REWARDS_VARIANT, graph_convention="oracle_sage")
        env.seed(3)
        env.reset()
        G = road_graph(env.sim)
        deliveries = 0
        for _ in range(200):
            a = greedy_action(env.sim, G)
            _, reward, done, _ = env.step(a)
            if reward == env.sim.rewards["drop-off"]:
                deliveries += 1
            if done:
                env.reset()
                G = road_graph(env.sim)
        self.assertGreater(deliveries, 0)

    def test_next_hop_is_none_at_destination_and_a_neighbour_otherwise(self):
        sim = make_sim(1, **PREDICTABLE5)
        G = road_graph(sim)
        loc = sim.taxi.location
        self.assertIsNone(next_hop(G, loc, loc))
        other = next(iter(G.neighbors(loc)))
        target = other
        path = nx.shortest_path(G, loc, target)
        self.assertEqual(next_hop(G, loc, target), path[1])


class TestToPlannerData(unittest.TestCase):
    def test_matches_env_to_graph_shapes_and_is_plan_ready(self):
        sim = make_sim(1, **PREDICTABLE5)
        data = to_planner_data(sim, graph_convention="oracle_sage")
        self.assertEqual(data.x.dtype, th.float32)
        self.assertEqual(data.edge_index.dtype, th.long)
        self.assertEqual(data.edge_attr.dtype, th.float32)
        self.assertEqual(data.mask.dtype, th.bool)
        self.assertEqual(tuple(data.global_features.shape), (1, 32))

        # actually usable as Planner.plan() input, not just shape-correct
        planner = Planner(graph_convention="oracle_sage")
        goal = sim.taxi.location  # a no-op move -- always legal
        projection, actions = planner.plan(data, goal)
        self.assertIsNotNone(projection)


class TestRunDepthCheckpoints(unittest.TestCase):
    def test_returns_checkpoint_history_matching_log_every(self):
        corpus = collect_graph_corpus("predictable5", episodes=1, steps_per_episode=25, seed=1, graph_convention="oracle_sage")
        vocab, checkpoints = run_depth(corpus, num_iterations=1, log_every=10, graph_convention="oracle_sage")
        self.assertEqual(len(checkpoints), 2)  # 25 graphs, log_every=10 -> checkpoints at 10, 20
        self.assertEqual([g for g, _v in checkpoints], [10, 20])
        # vocab sizes are non-decreasing (growing mode never shrinks)
        sizes = [v for _g, v in checkpoints]
        self.assertEqual(sizes, sorted(sizes))
        self.assertLessEqual(sizes[-1], len(vocab))


class TestCollectPolicyCorpus(unittest.TestCase):
    def test_includes_more_graphs_than_live_states_alone(self):
        """Corpus includes both live states AND planner projections - with
        goals_per_state > 0, the corpus must be strictly larger than
        episodes*steps_per_episode (the live-state-only count)."""
        n_states = 2 * 15
        corpus = collect_policy_corpus("predictable5", episodes=2, steps_per_episode=15, seed=2, graph_convention="oracle_sage", goals_per_state=2)
        self.assertGreater(len(corpus), n_states)

    def test_zero_goals_per_state_gives_exactly_live_states(self):
        corpus = collect_policy_corpus("predictable5", episodes=2, steps_per_episode=10, seed=2, graph_convention="oracle_sage", goals_per_state=0)
        self.assertEqual(len(corpus), 20)


class TestMeasureHeldOutOov(unittest.TestCase):
    def test_zero_oov_against_the_same_corpus_it_was_built_from(self):
        """Sanity/positive control: measuring OOV against the EXACT corpus a vocab was
        grown (then frozen) from must give 0% - every signature in it is, by
        construction, already in the vocab."""
        corpus = collect_graph_corpus("predictable5", episodes=1, steps_per_episode=30, seed=5, graph_convention="oracle_sage")
        vocab, _checkpoints = run_depth(corpus, num_iterations=1, log_every=10 ** 9, graph_convention="oracle_sage")
        freeze_vocab(vocab)
        fraction, total, oov = measure_held_out_oov(corpus, vocab, num_iterations=1, graph_convention="oracle_sage")
        self.assertEqual(oov, 0)
        self.assertEqual(fraction, 0.0)
        self.assertGreater(total, 0)

    def test_oov_fraction_in_valid_range(self):
        corpus_a = collect_graph_corpus("predictable5", episodes=1, steps_per_episode=10, seed=6, graph_convention="oracle_sage")
        vocab, _ = run_depth(corpus_a, num_iterations=2, log_every=10 ** 9, graph_convention="oracle_sage")
        freeze_vocab(vocab)
        corpus_b = collect_graph_corpus("predictable5", episodes=1, steps_per_episode=10, seed=999_999, graph_convention="oracle_sage")
        fraction, total, oov = measure_held_out_oov(corpus_b, vocab, num_iterations=2, graph_convention="oracle_sage")
        self.assertGreaterEqual(fraction, 0.0)
        self.assertLessEqual(fraction, 1.0)
        self.assertEqual(oov, int(fraction * total + 0.5))


class TestCollisionCheckBuildingBlocks(unittest.TestCase):
    def test_classify_goal_matches_planner_dispatch(self):
        sim = make_sim(1, **PREDICTABLE5)
        data = to_planner_data(sim, graph_convention="oracle_sage")
        state = graph_to_networkx(data)
        self.assertEqual(classify_goal(state, state.taxi.node), "dropoff")
        pid = next(iter(sim.passengers))
        self.assertEqual(classify_goal(state, pid), "pickup")
        other_location = next(
            n for n, d in state.graph.nodes(data=True)
            if d["type"] == "location" and n != state.taxi.location
        )
        self.assertEqual(classify_goal(state, other_location), "move")

    def test_state_key_distinguishes_different_taxi_locations(self):
        sim = make_sim(1, **PREDICTABLE5)
        data = to_planner_data(sim, graph_convention="oracle_sage")
        state = graph_to_networkx(data)
        key1 = state_key(state)

        planner = Planner(graph_convention="oracle_sage")
        other_location = next(
            n for n, d in state.graph.nodes(data=True)
            if d["type"] == "location" and n != state.taxi.location
        )
        projection, _actions = planner.plan(data, other_location)
        state2 = decode_state(projection.x, projection.edge_index, projection.edge_attr, "oracle_sage")
        key2 = state_key(state2)

        self.assertNotEqual(key1, key2)

    def test_state_key_identical_for_identical_states(self):
        sim = make_sim(1, **PREDICTABLE5)
        data1 = to_planner_data(sim, graph_convention="oracle_sage")
        data2 = to_planner_data(sim, graph_convention="oracle_sage")
        state1 = graph_to_networkx(data1)
        state2 = graph_to_networkx(data2)
        self.assertEqual(state_key(state1), state_key(state2))

    def test_wl_to_stable_partition_converges_and_is_deterministic(self):
        sim = make_sim(1, **PREDICTABLE5)
        data = to_planner_data(sim, graph_convention="oracle_sage")
        vocab1, vocab2 = {}, {}
        c1, h1 = wl_to_stable_partition(data.x, data.edge_index, data.edge_attr, vocab1, "oracle_sage", max_iterations=15)
        c2, h2 = wl_to_stable_partition(data.x, data.edge_index, data.edge_attr, vocab2, "oracle_sage", max_iterations=15)
        self.assertTrue(th.equal(c1, c2))
        self.assertTrue(th.equal(h1, h2))

    def test_hist_equal_pads_different_lengths_correctly(self):
        h1 = th.tensor([1.0, 2.0, 0.0])
        h2 = th.tensor([1.0, 2.0])
        self.assertTrue(_hist_equal(h1, h2))
        h3 = th.tensor([1.0, 3.0])
        self.assertFalse(_hist_equal(h1, h3))

    def test_select_candidates_exhaustive_returns_all_selectable(self):
        sim = make_sim(1, **PREDICTABLE5)
        data = to_planner_data(sim, graph_convention="oracle_sage")
        rng = np.random.RandomState(0)
        candidates = select_candidates(data.mask, exhaustive=True, k=3, rng=rng)
        self.assertEqual(sorted(candidates), sorted(data.mask.nonzero(as_tuple=True)[0].tolist()))

    def test_select_candidates_random_subset_respects_k(self):
        sim = make_sim(1, **PREDICTABLE5)
        data = to_planner_data(sim, graph_convention="oracle_sage")
        rng = np.random.RandomState(0)
        candidates = select_candidates(data.mask, exhaustive=False, k=3, rng=rng)
        self.assertLessEqual(len(candidates), 3)
        selectable = set(data.mask.nonzero(as_tuple=True)[0].tolist())
        self.assertTrue(set(candidates).issubset(selectable))


class TestKnownAnswerCollisionDepth(unittest.TestCase):
    """
    Known-answer test for the collision tool (Step 4 calibration item 4): a hand-built
    vilg graph with two "move" targets, A1 and B1, whose local neighbourhoods are
    identical up to a derived hop count L, and first differ at L+1.

    Construction: a hub location H with two branches, H-A1-A2 and H-B1-B2 (one road
    each), taxi sitting at H. A passenger P is placed AT A2 (in(P,A2)), destination H
    - B2 has no such passenger. A1 and B1 (one road-hop into each branch) are
    otherwise perfectly symmetric: same object type, same immediate "adjacent"
    propositions, same distance to the shared hub H and everything attached to it
    (taxi, in this case) - the ONLY asymmetry anywhere in the graph is the passenger
    at A2, absent from B2.

    Hop-count derivation ("2 hops per road" - vilg materialises each road as its own
    `adjacent` proposition NODE, so traversing one road edge costs 2 WL-graph hops:
    location -> adjacent-proposition -> location, not 1):
      - The passenger's distinguishing feature is its OWN `in(P, A2)` proposition
        node, which is not itself on the road chain - reaching it from A2 costs one
        MORE hop beyond reaching A2 itself.
      - A2 is 1 road-hop from A1 = 2 WL-hops (via the adjacent(A1,A2) proposition).
      - Reaching in(P,A2) from A1 therefore costs 2 (to A2) + 1 (to the passenger's
        own proposition) = 3 WL-hops.
      - So A1's colour first depends on the passenger's presence only once
        refinement has run enough rounds to reach 3 hops out - i.e. at L=3 refine()
        calls (L=3 total iterations). At L=2 (fewer iterations than the distance),
        A1 and B1 must still collide; at L=3, they must separate.
    This derivation was verified empirically (not just asserted): the construction
    below collides at L=0,1,2 and separates at L=3, exactly as derived.
    """

    LOCATION = [1, 0, 0]
    TAXI = [0, 1, 0]
    PASSENGER = [0, 0, 1]
    PRED_ADJACENT = [1, 0, 0]
    PRED_IN = [0, 1, 0]
    PRED_DESTINATION = [0, 0, 1]
    NONGOAL = [0, 0, 1]
    UNACHIEVED_GOAL = [0, 1, 0]

    # node ids: 0=T(taxi) 1=H 2=A1 3=A2 4=B1 5=B2 6=P(passenger)
    A1, B1 = 2, 4

    @classmethod
    def _obj_row(cls, attr):
        return attr + [0, 0, 0, 0, 0, 0]

    @classmethod
    def _prop_row(cls, pred, status):
        return [0, 0, 0] + pred + status

    def _build_graph(self):
        nodes = [
            self._obj_row(self.TAXI),        # 0 T
            self._obj_row(self.LOCATION),     # 1 H
            self._obj_row(self.LOCATION),     # 2 A1
            self._obj_row(self.LOCATION),     # 3 A2
            self._obj_row(self.LOCATION),     # 4 B1
            self._obj_row(self.LOCATION),     # 5 B2
            self._obj_row(self.PASSENGER),    # 6 P
            self._prop_row(self.PRED_ADJACENT, self.NONGOAL),      # 7  adjacent(H,A1)
            self._prop_row(self.PRED_ADJACENT, self.NONGOAL),      # 8  adjacent(A1,A2)
            self._prop_row(self.PRED_ADJACENT, self.NONGOAL),      # 9  adjacent(H,B1)
            self._prop_row(self.PRED_ADJACENT, self.NONGOAL),      # 10 adjacent(B1,B2)
            self._prop_row(self.PRED_IN, self.NONGOAL),            # 11 in(T,H)
            self._prop_row(self.PRED_IN, self.NONGOAL),            # 12 in(P,A2) -- the ONLY asymmetric node
            self._prop_row(self.PRED_DESTINATION, self.UNACHIEVED_GOAL),  # 13 destination(P,H)
        ]
        # (prop_idx, u, v) -- each expands to the 4 edges env_to_vilg_graph itself
        # produces per proposition: prop->u (pos1), prop->v (pos2), u->prop (pos1,
        # same label as its forward counterpart), v->prop (pos2, ditto).
        props = [
            (7, 1, 2), (8, 2, 3), (9, 1, 4), (10, 4, 5),
            (11, 0, 1), (12, 6, 3), (13, 6, 1),
        ]
        edges, edge_attrs = [], []
        for prop_idx, u, v in props:
            edges += [(prop_idx, u), (prop_idx, v), (u, prop_idx), (v, prop_idx)]
            edge_attrs += [[1, 0], [0, 1], [1, 0], [0, 1]]

        x = th.tensor(nodes, dtype=th.float)
        edge_index = th.tensor(edges, dtype=th.long).T
        edge_attr = th.tensor(edge_attrs, dtype=th.float)
        return x, edge_index, edge_attr

    def test_collides_at_derived_L_and_separates_at_L_plus_1(self):
        x, edge_index, edge_attr = self._build_graph()
        self.assertEqual(x.shape[0], 14)
        self.assertEqual(edge_index.shape[1], 28)

        from sage.domains.utils.wl_colours import wl_colours

        for L in [0, 1, 2]:
            colours, _hist = wl_colours(x, edge_index, edge_attr, num_iterations=L, vocab={}, frozen=False, graph_convention="vilg")
            self.assertEqual(
                colours[self.A1].item(), colours[self.B1].item(),
                f"A1/B1 expected to COLLIDE at L={L} (derived: distinguishing feature is 3 hops away)",
            )

        colours, _hist = wl_colours(x, edge_index, edge_attr, num_iterations=3, vocab={}, frozen=False, graph_convention="vilg")
        self.assertNotEqual(
            colours[self.A1].item(), colours[self.B1].item(),
            "A1/B1 expected to SEPARATE at L=3 (derived hop distance to the passenger's own proposition)",
        )

    def test_wl_to_stable_partition_also_separates_them(self):
        """The floor (stable-partition) measurement must agree with the fixed-L
        result: since the graph is finite and the distinguishing feature IS
        eventually reachable, A1/B1 must be separated by the time refinement
        converges, not just at some intermediate L."""
        x, edge_index, edge_attr = self._build_graph()
        vocab = {}
        colours, _hist = wl_to_stable_partition(x, edge_index, edge_attr, vocab, "vilg", max_iterations=15)
        self.assertNotEqual(colours[self.A1].item(), colours[self.B1].item())


if __name__ == "__main__":
    unittest.main()
