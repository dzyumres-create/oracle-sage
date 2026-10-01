"""
Tests for the ternary planner (sage/domains/gym_taxi/simulator/ternary_planner.py):
decode_object/_vilg/_atom, plan_on_facts, the tie-break, and plan_ternary end to end.

Run from the repo root with:
    PYTHONPATH=. python -m pytest tests/test_ternary_planner.py -v
Do not run while a training run is using the GPU.
"""
import copy
import random as pyrandom
import time
import unittest
from collections import defaultdict

import networkx as nx

import gym.utils.seeding as _seeding

if not hasattr(__import__("numpy").random.Generator, "randint"):
    class _RandintCompatGenerator:
        def __init__(self, generator):
            self._generator = generator

        def randint(self, low, high=None):
            if hasattr(self._generator, "randint"):
                return self._generator.randint(low, high) if high is not None else self._generator.randint(low)
            return self._generator.integers(low, high)

        def __getattr__(self, name):
            return getattr(self._generator, name)

    _original_np_random = _seeding.np_random

    def _np_random_with_randint(seed=None):
        generator, seed = _original_np_random(seed)
        return _RandintCompatGenerator(generator), seed

    _seeding.np_random = _np_random_with_randint

import gym as gym_module
import numpy as np
import torch as th
from torch_geometric.data import Data

from sage.domains.gym_taxi import REWARDS
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.simulator.planner import Planner
from sage.domains.gym_taxi.simulator.ternary_planner import (
    MAX_CLUSTER_COMBINATIONS,
    _stable_hash_bytes,
    _tie_break_index,
    decode_atom,
    decode_object,
    decode_vilg,
    enumerate_object_hypotheses,
    plan_on_facts,
    plan_ternary,
)
from sage.domains.gym_taxi.simulator.ternary_taxi_world import TernaryTaxiWorldSimulator
from sage.domains.gym_taxi.utils.config import CITY_TERNARY
from sage.domains.gym_taxi.utils.ternary_representations import (
    facts_to_atom_graph,
    facts_to_json,
    facts_to_object_graph,
    facts_to_vilg_graph,
    json_to_facts,
)
from sage.forks.stable_baselines3.stable_baselines3.common.env_util import make_vec_env
from sage.agent.async_vec_env import AsyncVecEnv
from sage.agent.graph_plan_feedback_policy import GNNPlanFeedbackPolicy
from sage.agent.plan_feedback_a2c import PlanFeedback_A2C
from sage.domains.utils import spaces as sage_spaces
from analysis.ternary_collision_counter import build_flipped_facts
from tests.test_ternary_world import crossed_state, scripted_policy


PLANNING_META = {"time": 0, "timeout": 2000, "planning": True}
ENCODERS = {"oracle_sage": facts_to_object_graph, "vilg": facts_to_vilg_graph, "atom": facts_to_atom_graph}
DECODERS = {"oracle_sage": decode_object, "vilg": decode_vilg, "atom": decode_atom}
CONVENTIONS = ("oracle_sage", "vilg", "atom")


# ==========================================================================================
# Shared helpers
# ==========================================================================================

def make_ternary_sim(seed, spawning=True, **overrides):
    scenario = dict(CITY_TERNARY)
    scenario.update(overrides)
    if not spawning:
        scenario["passenger_creation_probability"] = 0
    return TernaryTaxiWorldSimulator(
        np.random.RandomState(seed), observation_fn=lambda s: None, planning=True, **scenario,
    )


def to_data(node_feats, edge_feats, edge_index, mask, global_feats):
    g = Data(
        x=th.as_tensor(node_feats, dtype=th.float32),
        edge_index=th.as_tensor(edge_index, dtype=th.long),
        edge_attr=th.as_tensor(edge_feats, dtype=th.float32),
    )
    g.mask = th.as_tensor(mask, dtype=th.bool)
    g.global_features = th.as_tensor(global_feats, dtype=th.float32).unsqueeze(0)
    return g


def make_graph(sim, convention):
    meta = {"time": sim.time, "timeout": sim.timeout, "planning": sim.planning}
    return to_data(*ENCODERS[convention](sim.facts(), meta))


def clone_data(g):
    g2 = Data(x=g.x.clone(), edge_index=g.edge_index.clone(), edge_attr=g.edge_attr.clone())
    g2.mask = g.mask.clone()
    g2.global_features = g.global_features.clone()
    return g2


def execute(sim_copy, actions):
    """Runs the real (copied) sim forward through `actions` via sim.act() directly --
    the "executed" ground truth to compare a projection against."""
    for a in actions:
        sim_copy.act(int(a))
    return sim_copy


def execute_with_rewards(sim_copy, actions):
    """Like execute(), but also returns the reward from each action -- needed to
    detect delivery success/failure directly, since checking `pid not in
    sim.passengers` afterward is UNRELIABLE: after a successful delivery,
    _renumber_passengers often reassigns the delivered passenger's old id to a
    SURVIVING passenger (e.g. their own former buddy, since buddies are exactly the
    passengers most likely to still be present) -- so the id can look "still there"
    even though the ORIGINAL passenger was in fact delivered."""
    rewards = []
    for a in actions:
        _obs, reward, _done, _info = sim_copy.act(int(a))
        rewards.append(reward)
    return sim_copy, rewards


def classify_ternary_actions(sim, actions):
    """Classifies each action as 'move' (a location), 'pickup' (a live passenger not
    currently carried), or 'dropoff' (the taxi's own node, id 0) -- read off `sim`'s
    real, un-executed facts()."""
    facts = sim.facts()
    type_of = {}
    for predicate, args in facts:
        if predicate in ("location", "taxi", "passenger"):
            type_of[args[0]] = predicate
    kinds = []
    for a in actions:
        t = type_of.get(int(a))
        if t == "location":
            kinds.append("move")
        elif t == "taxi":
            kinds.append("dropoff")
        elif t == "passenger":
            kinds.append("pickup")
        else:
            raise ValueError(f"unrecognised action target {a} (type={t})")
    return kinds


def assert_ternary_actions_equivalent(test_case, sim, actions_a, actions_b):
    """Same length; non-move actions at the same positions/ids; every move step a real
    (or no-op) road edge per sim.roads; same final location -- deliberately not exact
    list equality, matching test_atom_planner.py's own nx.shortest_path tie-breaking
    rationale."""
    actions_a = [int(a) for a in actions_a]
    actions_b = [int(a) for a in actions_b]
    test_case.assertEqual(len(actions_a), len(actions_b), "plans differ in total length")

    kinds_a = classify_ternary_actions(sim, actions_a)
    kinds_b = classify_ternary_actions(sim, actions_b)
    test_case.assertEqual(kinds_a, kinds_b, "non-move actions occur at different positions/kinds")
    for i, kind in enumerate(kinds_a):
        if kind != "move":
            test_case.assertEqual(actions_a[i], actions_b[i], f"{kind} action id differs at position {i}")

    def assert_valid_move_sequence(actions, kinds):
        prev = sim.taxi.location
        for a, kind in zip(actions, kinds):
            if kind != "move":
                continue
            is_noop = prev == a
            is_adjacent = sim.roads.has_edge(prev, a)
            test_case.assertTrue(is_noop or is_adjacent, f"move from {prev} to {a} is not a real road edge")
            prev = a
        return prev

    final_a = assert_valid_move_sequence(actions_a, kinds_a)
    final_b = assert_valid_move_sequence(actions_b, kinds_b)
    test_case.assertEqual(final_a, final_b, "plans end at different final locations")


def sample_real_states(seeds=(0, 1, 2, 3, 4), sample_every=5, per_seed_target=110, random_prob=0.2):
    states = []
    for seed in seeds:
        sim = make_ternary_sim(seed)
        policy_rng = pyrandom.Random(seed + 4000)
        step = 0
        collected = 0
        while collected < per_seed_target:
            if step % sample_every == 0:
                states.append(sim.facts())
                collected += 1
            sim.act(scripted_policy(sim, policy_rng, random_prob=random_prob))
            step += 1
            if sim.done:
                sim = make_ternary_sim(seed * 100003 + step)
    return states


def sample_real_crossed_fact_pairs(seeds=(0, 1, 2, 3, 4), sample_every=5, per_seed_target=15, random_prob=0.2):
    """Real crossed states (taxi carrying p, buddy present) -- for destination-
    correctness measurement, NOT flip pairs (build_flipped_facts is reused elsewhere
    for the premise check, but here we just need naturally-occurring crossed states
    and their carried passenger's true destination)."""
    states = []
    for seed in seeds:
        sim = make_ternary_sim(seed)
        policy_rng = pyrandom.Random(seed + 5000)
        step = 0
        collected = 0
        while collected < per_seed_target:
            if step % sample_every == 0 and crossed_state(sim):
                states.append((sim.facts(), sim.taxi.passenger))
                collected += 1
            sim.act(scripted_policy(sim, policy_rng, random_prob=random_prob))
            step += 1
            if sim.done:
                sim = make_ternary_sim(seed * 100003 + step)
    return states


# ==========================================================================================
# (a) Decoder round-trips on 500+ real states
# ==========================================================================================

class TestDecoderRoundTrips(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.states = sample_real_states(per_seed_target=110)  # 5 seeds * 110 = 550
        assert len(cls.states) >= 500

    def test_atom_and_vilg_decode_exactly(self):
        for facts in self.states:
            for convention, decode in (("vilg", decode_vilg), ("atom", decode_atom)):
                graph = to_data(*ENCODERS[convention](facts, PLANNING_META))
                decoded = decode(graph)
                self.assertEqual(decoded, facts, f"{convention}: decode(encode(facts)) != facts")

    def test_true_facts_always_among_object_hypotheses(self):
        for facts in self.states:
            graph = to_data(*facts_to_object_graph(facts, PLANNING_META))
            hypotheses = enumerate_object_hypotheses(graph)
            self.assertIn(facts, hypotheses, "true facts not among the enumerated hypotheses")

    def test_unique_hypothesis_equals_true_facts(self):
        checked_unique = 0
        for facts in self.states:
            graph = to_data(*facts_to_object_graph(facts, PLANNING_META))
            hypotheses = enumerate_object_hypotheses(graph)
            if len(hypotheses) == 1:
                self.assertEqual(hypotheses[0], facts)
                checked_unique += 1
        self.assertGreater(checked_unique, 0, "no states had a unique hypothesis -- test is vacuous")

    def test_object_encode_decode_encode_matches_encode_exactly(self):
        for facts in self.states:
            nf, ef, ei, mask, gf = facts_to_object_graph(facts, PLANNING_META)
            graph = to_data(nf, ef, ei, mask, gf)
            decoded = decode_object(graph)
            nf2, ef2, ei2, mask2, gf2 = facts_to_object_graph(decoded, PLANNING_META)
            np.testing.assert_array_equal(nf, nf2)
            np.testing.assert_array_equal(ef, ef2)
            np.testing.assert_array_equal(ei, ei2)
            np.testing.assert_array_equal(mask, mask2)
            np.testing.assert_array_equal(gf, gf2)

    def test_report_ambiguity_frequency_buddy_vs_other(self):
        """Report only -- not a strict pass/fail beyond 'no crash'. Samples fresh
        (sim, facts) pairs (not the shared class-level facts-only sample) since
        classifying ambiguity as buddy-driven needs sim.passengers[*].buddy, which
        facts() deliberately never exposes."""
        n_states = 0
        n_ambiguous = 0
        n_buddy_driven = 0
        n_other = 0
        for seed in range(30):
            sim = make_ternary_sim(seed)
            policy_rng = pyrandom.Random(seed + 6000)
            for step in range(400):
                if step % 5 == 0:
                    facts = sim.facts()
                    graph = to_data(*facts_to_object_graph(facts, PLANNING_META))
                    hypotheses = enumerate_object_hypotheses(graph)
                    n_states += 1
                    if len(hypotheses) > 1:
                        n_ambiguous += 1
                        present = set(sim.passengers.keys())
                        has_buddy_pair = any(
                            sim.passengers[p].buddy is not None and sim.passengers[p].buddy in present
                            for p in present
                        )
                        if has_buddy_pair:
                            n_buddy_driven += 1
                        else:
                            n_other += 1
                sim.act(scripted_policy(sim, policy_rng, random_prob=0.2))
                if sim.done:
                    break
        print(
            f"\n[ambiguity frequency] states={n_states} ambiguous={n_ambiguous} "
            f"({n_ambiguous / n_states:.1%}) buddy-driven={n_buddy_driven} other={n_other}"
        )


def sample_real_crossed_sim_copies(seeds=(0, 1, 2, 3, 4), sample_every=5, per_seed_target=15, random_prob=0.2):
    """Deep-copied live sims at real crossed-state moments, paired with the carried
    passenger's id -- for TestDestinationCorrectness, which needs to EXECUTE plans
    against a real simulator, not just facts."""
    samples = []
    for seed in seeds:
        sim = make_ternary_sim(seed)
        policy_rng = pyrandom.Random(seed + 5000)
        step = 0
        collected = 0
        while collected < per_seed_target:
            if step % sample_every == 0 and crossed_state(sim):
                samples.append((copy.deepcopy(sim), sim.taxi.passenger))
                collected += 1
            sim.act(scripted_policy(sim, policy_rng, random_prob=random_prob))
            step += 1
            if sim.done:
                sim = make_ternary_sim(seed * 100003 + step)
    return samples


# ==========================================================================================
# (b) Ground truth: execute every plan in a copy of the live env, compare with the
#     planner's projection. Move/no-op goals (all 3 conventions) always match exactly.
#     Delivery goals for atom/vilg (exact decode) also always match exactly. Object's
#     delivery ground truth is characterised separately below (TestDestinationCorrectness),
#     since whether it matches depends on whether the hypothesis guessed right.
# ==========================================================================================

class TestGroundTruth(unittest.TestCase):
    def _assert_projection_matches_execution(self, convention, sim, goal):
        graph = make_graph(sim, convention)
        planner = Planner(graph_convention=convention, ternary=True)
        projection, actions = planner.plan(graph, goal)

        executed = execute(copy.deepcopy(sim), actions)
        meta = {"time": executed.time, "timeout": executed.timeout, "planning": executed.planning}
        expected_nf, expected_ef, expected_ei, expected_mask, _expected_gf = ENCODERS[convention](executed.facts(), meta)

        np.testing.assert_array_equal(projection.x.numpy(), expected_nf.astype(np.float32))
        np.testing.assert_array_equal(projection.edge_attr.numpy(), expected_ef.astype(np.float32))
        np.testing.assert_array_equal(projection.edge_index.numpy(), expected_ei.astype(np.int64))
        np.testing.assert_array_equal(projection.mask.numpy(), expected_mask)
        return projection, actions, executed

    def test_move_and_noop_ground_truth_all_conventions(self):
        for convention in CONVENTIONS:
            for seed in (0, 1, 2):
                sim = make_ternary_sim(seed, spawning=False)
                target = next(iter(sim.roads.successors(sim.taxi.location)))
                with self.subTest(convention=convention, seed=seed, goal="move"):
                    self._assert_projection_matches_execution(convention, sim, target)
                with self.subTest(convention=convention, seed=seed, goal="taxi_noop"):
                    self._assert_projection_matches_execution(convention, sim, sim.taxi.node)

    def test_delivery_ground_truth_atom_and_vilg_aboard_and_not(self):
        for convention in ("vilg", "atom"):
            for seed in (0, 1, 2, 3, 4):
                sim = make_ternary_sim(seed, spawning=False)
                pid = next(iter(sim.passengers))
                with self.subTest(convention=convention, seed=seed, case="not_yet_aboard"):
                    self._assert_projection_matches_execution(convention, copy.deepcopy(sim), pid)

                sim2 = copy.deepcopy(sim)
                origin = sim2.passengers[pid].location
                for step_node in nx.shortest_path(sim2.roads, sim2.taxi.location, origin)[1:]:
                    sim2._apply(step_node)
                sim2._apply(pid)  # pickup
                self.assertEqual(sim2.taxi.passenger, pid)
                with self.subTest(convention=convention, seed=seed, case="already_aboard"):
                    self._assert_projection_matches_execution(convention, sim2, pid)


# ==========================================================================================
# (c) Destination correctness
# ==========================================================================================

class TestDestinationCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = sample_real_crossed_sim_copies(per_seed_target=15)  # 5 seeds * 15 = 75
        assert len(cls.samples) >= 50

    def test_atom_and_vilg_always_correct_destination(self):
        for convention in ("vilg", "atom"):
            for sim, pid in self.samples:
                graph = make_graph(sim, convention)
                planner = Planner(graph_convention=convention, ternary=True)
                _projection, actions = planner.plan(graph, pid)
                _executed, rewards = execute_with_rewards(copy.deepcopy(sim), actions)
                self.assertEqual(rewards[-1], sim.rewards["drop-off"], f"{convention}: passenger {pid} was not delivered")

    def test_object_correct_roughly_half_on_crossed_states(self):
        successes = 0
        total = 0
        for sim, pid in self.samples:
            graph = make_graph(sim, "oracle_sage")
            planner = Planner(graph_convention="oracle_sage", ternary=True)
            _projection, actions = planner.plan(graph, pid)
            _executed, rewards = execute_with_rewards(copy.deepcopy(sim), actions)
            total += 1
            if rewards[-1] == sim.rewards["drop-off"]:
                successes += 1
        rate = successes / total
        print(f"\n[object destination correctness] {successes}/{total} = {rate:.1%} correct on crossed states")
        self.assertGreater(rate, 0.25, "object correctness rate implausibly low for a ~50/50 hypothesis")
        self.assertLess(rate, 0.75, "object correctness rate implausibly high for a ~50/50 hypothesis")

    def test_object_matches_executed_when_correct_else_still_aboard_when_wrong(self):
        """Per-sample, not just the aggregate rate above: when the tie-break happens to
        guess the carried passenger's true destination, the projection must match the
        executed env EXACTLY (the same exact-match bar atom/vilg always clear); when it
        guesses wrong, the real env must show a failed dropoff -- the passenger still
        aboard and still present, never silently delivered to the wrong place or lost."""
        n_correct = 0
        n_wrong = 0
        for sim, pid in self.samples:
            graph = make_graph(sim, "oracle_sage")
            planner = Planner(graph_convention="oracle_sage", ternary=True)
            projection, actions = planner.plan(graph, pid)

            # Disable further spawning on the EXECUTED copy only (the planning graph
            # above is unaffected): the projection jumps straight to the post-delivery
            # state in one shot, but a multi-action plan executed for real gives
            # try_spawn_pair() several chances to fire mid-plan (act() calls it after
            # every action) -- an uncontrolled spawn would add passengers the
            # projection never anticipated and make this exact-match comparison
            # meaningless, exactly why TestGroundTruth builds its sims with
            # spawning=False in the first place.
            sim_copy = copy.deepcopy(sim)
            sim_copy.pair_creation_probability = 0
            executed, rewards = execute_with_rewards(sim_copy, actions)

            if rewards[-1] == sim.rewards["drop-off"]:
                n_correct += 1
                meta = {"time": executed.time, "timeout": executed.timeout, "planning": executed.planning}
                expected_nf, expected_ef, expected_ei, expected_mask, _gf = facts_to_object_graph(executed.facts(), meta)
                np.testing.assert_array_equal(projection.x.numpy(), expected_nf.astype(np.float32))
                np.testing.assert_array_equal(projection.edge_attr.numpy(), expected_ef.astype(np.float32))
                np.testing.assert_array_equal(projection.edge_index.numpy(), expected_ei.astype(np.int64))
                np.testing.assert_array_equal(projection.mask.numpy(), expected_mask)
            else:
                n_wrong += 1
                self.assertEqual(executed.taxi.passenger, pid, "a wrong-guess dropoff must leave the passenger still aboard")
                self.assertIn(pid, executed.passengers, "a wrong-guess dropoff must not delete the passenger")

        print(f"\n[object exact-match check] {n_correct} correct (projection==executed exactly), "
              f"{n_wrong} wrong (still aboard, still present) out of {len(self.samples)}")
        self.assertGreater(n_correct, 0, "test is vacuous without at least one correct-guess sample")
        self.assertGreater(n_wrong, 0, "test is vacuous without at least one wrong-guess sample")

    def test_object_repeated_calls_on_same_observation_agree(self):
        for sim, pid in self.samples[:20]:
            graph1 = make_graph(sim, "oracle_sage")
            graph2 = make_graph(sim, "oracle_sage")  # independent re-encode of the SAME state
            planner = Planner(graph_convention="oracle_sage", ternary=True)
            _projection1, actions1 = planner.plan(graph1, pid)
            _projection2, actions2 = planner.plan(graph2, pid)
            self.assertEqual([int(a) for a in actions1], [int(a) for a in actions2])


# ==========================================================================================
# Tie-break hash stability
# ==========================================================================================

class TestTieBreakHashStability(unittest.TestCase):
    def test_same_hash_for_two_independent_json_conversions(self):
        sim = make_ternary_sim(0)
        policy_rng = pyrandom.Random(9000)
        for _ in range(30):
            sim.act(scripted_policy(sim, policy_rng, random_prob=0.2))
        facts = sim.facts()
        meta = {"time": sim.time, "timeout": sim.timeout, "planning": sim.planning}
        json_str = facts_to_json(facts, meta)

        facts_a, meta_a = json_to_facts(json_str)
        facts_b, meta_b = json_to_facts(json_str)
        graph_a = to_data(*facts_to_object_graph(facts_a, meta_a))
        graph_b = to_data(*facts_to_object_graph(facts_b, meta_b))
        self.assertEqual(_stable_hash_bytes(graph_a), _stable_hash_bytes(graph_b))

    @unittest.skipIf(not th.cuda.is_available(), "requires CUDA")
    def test_same_hash_cpu_and_cuda_copies(self):
        sim = make_ternary_sim(1)
        graph_cpu = make_graph(sim, "oracle_sage")
        graph_cuda = clone_data(graph_cpu)
        graph_cuda.x = graph_cuda.x.to("cuda")
        graph_cuda.edge_index = graph_cuda.edge_index.to("cuda")
        graph_cuda.edge_attr = graph_cuda.edge_attr.to("cuda")
        graph_cuda.global_features = graph_cuda.global_features.to("cuda")
        self.assertEqual(_stable_hash_bytes(graph_cpu), _stable_hash_bytes(graph_cuda))


# ==========================================================================================
# (e) No leak: ternary Data carries no attributes beyond x/edge_index/edge_attr/mask/
#     global_features
# ==========================================================================================

class TestNoAttributeLeak(unittest.TestCase):
    EXPECTED_KEYS = {"x", "edge_index", "edge_attr", "mask", "global_features"}

    def test_projection_carries_no_extra_attributes(self):
        for convention in CONVENTIONS:
            sim = make_ternary_sim(0)
            graph = make_graph(sim, convention)
            planner = Planner(graph_convention=convention, ternary=True)
            target = next(iter(sim.roads.successors(sim.taxi.location)))
            projection, _actions = planner.plan(graph, target)
            self.assertEqual(set(projection.keys()), self.EXPECTED_KEYS, f"{convention}: projection has unexpected keys")

    def test_input_graph_from_real_env_has_no_extra_attributes(self):
        """What planner.plan() actually receives (post Batch.to_data_list()) -- see
        the Step 0 investigation confirming this is exactly {x, edge_index, edge_attr,
        mask, global_features} today."""
        for convention in CONVENTIONS:
            sim = make_ternary_sim(0)
            graph = make_graph(sim, convention)
            self.assertEqual(set(graph.keys()), self.EXPECTED_KEYS)


# ==========================================================================================
# (d) Real path: GNNPlanFeedbackPolicy.forward() reaching the planner through
#     project_actions, guard removed. CPU + CUDA-gated.
# ==========================================================================================

POLICY_KWARGS = {
    "optimizer_class": th.optim.AdamW, "optimizer_kwargs": {"weight_decay": 0.0001},
    "ortho_init": False,
    "exploration_initial_eps": 0.0, "exploration_final_eps": 0.0, "exploration_fraction": 0.0,
    "shared_gnn": True,
    "layer_norm": False,
    "num_planning_choices": 5,  # >1, so this takes the project_actions/gnn_extractor2 path
    "features_extractor_kwargs": {"gnn_steps": 5},
}
MONITOR_KWARGS = {"info_keywords": ("len100", "len200")}


def make_ternary_env_factory(graph_convention):
    def make_env():
        return GraphTaxiEnv(
            representation="graph", scenario="city_ternary", mask=False,
            rewards=REWARDS["v1"], graph_convention=graph_convention, ternary=True,
        )
    return make_env


class TestRealPathThroughProjectActions(unittest.TestCase):
    def _build(self, convention, device="cpu"):
        env = make_vec_env(
            make_ternary_env_factory(convention), n_envs=1, seed=0,
            monitor_kwargs=MONITOR_KWARGS, vec_env_cls=AsyncVecEnv,
        )
        model = PlanFeedback_A2C(
            GNNPlanFeedbackPolicy, env, verbose=0, device=device,
            supported_action_spaces=(sage_spaces.BinaryAction, gym_module.spaces.Discrete, sage_spaces.Autoregressive),
            n_steps=5, policy_kwargs=POLICY_KWARGS,
        )
        return env, model

    def test_forward_cpu_all_conventions(self):
        for convention in CONVENTIONS:
            env, model = self._build(convention, device="cpu")
            obs = env.reset()
            actions, values, log_prob, explored, plans = model.policy.forward(obs)
            self.assertEqual(actions.device.type, "cpu")
            self.assertEqual(values.shape, (1, 1))
            self.assertEqual(len(plans), 1)  # one plan per env
            env.close()

    @unittest.skipIf(not th.cuda.is_available(), "requires CUDA")
    def test_forward_cuda_all_conventions(self):
        for convention in CONVENTIONS:
            env, model = self._build(convention, device="auto")
            device = model.device
            obs = env.reset()
            actions, values, log_prob, explored, plans = model.policy.forward(obs)
            self.assertEqual(actions.device.type, device.type)
            env.close()


# ==========================================================================================
# (f) Local CPU smoke run with --planner --feedback --shared-gnn
# ==========================================================================================

# exploration_fraction=0 (POLICY_KWARGS above, matching test_atom_wiring.py's own
# _get_latent()-only tests) hits a PRE-EXISTING, unrelated bug the moment collect_rollouts
# actually runs: GNNFeedbackPolicy._on_step -> get_linear_fn's schedule divides by
# end_fraction (== exploration_fraction) unconditionally once progress_remaining reaches
# 1 (ZeroDivisionError) -- never exercised before since no existing test calls
# model.learn() for a GNN*Policy, only forward()/_get_latent() directly. Reproduced
# identically for the OLD domain (atom convention, no ternary code involved). Using
# gnn_global.py's own real CLI default (0.1) here, since a smoke run's whole point is to
# exercise collect_rollouts for real, matching how gnn_global.py is actually invoked.
SMOKE_POLICY_KWARGS = dict(POLICY_KWARGS, exploration_fraction=0.1)


class TestPlannerSmokeRun(unittest.TestCase):
    def test_smoke_run_per_convention(self):
        planner_call_times = []
        for convention in CONVENTIONS:
            env = make_vec_env(
                make_ternary_env_factory(convention), n_envs=1, seed=1,
                monitor_kwargs=MONITOR_KWARGS, vec_env_cls=AsyncVecEnv,
            )
            model = PlanFeedback_A2C(
                GNNPlanFeedbackPolicy, env, verbose=0, device="cpu",
                supported_action_spaces=(sage_spaces.BinaryAction, gym_module.spaces.Discrete, sage_spaces.Autoregressive),
                n_steps=5, policy_kwargs=SMOKE_POLICY_KWARGS,
            )

            planner = env.envs[0].observation_space.planner
            original_plan = planner.plan
            call_times = []

            def timed_plan(graph, goal, _original=original_plan, _times=call_times):
                t0 = time.time()
                result = _original(graph, goal)
                _times.append(time.time() - t0)
                return result

            planner.plan = timed_plan

            num_env_steps = 300
            t0 = time.time()
            model.learn(total_timesteps=num_env_steps, log_interval=1000, callback=[])
            elapsed = time.time() - t0
            env.close()

            steps_per_sec = num_env_steps / elapsed
            mean_planner_ms = (sum(call_times) / len(call_times)) * 1000 if call_times else float("nan")
            planner_call_times.extend(call_times)
            print(
                f"\n[planner smoke run] {convention}: {num_env_steps} env steps in {elapsed:.1f}s "
                f"({steps_per_sec:.1f} steps/s), {len(call_times)} planner calls, "
                f"mean {mean_planner_ms:.2f}ms/call"
            )


if __name__ == "__main__":
    unittest.main()
