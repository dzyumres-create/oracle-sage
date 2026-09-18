"""
Tests for Step 2b: wiring the "atom" graph_convention into TaxiWorldSimulator,
GraphTaxiEnv, JsonGraph, Planner and gnn_global.py's CLI. Step 1 (the pure translator:
env_to_atoms / atoms_to_graph / graph_to_atoms / env_to_atom_graph) is covered by
tests/test_atom_encoding.py; this file only tests the NEW wiring added on top of it.

Base branch: cell2-vilg-gnn -- no WL wiring, no stale-edge fix, no optimizer-timing fix.
See tests/test_atom_encoding.py's own module docstring for why the stale-edge bug is
harmless to the atom encoding by construction.

Run from the repo root with:
    python -m pytest tests/test_atom_wiring.py -v
"""
import json
import math
import random
import unittest

import numpy as np
import torch as th

# --- numpy/gym compat shims for constructing a "city" (random_walls=True) env in this
# sandbox's drifted gym/numpy -- see tests/test_taxi_world_edges.py / docs/test_vilg.py
# for the same pattern on this branch (build_wl_vocab.py, which carries this shim on
# cell4, doesn't exist yet here on cell2-vilg-gnn's base). Production code under test
# does NOT depend on this -- only this test harness does, to drive a real GraphTaxiEnv.
import gym.utils.seeding as _seeding

if not hasattr(np.random.Generator, "randint"):
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

from sage.domains.gym_taxi.simulator.taxi_world import TaxiWorldSimulator
from sage.domains.gym_taxi.simulator.planner import Planner
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv, GRAPH_CONVENTIONS, GRAPH_CONVENTION_JSON_WIDTH
from sage.domains.gym_taxi.utils.config import CITY
from sage.domains.gym_taxi.utils.representations import env_to_atoms, graph_to_atoms
from sage.domains.gym_taxi import REWARDS
from sage.domains.utils.spaces import JsonGraph
from sage.agent.async_vec_env import AsyncVecEnv
from sage.forks.stable_baselines3.stable_baselines3.common.env_util import make_vec_env

CITY_SEEDS = [0, 300, 600, 900, 1200]


def make_atom_env(seed=None):
    env = GraphTaxiEnv(representation="graph", scenario="city", mask=False, rewards=REWARDS["v1"], graph_convention="atom")
    if seed is not None:
        env.seed(seed)
    env.reset()
    return env


def sample_action(sim):
    """Same safe-random-action policy used throughout Step 0/2a's measurements: never
    crashes (only single-hop moves, own-node dropoff attempts, current passengers)."""
    taxi_location = sim.taxi.location
    candidates = {
        c for c in sim.graph.successors(taxi_location)
        if sim.graph.nodes[c]["attr"] != [0, 0, 1] or c in sim.passengers
    }
    candidates.add(taxi_location)
    candidates.add(0)
    return int(np.random.choice(list(candidates)))


class TestDropoffDispatchTakesDeleteAndResortPath(unittest.TestCase):
    """attempt_dropoff's dispatch (taxi_world.py) is `if graph_convention == "vilg": ...
    else: remove_node + resort_passengers`. "atom" was never added as a special case
    there (only _get_state_json needed a new branch), so it should already take the
    oracle_sage delete-and-resort path with zero taxi_world.py changes -- this test
    confirms that's actually true, not just assumed."""

    def test_atom_dropoff_deletes_node_like_oracle_sage(self):
        from sage.domains.gym_taxi.simulator.taxi_world import Passenger

        sim = TaxiWorldSimulator(np.random.RandomState(0), **CITY, planning=True, graph_convention="atom")
        pid = next(iter(sim.passengers))
        loc = sim.taxi.location

        # relocate the passenger to the taxi's own location (mirrors
        # test_atom_encoding.py's relocate_passenger_to), then pick up
        old_location = sim.passengers[pid].location
        destination = sim.passengers[pid].destination
        sim.graph.remove_edge(pid, old_location)
        sim.graph.remove_edge(old_location, pid)
        sim.graph.add_edge(pid, loc, attr=[0, 1, 0, 1])
        sim.graph.add_edge(loc, pid, attr=[0, 1, 0, -1])
        sim.passengers[pid] = Passenger(loc, destination)
        sim.attempt_pickup(pid)  # sets passengers[pid].location = sim.taxi.node (0) internally

        # relocate the DESTINATION to the taxi's current location (mirrors
        # relocate_destination_to), leaving .location (= taxi.node, from pickup) untouched,
        # so attempt_dropoff's `assert passenger.location == self.taxi.node` still holds
        old_destination = sim.passengers[pid].destination
        location = sim.passengers[pid].location
        sim.graph.remove_edge(pid, old_destination)
        sim.graph.remove_edge(old_destination, pid)
        sim.graph.add_edge(pid, sim.taxi.location, attr=[0, 0, 1, 1])
        sim.graph.add_edge(sim.taxi.location, pid, attr=[0, 0, 1, -1])
        sim.passengers[pid] = Passenger(location, sim.taxi.location)

        n_nodes_before = sim.graph.number_of_nodes()
        sim.attempt_dropoff(0)

        self.assertNotIn(pid, dict(sim.graph.nodes))  # node deleted, not kept alive (vilg semantics)
        self.assertEqual(sim.graph.number_of_nodes(), n_nodes_before - 1)
        self.assertNotIn(pid, sim.passengers)
        # resort_passengers keeps node ids exactly 0..n-1 -- the invariant env_to_atoms asserts
        self.assertEqual(sorted(sim.graph.nodes), list(range(sim.graph.number_of_nodes())))


class TestObsSpaceUnchangedForOracleSageAndVilg(unittest.TestCase):
    """oracle_sage and vilg's observation_space (dims, width, dtype) must be
    byte-identical to before this task -- default JsonGraph(width=250000)."""

    def test_dims_width_dtype_unchanged(self):
        expected = {
            "oracle_sage": (3, 4),
            "vilg": (9, 2),
        }
        for conv, (node_dim, edge_dim) in expected.items():
            with self.subTest(conv=conv):
                env = GraphTaxiEnv(representation="graph", scenario="predictable5", mask=False, graph_convention=conv)
                self.assertEqual(env.observation_space.node_dimension, node_dim)
                self.assertEqual(env.observation_space.edge_dimension, edge_dim)
                self.assertEqual(env.observation_space.width, 250000)
                self.assertEqual(env.observation_space.dtype, np.dtype("U250000"))

    def test_default_jsongraph_width_unchanged(self):
        # every existing caller that never passes width= (NLE, Tradeoff, or JsonGraph()
        # called directly) must still get exactly the old behaviour
        space = JsonGraph()
        self.assertEqual(space.width, 250000)
        self.assertEqual(space.dtype, np.dtype("U250000"))


class TestGraphConventionsAndWidths(unittest.TestCase):
    def test_atom_dims_and_width(self):
        self.assertEqual(GRAPH_CONVENTIONS["atom"], (6, 4))
        self.assertEqual(GRAPH_CONVENTION_JSON_WIDTH["atom"], 600000)
        self.assertEqual(GRAPH_CONVENTION_JSON_WIDTH["oracle_sage"], 250000)
        self.assertEqual(GRAPH_CONVENTION_JSON_WIDTH["vilg"], 250000)

    def test_atom_env_observation_space(self):
        env = GraphTaxiEnv(representation="graph", scenario="predictable5", mask=False, graph_convention="atom")
        self.assertEqual(env.observation_space.node_dimension, 6)
        self.assertEqual(env.observation_space.edge_dimension, 4)
        self.assertEqual(env.observation_space.width, 600000)
        self.assertEqual(env.observation_space.dtype, np.dtype("U600000"))


class TestLengthCheckRaises(unittest.TestCase):
    def test_synthetic_overlong_json_raises_value_error(self):
        env = GraphTaxiEnv(representation="graph", scenario="predictable5", mask=False, graph_convention="oracle_sage")
        env.reset()
        overlong = "x" * (env.observation_space.width + 1)
        with self.assertRaises(ValueError):
            env._check_json_length(overlong)

    def test_exactly_at_width_does_not_raise(self):
        env = GraphTaxiEnv(representation="graph", scenario="predictable5", mask=False, graph_convention="oracle_sage")
        env.reset()
        exactly = "x" * env.observation_space.width
        env._check_json_length(exactly)  # must not raise

    def test_real_reset_and_step_never_raise_for_small_scenario(self):
        for conv in ["oracle_sage", "vilg", "atom"]:
            with self.subTest(conv=conv):
                env = GraphTaxiEnv(representation="graph", scenario="predictable5", mask=False, graph_convention=conv)
                env.reset()  # would raise ValueError here if it were going to
                env.step(0)  # or here


class TestLengthCheckNeverFiresForOracleSageAndVilgOnCity(unittest.TestCase):
    """Confirms Step 2a's claim empirically, via the real env.step()/reset() path (which
    is where _check_json_length actually lives), not just by re-citing the numbers.
    Capped at 300 steps/seed rather than the full 2000 used for the Step 2a report --
    oracle_sage/vilg's JSON length is governed by graph size, which is already at its
    steady state well before 300 steps (concurrent_passengers=20 saturates fast), so
    this exercises the same length regime for a fraction of the cost."""

    def test_no_value_error_across_seeds(self):
        for conv in ["oracle_sage", "vilg"]:
            for seed in CITY_SEEDS:
                with self.subTest(conv=conv, seed=seed):
                    env = GraphTaxiEnv(representation="graph", scenario="city", mask=False, rewards=REWARDS["v1"], graph_convention=conv)
                    env.seed(seed)
                    env.reset()
                    for _ in range(300):
                        action = sample_action(env.sim)
                        _, _, done, _ = env.step(action)  # raises here if it were ever going to
                        if done:
                            env.reset()

    def test_measured_maxima_below_width(self):
        # Step 2a's measured full-2000-step-episode maxima
        measured_max = {"oracle_sage": 48373, "vilg": 123604}
        for conv, max_len in measured_max.items():
            with self.subTest(conv=conv):
                self.assertLess(max_len, GRAPH_CONVENTION_JSON_WIDTH[conv])


class TestPlannerAtomRaisesNotImplemented(unittest.TestCase):
    def test_plan_raises_before_touching_graph_to_networkx(self):
        planner = Planner(graph_convention="atom")
        with self.assertRaises(NotImplementedError) as ctx:
            planner.plan(graph=None, goal=0)  # graph=None: if this reached graph_to_networkx it would AttributeError, not NotImplementedError
        self.assertIn("atom planner: Step 3", str(ctx.exception))

    def test_env_planner_is_atom_convention(self):
        env = GraphTaxiEnv(representation="graph", scenario="predictable5", mask=False, graph_convention="atom")
        self.assertEqual(env.observation_space.planner.graph_convention, "atom")
        with self.assertRaises(NotImplementedError):
            env.observation_space.planner.plan(graph=None, goal=0)


class TestAtomFullEpisodeVerification(unittest.TestCase):
    """Full (up to timeout=2000-step) random-valid-action-policy episodes with
    graph_convention="atom", on all 5 required seeds: every observation decodes (via
    observation_space.converter, the SAME path the policy uses) to x.shape[1]==6 and
    edge_attr.shape[1]==4, and graph_to_atoms of that decoded observation equals
    env_to_atoms(env.sim) at that exact step -- independent ground truth, not derived
    from the same code path being tested."""

    def test_every_observation_decodes_correctly_and_matches_ground_truth(self):
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed):
                env = make_atom_env(seed)
                converter = env.observation_space.converter
                steps_checked = 0
                for _ in range(CITY["timeout"]):
                    obs, reward, done, info = env.step(sample_action(env.sim))
                    batch = converter([(obs,)])  # json_to_graph expects a list of (json_str,) rows

                    self.assertEqual(batch.x.shape[1], 6)
                    self.assertEqual(batch.edge_attr.shape[1], 4)

                    decoded_atoms = graph_to_atoms(
                        batch.x.cpu().numpy(), batch.edge_index.cpu().numpy(), batch.edge_attr.cpu().numpy()
                    )
                    ground_truth_atoms = env_to_atoms(env.sim)
                    self.assertEqual(decoded_atoms, ground_truth_atoms)

                    steps_checked += 1
                    if done:
                        break
                self.assertGreater(steps_checked, 0)


class TestJsonRoundTripNotTruncated(unittest.TestCase):
    """The string length surviving a real vec-env buffer round trip (DummyVecEnv/
    AsyncVecEnv._save_obs, which writes into a fixed-width U600000 numpy buffer for
    "atom") must equal the raw JSON's length -- i.e. genuinely not truncated, not just
    "under the configured width" on paper."""

    def test_vec_env_buffer_preserves_full_length(self):
        for seed in CITY_SEEDS:
            with self.subTest(seed=seed):
                def _make(seed=seed):
                    def _make_env():
                        env = GraphTaxiEnv(representation="graph", scenario="city", mask=False, rewards=REWARDS["v1"], graph_convention="atom")
                        env.seed(seed)
                        return env
                    return _make_env

                vec_env = AsyncVecEnv([_make(seed)])
                buffered_obs = vec_env.reset()  # the only reset call -- populates buf_obs
                raw_obs = vec_env.envs[0].sim._get_state_json()  # same state, recomputed directly
                buffered_str = str(buffered_obs[0][0] if not isinstance(buffered_obs, dict) else buffered_obs[None][0][0])
                self.assertEqual(len(buffered_str), len(raw_obs))
                self.assertEqual(buffered_str, raw_obs)

                max_seen = len(raw_obs)
                for _ in range(50):
                    action = sample_action(vec_env.envs[0].sim)
                    # AsyncVecEnv.step needs a per-env `mask` (which envs actually step
                    # this call) -- always True here, single env, no planner involved.
                    buffered_obs, _, buf_done, buf_info = vec_env.step(np.array([action]), np.array([True]))
                    raw_obs = vec_env.envs[0].sim._get_state_json() if not buf_done[0] else buf_info[0]["terminal_observation"]
                    if buf_done[0]:
                        break

                    buffered_str = str(buffered_obs[0][0] if not isinstance(buffered_obs, dict) else buffered_obs[None][0][0])
                    self.assertEqual(len(buffered_str), len(raw_obs))
                    self.assertEqual(buffered_str, raw_obs)
                    max_seen = max(max_seen, len(raw_obs))

                self.assertGreater(max_seen, 0)
                vec_env.close()


class TestPolicyForwardPass(unittest.TestCase):
    """Builds GNNPlanFeedbackPolicy/PlanFeedback_A2C for graph_convention="atom" (as in
    Part A4) and runs one real forward pass on a real batch of atom observations, without
    calling the planner. Confirms output shapes and that the raw per-node action logits
    (the "energies" masked_segmented_softmax operates on) are finite exactly on
    type-atom rows and -inf everywhere else."""

    def test_forward_pass_shapes_and_mask_finiteness(self):
        from sage.agent.plan_feedback_a2c import PlanFeedback_A2C
        from sage.agent.graph_plan_feedback_policy import GNNPlanFeedbackPolicy
        from sage.agent.graph_policy import make_mask, masked_segmented_softmax
        from sage.domains.utils import spaces as sage_spaces
        import gym as gym_module

        def make_env():
            return GraphTaxiEnv(representation="graph", scenario="city", mask=False, rewards=REWARDS["v1"], graph_convention="atom")

        env = make_vec_env(make_env, n_envs=1, seed=0, monitor_kwargs={"info_keywords": ("len100", "len200")}, vec_env_cls=AsyncVecEnv)

        policy_kwargs = {
            "optimizer_class": th.optim.AdamW, "optimizer_kwargs": {"weight_decay": 0.0001},
            "ortho_init": False,
            "exploration_initial_eps": 0.0, "exploration_final_eps": 0.0, "exploration_fraction": 0.0,
            "shared_gnn": True,
            "layer_norm": False,
            "num_planning_choices": 5,
            "features_extractor_kwargs": {"gnn_steps": 5},
        }
        model = PlanFeedback_A2C(
            GNNPlanFeedbackPolicy, env, verbose=0,
            supported_action_spaces=(sage_spaces.BinaryAction, gym_module.spaces.Discrete, sage_spaces.Autoregressive),
            n_steps=5, policy_kwargs=policy_kwargs,
        )
        policy = model.policy

        obs = env.reset()
        batch, symbolic_batch = policy._get_latent(obs)

        self.assertEqual(batch.x.shape[1], 32)  # EMB_SIZE, post-gnn_extractor latent width
        self.assertEqual(batch.global_features.shape[1], 32)

        x_a1 = policy.action_net(batch.x).flatten()
        self.assertEqual(x_a1.shape[0], batch.x.shape[0])

        mask, data_splits, data_starts = make_mask(batch)
        n_obj = len(env.envs[0].sim.graph.nodes)
        self.assertEqual(int(mask.sum().item()), n_obj)

        # masked_segmented_softmax's own first two lines (graph_policy.py) -- `energies =
        # energies.clamp(-30, 30)` rebinds to a NEW tensor (clamp is not in-place), so the
        # function does NOT mutate its `energies` argument; to inspect the actual
        # pre-softmax "logits" the model treats as -inf outside the mask, replicate
        # those exact two lines here rather than assuming a mutation that doesn't happen.
        energies = x_a1.clamp(-30, 30).clone()
        energies[~mask.bool()] = -np.inf
        finite = th.isfinite(energies)
        self.assertTrue(th.equal(finite, mask.bool()), "action logits are not finite exactly on type-atom rows")

        # cross-check via the actual production function's real output: masked-out rows
        # must get exactly zero probability post-softmax (not just "-inf pre-softmax")
        probs = masked_segmented_softmax(x_a1, mask, batch.batch)
        self.assertTrue(th.all(probs[~mask.bool()] == 0))
        self.assertTrue(th.all(probs[mask.bool()] > 0))

        env.close()


if __name__ == "__main__":
    unittest.main()
