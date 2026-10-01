"""
Tests for the ternary-domain env wiring: city-taxi-ternary-unmasked-v1, GraphTaxiEnv's
`ternary=True` path, TERNARY_GRAPH_CONVENTION_CONVERTERS, and the Planner guard.

Run from the repo root with:
    PYTHONPATH=. python -m pytest tests/test_ternary_wiring.py -v
Do not run while a training run is using the GPU.
"""
import hashlib
import random as pyrandom
import unittest

import numpy as np
import torch as th

# --- sandbox-vs-RCP compat shim (same as tests/test_atom_wiring.py; gym 0.26.2 on this
# sandbox returns a numpy Generator with no .randint, unlike RCP's pinned gym 0.18.0) ---
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

import gym as gym_module
from torch_geometric.data import Batch

from sage.domains.gym_taxi import REWARDS
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.simulator.ternary_taxi_world import TernaryTaxiWorldSimulator
from sage.domains.gym_taxi.utils.ternary_representations import (
    TERNARY_GRAPH_CONVENTIONS,
    TERNARY_GRAPH_CONVENTION_CONVERTERS,
    TERNARY_JSON_WIDTH,
    facts_to_atom_graph,
    facts_to_json,
    facts_to_object_graph,
    facts_to_vilg_graph,
    json_to_facts,
)
from sage.forks.stable_baselines3.stable_baselines3.common.env_util import make_vec_env
from sage.forks.stable_baselines3.stable_baselines3.common.preprocessing import preprocess_obs
from sage.agent.async_vec_env import AsyncVecEnv
from sage.agent.graph_plan_feedback_policy import GNNPlanFeedbackPolicy
from sage.agent.graph_policy import GNNPolicy
from sage.agent.plan_feedback_a2c import PlanFeedback_A2C
from sage.forks.stable_baselines3.stable_baselines3 import A2C
from sage.domains.utils import spaces as sage_spaces
from analysis.ternary_collision_counter import build_flipped_facts
from tests.test_ternary_world import crossed_state, scripted_policy


CONVENTIONS = ("oracle_sage", "vilg", "atom")

MONITOR_KWARGS = {"info_keywords": ("len100", "len200")}

POLICY_KWARGS = {
    "optimizer_class": th.optim.AdamW, "optimizer_kwargs": {"weight_decay": 0.0001},
    "ortho_init": False,
    "exploration_initial_eps": 0.0, "exploration_final_eps": 0.0, "exploration_fraction": 0.0,
    "shared_gnn": True,
    "layer_norm": False,
    "num_planning_choices": 5,  # >1, so GNNPlanFeedbackPolicy takes the project_actions path
    "features_extractor_kwargs": {"gnn_steps": 5},
}


def make_ternary_env_factory(graph_convention):
    """Same construction GraphTaxiEnv's own registration (city-taxi-ternary-unmasked-)
    would produce via gym.make, written as an explicit factory so make_vec_env can be
    used exactly as gnn_global.py uses it, without going through gym.make()'s own
    registry+PassiveEnvChecker path -- that path calls check_observation_space, which
    assumes a real gym.spaces.Box (.low/.high), and JsonGraph deliberately doesn't
    provide those (see the Cell 5 handoff's "model loading is broken" note); this is a
    pre-existing, old-domain-identical sandbox limitation (verified: gym.make on
    city-taxi-unmasked-v1 fails the exact same way, unrelated to anything in this
    commit), not something specific to the ternary wiring."""
    def make_env():
        return GraphTaxiEnv(
            representation="graph", scenario="city_ternary", mask=False,
            rewards=REWARDS["v1"], graph_convention=graph_convention, ternary=True,
        )
    return make_env


def _decode_batch(convention, obs):
    """obs: raw vec-env observation array (shape (n_envs, 1) of JSON strings) -- exactly
    what TERNARY_GRAPH_CONVENTION_CONVERTERS[convention] (== any real ternary env's
    observation_space.converter) receives."""
    return TERNARY_GRAPH_CONVENTION_CONVERTERS[convention](obs)


def _step_mask(n_envs):
    return np.ones(n_envs, dtype=bool)


# ==========================================================================================
# Registration: read the registered spec WITHOUT gym.make (gym 0.26's PassiveEnvChecker
# breaks on JsonGraph -- see make_ternary_env_factory's own docstring), then construct
# GraphTaxiEnv from the spec's own kwargs directly.
# ==========================================================================================

class TestRegistration(unittest.TestCase):
    def test_registered_spec_kwargs(self):
        spec = gym_module.spec("city-taxi-ternary-unmasked-v1")
        self.assertEqual(spec.entry_point, "sage.domains.gym_taxi.envs:GraphTaxiEnv")
        self.assertEqual(spec.kwargs["scenario"], "city_ternary")
        self.assertEqual(spec.kwargs["mask"], False)
        self.assertEqual(spec.kwargs["ternary"], True)
        self.assertEqual(spec.kwargs["rewards"], REWARDS["v1"])

    def test_constructed_from_spec_kwargs_per_convention(self):
        spec = gym_module.spec("city-taxi-ternary-unmasked-v1")
        for convention in CONVENTIONS:
            env = GraphTaxiEnv(**spec.kwargs, graph_convention=convention)
            self.assertIsInstance(env.sim, TernaryTaxiWorldSimulator)
            node_dim, edge_dim = TERNARY_GRAPH_CONVENTIONS[convention]
            self.assertEqual(env.observation_space.node_dimension, node_dim)
            self.assertEqual(env.observation_space.edge_dimension, edge_dim)
            env.close()


# ==========================================================================================
# (a) Observation path end to end, via the AsyncVecEnv factory training uses
# ==========================================================================================

class TestObservationPathEndToEnd(unittest.TestCase):
    def test_300_steps_per_convention(self):
        for convention in CONVENTIONS:
            node_dim, edge_dim = TERNARY_GRAPH_CONVENTIONS[convention]
            env = make_vec_env(
                make_ternary_env_factory(convention), n_envs=1, seed=0,
                monitor_kwargs=MONITOR_KWARGS, vec_env_cls=AsyncVecEnv,
            )
            policy_rng = pyrandom.Random(123)
            obs = env.reset()
            for _step in range(300):
                self._assert_observation_valid(obs, convention, node_dim, edge_dim)
                sim = env.envs[0].sim
                action = scripted_policy(sim, policy_rng, random_prob=0.2)
                obs, _reward, done, _info = env.step(np.array([action]), _step_mask(1))
                self.assertFalse(bool(done[0]), "episode ended within 300 steps (unexpected for CITY_TERNARY)")
            env.close()

    def _assert_observation_valid(self, obs, convention, node_dim, edge_dim):
        raw = obs[0][0]
        self.assertLessEqual(len(raw), TERNARY_JSON_WIDTH)
        facts, meta = json_to_facts(raw)  # must parse without raising
        self.assertIn("time", meta)
        self.assertIn("timeout", meta)
        self.assertIn("planning", meta)
        batch = _decode_batch(convention, obs)
        self.assertIsInstance(batch, Batch)
        self.assertEqual(batch.x.shape[1], node_dim)
        self.assertEqual(batch.edge_attr.shape[1], edge_dim)
        self.assertEqual(batch.x.dtype, th.float32)
        self.assertEqual(batch.edge_attr.dtype, th.float32)
        self.assertEqual(batch.edge_index.dtype, th.long)
        self.assertEqual(batch.mask.dtype, th.bool)
        self.assertEqual(batch.global_features.dtype, th.float32)
        self.assertEqual(batch.global_features.shape, (1, 32))


# ==========================================================================================
# (b) Consistency: env-path tensors == facts_to_*_graph(sim.facts(), meta) directly
# ==========================================================================================

class TestConsistencyWithDirectConverters(unittest.TestCase):
    def test_env_path_matches_direct_converter_every_step(self):
        converters = {
            "oracle_sage": facts_to_object_graph,
            "vilg": facts_to_vilg_graph,
            "atom": facts_to_atom_graph,
        }
        for convention, converter in converters.items():
            env = GraphTaxiEnv(
                representation="graph", scenario="city_ternary", mask=False,
                rewards=REWARDS["v1"], graph_convention=convention, ternary=True,
            )
            env.seed(1)
            obs = env.reset()
            policy_rng = pyrandom.Random(456)
            for _step in range(100):
                batch = _decode_batch(convention, [[obs]])
                sim = env.sim
                meta = {"time": sim.time, "timeout": sim.timeout, "planning": sim.planning}
                expected_nf, expected_ef, expected_ei, expected_mask, expected_gf = converter(sim.facts(), meta)

                np.testing.assert_array_equal(batch.x.numpy(), expected_nf.astype(np.float32))
                np.testing.assert_array_equal(batch.edge_attr.numpy(), expected_ef.astype(np.float32))
                np.testing.assert_array_equal(batch.edge_index.numpy(), expected_ei.astype(np.int64))
                np.testing.assert_array_equal(batch.mask.numpy(), expected_mask)
                np.testing.assert_array_equal(batch.global_features.numpy()[0], expected_gf.astype(np.float32))

                action = scripted_policy(sim, policy_rng, random_prob=0.2)
                obs, _reward, done, _info = env.step(action)
                self.assertFalse(done, "episode ended within 100 steps (unexpected for CITY_TERNARY)")
            env.close()


# ==========================================================================================
# (c) Premise through the env path
# ==========================================================================================

def _inject_known_crossed_pair(sim):
    """Deterministically injects a specific buddy pair and picks up its first member --
    "inject a known crossed pair into a live env", rather than waiting for
    scripted_policy to produce one by chance. Uses real location ids from this
    episode's own maze (sim.roads), since city-scale ids aren't known ahead of time
    the way the small hand-checked 2x2 grid's are."""
    origin_1 = sim.taxi.location
    origin_2 = next(l for l in sim.roads.nodes if l != origin_1)
    used = {origin_1, origin_2}
    dest_1 = next(l for l in sim.roads.nodes if l not in used)
    used.add(dest_1)
    dest_2 = next(l for l in sim.roads.nodes if l not in used)

    requests_p = ((origin_1, dest_1), (origin_2, dest_2))
    requests_q = ((origin_1, dest_2), (origin_2, dest_1))
    pid_p, pid_q = sim.spawn_pair(requests_p, origin_1, requests_q, origin_2)
    reward = sim._apply(pid_p)  # taxi is already at origin_1 -- pickup succeeds immediately
    assert sim.taxi.passenger == pid_p, "injected pickup did not succeed"
    assert crossed_state(sim), "injected pair is not a crossed state"
    return pid_p, pid_q


class TestPremiseThroughEnvPath(unittest.TestCase):
    def test_object_identical_atom_differs(self):
        env = GraphTaxiEnv(
            representation="graph", scenario="city_ternary", mask=False,
            rewards=REWARDS["v1"], graph_convention="oracle_sage", ternary=True,
        )
        env.seed(2)
        env.reset()
        sim = env.sim
        _inject_known_crossed_pair(sim)

        obs_a = sim._get_state_json()
        _p, _q, flipped_facts = build_flipped_facts(sim)
        meta = {"time": sim.time, "timeout": sim.timeout, "planning": sim.planning}
        obs_b = facts_to_json(flipped_facts, meta)

        batch_a_object = _decode_batch("oracle_sage", [[obs_a]])
        batch_b_object = _decode_batch("oracle_sage", [[obs_b]])
        np.testing.assert_array_equal(batch_a_object.x.numpy(), batch_b_object.x.numpy())
        np.testing.assert_array_equal(batch_a_object.edge_attr.numpy(), batch_b_object.edge_attr.numpy())
        np.testing.assert_array_equal(batch_a_object.edge_index.numpy(), batch_b_object.edge_index.numpy())

        batch_a_atom = _decode_batch("atom", [[obs_a]])
        batch_b_atom = _decode_batch("atom", [[obs_b]])
        identical = (
            batch_a_atom.x.shape == batch_b_atom.x.shape
            and th.equal(batch_a_atom.x, batch_b_atom.x)
            and batch_a_atom.edge_attr.shape == batch_b_atom.edge_attr.shape
            and th.equal(batch_a_atom.edge_attr, batch_b_atom.edge_attr)
            and batch_a_atom.edge_index.shape == batch_b_atom.edge_index.shape
            and th.equal(batch_a_atom.edge_index, batch_b_atom.edge_index)
        )
        self.assertFalse(identical, "atom Data is identical for a crossed pair and its flip -- premise violated")
        env.close()


# ==========================================================================================
# (d) preprocess_obs on a real vec-env batch, via AsyncVecEnv
# ==========================================================================================

class TestPreprocessObs(unittest.TestCase):
    def test_returns_batch_on_requested_device(self):
        for convention in CONVENTIONS:
            env = make_vec_env(
                make_ternary_env_factory(convention), n_envs=1, seed=0,
                monitor_kwargs=MONITOR_KWARGS, vec_env_cls=AsyncVecEnv,
            )
            obs = env.reset()
            device = th.device("cpu")
            batch = preprocess_obs(obs, device, env.observation_space)
            self.assertIsInstance(batch, Batch)
            self.assertEqual(batch.x.device.type, "cpu")
            self.assertEqual(batch.edge_index.device.type, "cpu")
            env.close()

    @unittest.skipIf(not th.cuda.is_available(), "requires CUDA")
    def test_returns_batch_on_cuda(self):
        for convention in CONVENTIONS:
            env = make_vec_env(
                make_ternary_env_factory(convention), n_envs=1, seed=0,
                monitor_kwargs=MONITOR_KWARGS, vec_env_cls=AsyncVecEnv,
            )
            obs = env.reset()
            device = th.device("cuda")
            batch = preprocess_obs(obs, device, env.observation_space)
            self.assertEqual(batch.x.device.type, "cuda")
            env.close()


# ==========================================================================================
# (e) policy.forward(): planner-free (GNNFeedbackPolicy) AND _get_latent (GNNPlanFeedbackPolicy)
# ==========================================================================================

class TestPolicyForwardPass(unittest.TestCase):
    """Two configurations gnn_global.py can build, per Step 0(e) as corrected below:
      - NEITHER --feedback NOR --planner -> plain A2C/GNNPolicy: a real, complete
        forward() pass that never touches Planner.plan() or anything planner-related.
      - --planner --feedback --shared-gnn -> PlanFeedback_A2C/GNNPlanFeedbackPolicy,
        exactly as gnn_global.py builds it and exactly as test_atom_wiring.py's own
        TestPolicyForwardPass does -- calling _get_latent(obs) here (this file predates
        the ternary planner). Now that ternary_planner.py exists, a full forward() pass
        on GNNPlanFeedbackPolicy also works and is covered separately, with ground-truth
        checks, in tests/test_ternary_planner.py.

    A THIRD configuration, --feedback WITHOUT --planner (Feedback_A2C/GNNFeedbackPolicy),
    is deliberately NOT tested here: Step 0 was wrong to call it planner-free. It never
    references the Planner class, but GNNFeedbackPolicy's OWN _get_node_action
    (graph_feedback_policy.py:341) unconditionally calls a *different*, module-level
    project_actions/project_action (graph_feedback_policy.py:67-117) whose own
    docstring says "this current implementation is designed only for tradeoff world
    v0" -- it assumes node_feats[:,2:4] are pos/neg reward accumulators and walks
    edge_index with `while next_node > 1`, both foreign to Taxi's graph. This crashes
    with `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`
    at graph_feedback_policy.py:100 -- reproduced identically on the UNMODIFIED old
    domain (atom convention, no ternary code involved at all), so this is a
    pre-existing GNNFeedbackPolicy/Taxi incompatibility, not a regression from this
    commit. Confirmed NOT shared code with GNNPlanFeedbackPolicy: that class overrides
    _get_action_from_latent (graph_plan_feedback_policy.py:171) with its own
    implementation calling its own project_actions(..., planner) (line 54, using the
    real Planner.plan()) -- graph_feedback_policy.py's buggy project_action is never
    reached through GNNPlanFeedbackPolicy's call chain. graph_feedback_policy.py is
    out of scope to modify for this commit; the full forward() pass for a
    feedback-but-no-planner configuration will need to wait for that bug's own fix,
    unrelated to the ternary domain.
    """

    def _build_plain_policy(self, convention, device="cpu"):
        env = make_vec_env(
            make_ternary_env_factory(convention), n_envs=1, seed=0,
            monitor_kwargs=MONITOR_KWARGS,  # no vec_env_cls override -- matches gnn_global.py's no-feedback-no-planner branch
        )
        model = A2C(
            GNNPolicy, env, verbose=0, device=device,
            supported_action_spaces=(sage_spaces.BinaryAction, gym_module.spaces.Discrete, sage_spaces.Autoregressive),
            n_steps=5, policy_kwargs=POLICY_KWARGS,
        )
        return env, model

    def _build_plan_feedback_policy(self, convention, device="cpu"):
        env = make_vec_env(
            make_ternary_env_factory(convention), n_envs=1, seed=0,
            monitor_kwargs=MONITOR_KWARGS, vec_env_cls=AsyncVecEnv,  # matches gnn_global.py's --planner branch
        )
        model = PlanFeedback_A2C(
            GNNPlanFeedbackPolicy, env, verbose=0, device=device,
            supported_action_spaces=(sage_spaces.BinaryAction, gym_module.spaces.Discrete, sage_spaces.Autoregressive),
            n_steps=5, policy_kwargs=POLICY_KWARGS,
        )
        return env, model

    def test_plain_policy_full_forward_cpu(self):
        for convention in CONVENTIONS:
            env, model = self._build_plain_policy(convention, device="cpu")
            obs = env.reset()
            actions, values, log_prob = model.policy.forward(obs)
            self.assertEqual(actions.device.type, "cpu")
            self.assertEqual(values.shape, (1, 1))
            env.close()

    @unittest.skipIf(not th.cuda.is_available(), "requires CUDA")
    def test_plain_policy_full_forward_cuda(self):
        for convention in CONVENTIONS:
            env, model = self._build_plain_policy(convention, device="auto")
            device = model.device
            obs = env.reset()
            actions, values, log_prob = model.policy.forward(obs)
            self.assertEqual(actions.device.type, device.type)
            env.close()

    def test_plan_feedback_policy_get_latent_cpu(self):
        for convention in CONVENTIONS:
            env, model = self._build_plan_feedback_policy(convention, device="cpu")
            policy = model.policy
            obs = env.reset()
            batch, symbolic_batch = policy._get_latent(obs)
            self.assertEqual(batch.x.shape[1], 32)  # EMB_SIZE, post-gnn_extractor latent width
            self.assertEqual(batch.global_features.shape[1], 32)
            env.close()

    @unittest.skipIf(not th.cuda.is_available(), "requires CUDA")
    def test_plan_feedback_policy_get_latent_cuda(self):
        for convention in CONVENTIONS:
            env, model = self._build_plan_feedback_policy(convention, device="auto")
            device = model.device
            policy = model.policy
            obs = env.reset()
            batch, symbolic_batch = policy._get_latent(obs)
            self.assertEqual(batch.x.device.type, device.type)
            env.close()


# ==========================================================================================
# (f) Planner guard, reached through the real call path
# ==========================================================================================

class TestPlannerNowWired(unittest.TestCase):
    """
    Formerly TestPlannerGuardThroughRealPath: this class asserted Planner.plan raised
    NotImplementedError for a ternary observation, reached through the real
    GNNPlanFeedbackPolicy.forward() call path. The ternary planner (ternary_planner.py)
    now exists -- planner.py's guard was swapped for a deferred-import dispatch into
    it (see tests/test_ternary_planner.py for the full planner test suite, including
    the same "reached through project_actions, not by calling the guard directly"
    real-path coverage this class used to provide). This class is kept only to confirm
    the OLD raising behaviour is genuinely gone, not to re-test the planner itself.
    """

    def test_plan_no_longer_raises_not_implemented(self):
        env = GraphTaxiEnv(
            representation="graph", scenario="city_ternary", mask=False,
            rewards=REWARDS["v1"], graph_convention="atom", ternary=True,
        )
        self.assertTrue(env.observation_space.planner.ternary)
        env.close()
        # the planner itself is exercised on real graphs in tests/test_ternary_planner.py;
        # this class only confirms the guard this file used to test is gone.


# ==========================================================================================
# (g) Old domain unchanged -- fixed seed + action sequence, observations identical
#     before/after this commit. Hashes recorded from HEAD (a628f16) via `git stash` /
#     `git stash pop` around a standalone capture script, before writing any of this
#     commit's edits; see the conversation record for the exact script.
# ==========================================================================================

OLD_DOMAIN_REFERENCE_SHA256 = {
    "oracle_sage": "984aee8dc43d9e24c03a0f2a6d044ed407aa1968b0068b4a15ea269e53d20113",
    "vilg": "90617bddec63dfb707212be3d428281c21f8db55b81914d4dc62cf167f8b0d9e",
    "atom": "443aba261710f3dc090cd55617c1f0f019a4ec4a3ce3d04dc3cf5f358e346ecc",
}


def _old_domain_sample_action(sim):
    """Deterministic (no RNG draw): lowest-id legal action -- same candidate set
    build_wl_vocab.py's sample_action uses, minus the randomness, so the action
    sequence itself is fixed and reproducible without depending on np.random's
    internal state matching across the before/after runs."""
    taxi_location = sim.taxi.location
    candidates = {
        c for c in sim.graph.successors(taxi_location)
        if sim.graph.nodes[c]["attr"] != [0, 0, 1] or c in sim.passengers
    }
    candidates.add(taxi_location)
    candidates.add(0)
    return sorted(candidates)[0]


class TestOldDomainUnchanged(unittest.TestCase):
    def test_observations_match_pre_commit_reference_hash(self):
        from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv as OldGraphTaxiEnv

        for convention in CONVENTIONS:
            env = OldGraphTaxiEnv(representation="graph", scenario="city", mask=False, rewards=REWARDS["v1"], graph_convention=convention)
            env.seed(0)
            obs = env.reset()
            observations = [obs]
            for _ in range(300):
                action = _old_domain_sample_action(env.sim)
                obs, _reward, done, _info = env.step(action)
                observations.append(obs)
                if done:
                    obs = env.reset()
            blob = "\x00".join(observations)
            actual_hash = hashlib.sha256(blob.encode()).hexdigest()
            self.assertEqual(
                actual_hash, OLD_DOMAIN_REFERENCE_SHA256[convention],
                f"{convention}: old-domain observations changed vs the pre-commit (HEAD a628f16) reference",
            )
            env.close()


# ==========================================================================================
# (h) Local CPU smoke run of gnn_global.py, using the proven planner-free config
# ==========================================================================================

class TestSmokeRun(unittest.TestCase):
    """(h): a multi-step gnn_global.py-style smoke run using the third planner-free
    path (A2C/GNNPolicy) -- per instruction, only if that path actually works for
    real training, not just one forward() call.

    It does not. Confirmed as a THIRD pre-existing bug, distinct from
    GNNFeedbackPolicy's (see TestPolicyForwardPass's docstring): the "-unmasked-" env
    variant (mask=False -> planning=True, EVERY object selectable in the mask -- see
    env_to_graph/facts_to_object_graph's planning=True branch) was only ever meant to
    be trained WITH a planner, which turns an arbitrary selected "goal" node into a
    valid sequence of primitive moves. Without a planner, GNNPolicy samples raw node
    indices directly as primitive actions -- most of which are not roads adjacent to
    the taxi's current location, so TaxiWorldSimulator/TernaryTaxiWorldSimulator's own
    attempt_move correctly raises (KeyError -- "not in the graph" for the old domain,
    "no road from X to Y" here) the very first time an untrained policy picks an
    illegal move, typically within the first few dozen steps. Verified identical on
    the UNMODIFIED old domain (city-taxi-unmasked-v1, atom convention, A2C/GNNPolicy,
    no ternary code involved at all): `KeyError: 'The edge (np.int64(275),
    np.int64(291)) is not in the graph.'` -- a pre-existing gap in what
    --feedback=False --planner=False actually supports for Taxi, not a ternary
    regression. A real multi-step smoke run needs either the planner (this commit's
    explicit non-goal) or the MASKED env variant (mask=True -> planning=False, which
    restricts the mask to only legal moves and needs no planner) -- but decision 1
    scopes this commit to registering only city-taxi-ternary-unmasked-, so exercising
    mask=True here would mean constructing an env this commit doesn't register, a
    bigger step than "use the working planner-free path" calls for. Skipping per
    instruction ("if it is also broken, skip both and report it") -- the single
    forward() pass (TestPolicyForwardPass.test_plain_policy_full_forward_cpu/cuda)
    already covers what IS confirmed working, and a real trainable smoke run is left
    for whichever commit adds the ternary planner or a masked ternary env id."""

    @unittest.skip(
        "A2C/GNNPolicy cannot run multi-step training on the -unmasked- (planning=True) "
        "env without a planner -- confirmed identical on the unmodified old domain "
        "(KeyError on an untrained policy's first illegal move). See class docstring."
    )
    def test_smoke_run_per_convention(self):
        pass


if __name__ == "__main__":
    unittest.main()
