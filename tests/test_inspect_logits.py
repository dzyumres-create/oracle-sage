"""
Tests for analysis/inspect_logits.py (read-only model inspection).

Run from the repo root with:
    PYTHONPATH=. python -m pytest tests/test_inspect_logits.py -v
CPU only.
"""
import collections
import os
import pickle
import shutil
import tempfile
import unittest

import gym as gym_module
import numpy as np
import torch as th

# test_ternary_planner installs the gym-0.26 randint shim at import time.
import tests.test_ternary_planner  # noqa: F401

import analysis.inspect_logits as il
from sage.agent.async_vec_env import AsyncVecEnv
from sage.agent.graph_plan_feedback_policy import GNNPlanFeedbackPolicy
from sage.agent.plan_feedback_a2c import PlanFeedback_A2C
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.utils.compat import spec_kwargs
from sage.domains.utils import spaces as sage_spaces
from sage.forks.stable_baselines3.stable_baselines3.common.env_util import make_vec_env


def fixed_states(seeds=(11, 12)):
    """Initial decision states of a few seeded episodes, in the state-set format."""
    states = []
    for seed in seeds:
        env = il.make_env()
        env.np_random = np.random.RandomState(seed)
        obs = env.reset()
        states.append({"obs": obs, "occupied": 0, "carried": -1, "taxi_loc": int(env.sim.taxi.location),
                       "n_locations": 400, "time": 0})
    return states


class TestRoundTripLoad(unittest.TestCase):
    """A model saved the real way (PlanFeedback_A2C.save, with a non-empty AdamW state) loads
    through load_policy -- Cell 1 constructor + load_state_dict, no JsonGraph unpickling -- with
    exactly equal weights and optimizer state, and identical logits on fixed states."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_round_trip(self):
        th.manual_seed(1)
        env = make_vec_env(lambda: GraphTaxiEnv(**spec_kwargs(gym_module.spec(il.ENV_ID))), n_envs=1, seed=0,
                           vec_env_cls=AsyncVecEnv)
        model = PlanFeedback_A2C(
            GNNPlanFeedbackPolicy, env, verbose=0, device="cpu", learning_rate=il.LEARNING_RATE,
            supported_action_spaces=(sage_spaces.BinaryAction, gym_module.spaces.Discrete, sage_spaces.Autoregressive),
            policy_kwargs=dict(il.CELL1_POLICY_KWARGS, features_extractor_kwargs={"gnn_steps": 5}),
        )
        env.close()
        states = fixed_states()
        policy = model.policy
        # one AdamW step so the saved optimizer state is not empty
        batch, _symbolic = policy.extract_features(np.array([[s["obs"]] for s in states]), policy.device)
        nodes, _glob = policy.gnn_extractor(batch.x, batch.global_features, batch.edge_attr, batch.edge_index, batch.batch)
        policy.optimizer.zero_grad()
        policy.action_net(nodes).sum().backward()
        policy.optimizer.step()
        path = os.path.join(self.dir, "m")
        model.save(path)

        th.manual_seed(2)  # a differently initialised policy underneath
        loaded, data = il.load_policy(path + ".zip")
        saved = policy.state_dict()
        self.assertEqual(set(saved), set(loaded.state_dict()))
        for k, v in loaded.state_dict().items():
            self.assertTrue(th.equal(v, saved[k]), k)
        n_state = 0
        for p_saved, p_loaded in zip(policy.parameters(), loaded.parameters()):
            a, b = policy.optimizer.state.get(p_saved, {}), loaded.optimizer.state.get(p_loaded, {})
            self.assertEqual(set(a), set(b))
            for key in ("exp_avg", "exp_avg_sq"):
                if key in a:
                    self.assertTrue(th.equal(a[key], b[key]))
                    n_state += 1
        self.assertGreater(n_state, 0)
        before, after = il.forward_states(policy, states), il.forward_states(loaded, states)
        for r0, r1 in zip(before, after):
            self.assertTrue(np.array_equal(r0["logits"], r1["logits"]))
        prov = il.provenance(data)
        self.assertEqual(prov["kwargs_mismatch"], "")
        self.assertEqual(prov["episodes"], 0)


class TestClampStats(unittest.TestCase):
    def test_hand_checked(self):
        logits = np.array([31.0, 30.0, -30.0, -40.0, 0.0, 99.0])
        mask = np.array([True, True, True, True, True, False])  # 99 is invalid: ignored
        c = il.clamp_stats(logits, mask)
        # strictly outside [-30, 30] is clamped (zero gradient); exactly +-30 is not
        self.assertEqual(c, {"n_valid": 5, "above": 1, "below": 1, "all_clamped": False, "all_equal": False})

    def test_all_clamped_and_all_equal(self):
        self.assertEqual(il.clamp_stats(np.array([35.0, -31.0]), np.array([True, True])),
                         {"n_valid": 2, "above": 1, "below": 1, "all_clamped": True, "all_equal": False})
        # 40 and 50 both clamp to 30: every valid logit clamped AND the policy uniform
        self.assertEqual(il.clamp_stats(np.array([40.0, 50.0, -100.0]), np.array([True, True, False])),
                         {"n_valid": 2, "above": 2, "below": 0, "all_clamped": True, "all_equal": True})

    def test_entropy_after_clamp(self):
        # three valid nodes all above the clamp: uniform after it
        self.assertAlmostEqual(il.entropy(np.array([31.0, 45.0, 90.0]), np.array([True] * 3)), np.log(3), places=12)


class TestTopType(unittest.TestCase):
    def test_ties(self):
        types = np.array(["taxi", "location", "waiting_passenger", "waiting_passenger"], dtype=object)
        mask = np.ones(4, dtype=bool)
        self.assertEqual(il.top_type(np.array([0.0, 1.0, 5.0, 2.0]), mask, types, "absolute"), "waiting_passenger")
        # two passengers tied at the top: still a passenger
        self.assertEqual(il.top_type(np.array([0.0, 1.0, 5.0, 5.0]), mask, types, "absolute"), "waiting_passenger")
        # a passenger and a location tied
        self.assertEqual(il.top_type(np.array([0.0, 5.0, 5.0, 2.0]), mask, types, "absolute"), "mixed")
        # everything below -30: uniform under absolute, the true top under relative
        low = np.array([-40.0, -45.0, -31.0, -50.0])
        self.assertEqual(il.top_type(low, mask, types, "absolute"), "uniform")
        self.assertEqual(il.top_type(low, mask, types, "relative"), "waiting_passenger")


class TestSensitivity(unittest.TestCase):
    def test_zero_step_gives_zero_kl(self):
        th.manual_seed(3)
        policy = il.build_policy()
        states = fixed_states()
        before = {k: v.clone() for k, v in policy.state_dict().items()}
        self.assertEqual(il.sensitivity_kl(policy, states, step=0.0, n_directions=2), 0.0)
        self.assertGreater(il.sensitivity_kl(policy, states, step=il.LEARNING_RATE, n_directions=2), 0.0)
        for k, v in policy.state_dict().items():  # the inspected policy is never modified
            self.assertTrue(th.equal(v, before[k]), k)


class TestRestrictedUnpickler(unittest.TestCase):
    def test_only_allowed_globals(self):
        allowed = {("collections", "deque")}
        data = pickle.dumps(collections.deque([{"r": 1}]))
        self.assertEqual(list(il.RestrictedUnpickler(data, allowed).load()), [{"r": 1}])
        with self.assertRaises(pickle.UnpicklingError):
            il.RestrictedUnpickler(pickle.dumps(collections.OrderedDict()), allowed).load()
        with self.assertRaises(pickle.UnpicklingError):
            il.RestrictedUnpickler(pickle.dumps(os.getcwd), allowed).load()


if __name__ == "__main__":
    unittest.main()
