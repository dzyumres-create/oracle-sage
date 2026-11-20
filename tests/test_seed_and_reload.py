"""
F1 (paper-unified): --seed seeds the networks, and saved models reload.

Seeding: gnn_global passes --seed to the algorithm (algorithm_kwargs["seed"]), whose
_setup_model seeds python, numpy and torch before building the policy. Checked on CPU
only: PyG's scatter ops on CUDA use atomic adds, so GPU runs are repeatable only
approximately, never bit for bit. Single-threaded, as in tests/test_grad_norm_logging.py:
some multi-threaded CPU reductions sum in a run-dependent order.

Reload: sage.agent.model_io.load_policy rebuilds a saved policy (class and policy_kwargs
from the zip, spaces from a live env) and loads its weights and optimizer state; the
reloaded policy gives the same outputs as the one that was saved.

Run from the repo root (CPU; not while a training run uses the GPU):
    PYTHONPATH=. python -m pytest tests/test_seed_and_reload.py -v
"""
import os
import shutil
import tempfile
import unittest
from unittest import mock

import gym as gym_module
import numpy as np
import torch as th

import sage.experiments.gnn_global as gnn_global
from sage.agent.async_vec_env import AsyncVecEnv
from sage.agent.graph_plan_feedback_policy import GNNPlanFeedbackPolicy
from sage.agent.model_io import load_policy, read_zip
from sage.agent.plan_feedback_a2c import PlanFeedback_A2C
from sage.agent.wl_plan_feedback_policy import WLPlanFeedbackPolicy
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.utils.compat import spec_kwargs
from sage.domains.gym_taxi.utils.wl_vocab_cache import reset_wl_vocab_override
from sage.domains.utils import spaces as sage_spaces
from sage.forks.stable_baselines3.stable_baselines3.common.env_util import make_vec_env
from tests.test_paper_unified_base import POLICY_KWARGS, STD_ENV_ID, TERNARY_ENV_ID
from tests.test_ternary_wl_training_path import committed_vocab_path, wl_args

SUPPORTED = (sage_spaces.BinaryAction, gym_module.spaces.Discrete, sage_spaces.Autoregressive)


def env_factory(env_id, convention):
    kwargs = dict(spec_kwargs(gym_module.spec(env_id)), graph_convention=convention)
    return lambda: GraphTaxiEnv(**kwargs)


def build(env_id="city-taxi-ternary-unmasked-v1", convention="atom", seed=0, env_seed=0, wl=False, n_envs=2,
          n_steps=3):
    """A PlanFeedback_A2C as gnn_global builds it (CPU): envs seeded by make_vec_env, the
    algorithm given `seed` (None = unseeded, the pre-F1 behaviour)."""
    policy_kwargs = dict(POLICY_KWARGS)
    policy_cls = GNNPlanFeedbackPolicy
    if wl:
        gnn_global.configure_wl(wl_args(graph_convention=convention, wl_vocab_path=committed_vocab_path(convention)),
                                policy_kwargs)
        policy_cls = WLPlanFeedbackPolicy
    env = make_vec_env(env_factory(env_id, convention), n_envs=n_envs, seed=env_seed,
                       monitor_kwargs={"info_keywords": ("len100", "len200")}, vec_env_cls=AsyncVecEnv)
    model = PlanFeedback_A2C(policy_cls, env, verbose=0, device="cpu", seed=seed, n_steps=n_steps,
                             supported_action_spaces=SUPPORTED, policy_kwargs=policy_kwargs)
    return env, model


def params(model):
    return {k: v.detach().clone() for k, v in model.policy.state_dict().items()}


def same(a, b):
    return set(a) == set(b) and all(th.equal(a[k], b[k]) for k in a)


class SingleThread(unittest.TestCase):
    def setUp(self):
        self._threads = th.get_num_threads()
        th.set_num_threads(1)

    def tearDown(self):
        th.set_num_threads(self._threads)
        reset_wl_vocab_override()


class TestSeedWiring(unittest.TestCase):
    def test_main_passes_seed_to_the_algorithm(self):
        argv = ["--env-name", TERNARY_ENV_ID, "--graph-convention", "atom", "--planner", "--feedback", "--seed", "300"]
        with mock.patch.object(gnn_global, "run") as run:
            gnn_global.main(argv)
        variant = run.call_args[0][0]
        self.assertEqual(variant["seed"], 300)  # make_vec_env, as before
        self.assertEqual(variant["algorithm_kwargs"]["seed"], 300)  # the networks (F1)


class TestSeededInitialisation(SingleThread):
    def test_same_seed_same_weights_whatever_the_global_rng(self):
        for convention in ("oracle_sage", "vilg", "atom"):
            with self.subTest(convention=convention):
                th.manual_seed(11)
                env_a, a = build(convention=convention, seed=300)
                th.manual_seed(12)  # a different global state before the second build
                np.random.seed(12)
                env_b, b = build(convention=convention, seed=300)
                env_c, c = build(convention=convention, seed=600)
                self.assertTrue(same(params(a), params(b)))
                self.assertFalse(same(params(a), params(c)))
                for env in (env_a, env_b, env_c):
                    env.close()

    def test_wl_policy_seeded_too(self):
        env_a, a = build(convention="oracle_sage", seed=300, wl=True)
        th.manual_seed(5)
        env_b, b = build(convention="oracle_sage", seed=300, wl=True)
        self.assertTrue(same(params(a), params(b)))
        env_a.close()
        env_b.close()

    def test_env_streams_unchanged_by_the_algorithm_seed(self):
        """The algorithm re-seeds the envs with the seed+rank make_vec_env already used, before
        any reset, so the episodes are the same as without it (the pre-F1 behaviour)."""
        for env_id in (STD_ENV_ID, TERNARY_ENV_ID):
            with self.subTest(env=env_id):
                env_a, _a = build(env_id=env_id, convention="oracle_sage", seed=None, env_seed=7)
                env_b, _b = build(env_id=env_id, convention="oracle_sage", seed=7, env_seed=7)
                obs_a, obs_b = env_a.reset(), env_b.reset()
                self.assertEqual(obs_a.tolist(), obs_b.tolist())
                env_a.close()
                env_b.close()


class TestSameSeedSameFirstUpdate(SingleThread):
    def _learn(self, seed, convention):
        env, model = build(convention=convention, seed=seed, env_seed=seed)
        actions = []
        real_step = AsyncVecEnv.step

        def step_spy(self_env, acts, *rest):
            actions.append(np.array(acts))
            return real_step(self_env, acts, *rest)

        with mock.patch.object(AsyncVecEnv, "step", step_spy):
            model.learn(total_timesteps=model.n_envs * model.n_steps)  # one rollout, one update
        out = params(model), actions
        env.close()
        return out

    def test_first_update_repeats_on_cpu(self):
        for convention in ("oracle_sage", "atom"):
            with self.subTest(convention=convention):
                p1, acts1 = self._learn(300, convention)
                p2, acts2 = self._learn(300, convention)
                p3, _ = self._learn(600, convention)
                self.assertEqual(len(acts1), len(acts2))
                self.assertTrue(all(np.array_equal(x, y) for x, y in zip(acts1, acts2)))
                self.assertTrue(same(p1, p2), "same seed, different parameters after one update")
                self.assertFalse(same(p1, p3))


class TestSaveLoadSameOutput(SingleThread):
    """save -> model_io.load_policy -> identical weights, optimizer state and outputs (forward
    under a fixed torch seed, and evaluate_actions), for the GNN variants and ternary WL."""

    CASES = [(TERNARY_ENV_ID, c, False) for c in ("oracle_sage", "vilg", "atom")] + [
        (STD_ENV_ID, "oracle_sage", False), (TERNARY_ENV_ID, "oracle_sage", True)]

    def setUp(self):
        super().setUp()
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)
        super().tearDown()

    def test_round_trip(self):
        for env_id, convention, wl in self.CASES:
            with self.subTest(env=env_id, convention=convention, wl=wl):
                env, model = build(env_id=env_id, convention=convention, seed=1, env_seed=1, wl=wl)
                model.learn(total_timesteps=model.n_envs * model.n_steps)  # non-empty AdamW state
                path = os.path.join(self.dir, f"{convention}_{wl}")
                model.save(path)
                obs = env.reset()
                live = GraphTaxiEnv(**dict(spec_kwargs(gym_module.spec(env_id)), graph_convention=convention))

                th.manual_seed(99)  # the reloaded policy is built on top of different random weights
                loaded, data = load_policy(path + ".zip", live.observation_space, live.action_space)
                self.assertIs(type(loaded), type(model.policy))
                self.assertTrue(same(params(model), {k: v for k, v in loaded.state_dict().items()}))
                _data, _ps, saved_opt = read_zip(path + ".zip")
                loaded_opt = loaded.optimizer.state_dict()
                self.assertEqual(saved_opt["param_groups"], loaded_opt["param_groups"])
                for k, state in saved_opt["state"].items():
                    for name, value in state.items():
                        got = loaded_opt["state"][k][name]
                        self.assertTrue(th.equal(th.as_tensor(value), th.as_tensor(got)), f"optimizer {k}/{name}")

                model.policy.eval()
                outputs = []
                for policy in (model.policy, loaded):
                    th.manual_seed(5)
                    with th.no_grad():
                        actions, values, log_prob, _explored, plans = policy.forward(obs)
                        ev = policy.evaluate_actions(obs, actions)
                    outputs.append((actions, values, log_prob, [list(map(int, p)) for p in plans], ev))
                (a1, v1, l1, pl1, ev1), (a2, v2, l2, pl2, ev2) = outputs
                self.assertTrue(th.equal(a1, a2))
                self.assertTrue(th.equal(v1, v2))
                self.assertTrue(th.equal(l1, l2))
                self.assertEqual(pl1, pl2)
                for x, y in zip(ev1, ev2):
                    self.assertTrue((x is None and y is None) or th.equal(x, y))
                env.close()
                live.close()
                reset_wl_vocab_override()

    def test_override_kwargs(self):
        env, model = build(seed=1, env_seed=1)
        path = os.path.join(self.dir, "m")
        model.save(path)
        live = GraphTaxiEnv(**dict(spec_kwargs(gym_module.spec(TERNARY_ENV_ID)), graph_convention="atom"))
        loaded, _ = load_policy(path + ".zip", live.observation_space, live.action_space,
                                override_kwargs={"clamp_mode": "relative"})
        self.assertEqual(loaded.clamp_mode, "relative")
        self.assertTrue(same(params(model), dict(loaded.state_dict())))
        env.close()
        live.close()


if __name__ == "__main__":
    unittest.main()
