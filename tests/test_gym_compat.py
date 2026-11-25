"""
gym-0.26 compatibility for training on the Macs (sage/domains/gym_taxi/utils/compat.py).

On gym < 0.26 (RCP) compat.np_random is gym's own seeding.np_random and gnn_global hands
make_vec_env the env id, exactly as before. On gym >= 0.26:
  - compat.np_random is gym 0.18's seeding algorithm, so a seed gives RCP's episodes. Checked
    against RCP itself: the atom old-domain observation stream (a complete fact list of every
    state, 300 steps) hashes to the value recorded on RCP (gym 0.18);
  - envs are built like gym 0.18's gym.make (make_from_spec): no env checker, no wrappers.

  TestSeeding        legacy delegates to gym; new path == gym 0.18 (RCP's atom hash); seeds
                     validated; repeatable
  TestMakeFromSpec   unwrapped env, spec attached with merged kwargs, kwargs override, refuses
                     max_episode_steps; vec_env_id
  TestEndToEnd       gnn_global.main trains a few updates through the real make_vec_env (no
                     test wrapper), and the startup line names the compat path

Run from the repo root (CPU):
    PYTHONPATH=. python -m pytest tests/test_gym_compat.py -v
"""
import copy
import hashlib
import os
import shutil
import tempfile
import unittest
from unittest import mock

import gym
import numpy as np
import torch as th

import sage.experiments.gnn_global as gnn_global
from sage.agent.plan_feedback_a2c import PlanFeedback_A2C
from sage.domains.gym_taxi import REWARDS
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.utils import compat
from tests.test_ternary_wiring import OLD_DOMAIN_REFERENCE_SHA256, _old_domain_sample_action

NEW_GYM = not compat.LEGACY_GYM
TERNARY_ENV_ID = "city-taxi-ternary-unmasked-v1"


def atom_old_domain_hash(seed=0, num_steps=300):
    """analysis/old_domain_obs_hash.py's stream for the atom convention, with the env's own
    seeding (no test shim)."""
    env = GraphTaxiEnv(representation="graph", scenario="city", mask=False, rewards=REWARDS["v1"],
                       graph_convention="atom")
    env.seed(seed)
    observations = [env.reset()]
    for _ in range(num_steps):
        obs, _reward, done, _info = env.step(_old_domain_sample_action(env.sim))
        observations.append(obs)
        if done:
            env.reset()
    return hashlib.sha256("\x00".join(observations).encode()).hexdigest()


class TestSeeding(unittest.TestCase):
    def test_legacy_gym_delegates_to_gym(self):
        sentinel = (np.random.RandomState(1), 5)
        with mock.patch.object(compat, "LEGACY_GYM", True), \
                mock.patch("gym.utils.seeding.np_random", return_value=sentinel) as gym_np_random:
            self.assertIs(compat.np_random(5), sentinel)
        gym_np_random.assert_called_once_with(5)

    @unittest.skipUnless(NEW_GYM, "the gym 0.18 algorithm is only used on gym >= 0.26")
    def test_matches_rcp_gym018_atom_stream(self):
        self.assertEqual(atom_old_domain_hash(), OLD_DOMAIN_REFERENCE_SHA256["gym0.18"]["atom"])

    @unittest.skipUnless(NEW_GYM, "the gym 0.18 algorithm is only used on gym >= 0.26")
    def test_randomstate_repeatable_and_validated(self):
        a, seed_a = compat.np_random(300)
        b, _ = compat.np_random(np.int64(300))
        self.assertIsInstance(a, np.random.RandomState)
        self.assertEqual(seed_a, 300)
        self.assertEqual(a.randint(0, 1000, 20).tolist(), b.randint(0, 1000, 20).tolist())
        c, _ = compat.np_random(301)
        self.assertNotEqual(compat.np_random(300)[0].randint(0, 1000, 20).tolist(), c.randint(0, 1000, 20).tolist())
        unseeded, seed = compat.np_random()
        self.assertIsInstance(unseeded, np.random.RandomState)
        self.assertIsInstance(seed, int)
        for bad in (-1, 1.5, "3"):
            with self.assertRaises(ValueError):
                compat.np_random(bad)

    def test_env_seed_uses_compat(self):
        env = GraphTaxiEnv(**compat.spec_kwargs(gym.spec(TERNARY_ENV_ID)))
        with mock.patch.object(compat, "np_random", wraps=compat.np_random) as spy:
            self.assertEqual(env.seed(7), [7])
        spy.assert_called_once_with(7)
        self.assertTrue(hasattr(env.np_random, "randint"))


class TestMakeFromSpec(unittest.TestCase):
    def test_unwrapped_env_with_spec(self):
        env = compat.make_from_spec(TERNARY_ENV_ID, graph_convention="atom")
        self.assertIs(type(env), GraphTaxiEnv)  # no checker, no OrderEnforcing, no TimeLimit
        self.assertIs(env.unwrapped, env)
        self.assertEqual(env.spec.id, TERNARY_ENV_ID)
        self.assertEqual(env.graph_convention, "atom")
        registered = compat.spec_kwargs(gym.spec(TERNARY_ENV_ID))
        self.assertEqual(compat.spec_kwargs(env.spec), dict(registered, graph_convention="atom"))
        self.assertNotIn("graph_convention", registered)  # the registry itself is untouched
        env.close()

    def test_refuses_time_limit(self):
        spec = copy.deepcopy(gym.spec(TERNARY_ENV_ID))
        spec.max_episode_steps = 100
        with mock.patch.object(gym, "spec", return_value=spec):
            with self.assertRaises(NotImplementedError):
                compat.make_from_spec(TERNARY_ENV_ID)

    def test_vec_env_id(self):
        with mock.patch.object(compat, "LEGACY_GYM", True):
            self.assertEqual(compat.vec_env_id(TERNARY_ENV_ID), TERNARY_ENV_ID)
            self.assertIn("env_compat=off", compat.compat_label())
        with mock.patch.object(compat, "LEGACY_GYM", False):
            factory = compat.vec_env_id(TERNARY_ENV_ID)
            env = factory(graph_convention="vilg")
            self.assertIs(type(env), GraphTaxiEnv)
            self.assertEqual(env.graph_convention, "vilg")
            self.assertIn("env_compat=gym018_seeding+spec_make", compat.compat_label())
            env.close()


class TestEndToEnd(unittest.TestCase):
    """gnn_global.main as on the command line: the real make_vec_env, no test wrapper, a few
    updates on CPU, and the model saved."""

    def test_short_training_run(self):
        workdir = tempfile.mkdtemp()
        threads = th.get_num_threads()
        th.set_num_threads(1)
        cwd = os.getcwd()
        try:
            os.chdir(workdir)
            argv = ["--env-name", TERNARY_ENV_ID, "--planner", "--feedback", "--shared-gnn", "--num-processes", "2",
                    "--num-steps", "3", "--num-env-steps", "18", "--log-interval", "1", "--seed", "0",
                    "--checkpoint-interval", "0", "--allow-dirty",
                    "--save-dir", os.path.join(workdir, "models"), "--log-dir", os.path.join(workdir, "logs")]
            printed = []
            real_print = print

            def print_spy(*args, **kwargs):
                printed.append(" ".join(str(a) for a in args))
                return real_print(*args, **kwargs)

            with mock.patch("builtins.print", side_effect=print_spy), \
                    mock.patch.object(gnn_global, "PlanFeedback_A2C",
                                      lambda *a, **k: PlanFeedback_A2C(*a, **dict(k, device="cpu"))):
                gnn_global.main(argv)
            self.assertTrue(os.path.exists(os.path.join(workdir, "models", "final_model.zip")))
            startup = [p for p in printed if p.startswith("startup:")]
            self.assertEqual(len(startup), 1)
            self.assertIn(compat.compat_label(), startup[0])
        finally:
            os.chdir(cwd)
            th.set_num_threads(threads)
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
