"""
Tests for --log-grad-norms (Feedback_A2C.log_grad_norms): logging-only gradient norms
around the update's clip_grad_norm_, in both train() and _train_chunked().

The default (off) path is covered by tests/test_chunked_update.py's
TestDefaultUnchanged, which checks train() against a verbatim copy of the pre-change
train() -- gradients, parameters, logged values and RNG bitwise.

Run from the repo root with:
    PYTHONPATH=. python -m pytest tests/test_grad_norm_logging.py -v -s
Do not run while a training run is using the GPU.
"""
import unittest
from unittest import mock

import gym as gym_module
import numpy as np
import torch as th

from tests.test_chunked_update import TERNARY_ENV_ID, build_model_with_rollout, run_update
from tests.test_ternary_diagnostic import printed_table_rows
from tests.test_ternary_planner import POLICY_KWARGS

import sage.experiments.gnn_global as gnn_global
from sage.agent.async_vec_env import AsyncVecEnv
from sage.agent.graph_plan_feedback_policy import GNNPlanFeedbackPolicy
from sage.agent.plan_feedback_a2c import PlanFeedback_A2C
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.utils.compat import spec_kwargs
from sage.domains.utils import spaces as sage_spaces
from sage.forks.stable_baselines3.stable_baselines3.common import logger
from sage.forks.stable_baselines3.stable_baselines3.common.env_util import make_vec_env

GRAD_NORM_KEYS = ("train/grad_norm_head", "train/grad_norm_train", "train/grad_norm_total", "train/clip_coef")


def learn_from_seed(log_grad_norms, update_chunks=1, n_envs=2, n_steps=5, updates=3):
    """A short learn() from fixed seeds (torch, numpy, env). Returns the parameters after
    it and every key logged during it."""
    th.manual_seed(0)
    np.random.seed(0)
    env = make_vec_env(
        lambda: GraphTaxiEnv(**spec_kwargs(gym_module.spec(TERNARY_ENV_ID)), graph_convention="atom"),
        n_envs=n_envs, seed=0, monitor_kwargs={"info_keywords": gnn_global.info_keywords_for(TERNARY_ENV_ID)},
        vec_env_cls=AsyncVecEnv,
    )
    model = PlanFeedback_A2C(
        GNNPlanFeedbackPolicy, env, verbose=0, device="cpu",
        update_chunks=update_chunks, log_grad_norms=log_grad_norms,
        supported_action_spaces=(sage_spaces.BinaryAction, gym_module.spaces.Discrete, sage_spaces.Autoregressive),
        n_steps=n_steps, policy_kwargs=dict(POLICY_KWARGS, exploration_fraction=0.1),
    )
    logged = []
    real_record = logger.record

    def record_spy(key, value, *args, **kwargs):
        logged.append((key, value))
        return real_record(key, value, *args, **kwargs)

    with mock.patch.object(logger, "record", side_effect=record_spy):
        model.learn(total_timesteps=n_envs * n_steps * updates)
    params = {name: p.detach().clone() for name, p in model.policy.named_parameters()}
    env.close()
    return params, logged


class TestTrainingUnchanged(unittest.TestCase):
    """Parameters after a short learn() are bitwise identical with the flag on and off.
    An off-vs-off control first shows the run itself is reproducible, so a difference
    could only come from the flag.

    Single-threaded on purpose: with several CPU threads, some reduction kernels sum in a
    run-dependent order for some tensor sizes (measured: replaying one chunked update at
    N=7 differs by ~1e-8 between runs with 4 threads, never with 1). learn() hits such
    sizes as chunk sizes change between updates, and AdamW amplifies the noise, so
    multi-threaded runs aren't bitwise reproducible even without the flag."""

    def setUp(self):
        self._threads = th.get_num_threads()
        th.set_num_threads(1)

    def tearDown(self):
        th.set_num_threads(self._threads)

    def _check(self, update_chunks):
        off_a, logged_off = learn_from_seed(False, update_chunks)
        off_b, _ = learn_from_seed(False, update_chunks)
        on, logged_on = learn_from_seed(True, update_chunks)
        for name in off_a:
            self.assertTrue(th.equal(off_a[name], off_b[name]), f"control: {name} differs between two identical runs")
        for name in off_a:
            self.assertTrue(th.equal(off_a[name], on[name]), f"{name} changed by --log-grad-norms")

        keys_on = {key for key, _value in logged_on}
        keys_off = {key for key, _value in logged_off}
        self.assertTrue(set(GRAD_NORM_KEYS) <= keys_on)
        self.assertFalse(set(GRAD_NORM_KEYS) & keys_off)
        self.assertEqual(keys_on - keys_off, set(GRAD_NORM_KEYS), "the flag must add exactly these keys")
        n_updates = sum(1 for key, _value in logged_on if key == "train/n_updates")
        self.assertEqual(sum(1 for key, _value in logged_on if key == "train/clip_coef"), n_updates,
                         "one set of norms per update")
        return logged_on

    def test_train_path(self):
        logged = self._check(update_chunks=1)
        print("\n[grad norms, train(), 3 updates] " + "  ".join(
            f"{k.split('/')[1]}={v:.3f}" for k, v in logged if k in GRAD_NORM_KEYS))

    def test_chunked_path(self):
        self._check(update_chunks=2)


class TestLoggedValues(unittest.TestCase):
    """From one snapshotted update: grad_norm_total is exactly what clip_grad_norm_
    computes, head and train partition it (path_value_net is the only policy parameter
    outside the optimizer), and clip_coef is torch's own factor."""

    @classmethod
    def setUpClass(cls):
        cls.env, cls.model, cls.snapshot = build_model_with_rollout(n_envs=4, n_steps=5)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_values_match_clip_grad_norm(self):
        model = self.model
        outside = [n for n, p in model.policy.named_parameters()
                   if id(p) not in {id(q) for g in model.policy.optimizer.param_groups for q in g["params"]}]
        self.assertEqual(sorted(outside), ["path_value_net.path_value_net.bias", "path_value_net.path_value_net.weight"])

        real_clip = th.nn.utils.clip_grad_norm_
        for n in (1, 2):
            with self.subTest(update_chunks=n):
                clip_totals = []

                def clip_spy(params, max_norm, *args, **kwargs):
                    total = real_clip(params, max_norm, *args, **kwargs)
                    clip_totals.append(float(total))
                    return total

                model.log_grad_norms = True
                try:
                    with mock.patch.object(th.nn.utils, "clip_grad_norm_", side_effect=clip_spy):
                        logged = run_update(model, self.snapshot, n)["logged"]
                finally:
                    model.log_grad_norms = False
                head, trained, total, coef = (logged[k] for k in GRAD_NORM_KEYS)
                print(f"\n[grad norms, N={n}] head {head:.4f}  train {trained:.4f}  total {total:.4f}  clip_coef {coef:.4f}")
                self.assertEqual(len(clip_totals), 1)
                self.assertAlmostEqual(total, clip_totals[0], delta=1e-6 * max(total, 1.0))
                self.assertAlmostEqual(total ** 2, head ** 2 + trained ** 2, delta=1e-5 * max(total ** 2, 1.0))
                self.assertAlmostEqual(coef, min(1.0, model.max_grad_norm / (clip_totals[0] + 1e-6)), delta=1e-7)


class TestDisplayWidth(unittest.TestCase):
    def test_keys_print_in_full_and_once_among_all_train_keys(self):
        """Every key a flag-on update logs, through the SB3 fork's printed table: the
        four new keys must each appear in full exactly once."""
        _params, logged = learn_from_seed(True, update_chunks=1, updates=1)
        rows = printed_table_rows({key: value for key, value in logged if not isinstance(value, th.Tensor)})
        for key in GRAD_NORM_KEYS:
            shown = key.split("/", 1)[1]
            self.assertEqual(sum(1 for row_key, _v in rows if row_key == shown), 1, f"{key} not printed exactly once")


class TestGnnGlobalWiring(unittest.TestCase):
    BASE = ["--env-name", TERNARY_ENV_ID, "--graph-convention", "atom", "--planner", "--feedback",
            "--shared-gnn", "--num-processes", "2", "--checkpoint-interval", "0"]

    def _variant(self, argv):
        with mock.patch.object(gnn_global, "run") as run:
            gnn_global.main(argv)
        return run.call_args[0][0]

    def test_flag_reaches_algorithm_kwargs_only_when_set(self):
        self.assertNotIn("log_grad_norms", self._variant(self.BASE)["algorithm_kwargs"])
        self.assertIs(self._variant(self.BASE + ["--log-grad-norms"])["algorithm_kwargs"]["log_grad_norms"], True)

    def test_raises_without_feedback(self):
        no_feedback = [a for a in self.BASE if a != "--feedback"]
        with mock.patch.object(gnn_global, "run") as run:
            with self.assertRaises(ValueError):
                gnn_global.main(no_feedback + ["--log-grad-norms"])
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
