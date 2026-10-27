"""
Tests for --clamp-mode (sage/agent/graph_policy.py: masked_segmented_softmax / relative_clamp).

  TestAbsoluteUnchanged    absolute is the original function, bit for bit (values and gradients)
  TestRelativeInRange      relative == absolute wherever every valid logit is within (-30, 30):
                           pi and d(loss)/d(logits), on raw tensors and through the real
                           policy's forward and evaluate_actions
  TestRelativeBelowClamp   all valid logits below -30: absolute is uniform with zero gradient,
                           relative is not
  TestRelativeCuda         the scatter-max path on CUDA matches CPU (run on RCP)
  TestWiring               the flag reaches the policy kwargs (and the zip) only when relative;
                           the diagnostics' float64 recompute follows the mode

Run from the repo root with:
    PYTHONPATH=. python -m pytest tests/test_clamp_mode.py -v
Do not run while a training run is using the GPU.
"""
import os
import shutil
import tempfile
import unittest
from unittest import mock

import gym as gym_module
import numpy as np
import torch as th
from torch_geometric.utils import softmax

# test_ternary_planner installs the gym-0.26 randint shim at import time.
from tests.test_ternary_planner import POLICY_KWARGS
from tests.test_diag_occupancy import CELL1_ARGV, N_STEPS, diag_csvs, run_gnn_global

import analysis.inspect_logits as il
import sage.experiments.gnn_global as gnn_global
from sage.agent.async_vec_env import AsyncVecEnv
from sage.agent.graph_feedback_policy import GNNFeedbackPolicy
from sage.agent.graph_plan_feedback_policy import GNNPlanFeedbackPolicy
from sage.agent.graph_policy import RELATIVE_CLAMP_WINDOW, masked_segmented_softmax
from sage.agent.plan_feedback_a2c import PlanFeedback_A2C
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.utils.compat import spec_kwargs
from sage.domains.utils import spaces as sage_spaces
from sage.forks.stable_baselines3.stable_baselines3.common.env_util import make_vec_env

CUDA = th.cuda.is_available()


def original_masked_segmented_softmax(energies, mask, batch_ind):
    """graph_policy.masked_segmented_softmax before --clamp-mode existed, copied verbatim."""
    energies = energies.clamp(-30,30)
    energies[~mask.bool()] = -np.inf
    probs = softmax(energies, batch_ind)

    return probs.flatten()


def random_batch(sizes, low, high, seed, device="cpu", invalid_every=4):
    """Logits uniform in [low, high] for graphs of the given sizes; every invalid_every-th node
    masked out (never a graph's first node, so each graph keeps a valid one)."""
    g = th.Generator().manual_seed(seed)
    n = sum(sizes)
    logits = (low + (high - low) * th.rand(n, generator=g, dtype=th.float32)).to(device)
    batch = th.cat([th.full((k,), i, dtype=th.long) for i, k in enumerate(sizes)]).to(device)
    mask = th.ones(n, dtype=th.bool)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    for s, k in zip(starts, sizes):
        mask[s + 1:s + k:invalid_every] = False
    return logits, mask.to(device), batch


def probs_and_grad(fn, logits, mask, batch, weights, **kwargs):
    """pi and the gradient of sum(weights * log pi over valid nodes) w.r.t. the raw logits."""
    x = logits.clone().requires_grad_(True)
    p = fn(x, mask, batch, **kwargs)
    (weights[mask] * th.log(p[mask])).sum().backward()
    return p.detach(), x.grad.detach()


class TestAbsoluteUnchanged(unittest.TestCase):
    def test_bitwise_equal_to_original(self):
        for seed, (low, high) in enumerate([(-80.0, 80.0), (-29.0, 29.0), (-90.0, -31.0), (31.0, 90.0)]):
            logits, mask, batch = random_batch([7, 1, 23, 400], low, high, seed)
            weights = th.randn(logits.shape[0], generator=th.Generator().manual_seed(100 + seed))
            p0, g0 = probs_and_grad(original_masked_segmented_softmax, logits, mask, batch, weights)
            p1, g1 = probs_and_grad(masked_segmented_softmax, logits, mask, batch, weights)
            p2, g2 = probs_and_grad(masked_segmented_softmax, logits, mask, batch, weights, clamp_mode="absolute")
            for p, g in ((p1, g1), (p2, g2)):
                self.assertTrue(th.equal(p, p0) and th.equal(g, g0), f"range [{low}, {high}]")

    def test_unknown_mode_raises(self):
        logits, mask, batch = random_batch([3], -1.0, 1.0, 0)
        with self.assertRaises(ValueError):
            masked_segmented_softmax(logits, mask, batch, clamp_mode="relativ")


class TestRelativeInRange(unittest.TestCase):
    def test_tensors(self):
        """Valid logits spanning nearly the whole (-30, 30) -- wider than 30, so a [-30, 0]
        relative window would NOT match; the [-60, 0] one does."""
        for seed in range(5):
            logits, mask, batch = random_batch([5, 1, 40, 417], -29.99, 29.99, seed)
            weights = th.randn(logits.shape[0], generator=th.Generator().manual_seed(200 + seed))
            pa, ga = probs_and_grad(masked_segmented_softmax, logits, mask, batch, weights, clamp_mode="absolute")
            pr, gr = probs_and_grad(masked_segmented_softmax, logits, mask, batch, weights, clamp_mode="relative")
            self.assertTrue(th.allclose(pr, pa, rtol=0, atol=1e-6), (pr - pa).abs().max())
            self.assertTrue(th.allclose(gr, ga, rtol=0, atol=1e-6), (gr - ga).abs().max())
            self.assertTrue(th.equal(pr[~mask], th.zeros_like(pr[~mask])))
        self.assertEqual(RELATIVE_CLAMP_WINDOW, 60)

    def test_real_policy_forward_and_evaluate_actions(self):
        """The same weights built with clamp_mode absolute and relative: identical pi in the
        collection forward's stash and identical log_prob/entropy from evaluate_actions (whose
        gradients the loss uses), on real ternary states (a fresh policy's logits are small)."""
        env = il.make_env()
        states = [s["obs"] for s in il.load_or_build_states()["states"][:6]]
        obs = np.array([[s] for s in states])
        th.manual_seed(0)
        absolute = il.build_policy(env, "absolute")
        relative = il.build_policy(env, "relative")
        relative.load_state_dict(absolute.state_dict())
        self.assertEqual((absolute.clamp_mode, relative.clamp_mode), ("absolute", "relative"))
        out = {}
        for name, policy in (("abs", absolute), ("rel", relative)):
            policy.diag_stash = {}
            with th.no_grad():
                policy.forward(obs)
            out[name + "_p"] = policy.diag_stash["probs"]
            out[name + "_logits"] = policy.diag_stash["logits"]
            policy.diag_stash = None
            actions = th.tensor([0, 1, 2, 3, 401, 402])
            _values, log_prob, ent, _pv = policy.evaluate_actions(obs, actions)
            policy.zero_grad()
            (log_prob.sum() + ent).backward()
            out[name + "_lp"], out[name + "_ent"] = log_prob.detach(), ent.detach()
            out[name + "_grad"] = policy.action_net.weight.grad.clone()
        self.assertLess(out["abs_logits"].abs().max().item(), 30.0)
        self.assertTrue(th.allclose(out["rel_p"], out["abs_p"], rtol=0, atol=1e-6))
        self.assertTrue(th.allclose(out["rel_lp"], out["abs_lp"], rtol=0, atol=1e-5))
        self.assertTrue(th.allclose(out["rel_ent"], out["abs_ent"], rtol=0, atol=1e-5))
        self.assertTrue(th.allclose(out["rel_grad"], out["abs_grad"], rtol=1e-4, atol=1e-6))


class TestRelativeBelowClamp(unittest.TestCase):
    def test_all_valid_logits_below_minus_30(self):
        logits = th.tensor([-35.0, -40.0, -50.0, -31.0, -33.0, -60.0])
        mask = th.tensor([True, True, True, True, True, False])
        batch = th.zeros(6, dtype=th.long)
        weights = th.tensor([1.0, -2.0, 0.5, 3.0, -1.0, 0.0])
        pa, ga = probs_and_grad(masked_segmented_softmax, logits, mask, batch, weights, clamp_mode="absolute")
        pr, gr = probs_and_grad(masked_segmented_softmax, logits, mask, batch, weights, clamp_mode="relative")
        # absolute: every valid logit clamps to -30 -> uniform, and no gradient reaches any logit
        self.assertTrue(th.allclose(pa[mask], th.full((5,), 0.2)))
        self.assertTrue(th.equal(ga, th.zeros_like(ga)))
        # relative: the original ordering and ratios survive, and the gradient flows
        expected = th.softmax(logits[mask], 0)
        self.assertTrue(th.allclose(pr[mask], expected, rtol=1e-5, atol=1e-7))
        self.assertGreater(pr[mask].max().item() - pr[mask].min().item(), 0.5)
        self.assertGreater(gr[mask].abs().min().item(), 0.0)
        self.assertEqual(gr[5].item(), 0.0)

    def test_top_node_above_plus_30_keeps_its_gradient(self):
        # only the top node is above +30, and nothing saturates (pi_top well below 1 in float32)
        logits = th.tensor([31.0, 29.0, 28.0])
        mask = th.ones(3, dtype=th.bool)
        batch = th.zeros(3, dtype=th.long)
        weights = th.tensor([1.0, 0.0, 0.0])
        _pa, ga = probs_and_grad(masked_segmented_softmax, logits, mask, batch, weights, clamp_mode="absolute")
        _pr, gr = probs_and_grad(masked_segmented_softmax, logits, mask, batch, weights, clamp_mode="relative")
        self.assertEqual(ga[0].item(), 0.0)  # frozen at the +30 cap
        self.assertNotEqual(gr[0].item(), 0.0)

    def test_window_floor(self):
        """More than 60 below the top: clamped to top - 60 in relative mode."""
        logits = th.tensor([0.0, -100.0])
        p = masked_segmented_softmax(logits, th.ones(2, dtype=th.bool), th.zeros(2, dtype=th.long), clamp_mode="relative")
        self.assertAlmostEqual(p[1].item() / p[0].item(), float(np.exp(-60.0)), delta=1e-30)


@unittest.skipUnless(CUDA, "CUDA not available (run on RCP)")
class TestRelativeCuda(unittest.TestCase):
    def test_scatter_max_path_matches_cpu(self):
        for seed, (low, high) in enumerate([(-80.0, 80.0), (-90.0, -31.0), (-29.0, 29.0)]):
            logits, mask, batch = random_batch([7, 1, 23, 417, 3], low, high, seed)
            weights = th.randn(logits.shape[0], generator=th.Generator().manual_seed(300 + seed))
            pc, gc = probs_and_grad(masked_segmented_softmax, logits, mask, batch, weights, clamp_mode="relative")
            pg, gg = probs_and_grad(masked_segmented_softmax, logits.cuda(), mask.cuda(), batch.cuda(), weights.cuda(),
                                    clamp_mode="relative")
            self.assertTrue(th.allclose(pg.cpu(), pc, rtol=1e-5, atol=1e-7), f"probs [{low}, {high}]")
            self.assertTrue(th.allclose(gg.cpu(), gc, rtol=1e-4, atol=1e-6), f"grads [{low}, {high}]")


class TestWiring(unittest.TestCase):
    def _variant(self, argv):
        with mock.patch.object(gnn_global, "run") as run:
            gnn_global.main(argv)
        return run.call_args[0][0]

    def test_policy_kwarg_only_when_relative(self):
        self.assertNotIn("clamp_mode", self._variant(CELL1_ARGV)["algorithm_kwargs"]["policy_kwargs"])
        self.assertNotIn("clamp_mode", self._variant(CELL1_ARGV + ["--clamp-mode", "absolute"])["algorithm_kwargs"]["policy_kwargs"])
        self.assertEqual(self._variant(CELL1_ARGV + ["--clamp-mode", "relative"])["algorithm_kwargs"]["policy_kwargs"]["clamp_mode"],
                         "relative")
        with mock.patch.object(gnn_global, "run") as run:
            with self.assertRaises(ValueError):
                gnn_global.main([a for a in CELL1_ARGV if a != "--feedback"] + ["--clamp-mode", "relative"])
            run.assert_not_called()

    def test_policy_rejects_unknown_mode(self):
        env = il.make_env()
        with self.assertRaises(ValueError):
            GNNFeedbackPolicy(env.observation_space, env.action_space, lambda _: 1e-3,
                              **dict(il.CELL1_POLICY_KWARGS, features_extractor_kwargs={"gnn_steps": 5}, clamp_mode="max"))

    def test_saved_in_zip_and_loaded_back(self):
        d = tempfile.mkdtemp()
        try:
            env = make_vec_env(lambda: GraphTaxiEnv(**spec_kwargs(gym_module.spec(il.ENV_ID))), n_envs=1, seed=0,
                               vec_env_cls=AsyncVecEnv)
            model = PlanFeedback_A2C(
                GNNPlanFeedbackPolicy, env, verbose=0, device="cpu",
                supported_action_spaces=(sage_spaces.BinaryAction, gym_module.spaces.Discrete, sage_spaces.Autoregressive),
                policy_kwargs=dict(il.CELL1_POLICY_KWARGS, features_extractor_kwargs={"gnn_steps": 5}, clamp_mode="relative"),
            )
            env.close()
            self.assertEqual(model.policy.clamp_mode, "relative")
            model.save(os.path.join(d, "m"))
            policy, data = il.load_policy(os.path.join(d, "m.zip"))
            self.assertEqual(il.saved_policy_kwargs(data)["clamp_mode"], "relative")
            self.assertEqual(policy.clamp_mode, "relative")
            self.assertEqual(il.provenance(data)["kwargs_mismatch"], "")
        finally:
            shutil.rmtree(d)

    def test_diagnostics_follow_relative_mode(self):
        """A short gnn_global run with --clamp-mode relative --diag-occupancy: the diagnostics'
        float64 entropy (recomputed under the policy's mode) matches train/entropy per update."""
        workdir = tempfile.mkdtemp()
        threads = th.get_num_threads()
        th.set_num_threads(1)
        try:
            out = run_gnn_global(CELL1_ARGV + ["--num-processes", "2", "--log-interval", "5", "--num-env-steps", "60",
                                               "--clamp-mode", "relative", "--diag-occupancy"], workdir)
            self.assertEqual(out["model"].policy.clamp_mode, "relative")
            entropies = [v for k, v in out["records"] if k == "train/entropy"]
            rows = diag_csvs(workdir)["decisions"]
            for u, train_entropy in enumerate(entropies):
                mine = np.mean([float(r["entropy"]) for r in rows if int(r["update_idx"]) == u])
                self.assertAlmostEqual(mine, train_entropy, delta=1e-5 * abs(train_entropy))
        finally:
            th.set_num_threads(threads)
            shutil.rmtree(workdir, ignore_errors=True)


class TestInspectLogitsMirror(unittest.TestCase):
    def test_float64_log_policy_matches_torch(self):
        for mode in ("absolute", "relative"):
            for seed, (low, high) in enumerate([(-80.0, 80.0), (-90.0, -31.0), (-20.0, 20.0)]):
                logits, mask, batch = random_batch([300], low, high, seed)
                p = masked_segmented_softmax(logits.clone(), mask, batch, clamp_mode=mode)
                mirror = np.exp(il.log_policy(logits.double().numpy(), mask.numpy(), mode))
                self.assertTrue(np.allclose(mirror, p.double().numpy(), rtol=1e-5, atol=1e-9), (mode, low, high))


if __name__ == "__main__":
    unittest.main()
