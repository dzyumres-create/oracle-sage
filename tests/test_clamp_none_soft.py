"""
F3 (paper-unified): --clamp-mode none and soft (sage/agent/graph_policy.py).

  none: no bound on the action logits; log pi comes from a stable log-softmax
        (masked_segmented_log_softmax), because without a clamp a probability can underflow
        to 0 in float32, where log(p) gives -inf and the entropy NaN.
  soft: 30 * tanh(x / 30), bounded like the absolute clamp but never a zero gradient.
The default (absolute) is untouched: tests/test_clamp_mode.py checks it bit for bit against
the pre-clamp-mode function, and tests/test_paper_unified_base.py against the 5547edb outputs.

  TestInRange           none == absolute wherever every valid logit is within (-30, 30) (pi
                        and gradients, raw tensors and the real policy); soft == softmax of
                        30*tanh(x/30), inside (-30, 30), close to absolute for small logits
  TestOutsideClamp      logits beyond +-30: absolute's gradient is 0 there, none's and soft's
                        are not
  TestNoNaN             a logit gap of 200: none keeps log pi, entropy and gradients finite
                        (control: log of the plain softmax is -inf / NaN); the real policy with
                        huge logits; a short learn() under none and soft stays finite
  TestDiagMirror        the diagnostics' float64 recompute follows none and soft
  TestWiring            the flag reaches the policy kwargs only when not absolute; needs --feedback
  TestCuda              none and soft on CUDA match CPU (run on RCP)

Run from the repo root (CPU; not while a training run uses the GPU):
    PYTHONPATH=. python -m pytest tests/test_clamp_none_soft.py -v
"""
import unittest
from unittest import mock

import gym as gym_module
import numpy as np
import torch as th
from torch_geometric.utils import softmax

from tests.test_clamp_mode import CUDA, random_batch
from tests.test_diag_occupancy import CELL1_ARGV

import analysis.inspect_logits as il
import sage.experiments.gnn_global as gnn_global
from sage.agent.async_vec_env import AsyncVecEnv
from sage.agent.diag_occupancy import log_policy
from sage.agent.graph_plan_feedback_policy import GNNPlanFeedbackPolicy
from sage.agent.graph_policy import SOFT_CLAMP_SCALE, masked_segmented_log_softmax, masked_segmented_softmax
from sage.agent.plan_feedback_a2c import PlanFeedback_A2C
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.utils.compat import spec_kwargs
from sage.domains.utils import spaces as sage_spaces
from sage.forks.stable_baselines3.stable_baselines3.common.env_util import make_vec_env


def log_pi(x, mask, batch, mode):
    """log pi as the policy computes it under `mode` (see GNNFeedbackPolicy._choose_node); -inf
    on invalid nodes. For the log(p) modes only valid nodes go through log, as in the policy
    (log of an invalid node's p = 0 would put 0/0 = NaN into the gradient)."""
    if mode == "none":
        return masked_segmented_log_softmax(x, mask, batch)
    p = masked_segmented_softmax(x, mask, batch, clamp_mode=mode)
    return p.new_full(p.shape, -np.inf).masked_scatter(mask, th.log(p[mask]))


def pi_and_grad(logits, mask, batch, weights, mode):
    """pi and the gradient of sum(weights * log pi over valid nodes) w.r.t. the raw logits."""
    x = logits.clone().requires_grad_(True)
    lp = log_pi(x, mask, batch, mode)
    (weights[mask] * lp[mask]).sum().backward()
    return lp.detach().exp(), x.grad.detach()


def real_states(n=6):
    env = il.make_env()
    states = [s["obs"] for s in il.load_or_build_states()["states"][:n]]
    return env, np.array([[s] for s in states])


class TestInRange(unittest.TestCase):
    def test_none_equals_absolute_tensors(self):
        for seed in range(5):
            logits, mask, batch = random_batch([5, 1, 40, 417], -29.99, 29.99, seed)
            weights = th.randn(logits.shape[0], generator=th.Generator().manual_seed(500 + seed))
            pa, ga = pi_and_grad(logits, mask, batch, weights, "absolute")
            pn, gn = pi_and_grad(logits, mask, batch, weights, "none")
            self.assertTrue(th.allclose(pn, pa, rtol=0, atol=1e-6), (pn - pa).abs().max())
            self.assertTrue(th.allclose(gn, ga, rtol=0, atol=1e-5), (gn - ga).abs().max())
            self.assertTrue(th.equal(pn[~mask], th.zeros_like(pn[~mask])))
            # the probabilities sampling uses (masked_segmented_softmax) are exp(log pi)
            p_sampling = masked_segmented_softmax(logits.clone(), mask, batch, clamp_mode="none")
            self.assertTrue(th.allclose(p_sampling, pn, rtol=0, atol=1e-7))

    def test_soft_is_softmax_of_tanh_and_bounded(self):
        for seed, (low, high) in enumerate([(-200.0, 200.0), (-3.0, 3.0)]):
            logits, mask, batch = random_batch([9, 300], low, high, seed)
            ps = masked_segmented_softmax(logits.clone(), mask, batch, clamp_mode="soft")
            capped = SOFT_CLAMP_SCALE * th.tanh(logits / SOFT_CLAMP_SCALE)
            self.assertLess(capped.abs().max().item(), 30.0)
            ref = softmax(capped.masked_fill(~mask, -np.inf), batch).flatten()
            self.assertTrue(th.allclose(ps, ref, rtol=1e-6, atol=1e-9))
        # small logits: soft is close to absolute (tanh curvature only: rel. error ~ (x/30)^2/3)
        logits, mask, batch = random_batch([400], -3.0, 3.0, 7)
        pa = masked_segmented_softmax(logits.clone(), mask, batch, clamp_mode="absolute")
        ps = masked_segmented_softmax(logits.clone(), mask, batch, clamp_mode="soft")
        self.assertTrue(th.allclose(ps, pa, rtol=0.02, atol=1e-6), (ps / pa)[mask].sub(1).abs().max())

    def test_real_policy_forward_and_evaluate_actions(self):
        """Same weights under absolute and none (and soft): identical pi, log_prob, entropy and
        gradients on real ternary states (a fresh policy's logits are well inside +-30)."""
        env, obs = real_states()
        th.manual_seed(0)
        base = il.build_policy(env, "absolute")
        out = {}
        for mode in ("absolute", "none", "soft"):
            policy = il.build_policy(env, mode)
            policy.load_state_dict(base.state_dict())
            self.assertEqual(policy.clamp_mode, mode)
            policy.diag_stash = {}
            with th.no_grad():
                policy.forward(obs)
            p, logits = policy.diag_stash["probs"], policy.diag_stash["logits"]
            policy.diag_stash = None
            actions = th.tensor([0, 1, 2, 3, 401, 402])
            _v, log_prob, ent, _pv = policy.evaluate_actions(obs, actions)
            policy.zero_grad()
            (log_prob.sum() + ent).backward()
            out[mode] = (p, log_prob.detach(), ent.detach(), policy.action_net.weight.grad.clone(), logits)
        self.assertLess(out["absolute"][4].abs().max().item(), 30.0)
        pa, lpa, ea, ga, _ = out["absolute"]
        pn, lpn, en, gn, _ = out["none"]
        self.assertTrue(th.allclose(pn, pa, rtol=0, atol=1e-6))
        self.assertTrue(th.allclose(lpn, lpa, rtol=0, atol=1e-5))
        self.assertTrue(th.allclose(en, ea, rtol=0, atol=1e-5))
        # different float32 formulas (log of softmax vs log-softmax over ~420 nodes): the
        # weight gradients agree to ~2e-5 of their norm
        self.assertLess(((gn - ga).norm() / ga.norm()).item(), 1e-4)
        ps, lps, es, gs, _ = out["soft"]
        self.assertTrue(th.allclose(lps, lpa, rtol=0, atol=1e-3))
        self.assertTrue(th.allclose(es, ea, rtol=1e-3, atol=1e-4))


class TestOutsideClamp(unittest.TestCase):
    def test_gradients_beyond_plus_minus_30(self):
        logits = th.tensor([100.0, 50.0, 31.0, 0.0, -40.0, -45.0, 7.0])
        mask = th.tensor([True, True, True, True, True, True, False])
        batch = th.zeros(7, dtype=th.long)
        weights = th.tensor([1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 0.0])
        _pa, ga = pi_and_grad(logits, mask, batch, weights, "absolute")
        outside = th.tensor([True, True, True, False, True, True, False])
        self.assertTrue(th.equal(ga[outside], th.zeros_like(ga[outside])))  # frozen by the clamp
        for mode in ("none", "soft"):
            p, g = pi_and_grad(logits, mask, batch, weights, mode)
            self.assertTrue(th.isfinite(g).all(), mode)
            self.assertGreater(g[mask].abs().min().item(), 0.0, mode)
            self.assertEqual(g[6].item(), 0.0, mode)  # invalid node
            self.assertAlmostEqual(p[mask].sum().item(), 1.0, places=5)
        # none keeps the exact ordering and ratios of the raw logits
        pn, _ = pi_and_grad(logits, mask, batch, weights, "none")
        self.assertTrue(th.allclose(pn[mask], th.softmax(logits[mask].double(), 0).float(), rtol=1e-5, atol=1e-30))


class TestNoNaN(unittest.TestCase):
    def test_gap_of_200(self):
        logits = th.tensor([0.0, -200.0, -5.0, 3.0])
        mask = th.tensor([True, True, True, False])
        batch = th.zeros(4, dtype=th.long)
        # control: the plain softmax underflows to 0, so log(p) is -inf and the entropy NaN
        plain = softmax(logits.masked_fill(~mask, -np.inf), batch).flatten()
        self.assertEqual(plain[1].item(), 0.0)
        self.assertTrue(th.isnan((-plain[mask] * th.log(plain[mask])).sum()))
        # none: finite log pi, entropy and gradients, as _choose_node computes them
        x = logits.clone().requires_grad_(True)
        lp = masked_segmented_log_softmax(x, mask, batch)
        p = lp.exp()
        self.assertEqual(p[1].item(), 0.0)
        self.assertTrue(th.isfinite(lp[mask]).all())
        self.assertAlmostEqual(lp[1].item(), -200.0 - float(np.log(1 + np.exp(-5.0))), places=3)
        entropy = (-p[mask] * lp[mask]).sum()
        (lp[1] + entropy).backward()  # the log pi of an "executed" node that has p == 0
        self.assertTrue(th.isfinite(entropy))
        self.assertTrue(th.isfinite(x.grad).all())
        self.assertEqual(x.grad[3].item(), 0.0)

    def test_empty_graph_keeps_shift_zero(self):
        logits = th.tensor([1.0, 2.0, 5.0])
        mask = th.tensor([True, True, False])
        batch = th.tensor([0, 0, 1])  # graph 1 has no valid node
        lp = masked_segmented_log_softmax(logits, mask, batch)
        self.assertTrue(th.isfinite(lp[:2]).all())
        self.assertEqual(lp[2].item(), -np.inf)

    def _huge_logit_policy(self, mode):
        """A real policy whose action logits span hundreds (action_net scaled up)."""
        env, obs = real_states(4)
        th.manual_seed(1)
        policy = il.build_policy(env, mode)
        with th.no_grad():
            policy.action_net.weight.mul_(1e4)
            policy.action_net.bias.zero_()
        return policy, obs

    def test_real_policy_with_huge_logits(self):
        policy, obs = self._huge_logit_policy("none")
        policy.diag_stash = {}
        with th.no_grad():
            actions, _values, log_prob, _e, _plans = policy.forward(obs)
        logits, mask = policy.diag_stash["logits"], policy.diag_stash["mask"].bool()
        policy.diag_stash = None
        self.assertGreater(logits[mask].max().item() - logits[mask].min().item(), 200.0)
        self.assertTrue(th.isfinite(log_prob).all())
        # evaluate_actions on nodes the policy gives (numerically) zero probability
        lows = []
        batch, _sym = policy._get_latent(obs)
        x = policy.action_net(batch.x).flatten().detach()
        for g in range(len(obs)):
            idx = ((batch.batch == g) & mask).nonzero().flatten()
            lows.append(int(idx[x[idx].argmin()] - (batch.batch < g).sum()))
        _v, lp, ent, _pv = policy.evaluate_actions(obs, th.tensor(lows))
        policy.zero_grad()
        (lp.sum() + ent).backward()
        self.assertTrue(th.isfinite(lp).all() and th.isfinite(ent))
        self.assertTrue(all(th.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None))

    def test_short_learn_stays_finite(self):
        threads = th.get_num_threads()
        th.set_num_threads(1)
        try:
            for mode in ("none", "soft"):
                with self.subTest(mode=mode):
                    env = make_vec_env(lambda: GraphTaxiEnv(**spec_kwargs(gym_module.spec(il.ENV_ID))), n_envs=2,
                                       seed=0, vec_env_cls=AsyncVecEnv)
                    model = PlanFeedback_A2C(
                        GNNPlanFeedbackPolicy, env, verbose=0, device="cpu", seed=0, n_steps=3,
                        supported_action_spaces=(sage_spaces.BinaryAction, gym_module.spaces.Discrete,
                                                 sage_spaces.Autoregressive),
                        policy_kwargs=dict(il.CELL1_POLICY_KWARGS, features_extractor_kwargs={"gnn_steps": 5},
                                           clamp_mode=mode),
                    )
                    with th.no_grad():  # large logits from the start
                        model.policy.action_net.weight.mul_(1e3)
                    model.learn(total_timesteps=2 * 3 * 3)
                    env.close()
                    for name, p in model.policy.named_parameters():
                        self.assertTrue(th.isfinite(p).all(), f"{mode}: {name}")
        finally:
            th.set_num_threads(threads)


class TestDiagMirror(unittest.TestCase):
    def test_float64_log_policy_follows_none_and_soft(self):
        for mode in ("none", "soft"):
            for seed, (low, high) in enumerate([(-80.0, 80.0), (-20.0, 20.0)]):
                logits, mask, batch = random_batch([300], low, high, seed)
                p = masked_segmented_softmax(logits.clone(), mask, batch, clamp_mode=mode)
                mirror = np.exp(log_policy(logits.double().numpy(), mask.numpy(), mode))
                self.assertTrue(np.allclose(mirror, p.double().numpy(), rtol=1e-5, atol=1e-30), (mode, low, high))


class TestWiring(unittest.TestCase):
    def _variant(self, argv):
        with mock.patch.object(gnn_global, "run") as run:
            gnn_global.main(argv)
        return run.call_args[0][0]

    def test_flag_reaches_policy_kwargs(self):
        for mode in ("none", "soft"):
            kwargs = self._variant(CELL1_ARGV + ["--clamp-mode", mode])["algorithm_kwargs"]["policy_kwargs"]
            self.assertEqual(kwargs["clamp_mode"], mode)
            with mock.patch.object(gnn_global, "run") as run:
                with self.assertRaises(ValueError):
                    gnn_global.main([a for a in CELL1_ARGV if a != "--feedback"] + ["--clamp-mode", mode])
                run.assert_not_called()
        self.assertNotIn("clamp_mode", self._variant(CELL1_ARGV)["algorithm_kwargs"]["policy_kwargs"])

    def test_hybrid_actions_refuse_none(self):
        env, obs = real_states(1)
        policy = il.build_policy(env, "none")
        with self.assertRaises(NotImplementedError):
            policy._choose_hybrid(policy.action_net, None)


@unittest.skipUnless(CUDA, "CUDA not available (run on RCP)")
class TestCuda(unittest.TestCase):
    def test_none_and_soft_match_cpu(self):
        for mode in ("none", "soft"):
            for seed, (low, high) in enumerate([(-200.0, 200.0), (-29.0, 29.0)]):
                logits, mask, batch = random_batch([7, 1, 23, 417, 3], low, high, seed)
                weights = th.randn(logits.shape[0], generator=th.Generator().manual_seed(600 + seed))
                pc, gc = pi_and_grad(logits, mask, batch, weights, mode)
                pg, gg = pi_and_grad(logits.cuda(), mask.cuda(), batch.cuda(), weights.cuda(), mode)
                self.assertTrue(th.allclose(pg.cpu(), pc, rtol=1e-5, atol=1e-7), (mode, low, high))
                self.assertTrue(th.allclose(gg.cpu(), gc, rtol=1e-4, atol=1e-6), (mode, low, high))
                self.assertTrue(th.isfinite(gg).all())


if __name__ == "__main__":
    unittest.main()
