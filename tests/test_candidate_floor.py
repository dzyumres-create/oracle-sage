"""
--candidate-floor (sage/agent/graph_plan_feedback_policy.py: apply_candidate_floor, used by
GNNPlanFeedbackPolicy._choose_top_action and so by WLPlanFeedbackPolicy too).

With k candidates drawn without replacement, a near-certain actor fills slots 2..k with
near-zero-probability nodes, and the selector can execute them. The floor replaces every
candidate with pi < F * pi_max by the graph's top candidate; log pi stays that of the
executed node.

  TestApplyFloor      hand cases: low candidates replaced by the top one, the top one never;
                      nothing changes when every candidate is above the floor
  TestPolicy          default 0 is the old code path (never called, no RNG); a near-certain
                      actor executes its top node, with log pi of that node; a near-uniform
                      actor is unchanged by F = 0.01; the same draws as F = 0 (no RNG used)
  TestDiagAndWiring   the flag reaches the policy only when > 0, needs --feedback --planner,
                      names the diag run; the diag CSV / stdout report the replaced share
  TestCuda            CUDA matches CPU (run on RCP)
The default's outputs stay bitwise those of 5547edb: tests/test_paper_unified_base.py.

Run from the repo root (CPU; not while a training run uses the GPU):
    PYTHONPATH=. python -m pytest tests/test_candidate_floor.py -v
"""
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch as th

from tests.test_clamp_none_soft import real_states
from tests.test_diag_occupancy import CELL1_ARGV, diag_csvs, num, run_gnn_global

import analysis.inspect_logits as il
import sage.agent.graph_plan_feedback_policy as gpfp
import sage.experiments.gnn_global as gnn_global
from sage.agent.graph_plan_feedback_policy import apply_candidate_floor
from sage.agent.graph_policy import make_mask

CUDA = th.cuda.is_available()


def two_graphs(device="cpu"):
    """Graph 0: a near-certain actor over 5 nodes; graph 1: a flat one over 4 nodes."""
    pa = th.tensor([0.96, 0.03, 0.0001, 0.0099, 0.0, 0.3, 0.25, 0.25, 0.2], device=device)
    batch_ind = th.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], device=device)
    data_starts = th.tensor([0, 5], device=device)
    return pa, batch_ind, data_starts


class TestApplyFloor(unittest.TestCase):
    def test_low_candidates_replaced_by_top(self):
        pa, batch_ind, starts = two_graphs()
        a = th.tensor([[2, 1], [0, 3], [3, 2]])  # [k=3, n=2] local node ids, top of graph 0 drawn 2nd
        new, replaced = apply_candidate_floor(a, pa, starts, batch_ind, 0.01)
        # graph 0: pi_max 0.96, floor 0.0096 -> node 2 (1e-4) replaced by node 0; node 3 (0.0099) kept
        self.assertEqual(new[:, 0].tolist(), [0, 0, 3])
        self.assertEqual(replaced[:, 0].tolist(), [True, False, False])
        # graph 1: every candidate is above the floor -> unchanged
        self.assertEqual(new[:, 1].tolist(), [1, 3, 2])
        self.assertFalse(replaced[:, 1].any())

    def test_top_candidate_never_replaced(self):
        """The argmax node was not drawn and every candidate is below the floor: the highest-pi
        candidate stays, the others become copies of it."""
        pa, batch_ind, starts = two_graphs()
        a = th.tensor([[2], [4], [3]])  # graph 0 only: pi 1e-4, 0, 0.0099
        new, replaced = apply_candidate_floor(a, pa[:5], starts[:1], batch_ind[:5], 0.5)
        self.assertEqual(new[:, 0].tolist(), [3, 3, 3])
        self.assertEqual(replaced[:, 0].tolist(), [True, True, False])

    def test_nothing_below_floor_nothing_changes(self):
        pa, batch_ind, starts = two_graphs()
        a = th.tensor([[0, 0], [1, 1], [3, 2]])
        new, replaced = apply_candidate_floor(a, pa, starts, batch_ind, 0.01)
        self.assertTrue(th.equal(new, a))
        self.assertFalse(replaced.any())


def build(floor, sharpen=None):
    """A Cell 1 policy on 4 real ternary states. sharpen scales action_net up by that factor to
    make the actor near-certain; that needs --clamp-mode none, since under absolute the scaled
    logits all clamp to +30 and the actor turns uniform again (the floor works the same in
    every clamp mode)."""
    env, obs = real_states(4)
    th.manual_seed(0)
    policy = il.build_policy(env, "absolute" if sharpen is None else "none")
    if floor:
        policy.candidate_floor = floor  # same object as built with candidate_floor=floor (checked below)
    if sharpen is not None:
        with th.no_grad():
            policy.action_net.weight.mul_(sharpen)
    return policy, obs


def forward(policy, obs, seed=7):
    policy.diag_stash = {}
    th.manual_seed(seed)
    with th.no_grad():
        actions, _values, log_prob, _e, plans = policy.forward(obs)
    stash, policy.diag_stash = policy.diag_stash, None
    return actions, log_prob, plans, stash


class TestPolicy(unittest.TestCase):
    def test_kwarg_validated_and_stored(self):
        env, _obs = real_states(1)
        kwargs = dict(il.CELL1_POLICY_KWARGS, features_extractor_kwargs={"gnn_steps": 5})
        policy = gpfp.GNNPlanFeedbackPolicy(env.observation_space, env.action_space, lambda _: 1e-3,
                                            **dict(kwargs, candidate_floor=0.01))
        self.assertEqual(policy.candidate_floor, 0.01)
        self.assertEqual(il.build_policy(env).candidate_floor, 0.0)
        for bad in (-0.1, 1.0):
            with self.assertRaises(ValueError):
                gpfp.GNNPlanFeedbackPolicy(env.observation_space, env.action_space, lambda _: 1e-3,
                                           **dict(kwargs, candidate_floor=bad))

    def test_default_never_calls_the_floor(self):
        policy, obs = build(0.0)
        with mock.patch.object(gpfp, "apply_candidate_floor", side_effect=AssertionError("called")), \
                mock.patch.object(gpfp.logger, "record_mean") as record_mean:
            forward(policy, obs)
        self.assertNotIn("action_selection/floor_replaced", [c[0][0] for c in record_mean.call_args_list])

    def test_near_certain_actor_executes_its_top_node(self):
        """Scaled-up action_net: pi_top ~ 1, the other two draws are ~0. With F = 0.01 they are
        replaced, so the top node is executed, with log pi of that node."""
        policy, obs = build(0.01, sharpen=1e4)
        actions, log_prob, _plans, stash = forward(policy, obs)
        p, mask, batch_index = stash["probs"], stash["mask"].bool(), stash["batch_index"]
        starts = th.cat([th.zeros(1, dtype=th.long), th.cumsum(th.bincount(batch_index), 0)[:-1]])
        for g in range(len(obs)):
            nodes = (batch_index == g).nonzero().flatten()
            top = int(p[nodes].argmax())
            self.assertGreater(p[nodes].max().item(), 0.99)
            self.assertEqual(int(actions[g]), top)
            self.assertTrue(th.allclose(log_prob[g], th.log(p[starts[g] + top]), atol=1e-6))
            self.assertTrue(stash["floor_replaced"][:, g].sum() >= 1)
            self.assertEqual(stash["candidates"][:, g].tolist(), [top] * 3)
        # control, the same draws without the floor: the selector executes near-zero-pi nodes
        # (log pi ~ -1000 here, the B2 run-away pattern); slots the floor kept are identical
        policy.candidate_floor = 0.0
        actions0, log_prob0, _pl, stash0 = forward(policy, obs)
        kept = stash["floor_replaced"].logical_not()
        self.assertTrue(th.equal(stash0["candidates"][kept], stash["candidates"][kept]))
        tops = [int(p[(batch_index == g).nonzero().flatten()].argmax()) for g in range(len(obs))]
        off_top = [g for g in range(len(obs)) if int(actions0[g]) != tops[g]]
        self.assertTrue(off_top)
        self.assertLess(min(log_prob0.flatten()[g].item() for g in off_top), -100.0)

    def test_flat_actor_unchanged(self):
        """A fresh policy is close to uniform, so no candidate is below 1% of pi_max: F = 0.01
        changes nothing, and the floor draws no random numbers (same draws, same actions)."""
        policy, obs = build(0.0)
        a0, lp0, plans0, s0 = forward(policy, obs)
        policy.candidate_floor = 0.01
        a1, lp1, plans1, s1 = forward(policy, obs)
        self.assertFalse(s1["floor_replaced"].any())
        self.assertTrue(th.equal(s0["candidates"], s1["candidates"]))
        self.assertTrue(th.equal(a0, a1) and th.equal(lp0, lp1))
        self.assertEqual([list(p) for p in plans0], [list(p) for p in plans1])

    def test_evaluate_actions_unaffected(self):
        """The floor only changes which candidates the selector sees; evaluate_actions (the
        loss's log pi of the stored action) is identical with and without it."""
        policy, obs = build(0.0, sharpen=1e4)
        actions = th.tensor([0, 1, 2, 401])
        ref = policy.evaluate_actions(obs, actions)
        policy.candidate_floor = 0.01
        out = policy.evaluate_actions(obs, actions)
        for x, y in zip(ref, out):
            self.assertTrue(th.equal(x, y))


class TestDiagAndWiring(unittest.TestCase):
    def _variant(self, argv):
        with mock.patch.object(gnn_global, "run") as run:
            gnn_global.main(argv)
        return run.call_args[0][0]

    def test_flag(self):
        self.assertNotIn("candidate_floor", self._variant(CELL1_ARGV)["algorithm_kwargs"]["policy_kwargs"])
        self.assertNotIn("candidate_floor",
                         self._variant(CELL1_ARGV + ["--candidate-floor", "0"])["algorithm_kwargs"]["policy_kwargs"])
        v = self._variant(CELL1_ARGV + ["--candidate-floor", "0.01", "--diag-occupancy"])
        self.assertEqual(v["algorithm_kwargs"]["policy_kwargs"]["candidate_floor"], 0.01)
        self.assertTrue(v["algorithm_kwargs"]["diag_occupancy"]["run_name"].endswith("_s0_floor0.01"))
        with mock.patch.object(gnn_global, "run") as run:
            for argv in ([a for a in CELL1_ARGV if a != "--planner"] + ["--candidate-floor", "0.01"],
                         [a for a in CELL1_ARGV if a != "--feedback"] + ["--candidate-floor", "0.01"],
                         CELL1_ARGV + ["--candidate-floor", "1.5"]):
                with self.assertRaises(ValueError):
                    gnn_global.main(argv)
            run.assert_not_called()

    def test_diag_reports_replaced_share(self):
        """A short gnn_global run with --diag-occupancy and a floor of 0.99 (the fresh actor is
        nearly uniform, so a high floor makes it act): the decisions CSV's floor_replaced matches
        the candidates, and the logs carry action_selection/floor_replaced and diag/floor_rep_*."""
        workdir = tempfile.mkdtemp()
        threads = th.get_num_threads()
        th.set_num_threads(1)
        try:
            out = run_gnn_global(CELL1_ARGV + ["--num-processes", "2", "--log-interval", "5", "--num-env-steps", "30",
                                               "--candidate-floor", "0.99", "--diag-occupancy"], workdir)
            self.assertEqual(out["model"].policy.candidate_floor, 0.99)
            rows = diag_csvs(workdir, run_name="cell1_s0_floor0.99")["decisions"]
            self.assertTrue(rows)
            replaced = [int(r["floor_replaced"]) for r in rows]
            self.assertGreater(sum(replaced), 0)
            for r, n in zip(rows, replaced):
                nodes = [int(r[f"cand{k}_node"]) for k in (1, 2, 3)]
                # every replaced slot is a copy of the top candidate (cand1, the highest pi)
                self.assertGreaterEqual(nodes.count(nodes[0]), n + 1)
            means = {k for k, _v in out["means"]}
            self.assertIn("action_selection/floor_replaced", means)
            self.assertTrue({"diag/floor_rep_occ", "diag/floor_rep_emp"} & means)
        finally:
            th.set_num_threads(threads)
            shutil.rmtree(workdir, ignore_errors=True)


@unittest.skipUnless(CUDA, "CUDA not available (run on RCP)")
class TestCuda(unittest.TestCase):
    def test_matches_cpu(self):
        pa, batch_ind, starts = two_graphs()
        a = th.tensor([[2, 1], [0, 3], [3, 2]])
        cpu = apply_candidate_floor(a, pa, starts, batch_ind, 0.01)
        pa_c, batch_c, starts_c = two_graphs("cuda")
        gpu = apply_candidate_floor(a.cuda(), pa_c, starts_c, batch_c, 0.01)
        self.assertTrue(th.equal(cpu[0], gpu[0].cpu()) and th.equal(cpu[1], gpu[1].cpu()))

    def test_policy_on_cuda(self):
        policy, obs = build(0.01, sharpen=1e4)
        policy = policy.to("cuda")
        actions, log_prob, _plans, stash = forward(policy, obs)
        self.assertEqual(actions.device.type, "cuda")
        self.assertTrue(th.isfinite(log_prob).all())
        self.assertTrue(stash["floor_replaced"].any())


if __name__ == "__main__":
    unittest.main()
