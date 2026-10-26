"""
Tests for --diag-occupancy (sage/agent/diag_occupancy.py): logging-only per-decision and
per-update diagnostics for PlanFeedback_A2C on the ternary Taxi domain.

  TestBehaviourUnchanged   the key test: a real gnn_global run with the flag on is bitwise
                           identical to one with it off (parameters, gradients, every env
                           action, every non-diag logged value, final torch/numpy RNG states)
  TestGroundTruth          a scripted crossed pair: a forced fail_own, then the next decision
                           is occupied by the right passenger; CSV outcomes reconcile exactly
                           with the env's own per-episode dropoff counters
  TestConsistency          per-decision rows reproduce train/entropy, train/policy_loss and
                           every action_selection/* value the logger received; KL >= 0
  TestKLZeroWithoutLearning learning_rate = 0 gives KL == 0 exactly (CPU)
  TestWiring               flag validation, run naming, startup line, saved data

CPU, one thread throughout (see tests/test_chunked_update.py's single_cpu_thread for why).

Run from the repo root with:
    PYTHONPATH=. python -m pytest tests/test_diag_occupancy.py -v -s
Do not run while a training run is using the GPU.
"""
import csv
import functools
import io
import json
import math
import os
import random
import shutil
import tempfile
import unittest
import zipfile
from collections import Counter, defaultdict
from unittest import mock

import gym as gym_module
import networkx as nx
import numpy as np
import torch as th

# test_ternary_planner installs the gym-0.26 randint shim at import time.
from tests.test_ternary_planner import POLICY_KWARGS
from tests.test_chunked_update import single_cpu_thread
from tests.test_ternary_diagnostic import printed_table_rows

import sage.experiments.gnn_global as gnn_global
from sage.agent.async_vec_env import AsyncVecEnv
from sage.agent.diag_occupancy import OUTCOMES, STDOUT_KEYS, run_paths
from sage.agent.graph_plan_feedback_policy import GNNPlanFeedbackPolicy
from sage.agent.plan_feedback_a2c import PlanFeedback_A2C
from sage.domains.gym_taxi.envs.taxi_env import DIAG_INFO_KEYS, GraphTaxiEnv
from sage.domains.gym_taxi.simulator.ternary_planner import plan_on_facts
from sage.domains.gym_taxi.simulator.ternary_taxi_world import (
    DROPOFF_ATTEMPTS,
    DROPOFF_EMPTY,
    DROPOFF_FAILURES,
    DROPOFF_FAILURES_OWN_OTHER,
    TernaryTaxiWorldSimulator,
)
from sage.domains.gym_taxi.utils.compat import spec_kwargs
from sage.domains.gym_taxi.utils.ternary_representations import _ternary_observation_fn
from sage.domains.utils import spaces as sage_spaces
from sage.forks.stable_baselines3.stable_baselines3.common import logger
from sage.forks.stable_baselines3.stable_baselines3.common.env_util import make_vec_env

TERNARY_ENV_ID = "city-taxi-ternary-unmasked-v1"
N_STEPS = 5
# Cell 1's configuration (object encoding, shared GNN, 3 candidates, eps 0), small.
CELL1_ARGV = ["--env-name", TERNARY_ENV_ID, "--planner", "--feedback", "--shared-gnn",
              "--num-steps", str(N_STEPS), "--checkpoint-interval", "0", "--seed", "0"]
CELL1_POLICY_KWARGS = dict(POLICY_KWARGS, num_planning_choices=3, exploration_fraction=0.1)
ACTION_SELECTION_KEYS = ("prob_choice_1", "prob_choice_2", "prob_choice_3",
                         "choice_1_selected", "choice_2_selected", "choice_3_selected", "choice_likelihood")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(value):
    return float(value)


# ==========================================================================================
# Runner: gnn_global.main as on RCP, on CPU, from fixed seeds, recording everything
# ==========================================================================================

def make_vec_env_from_spec(env_id, **kwargs):
    """gym.make on gym 0.26 rejects JsonGraph, so build the same registered env from its spec
    (as tests/test_chunked_update.py's TestGnnGlobalWiring does)."""
    env_kwargs = kwargs.pop("env_kwargs", {}) or {}
    return make_vec_env(lambda: GraphTaxiEnv(**spec_kwargs(gym_module.spec(env_id)), **env_kwargs), **kwargs)


def run_gnn_global(argv, workdir, loss_grad_every=None):
    """gnn_global.main(argv) from cwd=workdir with torch/numpy/python seeded, the model on CPU,
    the final save captured instead of written (and, if given, the per-loss-term gradients every
    loss_grad_every updates instead of gnn_global's 5). Returns the model, every env.step call,
    every logger.record/record_mean call, and the RNG states at the end."""
    steps, records, means, captured = [], [], [], {}
    real_step = AsyncVecEnv.step
    real_record, real_record_mean = logger.record, logger.record_mean

    def step_spy(self, actions, mask):
        out = real_step(self, actions, mask)
        steps.append((np.array(actions), np.array(mask), out[1].copy(), out[2].copy()))
        return out

    def save_spy(model_self, path, *args, **kwargs):
        captured["model"] = model_self

    def record_spy(key, value, *args, **kwargs):
        records.append((key, value))
        return real_record(key, value, *args, **kwargs)

    def record_mean_spy(key, value, *args, **kwargs):
        means.append((key, value))
        return real_record_mean(key, value, *args, **kwargs)

    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        th.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        with mock.patch.object(gnn_global, "make_vec_env", side_effect=make_vec_env_from_spec), \
                mock.patch.object(gnn_global, "PlanFeedback_A2C", functools.partial(PlanFeedback_A2C, device="cpu")), \
                mock.patch.object(PlanFeedback_A2C, "save", save_spy), \
                mock.patch.object(AsyncVecEnv, "step", step_spy), \
                mock.patch.object(logger, "record", side_effect=record_spy), \
                mock.patch.object(logger, "record_mean", side_effect=record_mean_spy), \
                mock.patch.object(gnn_global, "DIAG_LOSS_GRAD_EVERY", loss_grad_every or gnn_global.DIAG_LOSS_GRAD_EVERY):
            gnn_global.main(argv + ["--log-dir", os.path.join(workdir, "logs", "cell1"),
                                    "--save-dir", os.path.join(workdir, "models")])
        rng = {"torch": th.get_rng_state(), "numpy": np.random.get_state()}
    finally:
        os.chdir(cwd)
    model = captured["model"]
    model.env.close()
    return {"model": model, "steps": steps, "records": records, "means": means, "rng": rng}


def diag_csvs(workdir, run_name="cell1_s0"):
    """The three CSVs a gnn_global --diag-occupancy run wrote under workdir (log dir .../cell1, seed 0)."""
    return {kind: read_csv(os.path.join(workdir, "chart_logs", "diag", f"{run_name}_{kind}.csv"))
            for kind in ("decisions", "bootstrap", "updates")}


class _TempDirCase(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.mkdtemp(prefix="diag_occupancy_")
        self._threads = th.get_num_threads()
        th.set_num_threads(1)

    def tearDown(self):
        th.set_num_threads(self._threads)
        shutil.rmtree(self.workdir, ignore_errors=True)

    def subdir(self, name):
        path = os.path.join(self.workdir, name)
        os.makedirs(path)
        return path


# ==========================================================================================
# 1. Behaviour identity
# ==========================================================================================

class TestBehaviourUnchanged(_TempDirCase):
    """~20 updates through gnn_global.main, flag off twice (the control: the run itself is
    reproducible here) and flag on once, with the per-loss-term gradients (th.autograd.grad on
    the retained graph) taken on EVERY update. Everything training touches must be bitwise equal."""

    N_ENVS, UPDATES = 2, 20
    # time/* depend on wall-clock time
    TIME_KEYS = ("time/fps", "time/time_elapsed")

    def _argv(self):
        return CELL1_ARGV + ["--num-processes", str(self.N_ENVS), "--log-interval", "5",
                             "--num-env-steps", str(self.N_ENVS * N_STEPS * self.UPDATES)]

    def _assert_same(self, a, b, what):
        pa = dict(a["model"].policy.named_parameters())
        pb = dict(b["model"].policy.named_parameters())
        self.assertEqual(set(pa), set(pb))
        for name in pa:
            self.assertTrue(th.equal(pa[name], pb[name]), f"{what}: parameter {name}")
            ga, gb = pa[name].grad, pb[name].grad
            self.assertEqual(ga is None, gb is None, f"{what}: grad {name}")
            if ga is not None:
                self.assertTrue(th.equal(ga, gb), f"{what}: grad {name} (incl. never-zeroed frozen grads)")
        self.assertEqual(len(a["steps"]), len(b["steps"]), f"{what}: number of env.step calls")
        for k, (sa, sb) in enumerate(zip(a["steps"], b["steps"])):
            for part, xa, xb in zip(("actions", "mask", "rewards", "dones"), sa, sb):
                self.assertTrue(np.array_equal(xa, xb), f"{what}: env.step call {k} {part}")
        self.assertTrue(th.equal(a["rng"]["torch"], b["rng"]["torch"]), f"{what}: final torch RNG state")
        na, nb = a["rng"]["numpy"], b["rng"]["numpy"]
        self.assertEqual(na[0], nb[0])
        self.assertTrue(np.array_equal(na[1], nb[1]) and na[2:] == nb[2:], f"{what}: final numpy RNG state")

        def plain(calls):
            return [(k, v) for k, v in calls if not k.startswith("diag/") and k not in self.TIME_KEYS]
        self.assertEqual(plain(a["records"]), plain(b["records"]), f"{what}: logger.record calls")
        self.assertEqual(plain(a["means"]), plain(b["means"]), f"{what}: logger.record_mean calls")

    def test_flag_on_is_bitwise_identical_to_flag_off(self):
        off = run_gnn_global(self._argv(), self.subdir("off"))
        control = run_gnn_global(self._argv(), self.subdir("control"))
        on = run_gnn_global(self._argv() + ["--diag-occupancy"], self.subdir("on"), loss_grad_every=1)
        self.assertEqual(off["model"]._n_updates, self.UPDATES)
        self._assert_same(off, control, "control (flag off twice)")
        self._assert_same(off, on, "flag on vs off")
        # flag off writes nothing and sets nothing
        self.assertFalse(os.path.exists(os.path.join(self.workdir, "off", "chart_logs")))
        self.assertNotIn("diag", vars(off["model"]))
        self.assertIsNone(off["model"].diag)
        # flag on did write, did log diag/*, and did take the per-term gradients on every update
        self.assertTrue(os.path.exists(os.path.join(self.workdir, "on", "chart_logs", "diag", "cell1_s0_decisions.csv")))
        self.assertTrue(any(k.startswith("diag/") for k, _ in on["means"]))
        updates = diag_csvs(os.path.join(self.workdir, "on"))["updates"]
        self.assertEqual(len(updates), self.UPDATES)
        for row in updates:
            for key in ("gn_pol", "gn_ent", "gn_val", "gn_pv", "cos_pol_pv"):
                self.assertNotEqual(row[key], "nan", f"update {row['update_idx']}: {key}")


# ==========================================================================================
# 2. Ground truth
# ==========================================================================================

# A crossed pair (rule 2): same origins, rotated destinations. p = 401 boards at 21, so its
# destination is 210; 390 is p's OTHER own destination.
CROSSED_PAIR = (((21, 210), (45, 390)), 21, ((21, 390), (45, 210)), 45)
P, P_DEST, P_OTHER_DEST = 401, 210, 390


class ScriptedEnv(GraphTaxiEnv):
    """The ternary env on an open 20x20 grid with only the crossed pair (no spawning), and an
    optional short timeout."""
    timeout = 2000

    def _init_simulator(self):
        return TernaryTaxiWorldSimulator(
            self.np_random, size=20, delivery_limit=100, concurrent_passengers=20, timeout=self.timeout,
            pair_creation_probability=0.0, requests_per_passenger=2, random_walls=False,
            rewards=self.rewards, planning=not self.mask, observation_fn=_ternary_observation_fn,
            initial_pair=CROSSED_PAIR,
        )


class ShortEpisodeEnv(GraphTaxiEnv):
    """The real ternary env (random maze, spawning) with 150-step episodes."""

    def _init_simulator(self):
        return TernaryTaxiWorldSimulator(
            self.np_random, **dict(self.scenario, timeout=150), rewards=self.rewards,
            planning=not self.mask, observation_fn=_ternary_observation_fn,
        )


def build_model(env_cls, n_envs, out_dir, seed=0, loss_grad_every=5, update_chunks=1):
    env = make_vec_env(
        lambda: env_cls(**spec_kwargs(gym_module.spec(TERNARY_ENV_ID)), diag_occupancy=True),
        n_envs=n_envs, seed=seed, monitor_kwargs={"info_keywords": gnn_global.info_keywords_for(TERNARY_ENV_ID)},
        vec_env_cls=AsyncVecEnv,
    )
    model = PlanFeedback_A2C(
        GNNPlanFeedbackPolicy, env, verbose=0, device="cpu", n_steps=N_STEPS,
        supported_action_spaces=(sage_spaces.BinaryAction, gym_module.spaces.Discrete, sage_spaces.Autoregressive),
        policy_kwargs=CELL1_POLICY_KWARGS, update_chunks=update_chunks,
        diag_occupancy={"out_dir": out_dir, "run_name": "t", "loss_grad_every": loss_grad_every},
    )
    return env, model


def script_forward(model, choose):
    """Wrap policy.forward: the real forward runs (so the stash, RNG and logging are real), then
    choose(env_idx, sim, node, plan) may replace an env's executed node and plan. Returns the
    list of overrides made."""
    real_forward = model.policy.forward
    made = []

    def forward(obs, deterministic=False):
        actions, values, log_probs, explored, plans = real_forward(obs, deterministic)
        plans = np.array(plans, dtype=object)
        actions = actions.clone().reshape(-1)
        for i, sim in enumerate(model.env.get_attr("sim")):
            override = choose(i, sim, int(actions[i]), plans[i])
            if override is not None:
                actions[i], plans[i] = override
                made.append((i, override[0]))
        return actions, values, log_probs, explored, plans

    model.policy.forward = forward
    return made


def path(sim, start, end):
    return nx.shortest_path(sim.roads, start, end)[1:]


class TestGroundTruth(_TempDirCase):

    def test_forced_fail_own_then_occupied_retry(self):
        """Decision 0: board p and drop it at its OTHER own destination (fail_own). Decision 1
        must then be occupied by p; deliver it. Decision 2 must be empty again."""
        env, model = build_model(ScriptedEnv, 1, self.workdir)
        script = [
            lambda sim: (P, path(sim, sim.taxi.location, 21) + [P] + path(sim, 21, P_OTHER_DEST) + [0]),
            lambda sim: (P, path(sim, sim.taxi.location, P_DEST) + [0]),
        ]
        calls = {"n": 0}

        def choose(i, sim, node, plan):
            calls["n"] += 1
            if calls["n"] <= len(script):
                return script[calls["n"] - 1](sim)
            return None

        script_forward(model, choose)
        model.learn(total_timesteps=N_STEPS)  # one rollout
        rows = read_csv(model.diag.paths["decisions"])
        env.close()

        self.assertEqual(len(rows), N_STEPS)
        r0, r1, r2 = rows[0], rows[1], rows[2]
        self.assertEqual((r0["occupied"], r0["carried_pid"], r0["selected_type"]), ("0", "-1", "waiting_passenger"))
        self.assertEqual((r0["outcome"], r0["dropoff_pid"], r0["pickup_failed"]), ("fail_own", str(P), "0"))
        self.assertEqual((r1["occupied"], r1["carried_pid"]), ("1", str(P)))
        self.assertEqual((r1["selected_type"], r1["selected_is_carried"], r1["same_node_as_prev_decision"]),
                         ("carried_passenger", "1", "1"))
        self.assertFalse(math.isnan(num(r1["p_carried"])))
        self.assertEqual(r1["outcome"], "delivered")
        # the dropoff is the plan's last step; the CSV keeps 9 significant digits
        self.assertAlmostEqual(num(r1["plan_reward"]), 0.99 ** (int(r1["plan_len"]) - 1), delta=1e-7)
        self.assertEqual((r2["occupied"], r2["carried_pid"], r2["same_node_as_prev_decision"]), ("0", "-1", "-1"))
        self.assertTrue(math.isnan(num(r2["p_carried"])))
        self.assertGreater(model.diag.state_checks, 0)
        self.assertEqual(model.diag.state_mismatches, 0)

    def test_outcomes_reconcile_with_episode_counters(self):
        """A short run with 150-step episodes, where most decisions are scripted to produce
        dropoffs of every kind (and some truncations). Per (env, episode), the CSV's outcome
        counts must equal the dropoff counters the env reports at the episode's end."""
        env, model = build_model(ShortEpisodeEnv, 2, self.workdir, seed=3)
        rng = random.Random(7)  # the test's own RNG; training's streams are untouched by it

        def choose(i, sim, node, plan):
            r = rng.random()
            if r < 0.2 or not sim.passengers:
                return None  # the policy's own choice
            if r < 0.35:
                return 0, [0]  # dropoff where the taxi stands (fail_empty when empty)
            pid = rng.choice(sorted(sim.passengers))
            dest = rng.choice([d for _o, d in sim.passengers[pid].requests] + [rng.randint(1, 400)])
            if pid == sim.taxi.passenger:
                return pid, path(sim, sim.taxi.location, dest) + [0]
            here = sim.passengers[pid].location
            return pid, path(sim, sim.taxi.location, here) + [pid] + path(sim, here, dest) + [0]

        made = script_forward(model, choose)
        terminal = defaultdict(list)  # env -> [episode info at each done]
        real_step = AsyncVecEnv.step

        def step_spy(self_env, actions, mask):
            out = real_step(self_env, actions, mask)
            for i in np.flatnonzero(mask):
                if out[2][i]:
                    terminal[i].append(out[3][i]["episode"])
            return out

        with mock.patch.object(AsyncVecEnv, "step", step_spy):
            model.learn(total_timesteps=2 * N_STEPS * 40)
        rows = read_csv(model.diag.paths["decisions"])
        env.close()

        counted = defaultdict(Counter)
        for r in rows:
            counted[(int(r["env_idx"]), int(r["episode_idx"]))][r["outcome"]] += 1
        n_episodes = sum(len(v) for v in terminal.values())
        self.assertGreaterEqual(n_episodes, 4, "too few finished episodes to compare")
        totals = Counter(r["outcome"] for r in rows)
        print(f"\n[reconcile] {len(rows)} decisions ({len(made)} scripted), {n_episodes} episodes; outcomes {dict(totals)}")
        for outcome in ("delivered", "fail_own", "fail_nondest", "fail_empty", "truncated"):
            self.assertGreater(totals[outcome], 0, f"no {outcome} decisions: the check would be vacuous")
        for i, episodes in terminal.items():
            for k, ep in enumerate(episodes):
                c = counted[(i, k)]
                with self.subTest(env=i, episode=k):
                    self.assertEqual(c["delivered"] + c["fail_own"] + c["fail_nondest"], ep[DROPOFF_ATTEMPTS])
                    self.assertEqual(c["fail_own"] + c["fail_nondest"], ep[DROPOFF_FAILURES])
                    self.assertEqual(c["fail_own"], ep[DROPOFF_FAILURES_OWN_OTHER])
                    self.assertEqual(c["fail_empty"], ep[DROPOFF_EMPTY])
        self.assertEqual(model.diag.state_mismatches, 0)


# ==========================================================================================
# 3. Consistency with what the logger already receives; 4. KL
# ==========================================================================================

def forwards_in_logger_order(decisions, bootstrap):
    """The per-env rows of every collection forward, in call order: each update's n_steps
    decision forwards, then its bootstrap forward."""
    by_forward = defaultdict(list)
    for r in decisions:
        by_forward[(int(r["update_idx"]), int(r["rollout_step"]))].append(r)
    for r in bootstrap:
        by_forward[(int(r["update_idx"]), N_STEPS)].append(r)
    return [sorted(by_forward[key], key=lambda r: int(r["env_idx"])) for key in sorted(by_forward)]


def action_selection_from_rows(rows, k=3):
    """_choose_top_action's action_selection/* values for one forward, recomputed from its rows.
    probs.max(0)/min(0) index the candidates in DRAW order and return the first extreme."""
    out = defaultdict(list)
    for r in rows:
        by_draw = sorted(range(1, k + 1), key=lambda j: int(r[f"cand{j}_draw"]))
        probs = [np.float32(num(r[f"cand{j}_p"])) for j in by_draw]
        index1 = int(np.argmax(probs))
        index3 = int(np.argmin(probs))
        rank = int(r["executed_rank"])
        chosen = int(r[f"cand{rank}_draw"])
        out["prob_choice_1"].append(probs[index1])
        out["prob_choice_3"].append(probs[index3])
        out["prob_choice_2"].append(sum(probs) - probs[index1] - probs[index3])
        out["c1"].append(chosen == index1)
        out["c3"].append(chosen == index3)
        out["choice_likelihood"].append(num(r[f"cand{rank}_p"]) / probs[index1])
    c1, c3 = np.mean(out["c1"]), np.mean(out["c3"])
    return {
        "prob_choice_1": np.mean(out["prob_choice_1"]), "prob_choice_2": np.mean(out["prob_choice_2"]),
        "prob_choice_3": np.mean(out["prob_choice_3"]), "choice_1_selected": c1, "choice_3_selected": c3,
        "choice_2_selected": 1 - c1 - c3, "choice_likelihood": np.mean(out["choice_likelihood"]),
    }


class TestConsistency(_TempDirCase):
    N_ENVS, UPDATES = 2, 20

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="diag_consistency_")
        threads = th.get_num_threads()
        th.set_num_threads(1)
        try:
            cls.out = run_gnn_global(
                CELL1_ARGV + ["--num-processes", str(cls.N_ENVS), "--log-interval", "5",
                              "--num-env-steps", str(cls.N_ENVS * N_STEPS * cls.UPDATES), "--diag-occupancy"],
                cls.dir)
        finally:
            th.set_num_threads(threads)
        cls.csv = diag_csvs(cls.dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_row_counts(self):
        self.assertEqual(len(self.csv["decisions"]), self.N_ENVS * N_STEPS * self.UPDATES)
        self.assertEqual(len(self.csv["bootstrap"]), self.N_ENVS * self.UPDATES)
        self.assertEqual([int(r["update_idx"]) for r in self.csv["updates"]], list(range(self.UPDATES)))
        self.assertTrue(all(r["outcome"] in OUTCOMES for r in self.csv["decisions"]))

    def test_entropy_and_policy_loss_match_train(self):
        """Mean per-decision entropy over a rollout == train/entropy of that update; the
        per-group policy-loss contributions sum to train/policy_loss. Collection runs on a batch
        of n_envs graphs, training on n_steps * n_envs, in float32 vs this float64 recompute."""
        entropies = [v for k, v in self.out["records"] if k == "train/entropy"]
        losses = [v for k, v in self.out["records"] if k == "train/policy_loss"]
        self.assertEqual(len(entropies), self.UPDATES)
        by_update = defaultdict(list)
        for r in self.csv["decisions"]:
            by_update[int(r["update_idx"])].append(num(r["entropy"]))
        for u in range(self.UPDATES):
            with self.subTest(update=u):
                self.assertAlmostEqual(np.mean(by_update[u]), entropies[u], delta=1e-5 * abs(entropies[u]))
                row = self.csv["updates"][u]
                self.assertAlmostEqual(num(row["ploss_occ"]) + num(row["ploss_empty"]), losses[u],
                                       delta=1e-6 + 1e-4 * abs(losses[u]))

    def test_action_selection_metrics_match_logger(self):
        """Every action_selection/* value _choose_top_action passed to logger.record_mean (one per
        forward, the bootstrap forward included) is reproduced from the per-candidate rows."""
        forwards = forwards_in_logger_order(self.csv["decisions"], self.csv["bootstrap"])
        logged = defaultdict(list)
        for k, v in self.out["means"]:
            if k.startswith("action_selection/"):
                logged[k.split("/", 1)[1]].append(v)
        # the last update's bootstrap forward ran, but its rows are written at close() like any other
        self.assertEqual(len(forwards), self.UPDATES * (N_STEPS + 1))
        for key in ACTION_SELECTION_KEYS:
            self.assertEqual(len(logged[key]), len(forwards), key)
        for n, rows in enumerate(forwards):
            expected = action_selection_from_rows(rows)
            for key in ACTION_SELECTION_KEYS:
                with self.subTest(forward=n, key=key):
                    self.assertAlmostEqual(expected[key], logged[key][n], delta=1e-6)

    def test_kl_nonnegative_and_updates_agree(self):
        kls = [num(r["kl"]) for r in self.csv["bootstrap"] if r["kl"] != "nan"]
        self.assertEqual(len(kls), self.N_ENVS * (self.UPDATES - 1))  # the last update has no "after"
        self.assertGreaterEqual(min(kls), -1e-12)
        self.assertGreater(max(kls), 0.0)
        last = self.csv["updates"][-1]
        self.assertEqual(last["kl_all"], "nan")
        for u, row in enumerate(self.csv["updates"][:-1]):
            mine = [num(r["kl"]) for r in self.csv["bootstrap"] if int(r["update_idx"]) == u]
            self.assertAlmostEqual(num(row["kl_all"]), np.mean(mine), delta=1e-12 + 1e-6 * np.mean(mine))

    def test_loss_term_gradients_every_fifth_update(self):
        for row in self.csv["updates"]:
            keys = ("gn_pol", "gn_ent", "gn_val", "gn_pv", "cos_pol_pv")
            if int(row["update_idx"]) % 5 == 0:
                for key in keys:
                    self.assertNotEqual(row[key], "nan", key)
                self.assertLessEqual(abs(num(row["cos_pol_pv"])), 1.0 + 1e-9)
            else:
                self.assertEqual([row[k] for k in keys], ["nan"] * len(keys))

    def test_grad_norms_recorded(self):
        for row in self.csv["updates"]:
            total = num(row["grad_norm_total"])
            # shared GNN: path_value_net is the only parameter outside the optimizer
            self.assertEqual(row["grad_norm_frozen"], row["grad_norm_head"])
            self.assertAlmostEqual(total ** 2, num(row["grad_norm_train"]) ** 2 + num(row["grad_norm_frozen"]) ** 2,
                                   delta=1e-5 * total ** 2)

    def test_stdout_keys_survive_truncation(self):
        logged = {k for k, _ in self.out["means"] if k.startswith("diag/")}
        self.assertTrue(logged <= {"diag/" + k for k in STDOUT_KEYS}, logged - {"diag/" + k for k in STDOUT_KEYS})
        rows = printed_table_rows({"diag/" + k: 0.5 for k in STDOUT_KEYS})
        self.assertEqual(sorted(k for k, _ in rows if k != "diag/"), sorted(STDOUT_KEYS))


class TestLossTermGradients(_TempDirCase):
    """The four per-term gradients (coefficients included) must add up to the shared GNN's
    actual pre-clip gradient -- in the default train() and, accumulated with the n_i/B chunk
    weights, in the chunked one. Every parameter here is in the optimizer, so its .grad after
    zero_grad + backward is exactly the update's gradient."""

    def _check(self, update_chunks):
        env, model = build_model(ScriptedEnv, 2, self.workdir, loss_grad_every=1, update_chunks=update_chunks)
        policy = model.policy
        params, seen = [], set()
        for module in (policy.gnn_extractor, policy.features_extractor):
            for p in module.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
        terms, actual = [], []
        real_on_loss_grads = model.diag.on_loss_grads
        real_clip = th.nn.utils.clip_grad_norm_

        def on_loss_grads(grads):
            terms.append({k: None if v is None else v.clone() for k, v in grads.items()})
            return real_on_loss_grads(grads)

        def clip(parameters, max_norm, *args, **kwargs):
            actual.append(th.cat([p.grad.detach().reshape(-1).clone() for p in params]))
            return real_clip(parameters, max_norm, *args, **kwargs)

        with mock.patch.object(model.diag, "on_loss_grads", side_effect=on_loss_grads), \
                mock.patch.object(th.nn.utils, "clip_grad_norm_", side_effect=clip):
            model.learn(total_timesteps=2 * N_STEPS * 3)
        env.close()
        self.assertEqual(len(terms), 3)
        self.assertEqual(len(actual), 3)
        for u, (t, g) in enumerate(zip(terms, actual)):
            total = t["pol"] + t["ent"] + t["val"] + t["pv"]
            with self.subTest(update=u, update_chunks=update_chunks):
                self.assertTrue(th.allclose(total, g, rtol=1e-4, atol=1e-7),
                                f"max |diff| {(total - g).abs().max().item():.2e}, |g| {g.norm().item():.2e}")
                self.assertGreater(t["pol"].norm().item(), 0.0)
                self.assertGreater(t["pv"].norm().item(), 0.0)
        rows = read_csv(model.diag.paths["updates"])
        for row, t in zip(rows, terms):
            self.assertAlmostEqual(num(row["gn_pv"]), t["pv"].double().norm().item(), delta=1e-6 * num(row["gn_pv"]))

    def test_default_train(self):
        self._check(update_chunks=1)

    def test_chunked_train(self):
        self._check(update_chunks=3)


class TestKLZeroWithoutLearning(_TempDirCase):
    def test_learning_rate_zero_gives_zero_kl(self):
        run_gnn_global(CELL1_ARGV + ["--num-processes", "2", "--log-interval", "5", "--num-env-steps", "60",
                                     "--learning-rate", "0", "--diag-occupancy"], self.workdir)
        boot = diag_csvs(self.workdir)["bootstrap"]
        checked = 0
        for r in boot:
            if r["kl"] != "nan":
                self.assertEqual(num(r["kl"]), 0.0)
                self.assertEqual(r["p_top_after"], r["p_top"])
                self.assertEqual(r["entropy_after"], r["entropy"])
                checked += 1
        self.assertEqual(checked, 2 * 5)


# ==========================================================================================
# Wiring
# ==========================================================================================

class TestWiring(unittest.TestCase):
    def _variant(self, argv):
        with mock.patch.object(gnn_global, "run") as run:
            gnn_global.main(argv)
        return run.call_args[0][0]

    def test_flag_reaches_env_and_algorithm_only_when_set(self):
        off = self._variant(CELL1_ARGV)
        self.assertFalse(off["diag_occupancy"])
        self.assertNotIn("diag_occupancy", off["algorithm_kwargs"])
        on = self._variant(CELL1_ARGV + ["--diag-occupancy", "--log-dir", "./logs/cell1_seed3/"])
        self.assertTrue(on["diag_occupancy"])
        self.assertEqual(on["algorithm_kwargs"]["diag_occupancy"],
                         {"out_dir": os.path.join("chart_logs", "diag"), "run_name": "cell1_seed3_s0",
                          "loss_grad_every": 5})

    def test_requires_ternary_feedback_planner(self):
        with mock.patch.object(gnn_global, "run") as run:
            for drop in ("--planner", "--feedback"):
                with self.assertRaises(ValueError):
                    gnn_global.main([a for a in CELL1_ARGV if a != drop] + ["--diag-occupancy"])
            with self.assertRaises(ValueError):
                gnn_global.main(["--env-name", "city-taxi-unmasked-v1", "--planner", "--feedback", "--diag-occupancy"])
            run.assert_not_called()
        with self.assertRaises(ValueError):
            GraphTaxiEnv(**spec_kwargs(gym_module.spec("city-taxi-unmasked-v1")), diag_occupancy=True)

    def test_env_info_fields_only_with_flag(self):
        for flag in (False, True):
            env = GraphTaxiEnv(**spec_kwargs(gym_module.spec(TERNARY_ENV_ID)), diag_occupancy=flag)
            env.seed(0)
            env.reset()
            _obs, _r, _d, info = env.step(int(env.sim.taxi.location))
            self.assertEqual(set(DIAG_INFO_KEYS) <= set(info), flag)

    def test_run_paths_never_reuse_files(self):
        d = tempfile.mkdtemp()
        try:
            first = run_paths(d, "r")
            open(first["decisions"], "w").close()
            second = run_paths(d, "r")
            self.assertTrue(second["decisions"].endswith("r_1_decisions.csv"))
        finally:
            shutil.rmtree(d)

    def test_startup_line(self):
        commit, dirty = gnn_global.git_provenance()
        line = gnn_global.startup_line(["--seed", "3"])
        self.assertTrue(line.startswith("startup: argv="))
        self.assertIn("main_args=['--seed', '3']", line)
        self.assertIn(f"commit={commit} dirty={dirty}", line)
        self.assertEqual(len(commit), 40)  # the test runs inside the repository
        out = io.StringIO()
        with mock.patch("sys.stdout", out), mock.patch.object(gnn_global, "run"):
            gnn_global.main(CELL1_ARGV)
        self.assertTrue(out.getvalue().startswith("startup: "))

    def test_saved_data_has_no_diagnostics(self):
        d = tempfile.mkdtemp()
        try:
            env, model = build_model(ScriptedEnv, 1, d)
            model.save(os.path.join(d, "m"))
            env.close()
            with zipfile.ZipFile(os.path.join(d, "m.zip")) as z:
                data = json.loads(z.read("data").decode())
            self.assertNotIn("diag", data)
        finally:
            shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
