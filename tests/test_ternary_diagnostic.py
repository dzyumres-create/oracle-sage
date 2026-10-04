"""
Tests for the ambiguous-delivery diagnostic (Task A): TernaryTaxiWorldSimulator's
per-episode dropoff counters and per-attempt DropoffRecords,
ternary_planner.object_hypothesis_count, and gnn_global.info_keywords_for.

Expected values were written into these tests before they were first run.

Run from the repo root with:
    PYTHONPATH=. python -m pytest tests/test_ternary_diagnostic.py -v -s
Do not run while a training run is using the GPU.
"""
import copy
import io
import math
import random as pyrandom
import unittest
from unittest import mock

import gym as gym_module
import numpy as np
import torch as th

# test_ternary_planner installs the gym-0.26 randint shim at import time (needed by
# PlanFeedback_A2C/make_vec_env below); importing its helpers here keeps one shim.
from tests.test_ternary_planner import make_graph, make_ternary_sim, POLICY_KWARGS
from tests.test_ternary_world import scripted_policy

from sage.agent.async_vec_env import AsyncVecEnv
from sage.agent.graph_plan_feedback_policy import GNNPlanFeedbackPolicy
from sage.agent.plan_feedback_a2c import PlanFeedback_A2C
from sage.domains.gym_taxi import REWARDS
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.simulator.planner import Planner
from sage.domains.gym_taxi.simulator.ternary_planner import _object_cluster_solutions, object_hypothesis_count
from sage.domains.gym_taxi.simulator.ternary_taxi_world import (
    DROPOFF_ATTEMPTS,
    DROPOFF_ATTEMPTS_AMBIGUOUS,
    DROPOFF_DIAGNOSTIC_KEYS,
    DROPOFF_EMPTY,
    DROPOFF_FAILURES,
    DROPOFF_FAILURES_AMBIGUOUS,
    DROPOFF_FAILURES_OWN_OTHER,
    DropoffRecord,
    TernaryTaxiWorldSimulator,
)
from sage.domains.gym_taxi.utils.compat import spec_kwargs
from sage.domains.gym_taxi.utils.ternary_representations import facts_to_json, json_to_ternary_graph_object
from sage.domains.utils import spaces as sage_spaces
from sage.experiments.gnn_global import info_keywords_for
from sage.forks.stable_baselines3.stable_baselines3.common import logger as sb3_logger
from sage.forks.stable_baselines3.stable_baselines3.common.env_util import make_vec_env
from sage.forks.stable_baselines3.stable_baselines3.common.logger import HumanOutputFormat

from analysis.old_domain_info_hash import old_domain_info_hashes

CONVENTIONS = ("oracle_sage", "vilg", "atom")
TERNARY_ENV_ID = "city-taxi-ternary-unmasked-v1"
OLD_ENV_ID = "city-taxi-unmasked-v1"


def counts_from(sim):
    return dict(sim.dropoff_counts)


# ==========================================================================================
# (a) Hand-checkable: a known crossed pair on a 2x2 grid, driven through sim.act
# ==========================================================================================

class TestHandChecked(unittest.TestCase):
    """
    2x2 grid, no walls: roads 1-2, 1-3, 2-4, 3-4 (both directions). Same injected pair as
    test_ternary_world.TestHandCheckedPair:

        p (pid 5): requests (1 -> 4), (3 -> 2); waits at 1 (the taxi's start)
        q (pid 6): requests (1 -> 2), (3 -> 4); waits at 3

    The object encoding sees req23 edges (1,4), (3,2), (1,2), (3,4) for both p and q, so
    the cluster {p, q} has two valid hypotheses -> ambiguous while both are present.
    After p is delivered, q (renumbered to 5) has only its own two req23 edges -> one
    hypothesis -> not ambiguous.

    Script and expected records (written before running):
      act(0)  nobody aboard                  -> DROPOFF_EMPTY only, no record
      act(5)  pick up p at 1                 -> p's true destination is 4
      act(2), act(0)  at 2 = p's OTHER dest  -> (5, 2, success=F, ambiguous=T, own_other=T)
      act(4), act(0)  at 4                   -> (5, 4, success=T, ambiguous=T, own_other=F)
      act(3), act(5)  pick up q (now 5) at 3 -> q's true destination is 4
      act(0)  at 3, not a q destination      -> (5, 3, success=F, ambiguous=F, own_other=F)
      act(4), act(0)  at 4                   -> (5, 4, success=T, ambiguous=F, own_other=F); done
    """

    EXPECTED_RECORDS = [
        DropoffRecord(5, 2, False, True, True),
        DropoffRecord(5, 4, True, True, False),
        DropoffRecord(5, 3, False, False, False),
        DropoffRecord(5, 4, True, False, False),
    ]
    EXPECTED_COUNTS = {
        DROPOFF_ATTEMPTS: 4,
        DROPOFF_ATTEMPTS_AMBIGUOUS: 2,
        DROPOFF_FAILURES: 2,
        DROPOFF_FAILURES_AMBIGUOUS: 1,
        DROPOFF_FAILURES_OWN_OTHER: 1,
        DROPOFF_EMPTY: 1,
    }

    def setUp(self):
        self.sim = TernaryTaxiWorldSimulator(
            np.random.RandomState(0),
            size=2,
            random_walls=False,
            delivery_limit=2,
            concurrent_passengers=2,
            requests_per_passenger=2,
            passenger_creation_probability=0,
            observation_fn=lambda s: None,
            initial_pair=(((1, 4), (3, 2)), 1, ((1, 2), (3, 4)), 3),
        )
        self.assertEqual(self.sim.taxi.location, 1)

    def test_records_and_counts(self):
        sim = self.sim
        zero = {key: 0 for key in DROPOFF_DIAGNOSTIC_KEYS}
        self.assertEqual(counts_from(sim), zero)

        _obs, _r, _d, info = sim.act(0)
        self.assertEqual(info, dict(zero, **{DROPOFF_EMPTY: 1}))
        self.assertEqual(sim.dropoff_records, [])

        sim.act(5)
        self.assertEqual(sim.taxi.passenger, 5)
        sim.act(2)
        _obs, reward, _d, info = sim.act(0)
        self.assertEqual(reward, sim.rewards["failed-action"])
        self.assertEqual(sim.dropoff_records, self.EXPECTED_RECORDS[:1])
        self.assertEqual(info, {DROPOFF_EMPTY: 1, DROPOFF_ATTEMPTS: 1, DROPOFF_ATTEMPTS_AMBIGUOUS: 1,
                                DROPOFF_FAILURES: 1, DROPOFF_FAILURES_AMBIGUOUS: 1,
                                DROPOFF_FAILURES_OWN_OTHER: 1})

        sim.act(4)
        _obs, reward, done, _info = sim.act(0)
        self.assertEqual(reward, sim.rewards["drop-off"])
        self.assertFalse(done)
        self.assertEqual(sorted(sim.passengers), [5])  # q renumbered 6 -> 5

        sim.act(3)
        sim.act(5)
        self.assertEqual(sim.taxi.passenger, 5)
        _obs, reward, _d, _info = sim.act(0)
        self.assertEqual(reward, sim.rewards["failed-action"])

        sim.act(4)
        _obs, reward, done, info = sim.act(0)
        self.assertEqual(reward, sim.rewards["drop-off"])
        self.assertTrue(done)

        self.assertEqual(sim.dropoff_records, self.EXPECTED_RECORDS)
        self.assertEqual(info, self.EXPECTED_COUNTS)
        self.assertEqual(counts_from(sim), self.EXPECTED_COUNTS)

    def test_object_hypothesis_count_directly(self):
        """The public function on the same two states: 2 hypotheses with both
        passengers present, 1 after p's delivery."""
        meta = {"time": 0, "timeout": 2000, "planning": True}
        from sage.domains.gym_taxi.utils.ternary_representations import facts_to_object_graph
        nf, ef, ei, _m, _g = facts_to_object_graph(self.sim.facts(), meta)
        self.assertEqual(object_hypothesis_count(nf, ei, ef, 5), 2)
        self.assertEqual(object_hypothesis_count(nf, ei, ef, 6), 2)
        with self.assertRaises(ValueError):
            object_hypothesis_count(nf, ei, ef, 1)  # a location, not a passenger

        for action in (5, 2, 4, 0):  # pick up p, drive 1 -> 2 -> 4, deliver
            self.sim.act(action)
        nf, ef, ei, _m, _g = facts_to_object_graph(self.sim.facts(), meta)
        self.assertEqual(object_hypothesis_count(nf, ei, ef, 5), 1)

    def test_diagnostic_changes_no_state(self):
        """Recording a dropoff attempt must not change the simulation: same facts,
        rewards and RNG stream as a run with the diagnostic stubbed out."""
        twin = copy.deepcopy(self.sim)
        twin._record_dropoff_attempt = lambda *args: None
        for action in (0, 5, 2, 0, 4, 0, 3, 5, 0, 4, 0):
            a = self.sim.act(action)
            b = twin.act(action)
            self.assertEqual(a[1:3], b[1:3])
            self.assertEqual(self.sim.facts(), twin.facts())
        self.assertEqual(self.sim.random.get_state()[1].tolist(), twin.random.get_state()[1].tolist())


# ==========================================================================================
# (b) Agreement on 500+ real states: env-side flag vs the planner's own decoder input path
# ==========================================================================================

def sample_carried_states(seeds=(0, 1, 2, 3, 4), sample_every=3, per_seed_target=110, random_prob=0.2):
    """(sim snapshot, carried pid) pairs from scripted play, sampled only while a
    passenger is aboard -- the only moment the diagnostic computes ambiguity."""
    samples = []
    for seed in seeds:
        sim = make_ternary_sim(seed)
        rng = pyrandom.Random(seed + 7000)
        step = collected = 0
        while collected < per_seed_target:
            if step % sample_every == 0 and sim.taxi.passenger is not None:
                samples.append((copy.deepcopy(sim), sim.taxi.passenger))
                collected += 1
            sim.act(scripted_policy(sim, rng, random_prob=random_prob))
            step += 1
            if sim.done:
                sim = make_ternary_sim(seed * 100003 + step)
    return samples


class TestAgreementWithDecoder(unittest.TestCase):
    """Expected (written before running): env flag == decoder flag on every state;
    roughly 58% of carried-passenger states ambiguous (analysis/README.md, Part 2 §2);
    cluster-level ambiguity == destination-level ambiguity on every state (the README's
    k=2 buddy-rotation argument), i.e. 0 disagreements."""

    @classmethod
    def setUpClass(cls):
        cls.samples = sample_carried_states()
        assert len(cls.samples) >= 500

    def test_env_flag_equals_decoder_on_policy_side_data(self):
        n_ambiguous = 0
        n_destination_disagree = 0
        for sim, pid in self.samples:
            env_flag = sim._object_ambiguous(pid)

            # The planner's real input path: compact JSON -> policy-side converter ->
            # Batch -> to_data_list() (float32/long torch tensors, not the env's numpy).
            meta = {"time": sim.time, "timeout": sim.timeout, "planning": sim.planning}
            data = json_to_ternary_graph_object([[facts_to_json(sim.facts(), meta)]]).to_data_list()[0]
            _types, _base, cluster_solutions = _object_cluster_solutions(data)
            solutions = next(s for cluster, s in cluster_solutions if pid in cluster)
            decoder_flag = len(solutions) > 1
            self.assertEqual(env_flag, decoder_flag, f"pid {pid}: env {env_flag} vs decoder {decoder_flag}")

            origin = sim.passengers[pid].picked_up_at
            destinations = {d for solution in solutions for o, d in solution[pid] if o == origin}
            if (len(destinations) > 1) != decoder_flag:
                n_destination_disagree += 1
            n_ambiguous += env_flag

        n = len(self.samples)
        print(f"\n[agreement] carried-passenger states={n} ambiguous={n_ambiguous} ({n_ambiguous / n:.1%}); "
              f"cluster-vs-destination ambiguity disagreements={n_destination_disagree}")
        self.assertEqual(n_destination_disagree, 0)


# ==========================================================================================
# (c) Behaviour per convention: planner-driven episodes, the convention's real planner
# ==========================================================================================

def choose_goal(sim, rng, mode):
    """mode "deliver": the carried passenger if any, else a random waiting passenger
    (never a waiting passenger while carrying). mode "any": a random passenger, waiting
    or carried -- includes the old planner's "pick up someone else while carrying" plans."""
    waiting = [pid for pid, p in sim.passengers.items() if p.location is not None]
    carried = sim.taxi.passenger
    if mode == "deliver":
        if carried is not None:
            return carried
        if waiting:
            return rng.choice(sorted(waiting))
    else:
        candidates = sorted(waiting + ([carried] if carried is not None else []))
        if candidates:
            return rng.choice(candidates)
    return rng.choice(sorted(sim.roads.successors(sim.taxi.location)))


def run_planner_driven(convention, mode, seeds, steps_per_seed):
    """Plans with Planner(convention, ternary=True).plan on that convention's graph,
    executes every action of the plan with sim.act, repeats. Returns summed
    dropoff_counts over all seeds."""
    totals = {key: 0 for key in DROPOFF_DIAGNOSTIC_KEYS}
    planner = Planner(graph_convention=convention, ternary=True)
    for seed in seeds:
        sim = make_ternary_sim(seed)
        rng = pyrandom.Random(seed + 9000)
        steps = 0
        while steps < steps_per_seed and not sim.done:
            goal = choose_goal(sim, rng, mode)
            _projection, actions = planner.plan(make_graph(sim, convention), goal)
            for action in actions:
                sim.act(int(action))
                steps += 1
                if sim.done or steps >= steps_per_seed:
                    break
        for key in DROPOFF_DIAGNOSTIC_KEYS:
            totals[key] += sim.dropoff_counts[key]
    return totals


def summarise(totals):
    n_amb = totals[DROPOFF_ATTEMPTS_AMBIGUOUS]
    n_not = totals[DROPOFF_ATTEMPTS] - n_amb
    fail_amb = totals[DROPOFF_FAILURES_AMBIGUOUS]
    fail_not = totals[DROPOFF_FAILURES] - fail_amb
    return {
        "attempts": totals[DROPOFF_ATTEMPTS],
        "n_amb": n_amb, "p_fail_amb": fail_amb / n_amb if n_amb else float("nan"),
        "n_not": n_not, "p_fail_not": fail_not / n_not if n_not else float("nan"),
        "failures": totals[DROPOFF_FAILURES],
        "own_other": totals[DROPOFF_FAILURES_OWN_OTHER],
        "own_other_frac": (totals[DROPOFF_FAILURES_OWN_OTHER] / totals[DROPOFF_FAILURES]
                           if totals[DROPOFF_FAILURES] else float("nan")),
        "empty": totals[DROPOFF_EMPTY],
    }


def format_summary(convention, mode, s):
    return (f"[{mode}] {convention:11s} attempts={s['attempts']:4d}  "
            f"P(fail|amb)={s['p_fail_amb']:.3f} (n={s['n_amb']})  "
            f"P(fail|not amb)={s['p_fail_not']:.3f} (n={s['n_not']})  "
            f"failures={s['failures']} own_other={s['own_other']} ({s['own_other_frac']:.2f})  empty={s['empty']}")


class TestBehaviourPerConvention(unittest.TestCase):
    """
    Expected (written before running), mode "deliver":
      object: P(fail | ambiguous) ~ 0.5 (two hypotheses, uniform tie-break), within a
        99.9% binomial window (|p - 0.5| <= 3.29 * sqrt(0.25 / n)), n >= 100; failures
        are almost all own-other-destination.
      atom, vilg: zero failures, hence zero own-other-destination failures.
    Mode "any" is report-only: it adds the old planner's "pick up someone else while
    carrying" plans, which can fail for EVERY convention -- including landing on the
    carried passenger's own other destination when the goal is its buddy.
    """

    SEEDS = (0, 1, 2, 3)
    STEPS_PER_SEED = 2000

    def test_deliver_mode(self):
        for convention in CONVENTIONS:
            s = summarise(run_planner_driven(convention, "deliver", self.SEEDS, self.STEPS_PER_SEED))
            print("\n" + format_summary(convention, "deliver", s))
            if convention == "oracle_sage":
                self.assertGreaterEqual(s["n_amb"], 100)
                window = 3.29 * math.sqrt(0.25 / s["n_amb"])
                self.assertLessEqual(abs(s["p_fail_amb"] - 0.5), window,
                                     f"P(fail|amb)={s['p_fail_amb']:.3f} outside 0.5 +/- {window:.3f}")
            else:
                self.assertEqual(s["own_other"], 0, f"{convention}: own-other-destination failures")
                self.assertEqual(s["failures"], 0, f"{convention}: dropoff failures with an exact decoder")

    def test_any_goal_mode_report_only(self):
        for convention in CONVENTIONS:
            s = summarise(run_planner_driven(convention, "any", (0, 1), 1500))
            print("\n" + format_summary(convention, "any", s))


# ==========================================================================================
# (d) Real path: gnn_global.py's info_keywords + make_vec_env/AsyncVecEnv + Monitor
# ==========================================================================================

SHORT_TIMEOUT = 120


def make_short_ternary_env_factory(convention):
    """The registered ternary env (constructed from its spec's kwargs via compat, as
    gnn_global.py's gym.make would), with the episode timeout shortened on this instance
    only -- a test-only scenario so an episode ends within the test."""
    def make_env():
        env = GraphTaxiEnv(**spec_kwargs(gym_module.spec(TERNARY_ENV_ID)), graph_convention=convention)
        env.scenario = dict(env.scenario, timeout=SHORT_TIMEOUT)
        return env
    return make_env


class TestRealPath(unittest.TestCase):
    def test_gnn_global_info_keywords(self):
        self.assertEqual(info_keywords_for(OLD_ENV_ID), ("len100", "len200"))
        self.assertEqual(info_keywords_for(TERNARY_ENV_ID), ("len100", "len200") + DROPOFF_DIAGNOSTIC_KEYS)

    def test_episode_counts_reach_monitor_episode_info(self):
        """One full (short) episode through make_vec_env + AsyncVecEnv + Monitor with
        gnn_global's info_keywords, driven by the real planner on the decoded
        observation. The terminal info's Monitor "episode" entry must carry exactly the
        finished episode's counters."""
        for convention in CONVENTIONS:
            env = make_vec_env(
                make_short_ternary_env_factory(convention), n_envs=1, seed=0,
                monitor_kwargs={"info_keywords": info_keywords_for(TERNARY_ENV_ID)}, vec_env_cls=AsyncVecEnv,
            )
            obs = env.reset()
            raw_env = env.envs[0].unwrapped
            planner = raw_env.observation_space.planner
            rng = pyrandom.Random(0)
            episode_info = None
            while episode_info is None:
                sim = raw_env.sim  # the episode's sim; replaced on auto-reset
                data = raw_env.observation_space.converter(obs).to_data_list()[0]
                _projection, actions = planner.plan(data, choose_goal(sim, rng, "deliver"))
                for action in actions:
                    obs, _r, dones, infos = env.step(np.array([int(action)]), np.array([True]))
                    if dones[0]:
                        episode_info = infos[0]["episode"]
                        finished_sim = sim
                        break
            with self.subTest(convention=convention):
                self.assertEqual(episode_info["l"], SHORT_TIMEOUT)
                for key in DROPOFF_DIAGNOSTIC_KEYS:
                    self.assertEqual(episode_info[key], finished_sim.dropoff_counts[key], key)
                self.assertGreater(episode_info[DROPOFF_ATTEMPTS], 0)
            env.close()

    def test_counts_logged_as_progress_keys_during_learn(self):
        """PlanFeedback_A2C.learn: the counters reach the training log as progress/<key>
        (OnPolicyAlgorithm logs every extra Monitor episode key), next to
        rollout/ep_rew_mean."""
        env = make_vec_env(
            make_short_ternary_env_factory("oracle_sage"), n_envs=1, seed=0,
            monitor_kwargs={"info_keywords": info_keywords_for(TERNARY_ENV_ID)}, vec_env_cls=AsyncVecEnv,
        )
        model = PlanFeedback_A2C(
            GNNPlanFeedbackPolicy, env, verbose=0, device="cpu",
            supported_action_spaces=(sage_spaces.BinaryAction, gym_module.spaces.Discrete, sage_spaces.Autoregressive),
            n_steps=5, policy_kwargs=dict(POLICY_KWARGS, exploration_fraction=0.1),
        )
        recorded = {}
        original_record = sb3_logger.record

        def capture(key, value, *args, **kwargs):
            recorded[key] = value
            return original_record(key, value, *args, **kwargs)

        with mock.patch.object(sb3_logger, "record", side_effect=capture):
            model.learn(total_timesteps=60, log_interval=1)
        env.close()

        self.assertGreater(len(model.ep_info_buffer), 0, "no episode finished during learn")
        self.assertIn("rollout/ep_rew_mean", recorded)
        for key in DROPOFF_DIAGNOSTIC_KEYS:
            self.assertIn(f"progress/{key}", recorded, key)
            for ep_info in model.ep_info_buffer:
                self.assertIn(key, ep_info)

        # Everything learn() recorded, rendered by the fork's own stdout table: every
        # diagnostic key must print in full, exactly once.
        rows = printed_table_rows({key: value for key, value in recorded.items() if not isinstance(value, th.Tensor)})
        for key in DROPOFF_DIAGNOSTIC_KEYS:
            self.assertEqual(sum(1 for row_key, _value in rows if row_key == key), 1, f"{key} not printed exactly once")


def printed_table_rows(key_values):
    """(key, value) pairs of the table the SB3 fork's HumanOutputFormat prints for
    key_values -- the printed log the training chart script reads. Keys are shown
    without their "progress/"-style group prefix, after the logger's own truncation."""
    out = io.StringIO()
    HumanOutputFormat(out).write(key_values, {key: None for key in key_values})
    rows = []
    for line in out.getvalue().splitlines():
        if line.startswith("|"):
            _left, key, value, _right = line.split("|")
            rows.append((key.strip(), value.strip()))
    return rows


class TestLoggedKeysSurviveTruncation(unittest.TestCase):
    """The fork's printed table shows a key's part after its first "/" indented by 3
    spaces, truncated to 23 characters, and stores rows in a dict keyed by the truncated
    text -- so a key longer than 20 characters is cut, and two that cut alike overwrite
    each other (the bug that hid one of the old dropoff_failures_* keys)."""

    def test_each_diagnostic_key_prints_in_full_and_once(self):
        key_values = {"rollout/ep_rew_mean": 0.5, "rollout/ep_len_mean": 100.0}
        expected = {}
        for i, key in enumerate(DROPOFF_DIAGNOSTIC_KEYS):
            key_values[f"progress/{key}"] = 1000 + i  # distinct values expose an overwrite
            expected[key] = str(1000 + i)
        rows = printed_table_rows(key_values)
        for key, value in expected.items():
            matches = [row_value for row_key, row_value in rows if row_key == key]
            self.assertEqual(matches, [value], f"progress/{key}: printed rows {rows}")

    def test_display_keys_fit_without_truncation(self):
        displayed = ["   " + key for key in DROPOFF_DIAGNOSTIC_KEYS]
        self.assertEqual([HumanOutputFormat._truncate(d) for d in displayed], displayed)
        self.assertEqual(len(set(displayed)), len(displayed))


# ==========================================================================================
# (e) Old domain: info dicts and Monitor CSV unchanged vs the pre-commit reference
# ==========================================================================================

# Computed with analysis/old_domain_info_hash.py; keyed by stack like
# test_ternary_wiring.OLD_DOMAIN_REFERENCE_SHA256 (env.seed differs by gym version).
# Each entry is (monitor_sha256, info_sha256).
#   gym0.26: Mac (gym 0.26.2); identical at 82c885f (before Task A) and with Task A applied.
#   gym0.18: RCP (gym 0.18.0); identical at the pre-ternary commit a628f16 and with
#            Task A applied, for every convention.
OLD_DOMAIN_INFO_REFERENCE_SHA256 = {
    "gym0.26": {
        "oracle_sage": ("9a3157f2768359c71817a3a9bdf4971f8b1f07682709c64d38ccd4cba560fbbc",
                        "37361c611b3f02d92f87049368fb3c50030c4bf3924626219beb5817a8a8ecf0"),
        "vilg": ("9a3157f2768359c71817a3a9bdf4971f8b1f07682709c64d38ccd4cba560fbbc",
                 "c60554f0cf090cad9b9f484631c39a8e53c3c8ef666eb076f4f28a82950a1a9c"),
        "atom": ("9a3157f2768359c71817a3a9bdf4971f8b1f07682709c64d38ccd4cba560fbbc",
                 "2b27206df4a5a0985c185fd82e709ccdd5e1f2cbf2592d093823329941d5c570"),
    },
    "gym0.18": {
        "oracle_sage": ("9a3157f2768359c71817a3a9bdf4971f8b1f07682709c64d38ccd4cba560fbbc",
                        "56e7b5932943bd524ab83d39cfc9e498d2d19174b6b394c80183e07fa7c6f27a"),
        "vilg": ("9a3157f2768359c71817a3a9bdf4971f8b1f07682709c64d38ccd4cba560fbbc",
                 "f1b23181a080adcad02c597291234b467fb3ac55b4f968757cc0749a125c5e6f"),
        "atom": ("9a3157f2768359c71817a3a9bdf4971f8b1f07682709c64d38ccd4cba560fbbc",
                 "121d53778c32495842f6dd3f198fc6615405e03486b429b6456649b849efae1c"),
    },
}


def _stack_key():
    return "gym" + ".".join(gym_module.__version__.split(".")[:2])


class TestOldDomainInfoUnchanged(unittest.TestCase):
    def test_info_and_monitor_match_pre_commit_reference(self):
        stack = _stack_key()
        references = OLD_DOMAIN_INFO_REFERENCE_SHA256.get(stack, {})
        missing = [c for c in CONVENTIONS if c not in references]
        if missing:
            self.skipTest(
                f"no old-domain info/Monitor reference for stack {stack!r} (gym {gym_module.__version__}), "
                f"conventions {missing}: compute them with "
                f"`PYTHONPATH=. python analysis/old_domain_info_hash.py <convention>` at commit 82c885f "
                f"on this stack and add them to OLD_DOMAIN_INFO_REFERENCE_SHA256[{stack!r}]"
            )
        # gnn_global.py's own choice for the old env -- the header the reference was
        # recorded with ("len100", "len200") must still be what it passes.
        keywords = info_keywords_for(OLD_ENV_ID)
        self.assertEqual(keywords, ("len100", "len200"))
        for convention in CONVENTIONS:
            with self.subTest(convention=convention):
                self.assertEqual(old_domain_info_hashes(convention, info_keywords=keywords), references[convention])


if __name__ == "__main__":
    unittest.main()
