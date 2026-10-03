"""
Tests for the Cell 1 oracle-decoder condition (Task B): GraphTaxiEnv(oracle_decoder=True),
json_to_ternary_graph_object_oracle / oracle_facts, Planner(oracle=True) /
plan_ternary(oracle=...), and gnn_global.py's --oracle-decoder.

The GNN sees the object encoding exactly as in Cell 1; the planner receives the TRUE
facts (oracle_facts) instead of the object hypothesis decoder's guess.

Expected values were written into these tests before they were first run.

Run from the repo root with:
    PYTHONPATH=. python -m pytest tests/test_oracle_decoder.py -v -s
Do not run while a training run is using the GPU.
"""
import copy
import random as pyrandom
import unittest
from unittest import mock

import gym as gym_module
import numpy as np
import torch as th

# test_ternary_planner installs the gym-0.26 randint shim at import time.
from tests.test_ternary_planner import (
    POLICY_KWARGS,
    execute_with_rewards,
    make_graph,
    make_ternary_sim,
    sample_real_crossed_sim_copies,
)
from tests.test_ternary_diagnostic import choose_goal, format_summary, summarise

import sage.experiments.gnn_global as gnn_global
import sage.domains.gym_taxi.simulator.ternary_planner as ternary_planner
from sage.agent.async_vec_env import AsyncVecEnv
from sage.agent.graph_plan_feedback_policy import GNNPlanFeedbackPolicy
from sage.agent.plan_feedback_a2c import PlanFeedback_A2C
from sage.domains.gym_taxi import REWARDS
from sage.domains.gym_taxi.envs.taxi_env import GRAPH_CONVENTION_CONVERTERS, GraphTaxiEnv
from sage.domains.gym_taxi.simulator.planner import Planner
from sage.domains.gym_taxi.simulator.ternary_taxi_world import DROPOFF_DIAGNOSTIC_KEYS
from sage.domains.gym_taxi.utils.compat import data_keys, spec_kwargs
from sage.domains.gym_taxi.utils.ternary_representations import (
    TERNARY_GRAPH_CONVENTION_CONVERTERS,
    facts_to_json,
    facts_to_object_graph,
    facts_to_oracle_facts,
    json_to_ternary_graph_object,
    json_to_ternary_graph_object_oracle,
    oracle_facts_to_facts,
)
from sage.domains.utils import spaces as sage_spaces
from sage.forks.stable_baselines3.stable_baselines3.common.env_util import make_vec_env

TERNARY_ENV_ID = "city-taxi-ternary-unmasked-v1"
OLD_ENV_ID = "city-taxi-unmasked-v1"
BASE_KEYS = {"x", "edge_index", "edge_attr", "mask", "global_features"}
CUDA = th.cuda.is_available()


def sim_json(sim):
    meta = {"time": sim.time, "timeout": sim.timeout, "planning": sim.planning}
    return facts_to_json(sim.facts(), meta)


def round_trip(converter, sims, device="cpu"):
    """The planner's real input path: JSON strings -> converter (which builds the Batch
    with Batch.from_data_list) -> .to(device) -> to_data_list()."""
    return converter([[sim_json(sim)] for sim in sims]).to(device).to_data_list()


def oracle_graph(sim, device="cpu"):
    return round_trip(json_to_ternary_graph_object_oracle, [sim], device)[0]


def make_oracle_env(**overrides):
    kwargs = dict(spec_kwargs(gym_module.spec(TERNARY_ENV_ID)), graph_convention="oracle_sage", oracle_decoder=True)
    kwargs.update(overrides)
    return GraphTaxiEnv(**kwargs)


def build_model(oracle_decoder, device="cpu"):
    env = make_vec_env(
        lambda: GraphTaxiEnv(**spec_kwargs(gym_module.spec(TERNARY_ENV_ID)), oracle_decoder=oracle_decoder),
        n_envs=2, seed=0, monitor_kwargs={"info_keywords": gnn_global.info_keywords_for(TERNARY_ENV_ID)},
        vec_env_cls=AsyncVecEnv,
    )
    model = PlanFeedback_A2C(
        GNNPlanFeedbackPolicy, env, verbose=0, device=device,
        supported_action_spaces=(sage_spaces.BinaryAction, gym_module.spaces.Discrete, sage_spaces.Autoregressive),
        n_steps=5, policy_kwargs=dict(POLICY_KWARGS, exploration_fraction=0.1),
    )
    return env, model


# ==========================================================================================
# (a) With the flag: true destinations, exact projections, zero own-other failures
# ==========================================================================================

class TestOracleDeliveries(unittest.TestCase):
    """
    Expected (written before running):
      crossed states: oracle planner delivers 75/75, every projection == executed env
        exactly; the hypothesis decoder on the same 75 delivers 31/75 (as measured in
        test_ternary_planner's TestDestinationCorrectness).
      planner-driven play: identical to atom/vilg in test_ternary_diagnostic (same seeds,
        goals and true-fact plans): "deliver" 180 attempts, 106 ambiguous, 74 not,
        0 failures; "any" 63 attempts, 0 failures.
    """

    @classmethod
    def setUpClass(cls):
        cls.samples = sample_real_crossed_sim_copies(per_seed_target=15)  # 75 real crossed states

    def test_crossed_states_true_destination_and_exact_projection(self):
        oracle_planner = Planner(graph_convention="oracle_sage", ternary=True, oracle=True)
        hypothesis_planner = Planner(graph_convention="oracle_sage", ternary=True)
        n_oracle = n_hypothesis = 0
        for sim, pid in self.samples:
            projection, actions = oracle_planner.plan(oracle_graph(sim), pid)
            executed = copy.deepcopy(sim)
            executed.pair_creation_probability = 0  # see test_ternary_planner: no mid-plan spawns
            executed, rewards = execute_with_rewards(executed, actions)
            self.assertEqual(rewards[-1], sim.rewards["drop-off"], f"oracle plan did not deliver pid {pid}")
            n_oracle += 1

            meta = {"time": executed.time, "timeout": executed.timeout, "planning": executed.planning}
            nf, ef, ei, mask, _gf = facts_to_object_graph(executed.facts(), meta)
            np.testing.assert_array_equal(projection.x.numpy(), nf.astype(np.float32))
            np.testing.assert_array_equal(projection.edge_attr.numpy(), ef.astype(np.float32))
            np.testing.assert_array_equal(projection.edge_index.numpy(), ei.astype(np.int64))
            np.testing.assert_array_equal(projection.mask.numpy(), mask)

            _p, hyp_actions = hypothesis_planner.plan(make_graph(sim, "oracle_sage"), pid)
            _e, hyp_rewards = execute_with_rewards(copy.deepcopy(sim), hyp_actions)
            n_hypothesis += hyp_rewards[-1] == sim.rewards["drop-off"]
        n = len(self.samples)
        print(f"\n[oracle crossed states] oracle delivered {n_oracle}/{n}; hypothesis decoder delivered {n_hypothesis}/{n}")

    def _run(self, mode, seeds, steps_per_seed):
        planner = Planner(graph_convention="oracle_sage", ternary=True, oracle=True)
        totals = {key: 0 for key in DROPOFF_DIAGNOSTIC_KEYS}
        for seed in seeds:
            sim = make_ternary_sim(seed)
            rng = pyrandom.Random(seed + 9000)  # same goal stream as test_ternary_diagnostic
            steps = 0
            while steps < steps_per_seed and not sim.done:
                _projection, actions = planner.plan(oracle_graph(sim), choose_goal(sim, rng, mode))
                for action in actions:
                    sim.act(int(action))
                    steps += 1
                    if sim.done or steps >= steps_per_seed:
                        break
            for key in DROPOFF_DIAGNOSTIC_KEYS:
                totals[key] += sim.dropoff_counts[key]
        return summarise(totals)

    def test_planner_driven_deliver_mode_no_failures(self):
        s = self._run("deliver", (0, 1, 2, 3), 2000)
        print("\n" + format_summary("oracle", "deliver", s))
        self.assertGreater(s["n_amb"], 0)
        self.assertEqual(s["failures"], 0)
        self.assertEqual(s["own_other"], 0)

    def test_planner_driven_any_goal_mode_report(self):
        s = self._run("any", (0, 1), 1500)
        print("\n" + format_summary("oracle", "any", s))
        self.assertEqual(s["own_other"], 0)


# ==========================================================================================
# (b) GNN unaffected: _get_latent identical with and without oracle_facts attached
# ==========================================================================================

class TestGnnUnaffected(unittest.TestCase):
    """
    On CPU the GNN outputs must be bitwise identical with and without oracle_facts.

    On CUDA they are compared with th.allclose(rtol=1e-5, atol=1e-6) instead: the GNN's
    message aggregation and global pooling are scatter-adds, which on the GPU use atomic
    float additions whose order varies from run to run, and float addition is not
    associative -- so two forwards of the SAME input can differ in the last bits.
    analysis/oracle_cuda_determinism.py checks that the with/without-oracle_facts
    difference is no larger than this run-to-run noise. The symbolic inputs involve no
    arithmetic and stay exact on both devices.
    """

    def _assert_close_or_equal(self, a, b, label, exact):
        max_abs = (a - b).abs().max().item() if a.numel() else 0.0
        print(f"\n[{label}] max abs diff with vs without oracle_facts: {max_abs:.3e}")
        if exact:
            self.assertTrue(th.equal(a, b), f"{label} differ (max abs diff {max_abs:.3e})")
        else:
            self.assertTrue(th.allclose(a, b, rtol=1e-5, atol=1e-6), f"{label} differ (max abs diff {max_abs:.3e})")

    def _assert_latents_equal(self, device):
        env, model = build_model(oracle_decoder=True, device=device)
        policy = model.policy
        obs = env.reset()
        with th.no_grad():
            with_oracle, sym_with = policy._get_latent(obs)
            policy.observation_space.converter = json_to_ternary_graph_object  # same obs, no oracle_facts
            try:
                without_oracle, sym_without = policy._get_latent(obs)
            finally:
                policy.observation_space.converter = json_to_ternary_graph_object_oracle
            values_with = policy.value_net(with_oracle.global_features)
            values_without = policy.value_net(without_oracle.global_features)
        env.close()

        self.assertEqual(set(data_keys(sym_with)) - set(data_keys(sym_without)), {"oracle_facts"})
        for name in ("x", "edge_index", "edge_attr", "global_features"):
            self.assertTrue(th.equal(getattr(sym_with, name), getattr(sym_without, name)), f"symbolic {name}")
        exact = th.device(device).type == "cpu"
        self._assert_close_or_equal(with_oracle.x, without_oracle.x, f"{device} latent node embeddings", exact)
        self._assert_close_or_equal(with_oracle.global_features, without_oracle.global_features,
                                    f"{device} latent globals", exact)
        self._assert_close_or_equal(values_with, values_without, f"{device} values", exact)

    def test_get_latent_identical_cpu(self):
        self._assert_latents_equal("cpu")

    @unittest.skipIf(not CUDA, "requires CUDA")
    def test_get_latent_identical_cuda(self):
        self._assert_latents_equal("cuda")


# ==========================================================================================
# (c) Attributes after a REAL Batch.from_data_list -> .to(device) -> to_data_list round trip
# ==========================================================================================

class TestAttributesAfterRoundTrip(unittest.TestCase):
    """Several different real states in one batch (different fact counts, so
    oracle_facts is ragged across graphs) -- the case PyG must split back correctly."""

    @classmethod
    def setUpClass(cls):
        cls.sims = []
        for seed in (0, 1, 2):
            sim = make_ternary_sim(seed)
            rng = pyrandom.Random(seed)
            for _ in range(40 * (seed + 1)):  # random moves; passengers keep spawning
                sim.act(rng.choice(sorted(sim.roads.successors(sim.taxi.location))))
            cls.sims.append(sim)
        assert len({len(sim.facts()) for sim in cls.sims}) > 1, "want ragged fact counts"

    def _check(self, device):
        plain = round_trip(json_to_ternary_graph_object, self.sims, device)
        oracle = round_trip(json_to_ternary_graph_object_oracle, self.sims, device)
        for sim, d_plain, d_oracle in zip(self.sims, plain, oracle):
            self.assertEqual(set(data_keys(d_plain)), BASE_KEYS)
            self.assertEqual(set(data_keys(d_oracle)), BASE_KEYS | {"oracle_facts"})
            self.assertEqual(d_oracle.oracle_facts.dtype, th.long)
            self.assertEqual(d_oracle.oracle_facts.device.type, th.device(device).type)
            np.testing.assert_array_equal(d_oracle.oracle_facts.cpu().numpy(), facts_to_oracle_facts(sim.facts()))
            self.assertEqual(oracle_facts_to_facts(d_oracle.oracle_facts), sim.facts())
            for name in BASE_KEYS:
                self.assertTrue(th.equal(getattr(d_plain, name), getattr(d_oracle, name)), name)

    def test_round_trip_cpu(self):
        self._check("cpu")

    @unittest.skipIf(not CUDA, "requires CUDA")
    def test_round_trip_cuda(self):
        self._check("cuda")


# ==========================================================================================
# (d) Validation
# ==========================================================================================

class TestValidation(unittest.TestCase):
    def test_env_raises_for_old_domain_and_other_conventions(self):
        with self.assertRaises(ValueError):
            GraphTaxiEnv(representation="graph", scenario="city", mask=False, rewards=REWARDS["v1"], oracle_decoder=True)
        for convention in ("vilg", "atom"):
            with self.assertRaises(ValueError):
                make_oracle_env(graph_convention=convention)
        self.assertTrue(make_oracle_env().observation_space.planner.oracle)

    def test_gnn_global_raises_for_old_env_and_other_conventions(self):
        with self.assertRaises(ValueError):
            gnn_global.validate_oracle_decoder(OLD_ENV_ID, "oracle_sage")
        for convention in ("vilg", "atom"):
            with self.assertRaises(ValueError):
                gnn_global.validate_oracle_decoder(TERNARY_ENV_ID, convention)
        gnn_global.validate_oracle_decoder(TERNARY_ENV_ID, "oracle_sage")  # no raise

        with mock.patch.object(gnn_global, "run") as run:
            for argv in (["--env-name", OLD_ENV_ID, "--oracle-decoder"],
                         ["--env-name", TERNARY_ENV_ID, "--graph-convention", "atom", "--oracle-decoder"]):
                with self.assertRaises(ValueError):
                    gnn_global.main(argv)
            run.assert_not_called()

    def test_planner_raises_for_non_ternary_or_other_conventions(self):
        with self.assertRaises(ValueError):
            Planner(graph_convention="oracle_sage", ternary=False, oracle=True)
        for convention in ("vilg", "atom"):
            with self.assertRaises(ValueError):
                Planner(graph_convention=convention, ternary=True, oracle=True)

    def test_plan_ternary_raises_oracle_true_without_oracle_facts(self):
        sim = make_ternary_sim(0)
        plain = round_trip(json_to_ternary_graph_object, [sim])[0]
        planner = Planner(graph_convention="oracle_sage", ternary=True, oracle=True)
        with self.assertRaisesRegex(ValueError, "oracle=True but the observation has no oracle_facts"):
            planner.plan(plain, sorted(sim.passengers)[0])

    def test_plan_ternary_raises_oracle_false_with_oracle_facts(self):
        sim = make_ternary_sim(0)
        planner = Planner(graph_convention="oracle_sage", ternary=True)
        with self.assertRaisesRegex(ValueError, "oracle=False but the observation carries oracle_facts"):
            planner.plan(oracle_graph(sim), sorted(sim.passengers)[0])


# ==========================================================================================
# (e) Real path: GNNPlanFeedbackPolicy.forward() through project_actions with the flag
# ==========================================================================================

class TestRealPath(unittest.TestCase):
    def _forward(self, device):
        env, model = build_model(oracle_decoder=True, device=device)
        obs = env.reset()
        seen_devices = []
        real_decode = ternary_planner.decode_oracle_facts

        def spy(graph):
            seen_devices.append(graph.oracle_facts.device.type)
            return real_decode(graph)

        def never(graph):
            raise AssertionError("the hypothesis decoder must not run with --oracle-decoder")

        with mock.patch.object(ternary_planner, "decode_oracle_facts", side_effect=spy), \
                mock.patch.dict(ternary_planner._DECODERS, {"oracle_sage": never}):
            with th.no_grad():
                actions, values, _log_prob, _explored, plans = model.policy.forward(obs)
            # also the training update's evaluate_actions path, which re-plans
            model.learn(total_timesteps=10, log_interval=None)
        env.close()

        self.assertEqual(actions.device.type, th.device(device).type)
        self.assertEqual(len(plans), 2)
        # forward(): num_planning_choices (5) candidates x 2 envs, then more during learn
        self.assertGreaterEqual(len(seen_devices), 10)
        self.assertEqual(set(seen_devices), {th.device(device).type})

    def test_forward_cpu(self):
        self._forward("cpu")

    @unittest.skipIf(not CUDA, "requires CUDA")
    def test_forward_cuda(self):
        self._forward("cuda")


# ==========================================================================================
# (f) Run naming
# ==========================================================================================

class TestRunNaming(unittest.TestCase):
    ARGS = ["--env-name", TERNARY_ENV_ID, "--save-dir", "./trained_models/cell1_seed0", "--log-dir", "./logs/cell1_seed0"]

    def _variant(self, argv):
        with mock.patch.object(gnn_global, "run") as run:
            gnn_global.main(argv)
        return run.call_args[0][0]

    def test_flag_changes_save_and_log_dirs(self):
        plain = self._variant(self.ARGS)
        oracle = self._variant(self.ARGS + ["--oracle-decoder"])
        self.assertEqual(plain["save_dir"], "./trained_models/cell1_seed0")
        self.assertEqual(plain["algorithm_kwargs"]["tensorboard_log"], "./logs/cell1_seed0")
        self.assertEqual(oracle["save_dir"], "./trained_models/cell1_seed0_oracle_decoder")
        self.assertEqual(oracle["algorithm_kwargs"]["tensorboard_log"], "./logs/cell1_seed0_oracle_decoder")
        self.assertFalse(plain["oracle_decoder"])
        self.assertTrue(oracle["oracle_decoder"])

    def test_default_dirs_also_suffixed(self):
        oracle = self._variant(["--env-name", TERNARY_ENV_ID, "--oracle-decoder"])
        self.assertEqual(oracle["save_dir"], "./trained_models_oracle_decoder")
        self.assertEqual(oracle["algorithm_kwargs"]["tensorboard_log"], "./logs_oracle_decoder")

    def test_run_passes_flag_to_the_env(self):
        """run() builds env_kwargs; constructing the registered env with them (as
        gym.make would) yields an oracle planner. Stops at make_vec_env."""
        captured = {}

        class Stop(Exception):
            pass

        def fake_make_vec_env(env_id, **kwargs):
            captured.update(kwargs, env_id=env_id)
            raise Stop()

        for flag in (False, True):
            variant = self._variant(self.ARGS + (["--oracle-decoder", "--planner"] if flag else ["--planner"]))
            with mock.patch.object(gnn_global, "make_vec_env", side_effect=fake_make_vec_env):
                with self.assertRaises(Stop):
                    gnn_global.run(variant)
            self.assertEqual(captured["env_kwargs"], {"oracle_decoder": True} if flag else {})
            env = GraphTaxiEnv(**spec_kwargs(gym_module.spec(captured["env_id"])), **captured["env_kwargs"])
            self.assertIs(env.observation_space.planner.oracle, flag)
            env.close()


# ==========================================================================================
# (g) Old domain and the non-oracle ternary env are wired exactly as before
#     (byte-identity of old-domain observations/info/Monitor output is checked by
#     test_ternary_wiring.TestOldDomainUnchanged and
#     test_ternary_diagnostic.TestOldDomainInfoUnchanged)
# ==========================================================================================

class TestWithoutFlagUnchanged(unittest.TestCase):
    def test_old_domain_env(self):
        for convention in ("oracle_sage", "vilg", "atom"):
            env = GraphTaxiEnv(representation="graph", scenario="city", mask=False, rewards=REWARDS["v1"],
                               graph_convention=convention)
            self.assertIs(env.observation_space.converter, GRAPH_CONVENTION_CONVERTERS[convention])
            self.assertFalse(env.observation_space.planner.oracle)
            self.assertFalse(env.observation_space.planner.ternary)
            env.close()

    def test_ternary_env_without_flag(self):
        for convention in ("oracle_sage", "vilg", "atom"):
            env = GraphTaxiEnv(**spec_kwargs(gym_module.spec(TERNARY_ENV_ID)), graph_convention=convention)
            self.assertIs(env.observation_space.converter, TERNARY_GRAPH_CONVENTION_CONVERTERS[convention])
            self.assertFalse(env.observation_space.planner.oracle)
            env.close()


if __name__ == "__main__":
    unittest.main()
