"""
W4 (paper-edges): --edge-mode wiring, from the CLI to the policy and back from a saved zip.

- CLI: --edge-mode reaches policy_kwargs only when not gr; mgr/egr runs get a "_mgr"/"_egr"
  suffix on --save-dir/--log-dir and an "edge mode:" startup line; the skipped and GNN-only
  combinations are refused.
- Observation and projection agree: in mgr/egr every GNN pass (the observation, the current
  state and every planner projection) goes through the mode's conversion, while the planner
  itself still receives Gr graphs.
- End to end: every new variant (MGr: ternary Object, standard/ternary Atom; EGr: all three
  representations, both domains) trains for two updates on the CPU, every parameter is in
  the optimizer, and save -> model_io.load_policy gives the same weights and outputs.
- WL refuses mgr/egr; the oracle decoder works with egr.

Run from the repo root (CPU; not while a training run uses the GPU):
    PYTHONPATH=. python -m pytest tests/test_edge_mode_policy.py -v
"""
import os
import shutil
import tempfile
import unittest
from unittest import mock

import gym as gym_module
import torch as th

import tests.test_ternary_planner  # noqa: F401  installs the gym-0.26 randint shim
import sage.agent.graph_policy as graph_policy
import sage.experiments.gnn_global as gnn_global
from sage.agent.async_vec_env import AsyncVecEnv
from sage.agent.graph_net import MultiMessagePassing, TypedMultiMessagePassing
from sage.agent.graph_plan_feedback_policy import GNNPlanFeedbackPolicy
from sage.agent.model_io import load_policy, read_zip, saved_policy_kwargs
from sage.agent.plan_feedback_a2c import PlanFeedback_A2C
from sage.agent.wl_plan_feedback_policy import WLPlanFeedbackPolicy
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.simulator.planner import Planner
from sage.domains.gym_taxi.utils.compat import spec_kwargs
from sage.domains.gym_taxi.utils.wl_vocab_cache import reset_wl_vocab_override
from sage.domains.utils import spaces as sage_spaces
from sage.forks.stable_baselines3.stable_baselines3.common.env_util import make_vec_env
from tests.test_paper_unified_base import POLICY_KWARGS, STD_ENV_ID, TERNARY_ENV_ID
from tests.test_ternary_wl_training_path import committed_vocab_path, wl_args

SUPPORTED = (sage_spaces.BinaryAction, gym_module.spaces.Discrete, sage_spaces.Autoregressive)

# every new variant: (env, convention, mode)
VARIANTS = [
    (TERNARY_ENV_ID, "oracle_sage", "mgr"),
    (STD_ENV_ID, "atom", "mgr"),
    (TERNARY_ENV_ID, "atom", "mgr"),
] + [(env, conv, "egr") for env in (STD_ENV_ID, TERNARY_ENV_ID) for conv in ("oracle_sage", "vilg", "atom")]


def short(env_id):
    return "ter" if env_id == TERNARY_ENV_ID else "std"


def env_kwargs(env_id, convention, **extra):
    return dict(spec_kwargs(gym_module.spec(env_id)), graph_convention=convention, **extra)


def build(env_id, convention, edge_mode, seed=1, n_envs=2, n_steps=3, policy_cls=GNNPlanFeedbackPolicy,
          extra_policy_kwargs=None, **env_extra):
    policy_kwargs = dict(POLICY_KWARGS)
    if edge_mode != "gr":
        policy_kwargs["edge_mode"] = edge_mode
    policy_kwargs.update(extra_policy_kwargs or {})
    kwargs = env_kwargs(env_id, convention, **env_extra)
    env = make_vec_env(lambda: GraphTaxiEnv(**kwargs), n_envs=n_envs, seed=seed,
                       monitor_kwargs={"info_keywords": ("len100", "len200")}, vec_env_cls=AsyncVecEnv)
    model = PlanFeedback_A2C(policy_cls, env, verbose=0, device="cpu", seed=seed, n_steps=n_steps,
                             supported_action_spaces=SUPPORTED, policy_kwargs=policy_kwargs)
    return env, model


def params(model_or_policy):
    policy = getattr(model_or_policy, "policy", model_or_policy)
    return {k: v.detach().clone() for k, v in policy.state_dict().items()}


class SingleThread(unittest.TestCase):
    def setUp(self):
        self._threads = th.get_num_threads()
        th.set_num_threads(1)

    def tearDown(self):
        th.set_num_threads(self._threads)
        reset_wl_vocab_override()


class TestCLI(unittest.TestCase):
    def _main(self, argv):
        with mock.patch.object(gnn_global, "run") as run, mock.patch("builtins.print") as printed:
            gnn_global.main(argv)
        lines = [" ".join(str(a) for a in c.args) for c in printed.call_args_list]
        return run.call_args[0][0], lines

    def _argv(self, env_id, convention, *extra):
        return ["--env-name", env_id, "--graph-convention", convention, "--planner", "--feedback",
                "--save-dir", "./trained_models/x", "--log-dir", "./logs/x"] + list(extra)

    def test_gr_default_changes_nothing(self):
        for extra in ((), ("--edge-mode", "gr")):
            variant, lines = self._main(self._argv(TERNARY_ENV_ID, "atom", *extra))
            self.assertNotIn("edge_mode", variant["algorithm_kwargs"]["policy_kwargs"])
            self.assertEqual(variant["save_dir"], "./trained_models/x")
            self.assertEqual(variant["algorithm_kwargs"]["tensorboard_log"], "./logs/x")
            self.assertFalse(any(line.startswith("edge mode:") for line in lines))

    def test_mgr_egr_reach_policy_kwargs_dirs_and_startup(self):
        for env_id, convention, mode in VARIANTS:
            with self.subTest(env=short(env_id), convention=convention, mode=mode):
                variant, lines = self._main(self._argv(env_id, convention, "--edge-mode", mode))
                self.assertEqual(variant["algorithm_kwargs"]["policy_kwargs"]["edge_mode"], mode)
                self.assertEqual(variant["save_dir"], f"./trained_models/x_{mode}")
                self.assertEqual(variant["algorithm_kwargs"]["tensorboard_log"], f"./logs/x_{mode}")
                self.assertTrue(lines[0].startswith("startup:") and f"'--edge-mode', '{mode}'" in lines[0])
                self.assertTrue(any(line.startswith(f"edge mode: {mode} (") for line in lines))

    def test_refused(self):
        for env_id, convention, mode in [(STD_ENV_ID, "vilg", "mgr"), (TERNARY_ENV_ID, "vilg", "mgr"),
                                         (STD_ENV_ID, "oracle_sage", "mgr")]:
            with self.subTest(env=short(env_id), convention=convention):
                with self.assertRaises(ValueError):
                    self._main(self._argv(env_id, convention, "--edge-mode", mode))
        wl = ["--wl-vocab-path", committed_vocab_path("oracle_sage"), "--wl-num-iterations", "1"]
        with self.assertRaises(ValueError):
            self._main(self._argv(TERNARY_ENV_ID, "oracle_sage", "--edge-mode", "egr", *wl))
        with self.assertRaises(SystemExit):  # argparse choices
            self._main(self._argv(TERNARY_ENV_ID, "atom", "--edge-mode", "multi"))


class TestGrUnchanged(SingleThread):
    def test_gr_builds_the_old_gnn(self):
        env, model = build(TERNARY_ENV_ID, "atom", "gr")
        self.assertEqual(model.policy.edge_mode, "gr")
        self.assertIsNone(model.policy.edge_spec)
        self.assertIs(type(model.policy.gnn_extractor.gnn), MultiMessagePassing)
        self.assertEqual(model.policy.gnn_extractor.gnn.gnns[0].f_mess[0].in_features, 32 + 9)
        env.close()


class TestObservationAndProjectionAgree(SingleThread):
    def test_every_gnn_pass_is_converted_and_the_planner_sees_gr(self):
        for env_id, convention, mode in [(TERNARY_ENV_ID, "atom", "mgr"), (TERNARY_ENV_ID, "atom", "egr"),
                                         (TERNARY_ENV_ID, "oracle_sage", "mgr"), (STD_ENV_ID, "vilg", "egr")]:
            with self.subTest(env=short(env_id), convention=convention, mode=mode):
                env, model = build(env_id, convention, mode)
                gr_width = env.observation_space.edge_dimension
                passes, conversions, planner_graphs = [], [], []
                real_forward = graph_policy.GNNExtractor.forward
                real_convert = getattr(graph_policy, "to_mgr" if mode == "mgr" else "split_by_type")
                real_plan = Planner.plan

                def forward_spy(self_ext, x, xg, edge_attr, edge_index, batch):
                    passes.append(edge_attr.shape)
                    return real_forward(self_ext, x, xg, edge_attr, edge_index, batch)

                def convert_spy(edge_index, edge_attr, spec):
                    conversions.append(edge_attr.shape)
                    return real_convert(edge_index, edge_attr, spec)

                def plan_spy(self_planner, graph, goal):
                    planner_graphs.append((graph.edge_attr.shape[1], graph.edge_index.t().tolist()))
                    return real_plan(self_planner, graph, goal)

                obs = env.reset()
                with mock.patch.object(graph_policy.GNNExtractor, "forward", forward_spy), \
                        mock.patch.object(graph_policy, "to_mgr" if mode == "mgr" else "split_by_type", convert_spy), \
                        mock.patch.object(Planner, "plan", plan_spy):
                    with th.no_grad():
                        model.policy.forward(obs)
                k = model.policy.num_planning_choices
                # observation + current state + k projections (each a batch over the envs)
                self.assertEqual(len(passes), 2 + k)
                self.assertEqual(conversions, passes)  # every pass converted, nothing else
                self.assertTrue(all(shape[1] == gr_width for shape in passes))  # converted from Gr
                self.assertEqual(len(planner_graphs), k * len(obs))
                for width, pairs in planner_graphs:
                    self.assertEqual(width, gr_width)
                    self.assertEqual(len(pairs), len(set(map(tuple, pairs))))  # Gr: one edge per pair
                env.close()


class TestEndToEndAndReload(SingleThread):
    def setUp(self):
        super().setUp()
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)
        super().tearDown()

    def test_train_save_load(self):
        for env_id, convention, mode in VARIANTS:
            with self.subTest(env=short(env_id), convention=convention, mode=mode):
                env, model = build(env_id, convention, mode)
                policy = model.policy
                spec = policy.edge_spec
                gnn = policy.gnn_extractor.gnn
                if mode == "egr":
                    self.assertIs(type(gnn), TypedMultiMessagePassing)
                    self.assertEqual(len(gnn.gnns[0].f_mess), spec.num_types)
                else:
                    self.assertIs(type(gnn), MultiMessagePassing)
                    self.assertEqual(gnn.gnns[0].f_mess[0].in_features, 32 + spec.num_types)
                in_optimizer = {id(p) for group in policy.optimizer.param_groups for p in group["params"]}
                self.assertTrue(all(id(p) in in_optimizer for p in policy.parameters()))

                before = params(model)
                model.learn(total_timesteps=2 * model.n_envs * model.n_steps)  # two updates
                after = params(model)
                self.assertTrue(all(bool(th.isfinite(v).all()) for v in after.values() if v.is_floating_point()))
                moved = [k for k in before if "gnn_extractor.gnn." in k and not th.equal(before[k], after[k])]
                self.assertTrue(moved, "the GNN did not train")

                path = os.path.join(self.dir, f"{short(env_id)}_{convention}_{mode}")
                model.save(path)
                data, _ps, _opt = read_zip(path + ".zip")
                self.assertEqual(saved_policy_kwargs(data)["edge_mode"], mode)
                live = GraphTaxiEnv(**env_kwargs(env_id, convention))
                th.manual_seed(99)
                loaded, _data = load_policy(path + ".zip", live.observation_space, live.action_space)
                self.assertEqual(loaded.edge_mode, mode)
                self.assertEqual(set(after), set(loaded.state_dict()))
                self.assertTrue(all(th.equal(after[k], v) for k, v in loaded.state_dict().items()))

                obs = env.reset()
                policy.eval()
                outputs = []
                for p in (policy, loaded):
                    th.manual_seed(5)
                    with th.no_grad():
                        actions, values, log_prob, _explored, plans = p.forward(obs)
                        ev = p.evaluate_actions(obs, actions)
                    outputs.append((actions, values, log_prob, ev))
                (a1, v1, l1, ev1), (a2, v2, l2, ev2) = outputs
                self.assertTrue(th.equal(a1, a2) and th.equal(v1, v2) and th.equal(l1, l2))
                for x, y in zip(ev1, ev2):
                    self.assertTrue((x is None and y is None) or th.equal(x, y))
                env.close()
                live.close()


class TestOtherPolicies(SingleThread):
    def test_wl_refuses_edge_modes(self):
        policy_kwargs = {}
        gnn_global.configure_wl(wl_args(graph_convention="oracle_sage",
                                        wl_vocab_path=committed_vocab_path("oracle_sage")), policy_kwargs)
        for mode in ("mgr", "egr"):
            with self.assertRaises(ValueError):
                build(TERNARY_ENV_ID, "oracle_sage", mode, policy_cls=WLPlanFeedbackPolicy,
                      extra_policy_kwargs=policy_kwargs)

    def test_oracle_decoder_with_egr(self):
        env, model = build(TERNARY_ENV_ID, "oracle_sage", "egr", oracle_decoder=True)
        model.learn(total_timesteps=model.n_envs * model.n_steps)
        self.assertIs(type(model.policy.gnn_extractor.gnn), TypedMultiMessagePassing)
        env.close()


if __name__ == "__main__":
    unittest.main()
