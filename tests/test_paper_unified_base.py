"""
F0 (paper-unified): the shared base every paper variant is trained from.

Checks, on the real code paths:
  1. default settings give the same policy outputs as the base commit (ternary-wl 5547edb)
     on fixed inputs, for every GNN variant on both domains and for ternary WL Object;
  2. every policy parameter is in the optimizer (path_value_net included);
  3. standard Object-Atom (vILG) edges are two-way, in the live observation and in the
     planner's projections;
  4. the dirty-tree guard: training refuses to start on a dirty tree unless --allow-dirty
     is given, and the startup line records the commit.

The other F0 checks already have tests on this branch, run alongside this file:
  - path_value_net in the optimizer: TestOptimizerHasEveryParameter below; on the real
    training path, tests/test_diag_occupancy.py (grad_norm_frozen is 0); for WL,
    tests/test_wl_plan_feedback_policy.py;
  - the stale-edge fix: tests/test_taxi_world_edges.py;
  - old-domain observation hashes: tests/test_ternary_wiring.py.

Run from the repo root (CPU; not while a training run uses the GPU):
    PYTHONPATH=. python -m pytest tests/test_paper_unified_base.py -v
"""
import base64
import gzip
import json
import os
import subprocess
import tempfile
import unittest
from unittest import mock

import gym as gym_module
import numpy as np
import torch as th

import sage.experiments.gnn_global as gnn_global
from sage.agent.async_vec_env import AsyncVecEnv
from sage.agent.graph_plan_feedback_policy import GNNPlanFeedbackPolicy
from sage.agent.plan_feedback_a2c import PlanFeedback_A2C
from sage.agent.wl_plan_feedback_policy import WLPlanFeedbackPolicy
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.utils.compat import spec_kwargs
from sage.domains.gym_taxi.utils.wl_vocab_cache import reset_wl_vocab_override
from sage.domains.utils import spaces as sage_spaces
from sage.forks.stable_baselines3.stable_baselines3.common.env_util import make_vec_env
from tests.test_ternary_wl_training_path import committed_vocab_path, wl_args

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(HERE, "fixtures", "policy_reference_5547edb.json.gz")
STD_ENV_ID = "city-taxi-unmasked-v1"
TERNARY_ENV_ID = "city-taxi-ternary-unmasked-v1"
# The fixture's policies, as gnn_global builds them for --planner --feedback --shared-gnn
POLICY_KWARGS = {"optimizer_class": th.optim.AdamW, "optimizer_kwargs": {"weight_decay": 0.0001},
                 "ortho_init": False, "exploration_initial_eps": 0.0, "exploration_final_eps": 0.0,
                 "exploration_fraction": 0.1, "shared_gnn": True, "layer_norm": False,
                 "num_planning_choices": 3, "features_extractor_kwargs": {"gnn_steps": 5}}
# float32 on CPU; a different torch/PyG build (RCP) may sum in another order
ATOL, RTOL = 1e-5, 1e-4


def from_jsonable(obj):
    """Inverse of analysis/make_policy_reference.py's to_jsonable: {"__tensor__": numpy dtype
    string, "shape", "data": base64 raw bytes} -> a tensor, bit for bit."""
    if isinstance(obj, dict):
        if "__tensor__" in obj:
            a = np.frombuffer(base64.b64decode(obj["data"]), dtype=np.dtype(obj["__tensor__"]))
            return th.from_numpy(a.astype(a.dtype.newbyteorder("=")).reshape(obj["shape"]))
        return {k: from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [from_jsonable(v) for v in obj]
    return obj


def load_reference():
    """The fixture (made by analysis/make_policy_reference.py at 5547edb). Gzipped JSON, not a
    torch/numpy pickle, so any torch or numpy version reads it (RCP: torch 1.7, numpy 1)."""
    with gzip.open(REFERENCE, "rt") as f:
        return from_jsonable(json.load(f))


def build_model(env_id, convention, wl):
    policy_kwargs = dict(POLICY_KWARGS)
    policy_cls = GNNPlanFeedbackPolicy
    if wl:
        gnn_global.configure_wl(wl_args(graph_convention=convention, wl_vocab_path=committed_vocab_path(convention)),
                                policy_kwargs)
        policy_cls = WLPlanFeedbackPolicy
    env_kwargs = dict(spec_kwargs(gym_module.spec(env_id)), graph_convention=convention)
    env = make_vec_env(lambda: GraphTaxiEnv(**env_kwargs), n_envs=2, seed=0,
                       monitor_kwargs={"info_keywords": ("len100", "len200")}, vec_env_cls=AsyncVecEnv)
    model = PlanFeedback_A2C(policy_cls, env, verbose=0, device="cpu",
                             supported_action_spaces=(sage_spaces.BinaryAction, gym_module.spaces.Discrete,
                                                      sage_spaces.Autoregressive),
                             n_steps=2, policy_kwargs=policy_kwargs)
    return env, model


class TestDefaultOutputsUnchanged(unittest.TestCase):
    """The base commit's policy outputs on fixed inputs (tests/fixtures/
    policy_reference_5547edb.json.gz: observations, weights, and the outputs recorded at
    5547edb), replayed on this commit with default settings.

    evaluate_actions (values, log-probs, entropy, discriminator values) is deterministic and
    always compared, on the recorded actions plus taxi and passenger goals. forward() samples
    with th.multinomial, so its actions and plans are compared only on the torch version
    the fixture was recorded with (the Mac's); other versions may draw differently."""

    @classmethod
    def setUpClass(cls):
        cls.reference = load_reference()

    def setUp(self):
        self._threads = th.get_num_threads()
        th.set_num_threads(1)

    def tearDown(self):
        th.set_num_threads(self._threads)
        reset_wl_vocab_override()

    def assertClose(self, a, b, what):
        if a is None or b is None:
            self.assertIs(a, b, what)
            return
        self.assertTrue(th.allclose(a, b, atol=ATOL, rtol=RTOL), f"{what}: max diff {(a - b).abs().max().item()}")

    def test_every_variant(self):
        for name, ref in self.reference.items():
            with self.subTest(case=name):
                env, model = build_model(ref["env_id"], ref["convention"], ref["wl"])
                try:
                    model.policy.load_state_dict(ref["state_dict"])
                    obs = np.array(ref["obs"])
                    th.manual_seed(123)
                    with th.no_grad():
                        actions, values, log_prob, _explored, plans = model.policy.forward(obs)
                        ev = model.policy.evaluate_actions(obs, ref["actions"])
                        extra = {label: model.policy.evaluate_actions(obs, e["actions"])
                                 for label, e in ref["extra"].items()}
                    if ref["torch_version"] == th.__version__:
                        self.assertEqual(actions.tolist(), ref["actions"].tolist())
                        self.assertEqual([list(map(int, p)) for p in plans], ref["plans"])
                        self.assertClose(values, ref["values"], "values")
                        self.assertClose(log_prob, ref["log_prob"], "log_prob")
                    for got, key in zip(ev, ("ev_values", "ev_log_prob", "ev_entropy", "ev_projected")):
                        self.assertClose(got, ref[key], key)
                    for label, e in ref["extra"].items():
                        for got, key in zip(extra[label], ("values", "log_prob", "entropy", "projected")):
                            self.assertClose(got, e[key], f"{label}/{key}")
                finally:
                    env.close()
                    reset_wl_vocab_override()


class TestOptimizerHasEveryParameter(unittest.TestCase):
    """The optimizer-timing fix (e828320): path_value_net (and an unshared gnn_extractor2)
    are built before the optimizer, so every policy parameter is trained -- for every GNN
    variant on both domains, shared and unshared discriminator GNN."""

    def test_every_gnn_variant(self):
        for env_id in (STD_ENV_ID, TERNARY_ENV_ID):
            for convention in ("oracle_sage", "vilg", "atom"):
                for shared in (True, False):
                    with self.subTest(env=env_id, convention=convention, shared_gnn=shared):
                        env = GraphTaxiEnv(**dict(spec_kwargs(gym_module.spec(env_id)), graph_convention=convention))
                        policy = GNNPlanFeedbackPolicy(env.observation_space, env.action_space, lambda _p: 3e-4,
                                                       **dict(POLICY_KWARGS, shared_gnn=shared))
                        in_optimizer = {id(p) for g in policy.optimizer.param_groups for p in g["params"]}
                        self.assertTrue(all(id(p) in in_optimizer for p in policy.path_value_net.parameters()))
                        missing = [n for n, p in policy.named_parameters() if id(p) not in in_optimizer]
                        self.assertEqual(missing, [])
                        env.close()


class TestVilgEdgesTwoWay(unittest.TestCase):
    """Standard-domain Object-Atom: every edge (u, v, label) has its mirror (v, u, label),
    in the live observation and in planner projections (move and delivery goals), so
    proposition nodes receive messages (c4692f0)."""

    def _assert_two_way(self, graph, what):
        ei = graph.edge_index.numpy()
        ea = graph.edge_attr.numpy()
        edges = {(int(u), int(v), tuple(a)) for (u, v), a in zip(ei.T, ea)}
        self.assertEqual(len(edges), ei.shape[1], f"{what}: duplicate edges")
        missing = [(u, v) for u, v, a in edges if (v, u, a) not in edges]
        self.assertEqual(missing, [], f"{what}: {len(missing)} one-way edges")
        has_incoming = set(int(v) for v in ei[1])
        self.assertTrue(all(i in has_incoming for i in range(graph.x.shape[0])), f"{what}: a node gets no messages")

    def test_observation_and_projections(self):
        env = make_vec_env(lambda: GraphTaxiEnv(**spec_kwargs(gym_module.spec(STD_ENV_ID)), graph_convention="vilg"),
                           n_envs=2, seed=3, monitor_kwargs={"info_keywords": ("len100", "len200")},
                           vec_env_cls=AsyncVecEnv)
        try:
            obs = env.reset()
            space = env.observation_space
            for i in range(len(obs)):
                graph = space.converter(obs[i:i + 1]).to_data_list()[0]
                self._assert_two_way(graph, f"obs {i}")
                planner = space.planner
                passengers = [n for n in range(graph.mask.numel()) if bool(graph.mask[n]) and graph.x[n].tolist()[:3] == [0, 0, 1]]
                self.assertTrue(passengers)
                for goal in [1] + passengers:
                    projection, _actions = planner.plan(graph, goal)
                    self._assert_two_way(projection, f"obs {i}, goal {goal}")
        finally:
            env.close()


class TestDirtyTreeGuard(unittest.TestCase):
    """run() refuses to start on a dirty tree (tracked changes, or untracked files under
    sage/) unless --allow-dirty; an unknown state (no git) also refuses. The startup line
    records the commit and the dirty state."""

    def _git(self, root, *args):
        subprocess.check_call(["git", "-C", root] + list(args), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def test_git_provenance_on_a_real_repo(self):
        with tempfile.TemporaryDirectory() as root:
            self._git(root, "init", "-q")
            self._git(root, "config", "user.email", "t@example.com")
            self._git(root, "config", "user.name", "t")
            os.makedirs(os.path.join(root, "sage"))
            with open(os.path.join(root, "sage", "a.py"), "w") as f:
                f.write("x = 1\n")
            self._git(root, "add", ".")
            self._git(root, "commit", "-q", "-m", "init")
            commit, dirty = gnn_global.git_provenance(root)
            self.assertEqual(len(commit), 40)
            self.assertEqual(dirty, [])
            with open(os.path.join(root, "notes.txt"), "w") as f:  # untracked, outside sage/: clean
                f.write("log\n")
            self.assertEqual(gnn_global.git_provenance(root)[1], [])
            with open(os.path.join(root, "sage", "b.py"), "w") as f:  # untracked code: dirty
                f.write("y = 2\n")
            self.assertEqual(len(gnn_global.git_provenance(root)[1]), 1)
            os.remove(os.path.join(root, "sage", "b.py"))
            with open(os.path.join(root, "sage", "a.py"), "w") as f:  # tracked change: dirty
                f.write("x = 2\n")
            self.assertEqual(len(gnn_global.git_provenance(root)[1]), 1)

    def test_git_provenance_unknown_without_repo(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(gnn_global.git_provenance(root), ("unknown", None))

    def test_guard(self):
        clean = ("a" * 40, [])
        dirty = ("a" * 40, [" M sage/agent/graph_policy.py"])
        unknown = ("unknown", None)
        for state, allow, refuses in ((clean, False, False), (dirty, False, True), (dirty, True, False),
                                      (unknown, False, True), (unknown, True, False)):
            with self.subTest(state=state, allow=allow):
                with mock.patch.object(gnn_global, "git_provenance", return_value=state):
                    if refuses:
                        with self.assertRaises(SystemExit):
                            gnn_global.require_clean_tree(allow)
                    else:
                        gnn_global.require_clean_tree(allow)

    def test_run_refuses_before_building_anything(self):
        variant = {"env_name": TERNARY_ENV_ID, "allow_dirty": False}
        with mock.patch.object(gnn_global, "git_provenance", return_value=("a" * 40, [" M x"])), \
                mock.patch.object(gnn_global, "make_vec_env") as mve:
            with self.assertRaises(SystemExit):
                gnn_global.run(variant)
            mve.assert_not_called()

    def test_flag_reaches_variant_and_startup_line(self):
        argv = ["--env-name", TERNARY_ENV_ID, "--graph-convention", "atom", "--planner", "--feedback"]
        for extra, expected in (([], False), (["--allow-dirty"], True)):
            with mock.patch.object(gnn_global, "run") as run, \
                    mock.patch.object(gnn_global, "git_provenance", return_value=("b" * 40, [])), \
                    mock.patch("builtins.print") as printed:
                gnn_global.main(argv + extra)
            self.assertIs(run.call_args[0][0]["allow_dirty"], expected)
            lines = [c[0][0] for c in printed.call_args_list if c[0] and str(c[0][0]).startswith("startup:")]
            self.assertEqual(len(lines), 1)
            self.assertIn("commit=" + "b" * 40, lines[0])
            self.assertIn("dirty=False", lines[0])


if __name__ == "__main__":
    unittest.main()
