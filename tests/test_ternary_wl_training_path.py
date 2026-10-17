"""
WL on the ternary training path (Cells 3, 4, 6 on city-taxi-ternary-unmasked-v1).

  - Decoder site (_ternary_json_to_data) and planner site (plan_ternary): with a ternary
    WL override configured, the attached wl_colours / wl_histogram equal a direct frozen
    wl_colours(domain="ternary") call on facts_to_*_graph output -- for live states, and
    for projections on the facts plan_on_facts projects to -- using a vocab decoded here
    from the JSON file, not through the production loaders.
  - Without an override nothing is attached (the ternary GNN path is unchanged; its
    byte-identity to ternary-gnn e84fb5d is checked with docs/tools/policy_input_harness.py
    --ternary); and the planner's actions and projection tensors are identical with and
    without WL attached.
  - OOV logging: a hand-built frozen vocab with exactly the buddies' L=1 signature
    removed gives a known OOV fraction (2 of the case's nodes, from the hand partitions in
    tests/test_wl_ternary.py), logged as wl/oov_fraction_decoder and
    wl/oov_fraction_planner for every convention; never under the old domain's wl_atom/.
  - gnn_global.configure_wl: a ternary WL run refuses a vocab of the wrong domain,
    convention or L, and needs --planner --feedback; it returns the startup line; the
    old-domain path is unchanged.
  - WLPlanFeedbackPolicy.forward end to end on a ternary batch for every convention, on
    CPU and (CUDA-gated) on CUDA; train/max_mem_gb is logged on CUDA only.

Run from the repo root with:
    PYTHONPATH=. python -m pytest tests/test_ternary_wl_training_path.py -v
"""
import argparse
import json
import os
import tempfile
import unittest

import gym as gym_module
import torch as th

import sage.domains.gym_taxi  # noqa: F401  (registers the env ids)
from sage.domains.utils import build_wl_vocab  # noqa: F401  (gym/numpy shims on the Mac stack)
from sage.agent.async_vec_env import AsyncVecEnv
from sage.agent.plan_feedback_a2c import PlanFeedback_A2C
from sage.agent.wl_plan_feedback_policy import WLPlanFeedbackPolicy
from sage.domains.gym_taxi import REWARDS
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.simulator.planner import Planner
from sage.domains.gym_taxi.simulator.ternary_planner import decode_atom, decode_object, decode_vilg, plan_on_facts
from sage.domains.gym_taxi.utils.ternary_representations import (
    TERNARY_GRAPH_CONVENTION_CONVERTERS,
    facts_to_atom_graph,
    facts_to_json,
    facts_to_object_graph,
    facts_to_vilg_graph,
)
from sage.domains.gym_taxi.utils.wl_vocab_cache import configure_wl_vocab_override, reset_wl_vocab_override
from sage.domains.utils import spaces as sage_spaces
from sage.domains.utils.build_wl_vocab import save_vocab
from sage.domains.utils.wl_colours import OOV_SIGNATURE, wl_colours
from sage.experiments import gnn_global
from sage.forks.stable_baselines3.stable_baselines3.common import logger as sb3_logger
from sage.forks.stable_baselines3.stable_baselines3.common.env_util import make_vec_env
from tests.test_wl_ternary import CASE_W, real_states
from tests.test_wl_vocab_domain import decode_entries

CONVENTIONS = ("oracle_sage", "vilg", "atom")
FACTS_TO_GRAPH = {"oracle_sage": facts_to_object_graph, "vilg": facts_to_vilg_graph, "atom": facts_to_atom_graph}
DECODERS = {"oracle_sage": decode_object, "vilg": decode_vilg, "atom": decode_atom}
META = {"time": 0, "timeout": 2000, "planning": True}
TERNARY_ENV = "city-taxi-ternary-unmasked-v1"


def committed_vocab_path(convention):
    return str(build_wl_vocab.ternary_default_out_path(convention, 1))


def independent_vocab(path):
    with open(path) as f:
        return decode_entries(json.load(f))


def decode(facts, convention, meta=META):
    """facts -> the env's JSON -> the policy-side converter, one unbatched Data."""
    return TERNARY_GRAPH_CONVENTION_CONVERTERS[convention]([[facts_to_json(facts, meta)]]).to_data_list()[0]


def direct_wl(facts, convention, vocab, meta=META):
    node_feats, edge_feats, edge_index, _mask, _g = FACTS_TO_GRAPH[convention](facts, meta)
    return wl_colours(
        th.as_tensor(node_feats, dtype=th.float32), th.as_tensor(edge_index, dtype=th.long),
        th.as_tensor(edge_feats, dtype=th.float32), num_iterations=1, vocab=vocab, frozen=True,
        graph_convention=convention, domain="ternary",
    )


def goals_for(facts):
    """Every kind of goal: the taxi (no-op), the taxi's own location, another location,
    each waiting passenger, the carried passenger if any."""
    taxi_location = next(args[1] for pred, args in facts if pred == "in" and args[0] == 0)
    passengers = [args[0] for pred, args in facts if pred == "passenger"]
    other = 1 if taxi_location != 1 else 2
    return [0, taxi_location, other] + passengers[:4]


STATES = None


def states():
    global STATES
    if STATES is None:
        STATES = real_states()
    return STATES


class OverrideTest(unittest.TestCase):
    def tearDown(self):
        reset_wl_vocab_override()


class TestAttachedColoursEqualDirectWL(OverrideTest):
    def test_decoder_and_planner_sites(self):
        for convention in CONVENTIONS:
            path = committed_vocab_path(convention)
            configure_wl_vocab_override(path, 1, graph_convention=convention, domain="ternary")
            vocab = independent_vocab(path)
            planner = Planner(graph_convention=convention, ternary=True)
            for name, facts in states():
                with self.subTest(convention=convention, state=name):
                    data = decode(facts, convention)
                    colours, histogram = direct_wl(facts, convention, vocab)
                    self.assertTrue(th.equal(data.wl_colours, colours))
                    self.assertTrue(th.equal(data.wl_histogram, histogram.unsqueeze(0)))
                    self.assertEqual(tuple(data.wl_histogram.shape), (1, len(vocab)))

                    decoded = DECODERS[convention](data)
                    if convention != "oracle_sage":
                        self.assertEqual(decoded, facts)  # exact decoders
                    for goal in goals_for(facts):
                        projection, _actions = planner.plan(data, goal)
                        _a, projected = plan_on_facts(decoded, goal)
                        colours, histogram = direct_wl(projected, convention, vocab)
                        self.assertTrue(th.equal(projection.wl_colours, colours), goal)
                        self.assertTrue(th.equal(projection.wl_histogram, histogram.unsqueeze(0)), goal)
            reset_wl_vocab_override()

    def test_batched_shapes(self):
        for convention in CONVENTIONS:
            configure_wl_vocab_override(committed_vocab_path(convention), 1, graph_convention=convention, domain="ternary")
            (_n1, f1), (_n2, f2) = states()[0], states()[1]
            batch = TERNARY_GRAPH_CONVENTION_CONVERTERS[convention](
                [[facts_to_json(f1, META)], [facts_to_json(f2, META)]])
            self.assertEqual(batch.wl_colours.shape[0], batch.x.shape[0])
            self.assertEqual(batch.wl_histogram.shape[0], 2)
            reset_wl_vocab_override()


class TestWithoutOverrideNothingChanges(OverrideTest):
    FIELDS = ("x", "edge_index", "edge_attr", "mask", "global_features")

    def test_no_wl_attributes_without_override(self):
        reset_wl_vocab_override()
        for convention in CONVENTIONS:
            planner = Planner(graph_convention=convention, ternary=True)
            name, facts = states()[0]
            data = decode(facts, convention)
            self.assertFalse(hasattr(data, "wl_colours") and data.wl_colours is not None)
            projection, _ = planner.plan(data, goals_for(facts)[2])
            self.assertIsNone(getattr(projection, "wl_colours", None))

    def test_planner_actions_and_projections_identical_with_and_without_wl(self):
        for convention in CONVENTIONS:
            planner = Planner(graph_convention=convention, ternary=True)
            for name, facts in states():
                reset_wl_vocab_override()
                plain = decode(facts, convention)
                plain_out = [planner.plan(plain, g) for g in goals_for(facts)]
                configure_wl_vocab_override(committed_vocab_path(convention), 1, graph_convention=convention,
                                            domain="ternary")
                with_wl = decode(facts, convention)
                for f in self.FIELDS:
                    self.assertTrue(th.equal(getattr(plain, f), getattr(with_wl, f)))
                for goal, (p_plain, a_plain) in zip(goals_for(facts), plain_out):
                    with self.subTest(convention=convention, state=name, goal=goal):
                        p_wl, a_wl = planner.plan(with_wl, goal)
                        self.assertEqual(list(a_wl), list(a_plain))
                        for f in self.FIELDS:
                            self.assertTrue(th.equal(getattr(p_wl, f), getattr(p_plain, f)), f)
                reset_wl_vocab_override()


class TestDomainGuard(OverrideTest):
    def test_standard_override_refused_on_ternary_path(self):
        configure_wl_vocab_override(str(build_wl_vocab.Path(build_wl_vocab.__file__).parent
                                        / "wl_vocab_taxi_city_vilg_L1_full.json"), 1, graph_convention="vilg")
        with self.assertRaises(ValueError):
            decode(states()[0][1], "vilg")


def hand_vocab_without_buddies(convention, path):
    """L=1 vocab grown on CASE_W, with the buddies' (rows 5 and 6, one class at L=1 in every
    encoding -- tests/test_wl_ternary.py's EXPECTED) final signature replaced by one that
    never occurs, keeping ids contiguous. Colouring CASE_W with it makes exactly rows 5
    and 6 OOV."""
    data = decode(CASE_W, convention)
    vocab = {}
    colours, _ = wl_colours(data.x, data.edge_index, data.edge_attr, num_iterations=1, vocab=vocab, frozen=False,
                            graph_convention=convention, domain="ternary")
    buddy_id = int(colours[5])
    assert int(colours[6]) == buddy_id and int((colours == buddy_id).sum()) == 2
    never = (10 ** 6, ())  # a refine signature no graph can produce: own colour 10**6 is no id
    vocab = {(never if v == buddy_id else k): v for k, v in vocab.items()}
    vocab[OOV_SIGNATURE] = len(vocab)
    save_vocab(vocab, path, convention, 1, domain="ternary")
    return data.x.shape[0]


class TestOovLogging(OverrideTest):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()
        super().tearDown()

    def _logged(self):
        return dict(sb3_logger.get_log_dict())

    def test_known_fraction_at_both_sites(self):
        for convention in CONVENTIONS:
            with self.subTest(convention=convention):
                path = os.path.join(self._tmpdir.name, f"{convention}.json")
                n_nodes = hand_vocab_without_buddies(convention, path)
                configure_wl_vocab_override(path, 1, graph_convention=convention, domain="ternary")
                sb3_logger.Logger.CURRENT.name_to_value.clear()
                sb3_logger.Logger.CURRENT.name_to_count.clear()
                data = decode(CASE_W, convention)
                Planner(graph_convention=convention, ternary=True).plan(data, 0)  # taxi goal: same facts
                logged = self._logged()
                self.assertAlmostEqual(logged["wl/oov_fraction_decoder"], 2 / n_nodes)
                self.assertAlmostEqual(logged["wl/oov_fraction_planner"], 2 / n_nodes)
                self.assertFalse(any(key.startswith("wl_atom/") for key in logged))
                reset_wl_vocab_override()

    def test_committed_vocab_zero_oov_on_a_corpus_state(self):
        """Positive control: a state from the vocab's own corpus seed logs 0."""
        for convention in CONVENTIONS:
            configure_wl_vocab_override(committed_vocab_path(convention), 1, graph_convention=convention, domain="ternary")
            sb3_logger.Logger.CURRENT.name_to_value.clear()
            sb3_logger.Logger.CURRENT.name_to_count.clear()
            sample = next(iter(build_wl_vocab.ternary_corpus(1, 10, 0, convention, max_steps=1)))
            TERNARY_GRAPH_CONVENTION_CONVERTERS[convention]([[sample.env.sim._get_state_json()]])
            self.assertEqual(self._logged()["wl/oov_fraction_decoder"], 0.0)
            reset_wl_vocab_override()


def wl_args(**overrides):
    args = dict(env_name=TERNARY_ENV, graph_convention="vilg", planner=True, feedback=True,
                wl_vocab_path=committed_vocab_path("vilg"), wl_num_iterations=1)
    args.update(overrides)
    return argparse.Namespace(**args)


class TestGnnGlobalConfigureWL(OverrideTest):
    def test_matching_ternary_run(self):
        for convention in CONVENTIONS:
            policy_kwargs = {}
            line = gnn_global.configure_wl(
                wl_args(graph_convention=convention, wl_vocab_path=committed_vocab_path(convention)), policy_kwargs)
            vocab_size = len(independent_vocab(committed_vocab_path(convention)))
            for part in (f"convention={convention}", f"env={TERNARY_ENV}", "L=1",
                         f"vocab={committed_vocab_path(convention)}", f"vocab_size={vocab_size}", "git_commit="):
                self.assertIn(part, line)
            self.assertEqual(policy_kwargs, {
                "wl_vocab_path": committed_vocab_path(convention), "wl_num_iterations": 1,
                "graph_convention": convention, "wl_domain": "ternary"})

    def test_refusals(self):
        standard_vilg = str(build_wl_vocab.Path(build_wl_vocab.__file__).parent / "wl_vocab_taxi_city_vilg_L1_full.json")
        cases = {
            "standard-domain vocab": wl_args(wl_vocab_path=standard_vilg),
            "other convention's vocab": wl_args(wl_vocab_path=committed_vocab_path("atom")),
            "wrong L": wl_args(wl_num_iterations=2),
            "no --planner": wl_args(planner=False),
            "no --feedback": wl_args(feedback=False),
            "ternary vocab on the standard env": wl_args(env_name="city-taxi-unmasked-v1"),
        }
        for name, args in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    gnn_global.configure_wl(args, {})
                reset_wl_vocab_override()

    def test_old_domain_path_unchanged(self):
        standard_vilg = str(build_wl_vocab.Path(build_wl_vocab.__file__).parent / "wl_vocab_taxi_city_vilg_L1_full.json")
        policy_kwargs = {}
        line = gnn_global.configure_wl(
            wl_args(env_name="city-taxi-unmasked-v1", wl_vocab_path=standard_vilg), policy_kwargs)
        self.assertIsNone(line)
        self.assertEqual(policy_kwargs, {"wl_vocab_path": standard_vilg, "wl_num_iterations": 1, "graph_convention": "vilg"})
        self.assertIsNone(gnn_global.configure_wl(wl_args(wl_vocab_path=None, wl_num_iterations=None), {}))


def build_wl_model(convention, device):
    path = committed_vocab_path(convention)
    args = wl_args(graph_convention=convention, wl_vocab_path=path)
    policy_kwargs = {"optimizer_class": th.optim.AdamW, "optimizer_kwargs": {"weight_decay": 0.0001},
                     "ortho_init": False, "exploration_initial_eps": 0.0, "exploration_final_eps": 0.0,
                     "exploration_fraction": 0.1, "shared_gnn": True, "layer_norm": False,
                     "num_planning_choices": 3, "features_extractor_kwargs": {"gnn_steps": 5}}
    gnn_global.configure_wl(args, policy_kwargs)

    def make_env():
        return GraphTaxiEnv("graph", "city_ternary", mask=False, rewards=REWARDS["v1"].copy(),
                            graph_convention=convention, ternary=True)

    env = make_vec_env(make_env, n_envs=2, seed=0, monitor_kwargs={"info_keywords": ("len100", "len200")},
                       vec_env_cls=AsyncVecEnv)
    model = PlanFeedback_A2C(
        WLPlanFeedbackPolicy, env, verbose=0, device=device,
        supported_action_spaces=(sage_spaces.BinaryAction, gym_module.spaces.Discrete, sage_spaces.Autoregressive),
        n_steps=2, policy_kwargs=policy_kwargs,
    )
    return env, model


class TestWLPolicyForward(OverrideTest):
    def _check(self, device):
        for convention in CONVENTIONS:
            with self.subTest(convention=convention):
                env, model = build_wl_model(convention, device)
                obs = env.reset()
                actions, values, log_prob, _explored, _plans = model.policy.forward(obs)
                self.assertEqual(actions.device.type, model.device.type)
                self.assertTrue(th.isfinite(values).all())
                self.assertTrue(th.isfinite(log_prob).all())
                for i, o in enumerate(obs):
                    facts = json.loads(o[0])["facts"]
                    n_obj = sum(1 for entry in facts if entry[0] in (0, 1, 2))  # taxi/location/passenger
                    self.assertLess(int(actions[i]), n_obj)  # an object, never a proposition/atom row
                env.close()
                reset_wl_vocab_override()

    def test_forward_cpu(self):
        self._check("cpu")

    @unittest.skipIf(not th.cuda.is_available(), "requires CUDA")
    def test_forward_cuda(self):
        self._check("cuda")


class TestMaxMemLogging(OverrideTest):
    def _one_update(self, device):
        env, model = build_wl_model("vilg", device)
        sb3_logger.Logger.CURRENT.name_to_value.clear()
        model.learn(total_timesteps=4, log_interval=10 ** 9)  # one rollout of 2 envs x 2 steps, one update
        logged = dict(sb3_logger.get_log_dict())
        env.close()
        return logged

    def test_cpu_does_not_log_max_mem(self):
        logged = self._one_update("cpu")
        self.assertIn("train/policy_loss", logged)
        self.assertNotIn("train/max_mem_gb", logged)

    @unittest.skipIf(not th.cuda.is_available(), "requires CUDA")
    def test_cuda_logs_max_mem(self):
        logged = self._one_update("cuda")
        self.assertGreater(logged["train/max_mem_gb"], 0.0)


if __name__ == "__main__":
    unittest.main()
