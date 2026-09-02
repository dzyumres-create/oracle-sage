"""
Tests for WLPlanFeedbackPolicy (sage/agent/wl_plan_feedback_policy.py): the
Taxi meta-controller with a WL-colour embedding encoder in place of the GNN
message-passing encoder, while the discriminator stays fully GNN.

Uses scenario="city" (Oracle-SAGE's real Taxi training env), matching the
frozen vocab (sage/domains/utils/wl_vocab_taxi_city_L1.json) WLPlanFeedbackPolicy
defaults to.

Run from the repo root with:
    python -m unittest tests.test_wl_plan_feedback_policy -v
"""
import unittest

import numpy as np
import torch as th

# Importing build_wl_vocab (not otherwise used here) applies its numpy/gym
# compatibility shims as an import side effect - needed to construct a
# "city" (random_walls=True) GraphTaxiEnv at all in this sandbox's drifted
# gym/numpy. See build_wl_vocab.py's own docstring for why. The production
# code under test does NOT depend on this - only this test harness does.
import sage.domains.utils.build_wl_vocab  # noqa: F401

from copy import deepcopy

from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.utils.representations import env_to_json
from sage.domains.gym_taxi.simulator.planner import Planner, graph_to_networkx
from sage.domains.utils.representations import json_to_graph
from sage.agent.graph_policy import GNNExtractor, EMB_SIZE
from sage.agent.graph_plan_feedback_policy import GNNPlanFeedbackPolicy
from sage.agent.wl_plan_feedback_policy import WLPlanFeedbackPolicy, WLEmbeddingExtractor
from torch_geometric.data import Batch


def make_city_env():
    env = GraphTaxiEnv(representation="graph", scenario="city", mask=False)
    env.reset()
    return env


def make_policy(env, shared_gnn, policy_cls=WLPlanFeedbackPolicy):
    # features_extractor_kwargs must be an explicit dict (not the None
    # default): GNNPolicy.__init__ unconditionally calls .pop("gnn_steps",
    # 5) on it, which crashes on None - a pre-existing quirk, not
    # introduced here (gnn_global.py always supplies a dict in practice).
    # ortho_init must be False: with it True, GNNPolicy._build()'s
    # ortho-init module_gains dict unconditionally references
    # self.action_net2, which is only ever set for MultiDiscrete/
    # Autoregressive action spaces - not GraphTaxiEnv's actual action_space
    # (plain gym.spaces.Discrete(6) for "city"; node selection is done via
    # per-node scores over batch.x, not via this nominal space's
    # cardinality). Real Taxi training also runs with the default
    # --ortho-init=False (gnn_global.py), so this matches real usage.
    return policy_cls(
        env.observation_space, env.action_space, lambda _: 1e-3,
        features_extractor_kwargs={}, shared_gnn=shared_gnn, ortho_init=False,
    )


class TestWLPlanFeedbackPolicyConstruction(unittest.TestCase):
    def test_wl_encoder_and_resized_value_net_with_independent_discriminator(self):
        env = make_city_env()
        policy = make_policy(env, shared_gnn=False)

        # meta-controller's encoder is the new WL module, not a GNNExtractor
        self.assertIsInstance(policy.gnn_extractor, WLEmbeddingExtractor)
        self.assertNotIsInstance(policy.gnn_extractor, GNNExtractor)

        # value_net resized to the frozen vocab's size + 1 (time_left
        # concatenated onto the histogram), not EMB_SIZE and not just
        # wl_vocab_size alone
        self.assertEqual(policy.value_net.in_features, policy.wl_vocab_size + 1)
        self.assertGreater(policy.wl_vocab_size, 0)

        # discriminator's encoder is a genuine, independent GNNExtractor -
        # not aliased to the new WL module - when shared_gnn=False
        self.assertIsInstance(policy.gnn_extractor2, GNNExtractor)
        self.assertIsNot(policy.gnn_extractor2, policy.gnn_extractor)

        # default wl_vocab_path points at the validated frozen vocab
        self.assertTrue(policy.wl_vocab_path.endswith("wl_vocab_taxi_city_L1.json"))

    def test_shared_gnn_true_no_longer_aliases_a_non_gnn_meta_controller(self):
        """
        Confirms the fix to the previously-known gap: GNNFeedbackPolicy.__init__
        (graph_feedback_policy.py:244-249) now only aliases
        gnn_extractor2 = gnn_extractor when self.gnn_extractor is actually a
        GNNExtractor. Since WLPlanFeedbackPolicy's meta-controller encoder
        is a WLEmbeddingExtractor (not a GNNExtractor), gnn_extractor2 must
        now ALWAYS be a fresh, independent GNNExtractor here, regardless of
        shared_gnn - the discriminator can no longer end up sharing weights
        with (or literally being the same object as) a non-GNN encoder.
        """
        env = make_city_env()
        policy = make_policy(env, shared_gnn=True)

        self.assertIsNot(policy.gnn_extractor2, policy.gnn_extractor)
        self.assertIsInstance(policy.gnn_extractor2, GNNExtractor)
        self.assertNotIsInstance(policy.gnn_extractor2, WLEmbeddingExtractor)

    def test_shared_gnn_true_still_aliases_for_the_original_gnn_only_policy(self):
        """
        Proves the fix doesn't regress the original, intended behavior:
        for a plain (non-WL) GNNPlanFeedbackPolicy, whose meta-controller
        encoder IS a GNNExtractor, shared_gnn=True must still make
        gnn_extractor2 literally the same object as gnn_extractor - exactly
        as Tradeoff/NLE/GNN-Taxi's existing, working configuration relies on.
        """
        env = make_city_env()
        policy = make_policy(env, shared_gnn=True, policy_cls=GNNPlanFeedbackPolicy)

        self.assertIsInstance(policy.gnn_extractor, GNNExtractor)
        self.assertIs(policy.gnn_extractor2, policy.gnn_extractor)
        self.assertIsInstance(policy.gnn_extractor2, GNNExtractor)


class TestWLPlanFeedbackPolicyGetLatent(unittest.TestCase):
    def test_get_latent_end_to_end_shapes_and_downstream_net_calls(self):
        env = make_city_env()
        policy = make_policy(env, shared_gnn=False)

        js = env_to_json(env.sim)
        # matches the (n_envs, 1) dtype=U250000 buffer shape a real vec-env
        # would produce (see sage/agent/async_vec_env.py)
        obs = np.array([[js]], dtype=np.dtype("U250000"))

        batch, symbolic_batch = policy._get_latent(obs)

        n_nodes = batch.x.shape[0]
        vocab_size = policy.wl_vocab_size

        # latent_nodes: [total_nodes, EMB_SIZE]
        self.assertEqual(tuple(batch.x.shape), (n_nodes, 32))
        self.assertEqual(batch.x.dtype, th.float32)
        # latent_global: [num_graphs, wl_vocab_size + 1] (wl_histogram concat time_left)
        self.assertEqual(tuple(batch.global_features.shape), (1, vocab_size + 1))
        self.assertEqual(batch.global_features.dtype, th.float32)

        # symbolic_batch (raw current state) is untouched - still raw one-hot x
        self.assertEqual(tuple(symbolic_batch.x.shape), (n_nodes, 3))

        # downstream consumers run without shape errors
        action_scores = policy.action_net(batch.x)
        self.assertEqual(tuple(action_scores.shape), (n_nodes, 1))

        value = policy.value_net(batch.global_features)
        self.assertEqual(tuple(value.shape), (1, 1))

    def test_latent_global_last_column_is_genuinely_the_original_time_left(self):
        """
        Confirms the concatenated time_left column holds the actual
        original value - not a zero, not a duplicate of a histogram
        entry, not misindexed - by independently recomputing the raw
        global_features via json_to_graph directly (bypassing
        WLPlanFeedbackPolicy entirely) as ground truth to compare against.
        """
        env = make_city_env()
        policy = make_policy(env, shared_gnn=False)

        js = env_to_json(env.sim)
        obs = np.array([[js]], dtype=np.dtype("U250000"))

        # ground truth, independent of _get_latent/WLPlanFeedbackPolicy
        raw_batch = json_to_graph([[js]])
        expected_time_left = raw_batch.global_features[:, 0:1].clone()

        batch, _ = policy._get_latent(obs)

        self.assertEqual(tuple(batch.global_features.shape), (1, policy.wl_vocab_size + 1))
        actual_time_left = batch.global_features[:, -1:]
        self.assertTrue(th.equal(actual_time_left, expected_time_left))
        # a freshly-reset env has time=0, so time_left == timeout/timeout == 1.0
        # exactly - a genuinely non-zero, specific value, not a placeholder
        self.assertEqual(actual_time_left.item(), 1.0)

        # and the histogram portion (everything but the last column) must
        # be untouched by the concatenation - still exactly batch.wl_histogram
        self.assertTrue(th.equal(batch.global_features[:, :-1], batch.wl_histogram))

    def test_gradients_reach_value_nets_time_left_weight(self):
        """
        Mirrors the embedding-table gradient check from the end-to-end
        smoke test, but for value_net's newly-added time_left input: after
        a real backward pass, the weight column connected to time_left
        (the last input feature, so the last column of value_net.weight,
        since nn.Linear.weight has shape [out_features, in_features]) must
        have received a real, non-zero, finite gradient - not be a dead
        weight that happens to have the right shape but no actual signal
        flowing into it.
        """
        env = make_city_env()
        policy = make_policy(env, shared_gnn=False)

        js = env_to_json(env.sim)
        obs = np.array([[js]], dtype=np.dtype("U250000"))

        batch, _ = policy._get_latent(obs)
        value = policy.value_net(batch.global_features)
        value.sum().backward()

        grad = policy.value_net.weight.grad
        self.assertIsNotNone(grad)
        self.assertFalse(th.isnan(grad).any())

        time_left_weight_grad = grad[:, -1]
        self.assertFalse(th.equal(time_left_weight_grad, th.zeros_like(time_left_weight_grad)))
        # deterministic exact value: d(value.sum())/d(weight[0,-1]) ==
        # time_left == 1.0 for this single, freshly-reset graph
        self.assertAlmostEqual(time_left_weight_grad.item(), 1.0, places=5)


def make_projected_batch(env):
    """
    Builds a real projected-state Batch via Planner.plan(), the same way
    project_actions does internally (planner.plan(deepcopy(state), goal)
    then Batch.from_data_list([...])), so tests can exercise
    _encode_projected_state against a genuine post-planner projection.
    """
    js = env_to_json(env.sim)
    raw_batch = json_to_graph([[js]])
    state_data = raw_batch.to_data_list()[0]

    nx_state = graph_to_networkx(state_data)
    assert len(nx_state.passengers) > 0, "TaxiWorldSimulator always adds one passenger at construction"
    goal = nx_state.passengers[0].node

    projection, _ = Planner().plan(deepcopy(state_data), goal)
    return Batch.from_data_list([projection])


class TestDiscriminatorReadsWlHistogram(unittest.TestCase):
    def test_path_value_net_resized_for_wl_vocab_not_emb_size(self):
        env = make_city_env()
        wl_policy = make_policy(env, shared_gnn=False, policy_cls=WLPlanFeedbackPolicy)
        gnn_policy = make_policy(env, shared_gnn=False, policy_cls=GNNPlanFeedbackPolicy)

        expected_wl_dim = (wl_policy.wl_vocab_size + 1) * 2
        self.assertEqual(wl_policy.path_value_net.path_value_net.in_features, expected_wl_dim)

        # baseline (GNN) policy must be completely unaffected by the new
        # input_dim parameter's default - still exactly EMB_SIZE*2
        self.assertEqual(gnn_policy.path_value_net.path_value_net.in_features, EMB_SIZE * 2)

    def test_encode_current_state_last_column_is_genuine_time_left(self):
        env = make_city_env()
        policy = make_policy(env, shared_gnn=False, policy_cls=WLPlanFeedbackPolicy)

        js = env_to_json(env.sim)
        # independent of _get_latent/_encode_current_state - a fresh parse
        symbolic_batch = json_to_graph([[js]])
        expected_time_left = symbolic_batch.global_features[:, 0:1].clone()
        expected_histogram = symbolic_batch.wl_histogram.clone()

        result = policy._encode_current_state(symbolic_batch)

        self.assertEqual(tuple(result.shape), (1, policy.wl_vocab_size + 1))
        self.assertTrue(th.equal(result[:, -1:], expected_time_left))
        self.assertEqual(result[:, -1:].item(), 1.0)  # freshly reset: time_left == 1.0 exactly
        self.assertTrue(th.equal(result[:, :-1], expected_histogram))

    def test_encode_projected_state_last_column_is_genuine_time_left(self):
        env = make_city_env()
        policy = make_policy(env, shared_gnn=False, policy_cls=WLPlanFeedbackPolicy)

        projected_batch = make_projected_batch(env)
        # ground truth read directly off the projected batch's own
        # attributes, BEFORE passing it through the method under test
        expected_time_left = projected_batch.global_features[:, 0:1].clone()
        expected_histogram = projected_batch.wl_histogram.clone()
        # a real delivery took at least one simulated step, so time_left
        # must have moved off the fresh-reset value of 1.0 - confirms this
        # is reading the ACTUAL (decremented) projected time, not a stale
        # or default value
        self.assertLess(expected_time_left.item(), 1.0)

        result = policy._encode_projected_state(projected_batch)

        self.assertEqual(tuple(result.shape), (1, policy.wl_vocab_size + 1))
        self.assertTrue(th.equal(result[:, -1:], expected_time_left))
        self.assertTrue(th.equal(result[:, :-1], expected_histogram))


class TestGnnExtractor2GenuinelyUnused(unittest.TestCase):
    def test_gnn_extractor2_never_invoked_in_wl_choose_top_action_path(self):
        """
        Spies on gnn_extractor2.forward (not just checking it's
        unreachable by code inspection) to prove the GNN dependency was
        genuinely removed from WLPlanFeedbackPolicy's discriminator path,
        not just made unreachable by accident. num_planning_choices
        defaults to 3 (>1), so this exercises the real
        _encode_current_state/_encode_projected_state path, not the
        num_planning_choices==1 shortcut.
        """
        env = make_city_env()
        policy = make_policy(env, shared_gnn=False, policy_cls=WLPlanFeedbackPolicy)

        def _raise_if_called(*args, **kwargs):
            raise AssertionError("gnn_extractor2 was called - should be fully unused here")
        policy.gnn_extractor2.forward = _raise_if_called

        js = env_to_json(env.sim)
        obs = np.array([[js]], dtype=np.dtype("U250000"))

        # must complete without the spy's AssertionError firing
        actions, values, log_prob, explored, plans = policy.forward(obs)
        self.assertIsNotNone(actions)

    def test_spy_mechanism_itself_actually_detects_a_real_call(self):
        """
        Positive control for the test above: the same spy technique, on
        the same code path, but for the GNN baseline (where gnn_extractor2
        genuinely IS called) - proves the spy isn't silently inert/always
        passing, i.e. the test above is a meaningful negative result.
        """
        env = make_city_env()
        policy = make_policy(env, shared_gnn=False, policy_cls=GNNPlanFeedbackPolicy)

        def _raise_if_called(*args, **kwargs):
            raise AssertionError("gnn_extractor2 was called")
        policy.gnn_extractor2.forward = _raise_if_called

        js = env_to_json(env.sim)
        obs = np.array([[js]], dtype=np.dtype("U250000"))

        with self.assertRaises(AssertionError):
            policy.forward(obs)


if __name__ == "__main__":
    unittest.main()
