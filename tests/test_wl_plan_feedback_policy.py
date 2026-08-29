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

from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.utils.representations import env_to_json
from sage.agent.graph_policy import GNNExtractor
from sage.agent.graph_plan_feedback_policy import GNNPlanFeedbackPolicy
from sage.agent.wl_plan_feedback_policy import WLPlanFeedbackPolicy, WLEmbeddingExtractor


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

        # value_net resized to the frozen vocab's size (not EMB_SIZE)
        self.assertEqual(policy.value_net.in_features, policy.wl_vocab_size)
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
        # latent_global: [num_graphs, wl_vocab_size], used as-is (batch.wl_histogram)
        self.assertEqual(tuple(batch.global_features.shape), (1, vocab_size))
        self.assertEqual(batch.global_features.dtype, th.float32)

        # symbolic_batch (raw current state) is untouched - still raw one-hot x
        self.assertEqual(tuple(symbolic_batch.x.shape), (n_nodes, 3))

        # downstream consumers run without shape errors
        action_scores = policy.action_net(batch.x)
        self.assertEqual(tuple(action_scores.shape), (n_nodes, 1))

        value = policy.value_net(batch.global_features)
        self.assertEqual(tuple(value.shape), (1, 1))


if __name__ == "__main__":
    unittest.main()
