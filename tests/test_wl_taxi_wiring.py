"""
Tests for the WL-colour wiring into the Taxi domain's graph construction:
- env_to_graph (sage/domains/gym_taxi/utils/representations.py)
- the JSON round-trip (sage/domains/utils/representations.py:
  graph_to_json/json_to_graph)

Uses scenario="city" (Oracle-SAGE's real Taxi training env), the same
scenario the frozen vocab (sage/domains/utils/wl_vocab_taxi_city_L1.json)
was built and validated against.

Run from the repo root with:
    python -m unittest tests.test_wl_taxi_wiring -v
"""
import unittest

import torch as th

# Importing build_wl_vocab (not otherwise used here) applies its numpy/gym
# compatibility shims as an import side effect - needed to construct a
# "city" (random_walls=True) GraphTaxiEnv at all in this sandbox's drifted
# gym/numpy. See build_wl_vocab.py's own docstring for why. The production
# wiring code under test does NOT depend on this - only this test harness does.
import sage.domains.utils.build_wl_vocab  # noqa: F401

from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.utils.representations import env_to_graph, env_to_json
from sage.domains.gym_taxi.utils.wl_vocab_cache import get_wl_vocab, NUM_ITERATIONS
from sage.domains.utils.representations import json_to_graph
from sage.domains.utils.wl_colours import wl_colours


def make_city_env():
    env = GraphTaxiEnv(representation="graph", scenario="city", mask=False)
    env.reset()
    return env


class TestEnvToGraphWlWiring(unittest.TestCase):
    def test_env_to_graph_returns_wl_fields_matching_direct_recomputation(self):
        env = make_city_env()
        node_feats, edge_feats, edge_index, mask, global_feats, wl_colour_ids, wl_histogram = env_to_graph(env.sim)

        n_nodes = len(node_feats)
        self.assertEqual(len(wl_colour_ids), n_nodes)
        self.assertEqual(len(wl_histogram), len(get_wl_vocab()))

        # recompute independently (bypassing env_to_graph's wiring) and compare
        x = th.as_tensor(node_feats, dtype=th.float)
        edge_index_t = th.as_tensor(edge_index, dtype=th.long)
        edge_attr_t = th.as_tensor(edge_feats, dtype=th.float)
        expected_colours, expected_hist = wl_colours(
            x, edge_index_t, edge_attr_t,
            num_iterations=NUM_ITERATIONS, vocab=get_wl_vocab(), frozen=True,
        )

        self.assertEqual(wl_colour_ids, expected_colours.tolist())
        self.assertEqual(wl_histogram, expected_hist.tolist())

    def test_env_to_graph_existing_fields_unchanged_in_position_and_shape(self):
        env = make_city_env()
        result = env_to_graph(env.sim)
        self.assertEqual(len(result), 7)
        node_feats, edge_feats, edge_index, mask, global_feats, wl_colour_ids, wl_histogram = result
        # existing fields keep their original shapes/semantics
        self.assertEqual(len(mask), len(node_feats))
        self.assertEqual(len(global_feats), 32)  # EMB_SIZE


class TestJsonRoundTrip(unittest.TestCase):
    def test_wl_fields_survive_json_round_trip_with_correct_dtype_and_shape(self):
        env = make_city_env()
        _, _, _, _, _, wl_colour_ids, wl_histogram = env_to_graph(env.sim)

        js = env_to_json(env.sim)
        batch = json_to_graph([[js]])

        self.assertTrue(hasattr(batch, "wl_colours"))
        self.assertTrue(hasattr(batch, "wl_histogram"))
        self.assertEqual(batch.wl_colours.dtype, th.long)
        self.assertEqual(batch.wl_histogram.dtype, th.float)
        self.assertEqual(tuple(batch.wl_colours.shape), (len(wl_colour_ids),))
        self.assertEqual(tuple(batch.wl_histogram.shape), (1, len(wl_histogram)))
        self.assertEqual(batch.wl_colours.tolist(), wl_colour_ids)
        self.assertEqual(batch.wl_histogram.squeeze(0).tolist(), wl_histogram)

    def test_non_taxi_five_field_json_is_unaffected(self):
        # Simulates gym_tradeoff/gym_nle's graph_to_json(*env_to_graph(env))
        # call pattern, whose env_to_graph still returns a plain 5-tuple.
        from sage.domains.utils.representations import graph_to_json
        import json

        js = graph_to_json([[1, 0, 0]], [[1, 0, 0, 1]], [[0], [0]], [True], [0.0] * 32)
        parsed = json.loads(js)
        self.assertNotIn("wl_colours", parsed)
        self.assertNotIn("wl_histogram", parsed)

        batch = json_to_graph([[js]])
        d = batch.to_data_list()[0]
        self.assertFalse(hasattr(d, "wl_colours"))
        self.assertFalse(hasattr(d, "wl_histogram"))


if __name__ == "__main__":
    unittest.main()
