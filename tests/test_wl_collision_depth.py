"""
Tests for sage/domains/utils/wl_collision_depth.py (depth-bucketed frozen collision rate
for a saved vocab), on the small predictable5 scenario.

Run from the repo root with:
    python -m pytest tests/test_wl_collision_depth.py -v
"""
import unittest

from sage.domains.utils.build_wl_vocab import MASK, REWARDS_VARIANT, to_planner_data
from sage.domains.utils.wl_collision_depth import measure_collisions_bucketed
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.utils.wl_colours import OOV_SIGNATURE, freeze_vocab, wl_colours
from sage.domains.utils.wl_depth_sweep import BUCKETS

SETTINGS = dict(scenario="predictable5", seed=3, episodes=1, sample_every=100, k=6, log=lambda *_: None)
CONVENTIONS = ("oracle_sage", "vilg", "atom")


def run(convention, vocab, num_iterations=1):
    return measure_collisions_bucketed(convention, vocab, num_iterations, **SETTINGS)


class TestWlCollisionDepth(unittest.TestCase):
    def test_states_bucketed_by_sampling_depth(self):
        results = run("oracle_sage", {OOV_SIGNATURE: 0})
        # predictable5 times out at 1000 steps: states at 0, 100, ..., 900
        expected = {b: sum(1 for s in range(0, 1000, 100) if b[0] <= s < b[1]) for b in BUCKETS}
        self.assertEqual({b: r["states"] for b, r in results.items()}, expected)

    def test_oov_only_vocab_collides_on_every_move_move_pair(self):
        """Every node resolves to OOV, so two projections with equal node counts (any
        two moves) have identical histograms: every differing move-move pair collides."""
        for convention in CONVENTIONS:
            with self.subTest(convention=convention):
                results = run(convention, {OOV_SIGNATURE: 0})
                pairs = sum(r["pairs"].get(("move", "move"), 0) for r in results.values())
                coll = sum(r["collisions"].get(("move", "move"), 0) for r in results.values())
                self.assertGreater(pairs, 0)
                self.assertEqual(coll, pairs)

    def test_real_vocab_separates_pairs_the_oov_vocab_cannot(self):
        """A vocab grown from one live state (then frozen) distinguishes at least some
        move-move pairs, and never reports more collisions than pairs."""
        for convention in CONVENTIONS:
            with self.subTest(convention=convention):
                env = GraphTaxiEnv(representation="graph", scenario="predictable5", mask=MASK,
                                   rewards=REWARDS_VARIANT, graph_convention=convention)
                env.seed(0)
                env.reset()
                d = to_planner_data(env.sim, graph_convention=convention)
                vocab = {}
                wl_colours(d.x, d.edge_index, d.edge_attr, num_iterations=1, vocab=vocab, graph_convention=convention)
                freeze_vocab(vocab)
                results = run(convention, vocab)
                pairs = sum(r["pairs"].get(("move", "move"), 0) for r in results.values())
                coll = sum(r["collisions"].get(("move", "move"), 0) for r in results.values())
                self.assertLess(coll, pairs)
                for r in results.values():
                    for pair_type, n in r["pairs"].items():
                        self.assertLessEqual(r["collisions"].get(pair_type, 0), n)


if __name__ == "__main__":
    unittest.main()
