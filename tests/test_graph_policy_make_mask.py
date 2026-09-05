"""
Equivalence test for the make_mask performance fix
(sage/agent/graph_policy.py): a prior investigation found the original
implementation iterated batch.batch (a GPU tensor) element-by-element in
pure Python via itertools.groupby - forcing a GPU-CPU sync per element,
with cost scaling with total node count (documented to turn a ~90min run
into 7+ hours on one seed). It's replaced with a vectorised
th.unique_consecutive-based implementation.

This test does NOT trust the fix by inspection - it runs the ORIGINAL
buggy implementation (preserved here verbatim, for comparison only) and
the current, fixed make_mask on IDENTICAL constructed batches, and asserts
all three return values (mask, data_splits, data_starts) match exactly.

Run from the repo root with:
    python -m unittest tests.test_graph_policy_make_mask -v
"""
import itertools
import unittest

import torch as th

from sage.agent.graph_policy import get_start_indices, make_mask


class _MockBatch:
    """Minimal stand-in for a torch_geometric Batch - make_mask only reads .mask and .batch."""

    def __init__(self, mask, batch):
        self.mask = mask
        self.batch = batch


def _original_make_mask(batch):
    """
    The pre-fix implementation, preserved verbatim (not imported from
    graph_policy.py, since it no longer exists there) purely so this test
    can prove the fix is behaviourally identical, not just trusted by
    inspection.
    """
    device = batch.mask.device
    data_splits = [sum(1 for _ in g) for _, g in itertools.groupby(batch.batch)]
    data_splits_tensor = th.tensor(data_splits, device=device)
    data_starts = get_start_indices(data_splits_tensor)
    return batch.mask.flatten(), data_splits, data_starts


def _assert_equivalent(test_case, batch):
    mask_old, splits_old, starts_old = _original_make_mask(batch)
    mask_new, splits_new, starts_new = make_mask(batch)

    test_case.assertTrue(th.equal(mask_old, mask_new), f"mask mismatch: {mask_old} vs {mask_new}")
    test_case.assertEqual(splits_old, splits_new, f"data_splits mismatch: {splits_old} vs {splits_new}")
    test_case.assertTrue(th.equal(starts_old, starts_new), f"data_starts mismatch: {starts_old} vs {starts_new}")

    return (mask_old, splits_old, starts_old), (mask_new, splits_new, starts_new)


class TestMakeMaskEquivalence(unittest.TestCase):
    def test_three_graphs_unequal_sizes(self):
        # 3 graphs of sizes 5, 2, 7 (unequal), 14 nodes total
        batch_index = th.tensor([0] * 5 + [1] * 2 + [2] * 7, dtype=th.long)
        mask = th.tensor([True, False, True, True, False,
                           True, True,
                           False, True, True, False, True, True, False])
        mock_batch = _MockBatch(mask=mask, batch=batch_index)

        old, new = _assert_equivalent(self, mock_batch)

        print(f"\n[test_three_graphs_unequal_sizes] old={old}")
        print(f"[test_three_graphs_unequal_sizes] new={new}")

        # also pin down the actual expected values by hand, independent of
        # either implementation, as a further sanity check
        self.assertEqual(new[1], [5, 2, 7])
        self.assertTrue(th.equal(new[2], th.tensor([0, 5, 7])))
        self.assertTrue(th.equal(new[0], mask))

    def test_five_graphs_including_size_one_boundary(self):
        # sizes 1, 1, 3, 1, 4 - exercises consecutive size-1 graphs, a case
        # where groupby's "consecutive equal VALUE" grouping and
        # unique_consecutive's "consecutive equal element" grouping could
        # plausibly diverge if either treated repeated single-node graphs
        # incorrectly (they don't - batch.batch is strictly non-decreasing
        # by construction - but worth verifying directly for this exact
        # boundary case rather than assuming).
        batch_index = th.tensor([0, 1, 2, 2, 2, 3, 4, 4, 4, 4], dtype=th.long)
        mask = th.rand(10) > 0.5
        mock_batch = _MockBatch(mask=mask, batch=batch_index)

        old, new = _assert_equivalent(self, mock_batch)

        print(f"\n[test_five_graphs_including_size_one_boundary] old={old}")
        print(f"[test_five_graphs_including_size_one_boundary] new={new}")

        self.assertEqual(new[1], [1, 1, 3, 1, 4])
        self.assertTrue(th.equal(new[2], th.tensor([0, 1, 2, 5, 6])))

    def test_single_graph(self):
        batch_index = th.zeros(6, dtype=th.long)
        mask = th.ones(6, dtype=th.bool)
        mock_batch = _MockBatch(mask=mask, batch=batch_index)

        old, new = _assert_equivalent(self, mock_batch)

        print(f"\n[test_single_graph] old={old}")
        print(f"[test_single_graph] new={new}")

        self.assertEqual(new[1], [6])
        self.assertTrue(th.equal(new[2], th.tensor([0])))


if __name__ == "__main__":
    unittest.main()
