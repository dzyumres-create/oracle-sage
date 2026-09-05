"""
Unit tests for sage.domains.utils.wl_colours.

Run from the repo root with:
    python -m unittest tests.test_wl_colours -v
"""
import unittest

import torch as th

from sage.domains.utils.wl_colours import (
    OOV_SIGNATURE,
    edge_labels,
    freeze_vocab,
    initial_colours,
    refine,
    wl_colours,
)


LOCATION = [1, 0, 0]
TAXI = [0, 1, 0]
PASSENGER = [0, 0, 1]

ROAD_FWD = [1, 0, 0, 1]
TETHER_FWD = [0, 1, 0, 1]
TETHER_BWD = [0, 1, 0, -1]
DEST_FWD = [0, 0, 1, 1]
DEST_BWD = [0, 0, 1, -1]


def make_taxi_graph():
    """
    4-node hand-built graph using the real Taxi edge-attr convention:
        node 0: taxi, currently at location 1
        node 1: location A
        node 2: location B
        node 3: passenger, at location B (node 2), heading to location A (node 1)
    Road edge between the two locations. See the hand-derivation in
    TestRefineByHand for the expected colours after 1 iteration.
    """
    x = th.tensor([TAXI, LOCATION, LOCATION, PASSENGER], dtype=th.float)

    edges = [
        (0, 1, TETHER_FWD),   # taxi -> its location
        (1, 0, TETHER_BWD),   # location -> taxi
        (3, 2, TETHER_FWD),   # passenger -> its location
        (2, 3, TETHER_BWD),   # location -> passenger
        (3, 1, DEST_FWD),     # passenger -> its destination
        (1, 3, DEST_BWD),     # destination -> passenger
        (1, 2, ROAD_FWD),     # road A -> B
        (2, 1, ROAD_FWD),     # road B -> A
    ]
    edge_index = th.tensor([[s, d] for s, d, _ in edges], dtype=th.long).T
    edge_attr = th.tensor([a for _, _, a in edges], dtype=th.float)
    return x, edge_index, edge_attr


def make_path_graph(n_locations):
    """
    A simple n-node path graph of `location` nodes, connected by bidirectional
    road edges: 0 - 1 - 2 - ... - (n-1).
    """
    x = th.tensor([LOCATION] * n_locations, dtype=th.float)
    edges = []
    for i in range(n_locations - 1):
        edges.append((i, i + 1, ROAD_FWD))
        edges.append((i + 1, i, ROAD_FWD))
    edge_index = th.tensor([[s, d] for s, d, _ in edges], dtype=th.long).T
    edge_attr = th.tensor([a for _, _, a in edges], dtype=th.float)
    return x, edge_index, edge_attr


class TestInitialColoursAndEdgeLabels(unittest.TestCase):
    def test_initial_colours_is_one_hot_argmax(self):
        x = th.tensor([LOCATION, TAXI, PASSENGER], dtype=th.float)
        colours = initial_colours(x)
        self.assertEqual(colours.dtype, th.long)
        self.assertTrue(th.equal(colours, th.tensor([0, 1, 2], dtype=th.long)))

    def test_edge_labels_encode_type_and_direction(self):
        edge_attr = th.tensor(
            [ROAD_FWD, TETHER_FWD, TETHER_BWD, DEST_FWD, DEST_BWD],
            dtype=th.float,
        )
        labels = edge_labels(edge_attr)
        # road,+1 -> 0 ; tether,+1 -> 2 ; tether,-1 -> 3 ; dest,+1 -> 4 ; dest,-1 -> 5
        self.assertTrue(
            th.equal(labels, th.tensor([0, 2, 3, 4, 5], dtype=th.long))
        )


class TestHandVerifiedRefinement(unittest.TestCase):
    """
    Part (a): hand-derive the colours after 1 iteration on make_taxi_graph()
    and check wl_colours produces exactly those ids, given a fresh vocab.

    Hand derivation (see PR description / commit message for full working):
      - x rows in order are [taxi, location, location, passenger], so with a
        fresh vocab, initial_colours' raw ids {1, 0, 0, 2} get mapped via the
        ("init", type_id) signature to vocab ids 0, 1, 1, 2 respectively
        (taxi type is encountered first -> 0, location type second -> 1,
        passenger type fourth -> 2). So mapped initial colours = [0, 1, 1, 2].
      - edge_labels for the 8 edges (in the order added by make_taxi_graph):
        [2, 3, 2, 3, 4, 5, 0, 0]
      - outgoing-edge (neighbour_colour, label) multisets:
          node0 (taxi):     [(1, 2)]
          node1 (locationA): [(0, 3), (1, 0), (2, 5)]
          node2 (locationB): [(1, 0), (2, 3)]
          node3 (passenger): [(1, 2), (1, 4)]
      - combined with each node's own current colour, all four signatures
        are distinct, so refine() assigns four new, distinct vocab ids in
        node order 0,1,2,3: since the vocab already has 3 entries from the
        init step (ids 0,1,2), the new ids are 3,4,5,6 respectively.
    """

    def test_one_iteration_matches_hand_derivation(self):
        x, edge_index, edge_attr = make_taxi_graph()
        vocab = {}

        colours, histogram = wl_colours(
            x, edge_index, edge_attr, num_iterations=1, vocab=vocab
        )

        self.assertEqual(colours.dtype, th.long)
        self.assertTrue(th.equal(colours, th.tensor([3, 4, 5, 6], dtype=th.long)))

        # 3 "init" signatures + 4 new refine signatures = 7 vocab entries.
        self.assertEqual(len(vocab), 7)

        self.assertEqual(histogram.shape, (7,))
        self.assertEqual(histogram.dtype, th.float)
        expected_histogram = th.tensor(
            [0, 0, 0, 1, 1, 1, 1], dtype=th.float
        )
        self.assertTrue(th.equal(histogram, expected_histogram))

    def test_refine_directly_matches_wl_colours_first_step(self):
        # Cross-check refine() in isolation against the same hand derivation,
        # bypassing wl_colours' init-vocab bookkeeping.
        x, edge_index, edge_attr = make_taxi_graph()
        node_colours = initial_colours(x)  # raw type ids: [1, 0, 0, 2]
        labels = edge_labels(edge_attr)
        vocab = {}

        new_colours = refine(node_colours, edge_index, labels, vocab)

        # Signatures are all distinct -> 4 fresh ids, assigned in node order.
        self.assertTrue(th.equal(new_colours, th.tensor([0, 1, 2, 3], dtype=th.long)))
        self.assertEqual(len(vocab), 4)


class TestIsomorphismInvariance(unittest.TestCase):
    """Part (b): structurally identical graphs get identical colours/histograms."""

    def test_identical_graphs_get_identical_colours_and_histograms(self):
        x1, edge_index1, edge_attr1 = make_taxi_graph()
        x2, edge_index2, edge_attr2 = make_taxi_graph()

        colours1, hist1 = wl_colours(x1, edge_index1, edge_attr1, num_iterations=5, vocab={})
        colours2, hist2 = wl_colours(x2, edge_index2, edge_attr2, num_iterations=5, vocab={})

        self.assertTrue(th.equal(colours1, colours2))
        self.assertTrue(th.equal(hist1, hist2))

    def test_relabelled_isomorphic_graph_matches_under_shared_vocab(self):
        """
        Build a second graph that is the same graph with locations 1 and 2
        swapped (a genuine relabelling/isomorphism, not a literal copy).
        Under a SHARED vocab, colours must match once un-permuted, and the
        resulting histograms must be identical.
        """
        x1, edge_index1, edge_attr1 = make_taxi_graph()

        # swap node labels 1 <-> 2 (locationA <-> locationB), 0 and 3 fixed.
        # Both location rows share identical features, so x is unchanged;
        # only the edge endpoints need relabelling.
        perm = {0: 0, 1: 2, 2: 1, 3: 3}
        x2 = x1.clone()
        edge_index2 = th.tensor(
            [[perm[int(s)], perm[int(d)]] for s, d in edge_index1.T.tolist()],
            dtype=th.long,
        ).T
        edge_attr2 = edge_attr1.clone()

        vocab = {}
        colours1, hist1 = wl_colours(x1, edge_index1, edge_attr1, num_iterations=1, vocab=vocab)
        colours2, hist2 = wl_colours(x2, edge_index2, edge_attr2, num_iterations=1, vocab=vocab)

        # un-permute colours2 back into graph-1's node ordering
        inverse_perm = [perm[i] for i in range(4)]  # this permutation is its own inverse (a transposition)
        colours2_unpermuted = colours2[inverse_perm]

        self.assertTrue(th.equal(colours1, colours2_unpermuted))
        self.assertTrue(th.equal(hist1, hist2))
        # no new signatures should have been introduced by the relabelled graph
        self.assertEqual(len(vocab), 7)


class TestNodeTypeChangeAffectsColours(unittest.TestCase):
    """Part (c): changing one node's type changes at least one colour."""

    def test_changing_one_node_type_changes_colours(self):
        x_baseline = th.tensor([LOCATION, LOCATION, LOCATION], dtype=th.float)
        x_changed = th.tensor([LOCATION, LOCATION, TAXI], dtype=th.float)

        edges = [(0, 1, ROAD_FWD), (1, 0, ROAD_FWD), (1, 2, ROAD_FWD), (2, 1, ROAD_FWD)]
        edge_index = th.tensor([[s, d] for s, d, _ in edges], dtype=th.long).T
        edge_attr = th.tensor([a for _, _, a in edges], dtype=th.float)

        colours_baseline, _ = wl_colours(
            x_baseline, edge_index, edge_attr, num_iterations=1, vocab={}
        )
        colours_changed, _ = wl_colours(
            x_changed, edge_index, edge_attr, num_iterations=1, vocab={}
        )

        self.assertFalse(th.equal(colours_baseline, colours_changed))
        # specifically, the changed node's own colour must differ
        self.assertNotEqual(colours_baseline[2].item(), colours_changed[2].item())


class TestVocabStability(unittest.TestCase):
    """
    Part (d): the vocab dict grows while new signatures are seen, and stays
    stable (ids reused, not reassigned) once a graph's signatures have all
    been seen before.
    """

    def test_vocab_grows_then_stabilises_and_reuses_ids_across_calls(self):
        shared_vocab = {}

        # Call 1: a 3-node path. Endpoints (degree 1) share one signature,
        # the middle node (degree 2) has a different one.
        x1, edge_index1, edge_attr1 = make_path_graph(3)
        colours1, _ = wl_colours(x1, edge_index1, edge_attr1, num_iterations=1, vocab=shared_vocab)

        # 1 "init" signature (all nodes are type `location`) + 2 refine
        # signatures (endpoint-shape, middle-shape) = 3 vocab entries.
        self.assertEqual(len(shared_vocab), 3)
        endpoint_colour = colours1[0].item()
        middle_colour = colours1[1].item()
        self.assertEqual(colours1[2].item(), endpoint_colour)  # other endpoint matches
        self.assertNotEqual(endpoint_colour, middle_colour)

        vocab_size_after_call1 = len(shared_vocab)

        # Call 2: a DIFFERENT (longer, 5-node) path graph, sharing the SAME
        # vocab. Its endpoints and interior nodes reproduce exactly the same
        # 1-hop signatures as call 1's path, so no new ids should be minted.
        x2, edge_index2, edge_attr2 = make_path_graph(5)
        colours2, _ = wl_colours(x2, edge_index2, edge_attr2, num_iterations=1, vocab=shared_vocab)

        self.assertEqual(len(shared_vocab), vocab_size_after_call1)  # unchanged: no new signatures

        # previously-seen colours are REUSED, not reassigned to new ids
        self.assertEqual(colours2[0].item(), endpoint_colour)
        self.assertEqual(colours2[4].item(), endpoint_colour)
        self.assertEqual(colours2[1].item(), middle_colour)
        self.assertEqual(colours2[2].item(), middle_colour)
        self.assertEqual(colours2[3].item(), middle_colour)


def make_star_graph():
    """
    4-node star: a centre node (degree 3) connected to three leaf nodes
    (each degree 1), all type `location`, road edges both directions.
    Used to construct a genuinely novel degree-3 neighbourhood signature
    that a 3-node path graph's vocab (max degree 2) could never have seen.
    """
    x = th.tensor([LOCATION] * 4, dtype=th.float)
    edges = []
    for leaf in (1, 2, 3):
        edges.append((0, leaf, ROAD_FWD))
        edges.append((leaf, 0, ROAD_FWD))
    edge_index = th.tensor([[s, d] for s, d, _ in edges], dtype=th.long).T
    edge_attr = th.tensor([a for _, _, a in edges], dtype=th.float)
    return x, edge_index, edge_attr


def build_and_freeze_path_vocab():
    """
    Shared setup for the frozen/OOV tests: build a vocab from a 3-node path
    graph (growing mode, 1 iteration), then freeze it.

    Resulting vocab (established by TestVocabStability, unchanged here):
        ("init", 0)                    -> 0   (location type)
        (0, ((0, 0),))                 -> 1   (S_end: degree-1 endpoint)
        (0, ((0, 0), (0, 0)))          -> 2   (S_mid: degree-2 middle)
        OOV_SIGNATURE                  -> 3   (added by freeze_vocab)
    So vocab_size == 4, and a degree-3 neighbourhood (never seen here) is
    genuinely novel.
    """
    vocab = {}
    x, edge_index, edge_attr = make_path_graph(3)
    wl_colours(x, edge_index, edge_attr, num_iterations=1, vocab=vocab)
    vocab_size = freeze_vocab(vocab)
    return vocab, vocab_size


class TestFrozenVocabAndOOV(unittest.TestCase):
    def test_freeze_vocab_reserves_oov_and_returns_vocab_size(self):
        vocab = {}
        x, edge_index, edge_attr = make_path_graph(3)
        wl_colours(x, edge_index, edge_attr, num_iterations=1, vocab=vocab)
        size_before = len(vocab)
        self.assertNotIn(OOV_SIGNATURE, vocab)

        vocab_size = freeze_vocab(vocab)

        self.assertIn(OOV_SIGNATURE, vocab)
        self.assertEqual(vocab[OOV_SIGNATURE], size_before)
        self.assertEqual(vocab_size, size_before + 1)
        self.assertEqual(len(vocab), vocab_size)

    def test_novel_signature_resolves_to_oov_and_vocab_is_unchanged(self):
        """Part (a): a genuinely novel node configuration maps to OOV, and
        nothing is added to the frozen vocab."""
        vocab, vocab_size = build_and_freeze_path_vocab()

        x, edge_index, edge_attr = make_star_graph()
        colours, _ = wl_colours(
            x, edge_index, edge_attr, num_iterations=1, vocab=vocab, frozen=True
        )

        # centre node (degree 3) has a neighbourhood never seen in the
        # 3-node path (max degree 2) -> must resolve to OOV
        self.assertEqual(colours[0].item(), vocab[OOV_SIGNATURE])
        # nothing was added: vocab size is exactly what freeze_vocab fixed
        self.assertEqual(len(vocab), vocab_size)

    def test_known_signature_still_resolves_normally_when_frozen(self):
        """Part (b): freezing doesn't break lookups for signatures already
        in the vocab - only unseen ones get OOV'd."""
        vocab, vocab_size = build_and_freeze_path_vocab()
        end_id = vocab[(0, ((0, 0),))]  # S_end's id, established pre-freeze

        x, edge_index, edge_attr = make_star_graph()
        colours, _ = wl_colours(
            x, edge_index, edge_attr, num_iterations=1, vocab=vocab, frozen=True
        )

        # the three leaves (degree 1, connected to a location) reproduce
        # S_end exactly, which WAS already in the vocab before freezing
        for leaf_colour in colours[1:]:
            self.assertEqual(leaf_colour.item(), end_id)
            self.assertNotEqual(leaf_colour.item(), vocab[OOV_SIGNATURE])
        self.assertEqual(len(vocab), vocab_size)

    def test_freeze_vocab_is_idempotent(self):
        """Part (c): calling freeze_vocab twice doesn't add a second OOV
        entry or change vocab_size."""
        vocab, vocab_size = build_and_freeze_path_vocab()
        vocab_before = dict(vocab)

        vocab_size_again = freeze_vocab(vocab)

        self.assertEqual(vocab_size_again, vocab_size)
        self.assertEqual(vocab, vocab_before)
        self.assertEqual(sum(1 for k in vocab if k == OOV_SIGNATURE), 1)

    def test_two_frozen_calls_on_different_graphs_have_same_histogram_length(self):
        """Part (d): frozen histograms always have the same (fixed) length,
        regardless of which graph produced them."""
        vocab, vocab_size = build_and_freeze_path_vocab()

        x1, edge_index1, edge_attr1 = make_path_graph(3)
        _, hist1 = wl_colours(
            x1, edge_index1, edge_attr1, num_iterations=1, vocab=vocab, frozen=True
        )

        x2, edge_index2, edge_attr2 = make_star_graph()
        _, hist2 = wl_colours(
            x2, edge_index2, edge_attr2, num_iterations=1, vocab=vocab, frozen=True
        )

        self.assertEqual(hist1.shape, hist2.shape)
        self.assertEqual(hist1.shape, (vocab_size,))
        self.assertEqual(len(vocab), vocab_size)  # still unchanged by either call

    def test_frozen_without_prior_freeze_raises_clear_error(self):
        """frozen=True with a vocab that was never passed through
        freeze_vocab (no OOV entry) must fail loudly, not silently misbehave."""
        vocab = {}
        x, edge_index, edge_attr = make_path_graph(3)
        wl_colours(x, edge_index, edge_attr, num_iterations=1, vocab=vocab)  # no freeze_vocab call
        self.assertNotIn(OOV_SIGNATURE, vocab)

        with self.assertRaises(ValueError):
            wl_colours(x, edge_index, edge_attr, num_iterations=1, vocab=vocab, frozen=True)

        node_colours = initial_colours(x)
        labels = edge_labels(edge_attr)
        with self.assertRaises(ValueError):
            refine(node_colours, edge_index, labels, vocab, frozen=True)


class TestWlColoursSmoke(unittest.TestCase):
    def test_default_num_iterations_runs_and_shapes_are_consistent(self):
        x, edge_index, edge_attr = make_taxi_graph()
        vocab = {}
        colours, histogram = wl_colours(x, edge_index, edge_attr, vocab=vocab)

        self.assertEqual(colours.shape, (4,))
        self.assertEqual(colours.dtype, th.long)
        self.assertEqual(histogram.shape, (len(vocab),))
        self.assertEqual(histogram.dtype, th.float)
        self.assertEqual(histogram.sum().item(), 4)


@unittest.skipUnless(th.cuda.is_available(), "requires CUDA - run on RCP to actually exercise this")
class TestWlColoursGpuDevicePlacement(unittest.TestCase):
    """
    Regression tests for the device-mismatch bug: wl_colours()/refine()
    used to construct their output tensors (th.empty/th.zeros) with no
    device= kwarg, so they always came back on CPU regardless of the
    input graph's actual device - fine for the live-state path (which
    round-trips through JSON and gets moved to GPU later anyway), but a
    hard crash for planner.py's projected-state path, which mutates an
    already-on-GPU batch directly with no later move.
    Skipped entirely (not just "passes vacuously") unless CUDA is
    genuinely available - meaningless on a CPU-only sandbox, must be run
    on RCP to actually exercise the fix.
    """

    def test_wl_colours_returns_tensors_on_the_input_device_not_cpu(self):
        device = th.device("cuda")
        x, edge_index, edge_attr = make_taxi_graph()
        x, edge_index, edge_attr = x.to(device), edge_index.to(device), edge_attr.to(device)

        colours, histogram = wl_colours(x, edge_index, edge_attr, num_iterations=1, vocab={})

        self.assertEqual(colours.device.type, "cuda")
        self.assertEqual(histogram.device.type, "cuda")

    def test_refine_returns_tensor_on_the_input_device_not_cpu(self):
        device = th.device("cuda")
        x, edge_index, edge_attr = make_taxi_graph()
        x, edge_index, edge_attr = x.to(device), edge_index.to(device), edge_attr.to(device)

        node_colours = initial_colours(x)
        labels = edge_labels(edge_attr)
        new_colours = refine(node_colours, edge_index, labels, {})

        self.assertEqual(new_colours.device.type, "cuda")

    def test_frozen_path_also_returns_tensors_on_the_input_device(self):
        device = th.device("cuda")
        x, edge_index, edge_attr = make_taxi_graph()
        x, edge_index, edge_attr = x.to(device), edge_index.to(device), edge_attr.to(device)

        vocab = {}
        wl_colours(x, edge_index, edge_attr, num_iterations=1, vocab=vocab)
        freeze_vocab(vocab)

        colours, histogram = wl_colours(x, edge_index, edge_attr, num_iterations=1, vocab=vocab, frozen=True)

        self.assertEqual(colours.device.type, "cuda")
        self.assertEqual(histogram.device.type, "cuda")

    def test_values_identical_between_cpu_and_gpu_only_device_differs(self):
        """
        Moving a graph to GPU must not change the computed colours or
        histogram values - only where the result tensors live. Reuses the
        hand-derived taxi graph fixture (see TestHandVerifiedRefinement)
        so the CPU side is itself already independently hand-verified.
        """
        x_cpu, edge_index_cpu, edge_attr_cpu = make_taxi_graph()
        device = th.device("cuda")
        x_gpu = x_cpu.to(device)
        edge_index_gpu = edge_index_cpu.to(device)
        edge_attr_gpu = edge_attr_cpu.to(device)

        colours_cpu, histogram_cpu = wl_colours(x_cpu, edge_index_cpu, edge_attr_cpu, num_iterations=1, vocab={})
        colours_gpu, histogram_gpu = wl_colours(x_gpu, edge_index_gpu, edge_attr_gpu, num_iterations=1, vocab={})

        self.assertTrue(th.equal(colours_cpu, colours_gpu.cpu()))
        self.assertTrue(th.equal(histogram_cpu, histogram_gpu.cpu()))


if __name__ == "__main__":
    unittest.main()
