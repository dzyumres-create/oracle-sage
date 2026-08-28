"""
.. module:: wl_colours
   :synopsis: Plain Weisfeiler-Leman colour refinement over directed graphs,
   as a pure-tensor building block intended to eventually replace
   Oracle-SAGE's GNN encoder for the Taxi domain.

   Nothing here depends on torch_geometric's Data/Batch classes, or on the
   environment/planner - functions take plain tensors in and return plain
   tensors out, with a `vocab` dict threaded through to keep colour ids
   stable and comparable across repeated calls (e.g. across environment
   steps, or across many graphs in a training run).
"""
from typing import Dict, Hashable, Tuple

import torch as th

OOV_SIGNATURE = "__OOV__"


def freeze_vocab(vocab: Dict[Hashable, int]) -> int:
    """
    Freezes `vocab` for read-only use: reserves one additional signature,
    `OOV_SIGNATURE` ("__OOV__"), if not already present, assigning it the
    next incrementing id exactly like any other new vocab entry. Idempotent
    - calling this twice on the same vocab is a no-op the second time.

    Call this once, offline, after accumulating a vocab across a
    representative sample of graphs (see `refine`/`wl_colours`'s `frozen`
    parameter for how the resulting vocab is then used read-only).

    :param vocab: signature -> colour id, mutated in place
    :return: vocab_size, i.e. len(vocab) after this call. This is the fixed
        size all future frozen-mode histograms from this vocab will use.
    """
    if OOV_SIGNATURE not in vocab:
        vocab[OOV_SIGNATURE] = len(vocab)
    return len(vocab)


def _resolve(signature: Hashable, vocab: Dict[Hashable, int], frozen: bool) -> int:
    """
    Looks up `signature` in `vocab`, returning its id. If `signature` is
    unseen: in growing mode (frozen=False) it is assigned the next
    incrementing id; in frozen mode it resolves to `vocab[OOV_SIGNATURE]`
    instead, and `vocab` is left untouched.
    """
    if signature in vocab:
        return vocab[signature]
    if frozen:
        return vocab[OOV_SIGNATURE]
    vocab[signature] = len(vocab)
    return vocab[signature]


def _check_frozen_vocab(vocab: Dict[Hashable, int], frozen: bool, caller: str) -> None:
    if frozen and OOV_SIGNATURE not in vocab:
        raise ValueError(
            f"{caller}() called with frozen=True but vocab has no "
            f'"{OOV_SIGNATURE}" entry - call freeze_vocab(vocab) first.'
        )


def initial_colours(x: th.Tensor) -> th.Tensor:
    """
    Assigns each node its initial WL colour from its one-hot node-type
    features.

    :param x: node features, shape [N, 3], one-hot
        [is_location, is_taxi, is_passenger]
    :return: initial colour per node, shape [N], dtype long. Values are
        simply the one-hot argmax (0=location, 1=taxi, 2=passenger) - this
        is a small, local id space, distinct from (and not looked up in)
        `vocab`; see `wl_colours` for how it is folded into the shared
        vocab id space.
    """
    return th.argmax(x, dim=1).long()


def edge_labels(edge_attr: th.Tensor) -> th.Tensor:
    """
    Combines edge type and direction into a single categorical edge label.

    edge_attr columns are [is_road, is_tether, is_destination, direction],
    where direction is +1 or -1 (each undirected connection in the Taxi
    graph is represented as a pair of edges with opposite direction and
    otherwise identical attr).

    Encoding: label = type_id * 2 + (0 if direction > 0 else 1), where
    type_id = argmax(edge_attr[:, 0:3]) in {0=road, 1=tether,
    2=destination}. So labels take values in {0, ..., 5}:
        0 = road,        direction +1
        1 = road,        direction -1
        2 = tether,       direction +1
        3 = tether,       direction -1
        4 = destination,  direction +1
        5 = destination,  direction -1

    :param edge_attr: edge attributes, shape [E, 4]
    :return: label per edge, shape [E], dtype long
    """
    type_id = th.argmax(edge_attr[:, 0:3], dim=1)
    direction_bit = (edge_attr[:, 3] <= 0).long()
    return type_id * 2 + direction_bit


def refine(
    node_colours: th.Tensor,
    edge_index: th.Tensor,
    edge_labels: th.Tensor,
    vocab: Dict[Hashable, int],
    frozen: bool = False,
) -> th.Tensor:
    """
    Runs one iteration of WL colour refinement over a directed graph.

    For each node v, the new colour is derived from v's current colour
    together with the multiset of (neighbour_colour, edge_label) pairs over
    v's OUTGOING edges (edges where v is edge_index[0]). Since the Taxi
    graph represents every undirected connection as a pair of
    opposite-direction edges, iterating over outgoing edges alone already
    captures both endpoints of every connection touching v.

    The (own_colour, sorted multiset) signature is looked up in `vocab`. In
    growing mode (frozen=False, the default) a signature seen for the first
    time (in this call or any previous call sharing the same `vocab`) is
    assigned the next incrementing integer id, and `vocab` is mutated in
    place so ids stay stable and comparable across repeated calls. In
    frozen mode (frozen=True) `vocab` is read-only: an unseen signature
    resolves to `vocab[OOV_SIGNATURE]` instead of being added - see
    `freeze_vocab`.

    :param node_colours: current colour per node, shape [N], dtype long
    :param edge_index: edge index, shape [2, E], edge_index[0] = source,
        edge_index[1] = destination
    :param edge_labels: label per edge (see `edge_labels`), shape [E]
    :param vocab: signature -> colour id, mutated in place unless `frozen`
    :param frozen: if True, treat `vocab` as read-only (see above).
        Requires `freeze_vocab(vocab)` to have been called first - raises
        ValueError otherwise.
    :return: new colour per node, shape [N], dtype long
    """
    _check_frozen_vocab(vocab, frozen, "refine")

    n = node_colours.shape[0]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    node_colours_list = node_colours.tolist()
    edge_labels_list = edge_labels.tolist()

    neighbours = [[] for _ in range(n)]
    for e, s in enumerate(src):
        d = dst[e]
        neighbours[s].append((node_colours_list[d], edge_labels_list[e]))

    new_colours = th.empty(n, dtype=th.long)
    for v in range(n):
        signature = (node_colours_list[v], tuple(sorted(neighbours[v])))
        new_colours[v] = _resolve(signature, vocab, frozen)

    return new_colours


def wl_colours(
    x: th.Tensor,
    edge_index: th.Tensor,
    edge_attr: th.Tensor,
    num_iterations: int = 5,
    vocab: Dict[Hashable, int] = None,
    frozen: bool = False,
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Runs WL colour refinement for `num_iterations` steps and returns the
    final per-node colours plus a histogram over the (shared) colour vocab.

    The raw type ids from `initial_colours` are first mapped into the same
    shared `vocab` id space used by `refine` (under a signature tagged
    `"init"`, so they cannot collide with refinement signatures, which are
    always 2-tuples of (int, tuple), nor with `OOV_SIGNATURE`, a plain
    string). This keeps every colour ever returned - regardless of
    `num_iterations`, including 0 - in one consistent id space, so the
    histogram is always indexed consistently.

    In growing mode (frozen=False, the default) `vocab` accumulates new
    signatures as they're encountered, so its size - and therefore the
    histogram's length - can grow from call to call; this is the intended
    way to build a vocab offline (see `sage/domains/utils/build_wl_vocab.py`).
    In frozen mode (frozen=True), `vocab` must already be frozen (see
    `freeze_vocab`) and is treated as strictly read-only: unseen signatures
    resolve to the reserved OOV id instead of growing `vocab`. Since a
    frozen `vocab` never changes size during (or across) frozen calls, the
    returned histogram's length - len(vocab) - is therefore FIXED across
    any number of frozen calls sharing that vocab, which is what makes it
    safe to feed as fixed-size input to a neural net layer.

    :param x: node features, shape [N, 3], see `initial_colours`
    :param edge_index: edge index, shape [2, E]
    :param edge_attr: edge attributes, shape [E, 4], see `edge_labels`
    :param num_iterations: number of refinement iterations, L (default 5)
    :param vocab: signature -> colour id, mutated in place (unless
        `frozen`) and shared across calls so that colour ids remain stable
        across graphs/calls. A fresh dict is created if None.
    :param frozen: if True, treat `vocab` as read-only (see above).
        Requires `freeze_vocab(vocab)` to have been called first - raises
        ValueError otherwise.
    :return: (colours, histogram)
        colours: final per-node colour id, shape [N], dtype long
        histogram: count of nodes with colour id i, shape [len(vocab)]
            after this call, dtype float, zero for ids not present in this
            graph
    """
    if vocab is None:
        vocab = {}

    _check_frozen_vocab(vocab, frozen, "wl_colours")

    type_colours = initial_colours(x).tolist()
    colours = th.empty(x.shape[0], dtype=th.long)
    for v, type_id in enumerate(type_colours):
        signature = ("init", type_id)
        colours[v] = _resolve(signature, vocab, frozen)

    labels = edge_labels(edge_attr)
    for _ in range(num_iterations):
        colours = refine(colours, edge_index, labels, vocab, frozen=frozen)

    histogram = th.zeros(len(vocab), dtype=th.float)
    for c in colours.tolist():
        histogram[c] += 1

    return colours, histogram
