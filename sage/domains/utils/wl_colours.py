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


def initial_colours_vilg(x: th.Tensor) -> th.Tensor:
    """
    Assigns each node its initial WL colour from vILG's 9-column node
    features (see env_to_vilg_graph, sage/domains/gym_taxi/utils/
    representations.py): columns 0-2 are an object-type one-hot (location/
    taxi/passenger), columns 3-5 are a predicate one-hot (adjacent/in/
    destination), columns 6-8 are a goal-status one-hot (achieved_goal/
    unachieved_goal/achieved_nongoal). Object nodes zero out columns 3-8;
    proposition nodes zero out columns 0-2 and always have exactly one "1"
    in each of the predicate and status blocks - i.e. a proposition node's
    row legitimately has TWO active columns at once, unlike `initial_colours`'
    plain one-hot.

    A blind `argmax(x, dim=1)` over the full 9-column vector - the
    behaviour this function replaces - always resolves to whichever active
    column has the lower index, which is always the predicate block (3-5)
    over the status block (6-8). That silently discards goal status:
    destination(p,l) achieved and destination(p,l) unachieved would collapse
    to the identical initial colour, and stay indistinguishable through
    every later `refine()` iteration, since `refine()` can only combine
    colours that already exist and cannot recover information this function
    already discarded.

    Instead, object and proposition nodes are decoded separately:
      - object nodes: colour = argmax(x[:, 0:3])  (0=location, 1=taxi,
        2=passenger) - same 0/1/2 id space `initial_colours` uses.
      - proposition nodes: colour = 3 + pred_id * 3 + status_id, where
        pred_id = argmax(x[:, 3:6]) and status_id = argmax(x[:, 6:9]) - a
        joint (predicate, status) id, offset past the 3 object-type ids so
        the two blocks never collide (GOOSE's Def 3.1 Fcat: one colour per
        (predicate, argument-status) combination).

    Worked example (see env_to_vilg_graph for the exact row construction):
        adjacent, achieved_nongoal:       pred_id=0, status_id=2 -> colour = 3+0*3+2 = 5
        destination(p,l), achieved:       pred_id=2, status_id=0 -> colour = 3+2*3+0 = 9
        destination(p,l), unachieved:     pred_id=2, status_id=1 -> colour = 3+2*3+1 = 10
    So an achieved and an unachieved destination proposition - identical in
    every other respect - now get distinct colours (9 vs 10), where the old
    whole-vector argmax gave both colour 3 (the predicate block's
    "destination" index) regardless of status.

    :param x: node features, shape [N, 9], see column layout above
    :return: initial colour per node, shape [N], dtype long. Values are in
        {0, ..., 11} (3 object-type ids, 9 = 3 predicates x 3 statuses
        proposition ids) - a small, local id space, distinct from (and not
        looked up in) `vocab`; see `wl_colours` for how it is folded into
        the shared vocab id space.
    """
    # sum(dim=1) > 0, not .any(dim=1): x is float (see wl_colours()'s
    # th.as_tensor(..., dtype=th.float) callers), and PyTorch <1.8's
    # .any(dim=...) only accepts uint8/bool input, raising RuntimeError on a
    # float tensor - this sandbox's torch silently accepts it, but RCP's
    # 1.7.1+cu110 does not. Do not "simplify" this back to .any(dim=1).
    is_object = x[:, 0:3].sum(dim=1) > 0
    obj_type = x[:, 0:3].argmax(dim=1)
    pred_id = x[:, 3:6].argmax(dim=1)
    status_id = x[:, 6:9].argmax(dim=1)
    prop_colour = 3 + pred_id * 3 + status_id
    return th.where(is_object, obj_type, prop_colour).long()


def edge_labels_vilg(edge_attr: th.Tensor) -> th.Tensor:
    """
    Combines vILG's edge argument-position into a categorical edge label.

    edge_attr columns are [position_1, position_2], a one-hot with no
    direction bit: unlike Oracle-SAGE's edges (always materialised as a
    forward/backward pair over the same object-object connection), vILG
    edges (see env_to_vilg_graph) only ever run proposition -> object, in
    one direction, so there is no reverse duplicate to disambiguate with a
    sign bit - the position one-hot alone is already unambiguous.

    :param edge_attr: edge attributes, shape [E, 2]
    :return: label per edge, shape [E], dtype long, values in {0, 1}
        (0 = position 1 / the proposition's "subject" argument, 1 =
        position 2 / the proposition's "value" argument)
    """
    return th.argmax(edge_attr, dim=1).long()


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

    new_colours = th.empty(n, dtype=th.long, device=node_colours.device)
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
    graph_convention: str = "oracle_sage",
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Runs WL colour refinement for `num_iterations` steps and returns the
    final per-node colours plus a histogram over the (shared) colour vocab.

    `graph_convention` selects which (initial_colours, edge_labels) decoder
    pair to use - "oracle_sage" (default, unchanged) uses `initial_colours`/
    `edge_labels`; "vilg" uses `initial_colours_vilg`/`edge_labels_vilg`
    instead, since vILG's node/edge feature layout (see env_to_vilg_graph,
    sage/domains/gym_taxi/utils/representations.py) is shaped differently
    from Oracle-SAGE's. This is threaded through explicitly by the caller
    (which already knows its own graph_convention - see
    sage/domains/gym_taxi/utils/representations.py and sage/domains/
    gym_taxi/simulator/planner.py) rather than auto-detected from tensor
    width, since auto-detection by shape is fragile and could silently
    misfire if the two conventions' dimensions ever coincide.

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

    :param x: node features - shape [N, 3] for "oracle_sage" (see
        `initial_colours`), shape [N, 9] for "vilg" (see
        `initial_colours_vilg`)
    :param edge_index: edge index, shape [2, E]
    :param edge_attr: edge attributes - shape [E, 4] for "oracle_sage" (see
        `edge_labels`), shape [E, 2] for "vilg" (see `edge_labels_vilg`)
    :param num_iterations: number of refinement iterations, L (default 5)
    :param vocab: signature -> colour id, mutated in place (unless
        `frozen`) and shared across calls so that colour ids remain stable
        across graphs/calls. A fresh dict is created if None.
    :param frozen: if True, treat `vocab` as read-only (see above).
        Requires `freeze_vocab(vocab)` to have been called first - raises
        ValueError otherwise.
    :param graph_convention: "oracle_sage" (default) or "vilg" - selects
        the decoder pair, see above.
    :return: (colours, histogram)
        colours: final per-node colour id, shape [N], dtype long
        histogram: count of nodes with colour id i, shape [len(vocab)]
            after this call, dtype float, zero for ids not present in this
            graph
    """
    if graph_convention not in ("oracle_sage", "vilg"):
        raise ValueError(
            f'wl_colours() got graph_convention={graph_convention!r}; expected '
            f'"oracle_sage" or "vilg".'
        )

    if vocab is None:
        vocab = {}

    _check_frozen_vocab(vocab, frozen, "wl_colours")

    if graph_convention == "vilg":
        type_colours = initial_colours_vilg(x).tolist()
    else:
        type_colours = initial_colours(x).tolist()
    colours = th.empty(x.shape[0], dtype=th.long, device=x.device)
    for v, type_id in enumerate(type_colours):
        signature = ("init", type_id)
        colours[v] = _resolve(signature, vocab, frozen)

    if graph_convention == "vilg":
        labels = edge_labels_vilg(edge_attr)
    else:
        labels = edge_labels(edge_attr)
    for _ in range(num_iterations):
        colours = refine(colours, edge_index, labels, vocab, frozen=frozen)

    # bincount, not a Python loop: colours is always th.long with every value
    # in [0, len(vocab)) (see _resolve - its only two return paths are
    # vocab[signature] or vocab[OOV_SIGNATURE], both valid ids), so
    # minlength=len(vocab) is exact, never padding-short. .float() matches
    # the histogram dtype every downstream consumer (concatenation, policy
    # net input) already expects.
    histogram = th.bincount(colours, minlength=len(vocab)).float()

    return colours, histogram
