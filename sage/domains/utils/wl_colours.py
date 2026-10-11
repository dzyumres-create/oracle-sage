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


def initial_colours_atom(x: th.Tensor) -> th.Tensor:
    """
    Assigns each node its initial WL colour from the atom encoding's node features (see
    atoms_to_graph, sage/domains/gym_taxi/utils/representations.py): a 6-column one-hot
    over ATOM_PREDICATES (location/taxi/passenger/adjacent/in/destination).

    Unlike vILG's node features (initial_colours_vilg, disjoint object-type/predicate/
    status blocks that can have several simultaneously-active columns), every atom row -
    type atom or proposition atom alike - has EXACTLY one active column: they all share
    the same flat one-hot encoding, not disjoint feature blocks. A plain argmax is
    therefore lossless here, unlike the whole-vector argmax initial_colours_vilg's
    docstring warns against - but that losslessness is asserted explicitly below rather
    than silently assumed, since a future encoding drift in atoms_to_graph (e.g. a
    genuinely multi-hot row) would otherwise resolve to a wrong-but-plausible colour
    instead of a loud failure.

    :param x: node features, shape [N, 6], one-hot over ATOM_PREDICATES
    :return: initial colour per node, shape [N], dtype long, values in {0, ..., 5}
    """
    if not th.all(x.sum(dim=1) == 1):
        raise ValueError(
            "initial_colours_atom expects exactly one-hot rows (atoms_to_graph's "
            "node_feats); got at least one row with a column sum != 1."
        )
    return th.argmax(x, dim=1).long()


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


def edge_labels_atom(edge_attr: th.Tensor) -> th.Tensor:
    """
    Combines the atom encoding's 4-bit multi-hot edge label (ATOM_LABELS = [(1,1),(1,2),
    (2,1),(2,2)], see atoms_to_graph, sage/domains/gym_taxi/utils/representations.py)
    into a single categorical label, as an integer BITMASK - NOT argmax, unlike
    edge_labels/edge_labels_vilg. Those two are single-bit one-hots by construction, so
    argmax is lossless for them; an atom-atom edge can instead carry MULTIPLE
    simultaneous labels at once (e.g. two arity-2 atoms sharing both their position-1 AND
    position-2 arguments sets both the (1,1) and (2,2) bits - see atoms_to_graph's
    np.maximum.at merge step, which explicitly preserves co-occurring labels rather than
    picking one). Taking argmax here would silently collapse a genuinely multi-label edge
    onto just its lowest-set bit, discarding real structure Horcik et al.'s Def. 2
    requires WL to see.

    bit_i * 2^i over the 4 columns instead maps each of the 16 possible on/off
    combinations to a distinct integer in {1, ..., 15} (0 is impossible - see the
    non-zero assertion below) - a plain, hashable, comparable colour like any other, with
    no further meaning attached to its numeric value (it is never decoded back to bits).

    :param edge_attr: edge attributes, shape [E, 4], 0/1 multi-hot over ATOM_LABELS
    :return: label per edge, shape [E], dtype long, values in {1, ..., 15}
    """
    if not th.all((edge_attr == 0) | (edge_attr == 1)):
        raise ValueError(
            "edge_labels_atom expects a 0/1 multi-hot edge_attr (atoms_to_graph's "
            "edge_feats); got a value that is neither 0 nor 1."
        )
    bits = th.as_tensor([1, 2, 4, 8], dtype=edge_attr.dtype, device=edge_attr.device)
    labels = (edge_attr * bits).sum(dim=1).long()
    if not th.all(labels > 0):
        raise ValueError(
            "edge_labels_atom expects every edge to have at least one active label bit "
            "(atoms_to_graph never emits an all-zero edge_feats row - every edge exists "
            "because two atoms share an argument at some (i, j) position); got an edge "
            "with an all-zero label."
        )
    return labels


# --- ternary-predicate domain (domain="ternary") -----------------------------------------
#
# Decoders for the graphs built by sage/domains/gym_taxi/utils/ternary_representations.py
# (facts_to_object_graph / facts_to_vilg_graph / facts_to_atom_graph). Their column layouts
# are documented in wl_colours' docstring. Unlike the standard decoders above, every one of
# these validates its input (0/1 or +-1 values, one-hot blocks) and raises on anything else,
# so a malformed or foreign graph can never be decoded into plausible-looking colours.

def _require_binary(tensor: th.Tensor, what: str) -> None:
    if not th.all((tensor == 0) | (tensor == 1)):
        raise ValueError(f"{what}: expected 0/1 values; got a value that is neither 0 nor 1.")


def _require_one_hot_rows(tensor: th.Tensor, what: str) -> None:
    _require_binary(tensor, what)
    if not th.all(tensor.sum(dim=1) == 1):
        raise ValueError(f"{what}: expected exactly one active column per row.")


def initial_colours_ternary_object(x: th.Tensor) -> th.Tensor:
    """
    Ternary object encoding: x is the 3-column object-type one-hot [location, taxi,
    passenger]. Colour = the active column (0=location, 1=taxi, 2=passenger).

    :param x: node features, shape [N, 3]
    :return: initial colour per node, shape [N], dtype long, values in {0, 1, 2}
    """
    _require_one_hot_rows(x, "initial_colours_ternary_object (x [N, 3])")
    return th.argmax(x, dim=1).long()


def edge_labels_ternary_object(edge_attr: th.Tensor) -> th.Tensor:
    """
    Ternary object encoding: the FULL 10-column edge vector as one integer key. Columns
    0-8 (road, at, picked_up_at, req12, req21, req13, req31, req23, req32) are 0/1 bits,
    several of which can be set on one directed pair (labels merged by the converter);
    column 9 is the direction, +1 or -1. Key = sum(bit_k * 2^k, k = 0..8) + 512 if the
    direction is -1, so every distinct vector gets a distinct key (nothing is argmax'ed
    away).

    :param edge_attr: edge attributes, shape [E, 10]
    :return: label per edge, shape [E], dtype long, values in {1, ..., 1023}
    """
    bits = edge_attr[:, 0:9]
    direction = edge_attr[:, 9]
    _require_binary(bits, "edge_labels_ternary_object (edge_attr[:, 0:9])")
    if not th.all((direction == 1) | (direction == -1)):
        raise ValueError("edge_labels_ternary_object: the direction column (edge_attr[:, 9]) must be +1 or -1.")
    if not th.all(bits.sum(dim=1) > 0):
        raise ValueError(
            "edge_labels_ternary_object: every object edge exists because some fact labels it; "
            "got an edge with no label bit set."
        )
    weights = th.as_tensor([1 << k for k in range(9)], dtype=edge_attr.dtype, device=edge_attr.device)
    return (bits * weights).sum(dim=1).long() + (direction < 0).long() * 512


def initial_colours_ternary_vilg(x: th.Tensor) -> th.Tensor:
    """
    Ternary object-atom (vILG) encoding: x has 10 columns -- 0-2 object type [location,
    taxi, passenger], 3-6 predicate [adjacent, in, picked_up_at, request], 7-9 goal status
    [achieved_goal, unachieved_goal, achieved_nongoal]. An object row is a one-hot in 0-2
    with 3-9 zero; a proposition row has 0-2 zero and exactly one active column in each of
    3-6 and 7-9. Colour: object -> its type id (0-2); proposition ->
    3 + predicate_id * 3 + status_id (3-14), the joint (predicate, status) id, as in
    initial_colours_vilg.

    :param x: node features, shape [N, 10]
    :return: initial colour per node, shape [N], dtype long, values in {0, ..., 14}
    """
    _require_binary(x, "initial_colours_ternary_vilg (x [N, 10])")
    obj, pred, status = x[:, 0:3], x[:, 3:7], x[:, 7:10]
    obj_n, pred_n, status_n = obj.sum(dim=1), pred.sum(dim=1), status.sum(dim=1)
    is_object = (obj_n == 1) & (pred_n == 0) & (status_n == 0)
    is_proposition = (obj_n == 0) & (pred_n == 1) & (status_n == 1)
    if not th.all(is_object | is_proposition):
        raise ValueError(
            "initial_colours_ternary_vilg: every row must be an object row (one-hot in columns "
            "0-2, zeros elsewhere) or a proposition row (zeros in 0-2, one-hot in 3-6 and in 7-9)."
        )
    proposition_colour = 3 + pred.argmax(dim=1) * 3 + status.argmax(dim=1)
    return th.where(is_object, obj.argmax(dim=1), proposition_colour).long()


def edge_labels_ternary_vilg(edge_attr: th.Tensor) -> th.Tensor:
    """
    Ternary object-atom (vILG) encoding: edge_attr is the argument-position one-hot
    [position 1, position 2, position 3] (both directions of an object<->proposition edge
    carry the same position). Label = position - 1.

    :param edge_attr: edge attributes, shape [E, 3]
    :return: label per edge, shape [E], dtype long, values in {0, 1, 2}
    """
    _require_one_hot_rows(edge_attr, "edge_labels_ternary_vilg (edge_attr [E, 3])")
    return th.argmax(edge_attr, dim=1).long()


def initial_colours_ternary_atom(x: th.Tensor) -> th.Tensor:
    """
    Ternary atom encoding: x is the 7-column predicate one-hot [location, taxi, passenger,
    adjacent, in, request, picked_up_at] (ATOM_PREDICATES_TERNARY -- note request precedes
    picked_up_at here, unlike the vilg predicate block). Colour = the active column.

    :param x: node features, shape [N, 7]
    :return: initial colour per node, shape [N], dtype long, values in {0, ..., 6}
    """
    _require_one_hot_rows(x, "initial_colours_ternary_atom (x [N, 7])")
    return th.argmax(x, dim=1).long()


def edge_labels_ternary_atom(edge_attr: th.Tensor) -> th.Tensor:
    """
    Ternary atom encoding: edge_attr is a 9-bit multi-hot over the argument-position pairs
    (i, j), i, j in 1..3, column (i-1)*3 + (j-1): a -> b has bit (i, j) when argument i of
    atom a equals argument j of atom b. Label = the integer bitmask sum(bit_k * 2^k), so
    co-occurring pairs are all kept (as in edge_labels_atom, generalised from 4 bits).

    :param edge_attr: edge attributes, shape [E, 9]
    :return: label per edge, shape [E], dtype long, values in {1, ..., 511}
    """
    _require_binary(edge_attr, "edge_labels_ternary_atom (edge_attr [E, 9])")
    if not th.all(edge_attr.sum(dim=1) > 0):
        raise ValueError(
            "edge_labels_ternary_atom: every atom edge exists because two atoms share an "
            "argument; got an edge with no label bit set."
        )
    weights = th.as_tensor([1 << k for k in range(9)], dtype=edge_attr.dtype, device=edge_attr.device)
    return (edge_attr * weights).sum(dim=1).long()


DOMAINS = ("standard", "ternary")

# Decoder pair per (domain, graph_convention), and the (node, edge) feature widths each one
# accepts. The widths are those of GraphTaxiEnv's GRAPH_CONVENTIONS (standard) and
# TERNARY_GRAPH_CONVENTIONS (ternary); every pair of entries differs in at least one width,
# which is what lets wl_colours reject a graph from the other domain.
DECODERS = {
    "standard": {
        "oracle_sage": (initial_colours, edge_labels),
        "vilg": (initial_colours_vilg, edge_labels_vilg),
        "atom": (initial_colours_atom, edge_labels_atom),
    },
    "ternary": {
        "oracle_sage": (initial_colours_ternary_object, edge_labels_ternary_object),
        "vilg": (initial_colours_ternary_vilg, edge_labels_ternary_vilg),
        "atom": (initial_colours_ternary_atom, edge_labels_ternary_atom),
    },
}
FEATURE_WIDTHS = {
    "standard": {"oracle_sage": (3, 4), "vilg": (9, 2), "atom": (6, 4)},
    "ternary": {"oracle_sage": (3, 10), "vilg": (10, 3), "atom": (7, 9)},
}


def _check_widths(x: th.Tensor, edge_attr: th.Tensor, graph_convention: str, domain: str) -> None:
    node_width, edge_width = FEATURE_WIDTHS[domain][graph_convention]
    if x.dim() != 2 or x.shape[1] != node_width or edge_attr.dim() != 2 or edge_attr.shape[1] != edge_width:
        raise ValueError(
            f"wl_colours(graph_convention={graph_convention!r}, domain={domain!r}) expects x of shape "
            f"[N, {node_width}] and edge_attr of shape [E, {edge_width}]; got x {tuple(x.shape)} and "
            f"edge_attr {tuple(edge_attr.shape)}. A graph from the other domain (or convention) is "
            f"never decoded with this decoder pair."
        )


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

    new_colours = _refine_list(
        node_colours.tolist(), edge_index[0].tolist(), edge_index[1].tolist(), edge_labels.tolist(), vocab, frozen
    )
    return th.tensor(new_colours, dtype=th.long, device=node_colours.device)


def _refine_list(node_colours_list, src, dst, edge_labels_list, vocab, frozen):
    """
    `refine` on plain Python lists: returns the new colour ids as a list. Signatures are
    resolved in node order (v = 0..N-1), so in growing mode new ids are assigned in exactly
    the order `refine` has always assigned them. Kept list-based (one tensor built by the
    caller at the end, instead of one tensor write per node) purely for speed - per-node
    tensor writes dominated wl_colours' run time.
    """
    neighbours = [[] for _ in range(len(node_colours_list))]
    for e, s in enumerate(src):
        neighbours[s].append((node_colours_list[dst[e]], edge_labels_list[e]))

    return [
        _resolve((own, tuple(sorted(nbrs))), vocab, frozen)
        for own, nbrs in zip(node_colours_list, neighbours)
    ]


def wl_colours(
    x: th.Tensor,
    edge_index: th.Tensor,
    edge_attr: th.Tensor,
    num_iterations: int = 5,
    vocab: Dict[Hashable, int] = None,
    frozen: bool = False,
    graph_convention: str = "oracle_sage",
    domain: str = "standard",
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Runs WL colour refinement for `num_iterations` steps and returns the
    final per-node colours plus a histogram over the (shared) colour vocab.

    `graph_convention` selects which (initial_colours, edge_labels) decoder
    pair to use - "oracle_sage" (default, unchanged) uses `initial_colours`/
    `edge_labels`; "vilg" uses `initial_colours_vilg`/`edge_labels_vilg`;
    "atom" uses `initial_colours_atom`/`edge_labels_atom` - since each
    convention's node/edge feature layout (see env_to_vilg_graph /
    atoms_to_graph, sage/domains/gym_taxi/utils/representations.py) is
    shaped differently. This is threaded through explicitly by the caller
    (which already knows its own graph_convention - see
    sage/domains/gym_taxi/utils/representations.py and sage/domains/
    gym_taxi/simulator/planner.py) rather than auto-detected from tensor
    width, since auto-detection by shape is fragile and could silently
    misfire if the two conventions' dimensions ever coincide.

    `domain` selects the domain the graph comes from, explicitly, like
    graph_convention: "standard" (default: the Taxi domain above, unchanged)
    or "ternary" (the ternary-predicate domain, sage/domains/gym_taxi/utils/
    ternary_representations.py), whose three conventions use the decoder pairs
    initial_colours_ternary_*/edge_labels_ternary_* below. In BOTH domains the
    feature widths are checked first (FEATURE_WIDTHS) and anything else raises
    ValueError, so a ternary graph can never be decoded with a standard decoder
    or vice versa: [N, 3]/[E, 4], [N, 9]/[E, 2] and [N, 6]/[E, 4] are the
    standard oracle_sage/vilg/atom widths; the ternary ones follow.

    Ternary column layouts, read from the converters in
    ternary_representations.py (facts_to_object_graph, facts_to_vilg_graph,
    facts_to_atom_graph). Every relational fact contributes to every
    convention; rows are objects first (row k = object k: taxi 0, locations,
    passengers), then, for vilg one row per relational fact and for atom one
    row per fact, in facts() order.

      object ("oracle_sage"), x [N, 3], edge_attr [E, 10]
        x:  0 location | 1 taxi | 2 passenger                         (one-hot)
        edge_attr, one row per directed object pair, labels merged (0/1):
            0 road        adjacent(a,b): a->b and b->a
            1 at          in(x,y): x->y, and y->x with dir -1
            2 picked_up_at  picked_up_at(p,l): p->l, and l->p with dir -1
            3 req12  4 req21  5 req13  6 req31  7 req23  8 req32
                          request(p,o,d): req_ij on the edge from argument i
                          to argument j (p=1, o=2, d=3), both directions
            9 dir         +1, or -1 if the pair is the reverse of an in or
                          picked_up_at fact (wins over any +1 label merged
                          onto the same pair)
      object-atom ("vilg"), x [N, 10], edge_attr [E, 3]
        x:  object rows      0 location | 1 taxi | 2 passenger, 3-9 zero
            proposition rows 0-2 zero,
                             3 adjacent | 4 in | 5 picked_up_at | 6 request,
                             7 achieved_goal | 8 unachieved_goal |
                             9 achieved_nongoal (always 9: the ternary domain
                             has no goal predicate)
        edge_attr: 0 position 1 | 1 position 2 | 2 position 3 (one-hot); an
            edge joins a proposition and its k-th argument, in BOTH directions,
            with the same position-k label
      atom ("atom"), x [N, 7], edge_attr [E, 9]
        x:  0 location | 1 taxi | 2 passenger | 3 adjacent | 4 in | 5 request |
            6 picked_up_at                                            (one-hot)
        edge_attr: column (i-1)*3 + (j-1) for the argument-position pair (i, j),
            i, j in 1..3, i.e. 0 (1,1) | 1 (1,2) | 2 (1,3) | 3 (2,1) | 4 (2,2) |
            5 (2,3) | 6 (3,1) | 7 (3,2) | 8 (3,3); a->b (a != b) has bit (i, j)
            when argument i of atom a equals argument j of atom b (multi-hot)

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
        `initial_colours_vilg`), shape [N, 6] for "atom" (see
        `initial_colours_atom`)
    :param edge_index: edge index, shape [2, E]
    :param edge_attr: edge attributes - shape [E, 4] for "oracle_sage" (see
        `edge_labels`), shape [E, 2] for "vilg" (see `edge_labels_vilg`),
        shape [E, 4] for "atom" (see `edge_labels_atom` - same width as
        oracle_sage's edge_attr, but a 4-bit multi-hot, not
        [is_road,is_tether,is_destination,direction])
    :param num_iterations: number of refinement iterations, L (default 5)
    :param vocab: signature -> colour id, mutated in place (unless
        `frozen`) and shared across calls so that colour ids remain stable
        across graphs/calls. A fresh dict is created if None.
    :param frozen: if True, treat `vocab` as read-only (see above).
        Requires `freeze_vocab(vocab)` to have been called first - raises
        ValueError otherwise.
    :param graph_convention: "oracle_sage" (default), "vilg", or "atom" -
        selects the decoder pair, see above.
    :param domain: "standard" (default) or "ternary" - selects the domain's
        decoder pairs and feature widths, see above.
    :return: (colours, histogram)
        colours: final per-node colour id, shape [N], dtype long
        histogram: count of nodes with colour id i, shape [len(vocab)]
            after this call, dtype float, zero for ids not present in this
            graph
    """
    if graph_convention not in ("oracle_sage", "vilg", "atom"):
        raise ValueError(
            f'wl_colours() got graph_convention={graph_convention!r}; expected '
            f'"oracle_sage", "vilg", or "atom".'
        )
    if domain not in DOMAINS:
        raise ValueError(f'wl_colours() got domain={domain!r}; expected "standard" or "ternary".')

    if vocab is None:
        vocab = {}

    _check_frozen_vocab(vocab, frozen, "wl_colours")
    _check_widths(x, edge_attr, graph_convention, domain)
    initial_colours_fn, edge_labels_fn = DECODERS[domain][graph_convention]

    type_colours = initial_colours_fn(x).tolist()
    # Colours are kept as a Python list across iterations and turned into a tensor once at
    # the end (see _refine_list); resolution order, and so every id, is unchanged.
    colours = [_resolve(("init", type_id), vocab, frozen) for type_id in type_colours]

    labels = edge_labels_fn(edge_attr)
    src, dst, labels = edge_index[0].tolist(), edge_index[1].tolist(), labels.tolist()
    for _ in range(num_iterations):
        colours = _refine_list(colours, src, dst, labels, vocab, frozen)
    colours = th.tensor(colours, dtype=th.long, device=x.device)

    # bincount, not a Python loop: colours is always th.long with every value
    # in [0, len(vocab)) (see _resolve - its only two return paths are
    # vocab[signature] or vocab[OOV_SIGNATURE], both valid ids), so
    # minlength=len(vocab) is exact, never padding-short. .float() matches
    # the histogram dtype every downstream consumer (concatenation, policy
    # net input) already expects.
    histogram = th.bincount(colours, minlength=len(vocab)).float()

    return colours, histogram
