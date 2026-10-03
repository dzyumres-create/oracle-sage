"""
.. module:: ternary_representations
   :synopsis: Graph-convention converters for the ternary-predicate Taxi domain
   (TernaryTaxiWorldSimulator) -- object (oracle_sage), vILG (object-atom) and atom
   (Horcik et al., AAAI-25, Def. 2) encodings, all built from facts(), never from a
   networkx graph (facts() is TernaryTaxiWorldSimulator's own single query interface;
   see ternary_taxi_world.py's own docstring for its ordering/predicate contract).

   A NEW module, not a modification of either representations.py: neither
   sage/domains/gym_taxi/utils/representations.py nor sage/domains/utils/
   representations.py is touched, so the old (non-ternary) domain's converters stay
   byte-identical and this commit cherry-picks cleanly onto any base branch that
   already has representations.py (cell5-atom-gnn, cell6-wl-atom, ...).

   Every facts_to_*_graph is a PURE function of (facts, meta), where
   meta = {"time": int, "timeout": int, "planning": bool} -- no env object, no
   simulator instance. This is deliberate: the (future) planner needs to re-encode
   PROJECTED states (a fact list plus a hypothetical time), which never have a live
   env to read from, and these same three functions are what it will call.

   Output contract: every facts_to_*_graph returns the same 5-tuple shape its
   old-domain, single-object-graph counterpart returns on cell5-atom-gnn --
   (node_feats, edge_feats, edge_index, mask, global_feats) -- as plain numpy, with
   the SAME dtypes those old builders use: node_feats/edge_feats/global_feats
   float64, edge_index int64, mask bool. This is NOT what a torch consumer wants
   directly -- json_to_atom_graph (representations.py:482-520) already establishes
   the pattern the policy-side wrapper below follows: explicit
   `th.as_tensor(..., dtype=th.float32)` / `dtype=th.long` casts, exactly because
   these numpy arrays are float64/int64, not float32/int64 -- unlike json_to_graph's
   plain-Python-list JSON round trip, which gets float32 "for free" from torch's
   default dtype on python float lists. There is no such free ride here:
   json_to_ternary_graph_object/_vilg/_atom (bottom of this module) cast explicitly,
   the same way json_to_atom_graph already does.

   Every converter validates its input via _validate_facts before doing anything else
   (see its docstring): malformed input raises immediately, it is never silently
   dropped or reinterpreted. Likewise, any relational fact whose predicate/arity this
   module does not have an explicit rule for raises ValueError -- e.g. an unrecognised
   predicate, or (in the atom builder) an argument position beyond ATOM_MAX_ARITY.
"""
import json

import numpy as np
import scipy.sparse as sp
import torch as th
from torch_geometric.data import Data, Batch

from sage.domains.utils.representations import EMB_SIZE, json_default


# --- facts() contract (mirrors ternary_taxi_world.py's own FACT_PREDICATES/RELATION_PREDICATES) ---

TYPE_PREDICATES = ("taxi", "location", "passenger")
RELATIONAL_PREDICATES = ("adjacent", "in", "request", "picked_up_at")
VALID_PREDICATES = TYPE_PREDICATES + RELATIONAL_PREDICATES

PREDICATE_ARITY = {
    "taxi": 1,
    "location": 1,
    "passenger": 1,
    "adjacent": 2,
    "in": 2,
    "picked_up_at": 2,
    "request": 3,
}

# Canonical predicate index order for the compact wire format (facts_to_json) -- any
# fixed order works since it's only used for index<->name round-tripping, but this one
# groups type atoms first (matching facts()'s own emission order) for readability.
TERNARY_PREDICATES = TYPE_PREDICATES + RELATIONAL_PREDICATES


def _validate_facts(facts):
    """
    Validates facts()'s own contract before any converter touches it:
      1. every predicate is one of the 7 valid ones (VALID_PREDICATES);
      2. every fact has exactly the arity PREDICATE_ARITY says its predicate has;
      3. type atoms (taxi/location/passenger) form a PREFIX of `facts` -- none may
         appear after the first relational fact;
      4. that prefix's object ids are EXACTLY 0..n_obj-1, IN THAT ORDER -- this is
         the "row k is object k" contract every converter below relies on;
      5. every relational fact's arguments are all existing object ids, i.e. in
         [0, n_obj).

    Raises ValueError, never returns a partial/best-effort result, on any violation
    -- a shuffled or gapped fact list must never be silently accepted.

    :param facts: list of (predicate, args_tuple), as TernaryTaxiWorldSimulator.facts()
        returns
    :return: n_obj, the number of type atoms (== the object-id space [0, n_obj))
    """
    n = len(facts)
    i = 0
    while i < n and facts[i][0] in TYPE_PREDICATES:
        i += 1
    n_obj = i

    type_ids = []
    for entry in facts[:n_obj]:
        predicate, args = entry
        if len(args) != PREDICATE_ARITY[predicate]:
            raise ValueError(
                f"type atom {entry!r} has arity {len(args)}, expected {PREDICATE_ARITY[predicate]}"
            )
        type_ids.append(args[0])
    if type_ids != list(range(n_obj)):
        raise ValueError(
            f"type atoms must be object ids 0..{n_obj - 1} in that exact order; got {type_ids}"
        )

    for entry in facts[n_obj:]:
        predicate, args = entry
        if predicate not in VALID_PREDICATES:
            raise ValueError(f"unrecognised predicate {predicate!r}; expected one of {VALID_PREDICATES}")
        if predicate in TYPE_PREDICATES:
            raise ValueError(
                f"type atom {entry!r} found after the type-atom prefix (position "
                f"{n_obj + facts[n_obj:].index(entry)}) -- type atoms must all come first"
            )
        if len(args) != PREDICATE_ARITY[predicate]:
            raise ValueError(
                f"{predicate} fact {entry!r} has arity {len(args)}, expected {PREDICATE_ARITY[predicate]}"
            )
        for arg in args:
            if not (isinstance(arg, int) and 0 <= arg < n_obj):
                raise ValueError(f"fact {entry!r} references object id {arg!r}, not in [0, {n_obj})")

    return n_obj


def _global_feats(meta):
    global_feats = np.zeros(EMB_SIZE, dtype=np.float64)
    global_feats[0] = (meta["timeout"] - meta["time"]) / meta["timeout"]
    return global_feats


def _taxi_location_from_facts(facts):
    for predicate, args in facts:
        if predicate == "in" and args[0] == 0:
            return args[1]
    raise ValueError("facts has no in(0, l) fact for the taxi (taxi id is always 0)")


def _planning_false_mask(facts, n_obj):
    """
    The planning=False (masked-action) restriction object/vilg both had on the old
    domain (env_to_graph:61-67, env_to_vilg_graph:156-161): the taxi's own location,
    every road-adjacent location, and anything else currently `in` that location
    (which always includes the taxi's own node itself, via `in(0, taxi_location)`) --
    "the taxi's current position and any adjacent positions, any passengers in the
    taxi's location, and the taxi" (env_to_graph:58-60's own comment). Derived purely
    from facts() -- no graph object needed, since `adjacent`/`in` ARE the graph.

    :return: bool array, shape [n_obj]
    """
    taxi_location = _taxi_location_from_facts(facts)
    mask = np.zeros(n_obj, dtype=bool)
    mask[taxi_location] = True
    for predicate, args in facts:
        if predicate == "adjacent" and args[0] == taxi_location:
            mask[args[1]] = True
        elif predicate == "in" and args[1] == taxi_location:
            mask[args[0]] = True
    return mask


# ==========================================================================================
# Object encoding (oracle_sage) -- decision 4
# ==========================================================================================

_OBJECT_TYPE_INDEX = {"location": 0, "taxi": 1, "passenger": 2}  # matches _OBJECT_ATTR_TO_PREDICATE's inverse

# 10-dim edge feature layout. road/at/picked_up_at/req12/req21/req13/req31/req23/req32 are
# 0/1 multi-hot bits (merged via elementwise-max when several facts label the same directed
# pair); dir is +1/-1 -- see _add_object_edge_label's docstring for the merge/priority rule.
OBJECT_EDGE_LABELS = ("road", "at", "picked_up_at", "req12", "req21", "req13", "req31", "req23", "req32", "dir")
_ROAD, _AT, _PICKED_UP_AT, _REQ12, _REQ21, _REQ13, _REQ31, _REQ23, _REQ32, _DIR = range(len(OBJECT_EDGE_LABELS))


def _add_object_edge_label(edges, u, v, bit, dir_value):
    """
    Accumulates one labeled directed edge (u, v) into `edges` (a plain dict keyed by
    (u, v), values a mutable length-10 list) -- NEVER writes into a networkx graph one
    edge at a time (decision 4's own constraint: a second `nx.DiGraph.add_edge(u, v,
    attr=...)` call on an existing pair REPLACES its attr rather than merging it, which
    is exactly wrong once two different predicates can legitimately share a directed
    pair -- e.g. a request(p, o, d) whose (o, d) happens to also be an adjacent(o, d)
    road, or whose (p, o) happens to also be the passenger's current in(p, o)).

    `dir` gets special handling, NOT a plain OR: every row starts at +1 (decision 4's
    default), and is pinned to -1 the first time ANY contributing fact calls this with
    dir_value=-1 (a reverse-of-in/picked_up_at edge) -- and is NEVER reset back to +1
    afterwards, regardless of what other (always dir_value=+1) labels later merge onto
    the same pair. This matters concretely: a waiting passenger's current location is
    typically also one of their own request origins, so in(p, l) and request(p, l, d)
    both touch the SAME directed pair (l, p) -- in's reverse wants dir=-1 there, while
    request's (o, p) reverse (req21) would, considered alone, want dir=+1. Decision 4's
    rule ("-1 IFF reverse of an in/picked_up_at fact") is about the EDGE, not the
    label, so -1 wins regardless of merge order -- see
    tests/test_ternary_representations.py's hand-checked (1, 5)/(3, 6) cases, which
    exercise exactly this.
    """
    row = edges.get((u, v))
    if row is None:
        row = [0] * (len(OBJECT_EDGE_LABELS) - 1) + [1]  # dir defaults to +1
        edges[(u, v)] = row
    row[bit] = 1
    if dir_value == -1:
        row[_DIR] = -1


def facts_to_object_graph(facts, meta):
    """
    Object encoding (Oracle-SAGE's own convention), generalised to the ternary
    predicates -- decision 4. Node features: 3-dim one-hot [location, taxi, passenger],
    unchanged from the old domain. Edge features: 10-dim, OBJECT_EDGE_LABELS -- every
    relational fact contributes a forward AND a reverse directed edge (arg i -> arg j
    for i < j, and its reverse), merged (never overwritten) when two facts label the
    same directed pair. adjacent and request edges always carry dir=+1 in both
    directions (there is no "forward"/"reverse" notion for a symmetric road, and
    request's own req_ij/req_ji column pair already encodes direction); only in/
    picked_up_at distinguish a forward (+1, contained -> container) direction from
    its reverse (-1) -- see _add_object_edge_label for how that combines with request
    when both land on the same pair.

    :param facts: TernaryTaxiWorldSimulator.facts() output
    :param meta: {"time", "timeout", "planning"}
    :return: node_feats [n_obj, 3] float64, edge_feats [E, 10] float64,
        edge_index [2, E] int64, mask [n_obj] bool, global_feats [EMB_SIZE] float64
    """
    n_obj = _validate_facts(facts)

    node_feats = np.zeros((n_obj, 3), dtype=np.float64)
    for i, (predicate, _args) in enumerate(facts[:n_obj]):
        node_feats[i, _OBJECT_TYPE_INDEX[predicate]] = 1

    edges = {}
    for predicate, args in facts[n_obj:]:
        if predicate == "adjacent":
            a, b = args
            _add_object_edge_label(edges, a, b, _ROAD, 1)
            _add_object_edge_label(edges, b, a, _ROAD, 1)
        elif predicate == "in":
            x, y = args
            _add_object_edge_label(edges, x, y, _AT, 1)
            _add_object_edge_label(edges, y, x, _AT, -1)
        elif predicate == "picked_up_at":
            p, l = args
            _add_object_edge_label(edges, p, l, _PICKED_UP_AT, 1)
            _add_object_edge_label(edges, l, p, _PICKED_UP_AT, -1)
        elif predicate == "request":
            p, o, d = args
            _add_object_edge_label(edges, p, o, _REQ12, 1)
            _add_object_edge_label(edges, o, p, _REQ21, 1)
            _add_object_edge_label(edges, p, d, _REQ13, 1)
            _add_object_edge_label(edges, d, p, _REQ31, 1)
            _add_object_edge_label(edges, o, d, _REQ23, 1)
            _add_object_edge_label(edges, d, o, _REQ32, 1)
        else:
            raise ValueError(f"facts_to_object_graph: unhandled relational predicate {predicate!r}")

    if edges:
        pairs = sorted(edges.keys())
        edge_index = np.array(pairs, dtype=np.int64).T
        edge_feats = np.array([edges[pair] for pair in pairs], dtype=np.float64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_feats = np.zeros((0, len(OBJECT_EDGE_LABELS)), dtype=np.float64)

    if meta["planning"]:
        mask = np.ones(n_obj, dtype=bool)
    else:
        mask = _planning_false_mask(facts, n_obj)

    return node_feats, edge_feats, edge_index, mask, _global_feats(meta)


# ==========================================================================================
# vILG / object-atom encoding -- decision 5
# ==========================================================================================

_VILG_NODE_WIDTH = 10
_VILG_EDGE_WIDTH = 3
# Keeps the old builder's first two positions (adjacent, in); destination's old single
# slot is replaced by two new predicates (picked_up_at, request) -- decision 5.
VILG_PREDICATE_ORDER = ("adjacent", "in", "picked_up_at", "request")
_VILG_PREDICATE_INDEX = {p: i for i, p in enumerate(VILG_PREDICATE_ORDER)}
# achieved_propositional_nongoal -- confirmed at representations.py:137 (env_to_vilg_graph),
# NOT guessed: every non-destination proposition already gets exactly this status today,
# and the ternary domain has no goal predicate at all (destination is gone), so EVERY
# ternary proposition gets it unconditionally.
VILG_NONGOAL_STATUS = (0, 0, 1)


def facts_to_vilg_graph(facts, meta):
    """
    vILG / object-atom encoding, generalised to the ternary predicates -- decision 5,
    option A: object nodes first (row k = object k, matching facts_to_object_graph and
    facts_to_atom_graph), then one proposition node per relational fact, in the order
    those facts appear in `facts`.

    Node features (10-dim): object rows are object-type one-hot(3) + zeros(7);
    proposition rows are zeros(3) + predicate one-hot(4, VILG_PREDICATE_ORDER) +
    VILG_NONGOAL_STATUS(3) -- every proposition, unconditionally (no goal predicate
    exists in this domain; see VILG_NONGOAL_STATUS's own note). Delivered passengers
    are simply absent from `facts` (TernaryTaxiWorldSimulator deletes them on
    delivery) -- there is no "retained but delivered" node to special-case, unlike the
    old vilg builder's destination-goal bookkeeping.

    Edges (3-dim position one-hot): object <-> proposition, MIRRORED (both directions
    carry the SAME label, unlike the old domain's one-directional vilg on this branch)
    -- for a relational fact's k-th argument (1-indexed), both
    (proposition -> object) and (object -> proposition) get position-k's one-hot bit.

    :param facts: TernaryTaxiWorldSimulator.facts() output
    :param meta: {"time", "timeout", "planning"}
    :return: node_feats [n_obj + n_rel, 10] float64, edge_feats [E, 3] float64,
        edge_index [2, E] int64, mask [n_obj + n_rel] bool, global_feats [EMB_SIZE] float64
    """
    n_obj = _validate_facts(facts)
    n_total = len(facts)

    node_feats = np.zeros((n_total, _VILG_NODE_WIDTH), dtype=np.float64)
    for i, (predicate, _args) in enumerate(facts[:n_obj]):
        node_feats[i, _OBJECT_TYPE_INDEX[predicate]] = 1

    rows, cols, positions = [], [], []
    for i, (predicate, args) in enumerate(facts[n_obj:], start=n_obj):
        if predicate not in _VILG_PREDICATE_INDEX:
            raise ValueError(f"facts_to_vilg_graph: unhandled relational predicate {predicate!r}")
        node_feats[i, 3 + _VILG_PREDICATE_INDEX[predicate]] = 1
        node_feats[i, 3 + len(VILG_PREDICATE_ORDER):] = VILG_NONGOAL_STATUS
        for position, obj in enumerate(args, start=1):
            if position > _VILG_EDGE_WIDTH:
                raise ValueError(
                    f"facts_to_vilg_graph: fact {(predicate, args)!r} has argument position "
                    f"{position} > max position {_VILG_EDGE_WIDTH}"
                )
            rows += [i, obj]
            cols += [obj, i]
            positions += [position, position]

    if rows:
        edge_index = np.array([rows, cols], dtype=np.int64)
        edge_feats = np.zeros((len(positions), _VILG_EDGE_WIDTH), dtype=np.float64)
        for e, position in enumerate(positions):
            edge_feats[e, position - 1] = 1
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_feats = np.zeros((0, _VILG_EDGE_WIDTH), dtype=np.float64)

    mask = np.zeros(n_total, dtype=bool)
    if meta["planning"]:
        mask[:n_obj] = True
    else:
        mask[:n_obj] = _planning_false_mask(facts, n_obj)

    return node_feats, edge_feats, edge_index, mask, _global_feats(meta)


# ==========================================================================================
# Atom encoding (Horcik et al., AAAI-25, Def. 2) -- decision 6
# ==========================================================================================

ATOM_PREDICATES_TERNARY = ("location", "taxi", "passenger", "adjacent", "in", "request", "picked_up_at")
ATOM_MAX_ARITY = 3


def _atoms_to_graph_generic(atoms, predicates, max_arity):
    """
    The Horcik et al. Def. 2 atom-encoding algorithm -- exactly cell5's own
    atoms_to_graph (representations.py:260-340)'s documented P_1..P_k sparse-
    incidence-matrix construction -- generalised over an explicit predicate list (for
    the node one-hot) and max_arity (the label space is every (i, j) for i, j in
    1..max_arity). facts_to_atom_graph below calls this with
    (ATOM_PREDICATES_TERNARY, 3). tests/test_ternary_representations.py's regression
    test calls it with cell5's OWN (ATOM_PREDICATES, 2) fed cell5's OWN
    env_to_atoms(env) output, and checks the result is bit-for-bit identical to
    atoms_to_graph(atoms) -- proving this generalisation is correct by sharing the
    exact same code path at max_arity=2, not by an independent reimplementation that
    could drift from cell5's.

    Raises ValueError if any atom's arity exceeds max_arity (never silently drops the
    excess argument positions).

    :param atoms: list of (predicate, args_tuple)
    :param predicates: ordered tuple of predicate names for the node one-hot
    :param max_arity: label space is every (i, j) for i, j in 1..max_arity
    :return: (node_feats [N, len(predicates)], edge_feats [E, max_arity**2], edge_index [2, E])
    """
    n = len(atoms)
    node_feats = np.zeros((n, len(predicates)), dtype=np.float64)
    for i, (predicate, _args) in enumerate(atoms):
        if predicate not in predicates:
            raise ValueError(f"_atoms_to_graph_generic: unrecognised predicate {predicate!r}")
        node_feats[i, predicates.index(predicate)] = 1

    P = {}
    for p in range(1, max_arity + 1):
        objs, ats = [], []
        for i, (_predicate, args) in enumerate(atoms):
            if len(args) > max_arity:
                raise ValueError(
                    f"_atoms_to_graph_generic: atom {atoms[i]!r} has arity {len(args)} > max_arity {max_arity}"
                )
            if len(args) >= p:
                objs.append(args[p - 1])
                ats.append(i)
        P[p] = sp.csr_matrix((np.ones(len(objs)), (objs, ats)), shape=(n, n))

    labels = [(i, j) for i in range(1, max_arity + 1) for j in range(1, max_arity + 1)]
    n_labels = len(labels)
    rows_list, cols_list, bits_list = [], [], []
    for bit, (i, j) in enumerate(labels):
        mat = (P[i].T @ P[j]).tocoo()
        keep = (mat.row != mat.col) & (mat.data > 0)
        rows, cols = mat.row[keep], mat.col[keep]
        bits = np.zeros((rows.shape[0], n_labels), dtype=np.float64)
        bits[:, bit] = 1
        rows_list.append(rows)
        cols_list.append(cols)
        bits_list.append(bits)

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    bits = np.concatenate(bits_list, axis=0)

    if rows.shape[0] == 0:
        return node_feats, np.zeros((0, n_labels), dtype=np.float64), np.zeros((2, 0), dtype=np.int64)

    pair_ids = rows.astype(np.int64) * n + cols.astype(np.int64)
    unique_pairs, inverse = np.unique(pair_ids, return_inverse=True)
    edge_feats = np.zeros((unique_pairs.shape[0], n_labels), dtype=np.float64)
    np.maximum.at(edge_feats, inverse, bits)
    edge_index = np.stack([unique_pairs // n, unique_pairs % n]).astype(np.int64)

    return node_feats, edge_feats, edge_index


def facts_to_atom_graph(facts, meta):
    """
    Atom encoding, generalised to arity 3 -- decision 6. `facts` IS the atom list
    (env_to_atoms' equivalent step is unnecessary here: TernaryTaxiWorldSimulator's
    own facts() already returns type atoms first in object-id order, exactly the
    "row k is object k" contract the old env_to_atoms has to derive by sorting a
    networkx graph). See _atoms_to_graph_generic for the algorithm itself.

    Only meta["planning"]=True is implemented, mirroring cell5's own
    env_to_atom_graph (representations.py:414-447) exactly: mask is True on type-atom
    rows only, and planning=False (masked-action space) raises NotImplementedError --
    not needed for this project (see Step 0's confirmation that --planner runs with
    env.planning=True for all three conventions in training).

    :param facts: TernaryTaxiWorldSimulator.facts() output
    :param meta: {"time", "timeout", "planning"}
    :return: node_feats [n, 7] float64, edge_feats [E, 9] float64, edge_index [2, E] int64,
        mask [n] bool, global_feats [EMB_SIZE] float64
    """
    if not meta["planning"]:
        raise NotImplementedError(
            "facts_to_atom_graph only supports meta['planning']=True, mirroring cell5's "
            "env_to_atom_graph; the planning=False (masked action space) path is not "
            "needed for this project."
        )

    n_obj = _validate_facts(facts)
    node_feats, edge_feats, edge_index = _atoms_to_graph_generic(facts, ATOM_PREDICATES_TERNARY, ATOM_MAX_ARITY)

    mask = np.zeros(node_feats.shape[0], dtype=bool)
    mask[:n_obj] = True

    return node_feats, edge_feats, edge_index, mask, _global_feats(meta)


# ==========================================================================================
# Compact JSON wire format -- Cell 5's atoms_to_json pattern, applied to facts directly
# ==========================================================================================

def facts_to_json(facts, meta):
    """
    Compact serialisation: the flat facts list (predicate index into
    TERNARY_PREDICATES + integer args) plus meta -- NOT any convention's expanded
    node_feats/edge_feats/edge_index, following atoms_to_json's own rationale
    (representations.py:450-479): the compact source-of-truth is what's cheap to
    serialise every env step, and every convention's tensors are rebuilt from it only
    when actually needed (the future policy-side wrapper -- see this module's
    docstring).

    :param facts: list of (predicate, args_tuple)
    :param meta: {"time", "timeout", "planning"} -- serialised as-is
    :return: JSON string
    """
    return json.dumps(
        {
            "facts": [[TERNARY_PREDICATES.index(predicate)] + [int(a) for a in args] for predicate, args in facts],
            "meta": meta,
        },
        default=json_default,
    )


def json_to_facts(s):
    """
    Inverse of facts_to_json.

    :param s: JSON string, as produced by facts_to_json
    :return: (facts, meta)
    """
    decoded = json.loads(s)
    facts = [(TERNARY_PREDICATES[entry[0]], tuple(entry[1:])) for entry in decoded["facts"]]
    return facts, decoded["meta"]


# (node_dimension, edge_dimension) per ternary graph-construction convention. Deliberately
# a SEPARATE dict from taxi_env.py's own GRAPH_CONVENTIONS (never re-keyed into it) --
# GraphTaxiEnv selects between the two dicts (and their matching converters/width) by its
# own `ternary` flag, so the old domain's dict/behaviour stays byte-identical.
TERNARY_GRAPH_CONVENTIONS = {
    "oracle_sage": (3, 10),
    "vilg": (10, 3),
    "atom": (7, 9),
}

# JsonGraph's string-observation width for the ternary domain -- a single constant, not
# one per convention like GRAPH_CONVENTION_JSON_WIDTH: facts_to_json's wire format is the
# SAME for every convention (the convention only matters on the decode side), so there is
# only one JSON size to bound. Measured max over 5 seeds x up to 2000 steps on
# CITY_TERNARY: ~19,580 chars (tests/test_ternary_representations.py's
# test_max_json_length_over_5_seeds_full_episodes) -- comfortably under this, same ~13x
# headroom the old domain's shared 250000 default gives oracle_sage/vilg/atom.
TERNARY_JSON_WIDTH = 250000


# ==========================================================================================
# Env-side observation hook and policy-side decoders -- the wiring commit
# ==========================================================================================

def _ternary_observation_fn(sim):
    """
    TernaryTaxiWorldSimulator.observation_fn (decision 3): serialises facts()+meta,
    identically regardless of graph_convention -- the convention only matters on the
    DECODE side (json_to_ternary_graph_object/_vilg/_atom below). A plain MODULE-LEVEL
    function, not a lambda or closure: TernaryTaxiWorldSimulator instances (and
    therefore this callback) must be picklable for AsyncVecEnv's multiprocessing (used
    whenever --planner is set), and lambdas/closures are not.

    :param sim: a TernaryTaxiWorldSimulator (this is exactly env.act()'s own
        `self.observation_fn(self)` call -- see ternary_taxi_world.py)
    :return: JSON string (facts_to_json's wire format)
    """
    meta = {"time": sim.time, "timeout": sim.timeout, "planning": sim.planning}
    return facts_to_json(sim.facts(), meta)


# ==========================================================================================
# Oracle-decoder condition: the true facts as one fixed-width int64 tensor per graph
# ==========================================================================================

# Column 0 is the predicate's index into TERNARY_PREDICATES; columns 1.. hold its
# arguments, padded with -1 up to the largest arity (request's 3).
ORACLE_FACTS_WIDTH = 1 + max(PREDICATE_ARITY.values())
ORACLE_FACTS_PAD = -1


def facts_to_oracle_facts(facts):
    """
    Encodes a facts list as an int64 array [n_facts, ORACLE_FACTS_WIDTH] -- rectangular,
    so it can ride on a torch_geometric Data as a plain tensor attribute and survive
    Batch.from_data_list / .to(device) / to_data_list (PyG concatenates it along dim 0
    and splits it back per graph; its name contains neither "index" nor "face", so PyG
    never offsets its values).

    :param facts: list of (predicate, args_tuple)
    :return: np.int64 array
    """
    out = np.full((len(facts), ORACLE_FACTS_WIDTH), ORACLE_FACTS_PAD, dtype=np.int64)
    for row, (predicate, args) in enumerate(facts):
        out[row, 0] = TERNARY_PREDICATES.index(predicate)
        out[row, 1:1 + len(args)] = args
    return out


def oracle_facts_to_facts(oracle_facts):
    """
    Inverse of facts_to_oracle_facts. Accepts a numpy array or a tensor on any device.

    :return: list of (predicate, args_tuple) with plain-int args
    """
    if isinstance(oracle_facts, th.Tensor):
        oracle_facts = oracle_facts.detach().cpu().numpy()
    facts = []
    for row in np.asarray(oracle_facts).tolist():
        args = tuple(int(a) for a in row[1:] if a != ORACLE_FACTS_PAD)
        facts.append((TERNARY_PREDICATES[int(row[0])], args))
    return facts


def _ternary_json_to_data(js, facts_to_graph_fn, attach_oracle_facts=False):
    """
    Shared body for json_to_ternary_graph_object/_vilg/_atom: decodes one convention's
    worth of compact-facts JSON into a torch_geometric Batch, following
    json_to_atom_graph's exact pattern (representations.py:482-520) -- same batching,
    same explicit float32/long casts (facts_to_*_graph returns float64/int64 numpy,
    per this module's own contract; there is no free dtype ride here the way
    json_to_graph's plain-Python-list path gets), same
    global_features.unsqueeze(0)-per-graph-then-batch shape.

    :param js: list of json objects representing a vector of environments (same
        calling convention as json_to_graph/json_to_atom_graph: each element indexable
        as `j[0]`)
    :param facts_to_graph_fn: facts_to_object_graph / facts_to_vilg_graph / facts_to_atom_graph
    :param attach_oracle_facts: also attach the TRUE facts as oracle_facts (int64
        [n_facts, ORACLE_FACTS_WIDTH]) -- the oracle-decoder condition only. Read by
        the planner alone; no feature extractor, GNN or head reads it.
    :return: Batch
    """
    data = []
    for j in js:
        facts, meta = json_to_facts(j[0])
        node_feats, edge_feats, edge_index, mask, global_feats = facts_to_graph_fn(facts, meta)
        d = Data(
            x=th.as_tensor(node_feats, dtype=th.float32),
            edge_attr=th.as_tensor(edge_feats, dtype=th.float32),
            edge_index=th.as_tensor(edge_index, dtype=th.long),
        )
        d.mask = th.as_tensor(mask, dtype=th.bool)
        d.global_features = th.as_tensor(global_feats, dtype=th.float32).unsqueeze(0)
        if attach_oracle_facts:
            d.oracle_facts = th.as_tensor(facts_to_oracle_facts(facts), dtype=th.long)
        data.append(d)
    return Batch.from_data_list(data)


def json_to_ternary_graph_object(js):
    """Policy-side decoder for graph_convention="oracle_sage" on the ternary domain --
    see _ternary_json_to_data."""
    return _ternary_json_to_data(js, facts_to_object_graph)


def json_to_ternary_graph_object_oracle(js):
    """Policy-side decoder for the oracle-decoder condition: exactly
    json_to_ternary_graph_object's Data (the GNN sees the object encoding as in Cell 1),
    plus the true facts as oracle_facts for the planner."""
    return _ternary_json_to_data(js, facts_to_object_graph, attach_oracle_facts=True)


def json_to_ternary_graph_vilg(js):
    """Policy-side decoder for graph_convention="vilg" on the ternary domain -- see
    _ternary_json_to_data."""
    return _ternary_json_to_data(js, facts_to_vilg_graph)


def json_to_ternary_graph_atom(js):
    """Policy-side decoder for graph_convention="atom" on the ternary domain -- see
    _ternary_json_to_data. facts_to_atom_graph itself raises NotImplementedError for
    meta["planning"]=False; nothing here needs to special-case that."""
    return _ternary_json_to_data(js, facts_to_atom_graph)


# JsonGraph's converter (JSON string -> Batch) per ternary graph-construction convention --
# the ternary analogue of taxi_env.py's own GRAPH_CONVENTION_CONVERTERS, kept as a
# separate dict for the same reason TERNARY_GRAPH_CONVENTIONS is.
TERNARY_GRAPH_CONVENTION_CONVERTERS = {
    "oracle_sage": json_to_ternary_graph_object,
    "vilg": json_to_ternary_graph_vilg,
    "atom": json_to_ternary_graph_atom,
}
