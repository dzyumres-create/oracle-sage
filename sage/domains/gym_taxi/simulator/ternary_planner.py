"""
.. module:: ternary_planner
   :synopsis: The ternary-domain Planner.plan implementation, for all three graph
   conventions, via ONE shared fact-level design (Cell 5's decode -> plan -> re-encode
   pattern, generalised from plan_atom's atom-list intermediate to a fact-list
   intermediate that works identically regardless of which convention produced it):

       decode(Data) -> facts            (convention-specific: decode_object/_vilg/_atom)
       plan_on_facts(facts, goal)       (SHARED -- knows nothing about graph conventions)
       facts_to_*_graph(projected_facts, meta) -> re-encode, then cast + increment_timer

   atom and vilg decode EXACTLY (decode(encode(facts)) == facts): both retain the full
   request/picked_up_at structure as explicit graph elements. Object encoding cannot:
   its edges only record which OBJECTS co-occur in a request, not which argument
   position each one filled, so request(p, o, d) triples are not uniquely recoverable
   -- decode_object returns a HYPOTHESIS fact set, chosen so that re-encoding it
   reproduces the input graph's edges EXACTLY. Where more than one hypothesis does
   this (crossed pairs), _tie_break_index picks one via a stable hash of the
   observation itself (see its own docstring) -- this is the ONLY place information
   loss enters this module; nothing else here is approximate.

   planner.py's Planner.plan dispatches into plan_ternary via a deferred import (this
   module imports increment_timer FROM planner.py, so a module-level import the other
   way would be circular -- see planner.py's own comment at the dispatch site).

   Per decision 5: this module NEVER reads anything but graph.x/edge_index/edge_attr/
   global_features (mask is written on the projection, never read on the input) --
   no facts JSON, no attribute attached to Data by any other part of the codebase.
"""
import hashlib
import itertools
from collections import defaultdict

import networkx as nx
import numpy as np
import torch as th
from torch_geometric.data import Data

from sage.domains.gym_taxi.simulator.planner import increment_timer
from sage.domains.gym_taxi.utils.ternary_representations import (
    ATOM_MAX_ARITY,
    ATOM_PREDICATES_TERNARY,
    OBJECT_EDGE_LABELS,
    RELATIONAL_PREDICATES,
    TYPE_PREDICATES,
    VILG_PREDICATE_ORDER,
    facts_to_atom_graph,
    facts_to_object_graph,
    facts_to_vilg_graph,
    _validate_facts,
)

# A cluster whose per-passenger local-matching combinations multiply past this raises,
# rather than silently doing an expensive search -- measured (see the ternary-gnn
# conversation record) at a mean of 4 and a max of 4 combinations for real clusters
# (which are themselves almost always size 1 or 2) over 800 real CITY_TERNARY states,
# so this is a generous safety margin, not a realistic ceiling.
MAX_CLUSTER_COMBINATIONS = 10000

# Fixed, arbitrary, module-level -- decision 3's tie-break must be the same across
# process restarts (reproducibility across runs), so this must never be seeded from
# wall-clock time, PID, or anything else that varies per run.
_TIE_BREAK_SEED = 20260930


def _to_numpy(value):
    if isinstance(value, th.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


# ==========================================================================================
# Decoders: Data -> facts
# ==========================================================================================

def decode_atom(graph):
    """
    Exact decoder for the ternary atom convention -- the direct generalisation of
    representations.py's graph_to_atoms from arity 2 (ATOM_LABELS, 4 bits) to arity 3
    (9 bits): for each non-type-atom row a and each argument position i in
    1..ATOM_MAX_ARITY, a's i-th argument is the object id of whichever type-atom row b
    has an edge a->b labelled (i, 1) -- a type atom's own sole argument is always at
    position 1, regardless of which position of `a` it fills. `facts` IS the atom list
    here (env_to_atoms' sorting step is unnecessary: facts() already returns type
    atoms first in object-id order).

    :param graph: Data with x [N, 7] (ATOM_PREDICATES_TERNARY one-hot), edge_index
        [2, E], edge_attr [E, 9] (3x3 position-pair multi-hot). Tensors on any device.
    :return: list of (predicate, args_tuple), in row order
    """
    x = _to_numpy(graph.x)
    edge_index = _to_numpy(graph.edge_index)
    edge_attr = _to_numpy(graph.edge_attr)
    n = x.shape[0]

    predicate_idx = np.argmax(x, axis=1)
    row_predicates = [ATOM_PREDICATES_TERNARY[i] for i in predicate_idx]
    is_type_atom = np.array([p in TYPE_PREDICATES for p in row_predicates], dtype=bool)

    labels = [(i, j) for i in range(1, ATOM_MAX_ARITY + 1) for j in range(1, ATOM_MAX_ARITY + 1)]
    pos1_bit = {i: labels.index((i, 1)) for i in range(1, ATOM_MAX_ARITY + 1)}

    src, dst = edge_index[0], edge_index[1]
    dst_is_type = is_type_atom[dst]

    args_by_position = {}
    for i in range(1, ATOM_MAX_ARITY + 1):
        bit = pos1_bit[i]
        keep = dst_is_type & (edge_attr[:, bit] > 0)
        arr = np.full(n, -1, dtype=np.int64)
        arr[src[keep]] = dst[keep]
        args_by_position[i] = arr

    facts = []
    for row in range(n):
        if is_type_atom[row]:
            facts.append((row_predicates[row], (int(row),)))
            continue
        args = []
        for i in range(1, ATOM_MAX_ARITY + 1):
            v = args_by_position[i][row]
            if v != -1:
                args.append(int(v))
        facts.append((row_predicates[row], tuple(args)))
    return facts


def decode_vilg(graph):
    """
    Exact decoder for the ternary vilg convention -- generalises
    planner.py's graph_to_networkx_vilg's position-dict approach from 2 positions to 3
    (VILG_PREDICATE_ORDER's own edge width), reading straight off the MIRRORED edges
    (both directions carry the same position label here, unlike the old one-directional
    vilg -- so either direction can be read; this reads proposition -> object edges,
    i.e. edges whose source is a proposition row).

    :param graph: Data with x [N, 10] (object one-hot(3) + predicate one-hot(4) +
        status(3)), edge_index [2, E], edge_attr [E, 3] (position one-hot). Tensors on
        any device.
    :return: list of (predicate, args_tuple), in row order (object rows first, per
        facts_to_vilg_graph's own "option A" layout)
    """
    x = _to_numpy(graph.x)
    edge_index = _to_numpy(graph.edge_index)
    edge_attr = _to_numpy(graph.edge_attr)
    n = x.shape[0]
    n_positions = edge_attr.shape[1]

    is_object = np.all(x[:, 3:] == 0, axis=1)

    position_target = {p: {} for p in range(1, n_positions + 1)}
    for e in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, e]), int(edge_index[1, e])
        if is_object[src]:
            continue  # only read proposition -> object edges
        row = edge_attr[e]
        for p in range(1, n_positions + 1):
            if row[p - 1] == 1:
                position_target[p][src] = dst

    facts = []
    for i in range(n):
        if is_object[i]:
            obj_type_idx = int(np.argmax(x[i, 0:3]))
            predicate = ("location", "taxi", "passenger")[obj_type_idx]
            facts.append((predicate, (i,)))
            continue
        pred_idx = int(np.argmax(x[i, 3:3 + len(VILG_PREDICATE_ORDER)]))
        predicate = VILG_PREDICATE_ORDER[pred_idx]
        args = []
        for p in range(1, n_positions + 1):
            if i in position_target[p]:
                args.append(int(position_target[p][i]))
        facts.append((predicate, tuple(args)))
    return facts


_ROAD = OBJECT_EDGE_LABELS.index("road")
_AT = OBJECT_EDGE_LABELS.index("at")
_PICKED_UP_AT = OBJECT_EDGE_LABELS.index("picked_up_at")
_REQ12 = OBJECT_EDGE_LABELS.index("req12")
_REQ13 = OBJECT_EDGE_LABELS.index("req13")
_REQ23 = OBJECT_EDGE_LABELS.index("req23")
_DIR = OBJECT_EDGE_LABELS.index("dir")


def _object_graph_clusters(graph):
    """
    Shared extraction for enumerate_object_hypotheses/decode_object: per passenger p,
    origins O_p come from req12 edges sourced at p, destinations D_p from req13 edges
    sourced at p (both exact -- the edge's source endpoint IS p, so there is no
    ambiguity in WHICH passenger they belong to). The PAIRING between O_p and D_p is
    what's lost: a candidate (o, d) pair is one with a real req23 edge. Passengers are
    clustered by UNION-FIND OVER SHARED CANDIDATE EDGES (two passengers are linked iff
    some (o, d) pair is a candidate for BOTH of them) -- NOT by shared location, which
    is far coarser: with ~20 passengers x 4 locations each, incidental single-location
    overlaps between otherwise-unrelated passengers are common, but essentially never
    correspond to a genuine req23-edge collision (measured: edge-based clusters are
    size 1 or 2 in every one of 800 real states sampled under scripted play;
    location-based clustering on the same states produced clusters up to size 13).

    Clusters are independent (by construction -- no candidate edge crosses a cluster
    boundary): each cluster's own request facts can be decided without looking at any
    other cluster. This matters in practice, not just in theory -- under an untrained
    policy that rarely delivers, the passenger pool saturates its cap and MANY
    independent 2-passenger (buddy) clusters can be alive simultaneously (measured: 10
    simultaneous clusters after 500 near-random steps at concurrent_passengers=20,
    each with 2 local solutions), so a naive cartesian product across ALL clusters
    would enumerate 2**10 = 1024 full hypotheses even though each cluster's own
    ambiguity is tiny -- see enumerate_object_hypotheses vs decode_object's own
    docstrings for how each of them handles this.

    :return: (type_atoms, base_relational, clusters, local_matchings, cluster_edges)
        -- clusters: list of passenger-id lists; local_matchings: {passenger:
        [valid local (origin,dest) pair tuples]}; cluster_edges: {cluster index:
        the set of req23 edges that cluster must explain}
    :raises ValueError: malformed graph (mismatched origin/destination counts, a
        passenger with no valid local matching, or a single cluster whose own
        LOCAL combination count exceeds MAX_CLUSTER_COMBINATIONS)
    """
    x = _to_numpy(graph.x)
    edge_index = _to_numpy(graph.edge_index)
    edge_attr = _to_numpy(graph.edge_attr)
    n_obj = x.shape[0]

    type_predicate = []
    for i in range(n_obj):
        row = x[i]
        if row[0] == 1:
            type_predicate.append("location")
        elif row[1] == 1:
            type_predicate.append("taxi")
        elif row[2] == 1:
            type_predicate.append("passenger")
        else:
            raise ValueError(f"unrecognised object node one-hot {row!r} for node {i}")
    passenger_ids = [i for i in range(n_obj) if type_predicate[i] == "passenger"]

    adjacent_pairs = set()
    in_pairs = set()
    picked_up_at_pairs = set()
    req12 = defaultdict(set)
    req13 = defaultdict(set)
    req23_edges = set()

    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        row = edge_attr[k]
        if row[_ROAD] == 1:
            adjacent_pairs.add((u, v))
        if row[_AT] == 1 and row[_DIR] == 1:
            in_pairs.add((u, v))
        if row[_PICKED_UP_AT] == 1 and row[_DIR] == 1:
            picked_up_at_pairs.add((u, v))
        if row[_REQ12] == 1:
            req12[u].add(v)
        if row[_REQ13] == 1:
            req13[u].add(v)
        if row[_REQ23] == 1:
            req23_edges.add((u, v))

    type_atoms = [(type_predicate[i], (i,)) for i in range(n_obj)]
    base_relational = (
        [("adjacent", (u, v)) for u, v in adjacent_pairs]
        + [("in", (u, v)) for u, v in in_pairs]
        + [("picked_up_at", (u, v)) for u, v in picked_up_at_pairs]
    )

    origin_of = {p: sorted(req12.get(p, ())) for p in passenger_ids}
    dest_of = {p: sorted(req13.get(p, ())) for p in passenger_ids}
    for p in passenger_ids:
        if len(origin_of[p]) != len(dest_of[p]):
            raise ValueError(
                f"passenger {p}: {len(origin_of[p])} req12 origins but {len(dest_of[p])} "
                f"req13 destinations -- object graph is malformed"
            )

    candidate_edges = {
        p: set((o, d) for o in origin_of[p] for d in dest_of[p] if (o, d) in req23_edges)
        for p in passenger_ids
    }
    for p in passenger_ids:
        if len(candidate_edges[p]) < len(origin_of[p]):
            raise ValueError(
                f"passenger {p}: only {len(candidate_edges[p])} candidate req23 edges for "
                f"{len(origin_of[p])} requests -- object graph is malformed"
            )

    parent = {p: p for p in passenger_ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    edge_to_passengers = defaultdict(set)
    for p in passenger_ids:
        for e in candidate_edges[p]:
            edge_to_passengers[e].add(p)
    for members in edge_to_passengers.values():
        members = list(members)
        for m in members[1:]:
            union(members[0], m)

    cluster_map = defaultdict(list)
    for p in passenger_ids:
        cluster_map[find(p)].append(p)
    clusters = list(cluster_map.values())

    local_matchings = {}
    cluster_edges = {}
    for idx, cluster in enumerate(clusters):
        edges = set()
        for p in cluster:
            edges |= candidate_edges[p]
        cluster_edges[idx] = edges

        for p in cluster:
            origins, dests = origin_of[p], dest_of[p]
            valid = []
            for perm in itertools.permutations(dests):
                pairs = tuple(zip(origins, perm))
                if all(pair in candidate_edges[p] for pair in pairs):
                    valid.append(pairs)
            if not valid:
                raise ValueError(f"passenger {p}: no valid local request matching reproduces its own req23 edges")
            local_matchings[p] = valid

        combo_count = 1
        for p in cluster:
            combo_count *= len(local_matchings[p])
        if combo_count > MAX_CLUSTER_COMBINATIONS:
            raise ValueError(
                f"object hypothesis decoder: cluster {sorted(cluster)} has {combo_count} "
                f"candidate combinations, exceeding the {MAX_CLUSTER_COMBINATIONS} limit"
            )

    return type_atoms, base_relational, clusters, local_matchings, cluster_edges


def _cluster_solutions(cluster, local_matchings, edges):
    """Every combination of this cluster's own per-passenger local matchings whose
    union reproduces `edges` exactly -- see _object_graph_clusters' docstring. A
    cluster's own combo_count (checked in _object_graph_clusters) already bounds this
    search, so this itself never needs a separate guard."""
    solutions = []
    for combo in itertools.product(*(local_matchings[p] for p in cluster)):
        union_edges = set()
        for pairs in combo:
            union_edges.update(pairs)
        if union_edges == edges:
            solutions.append(dict(zip(cluster, combo)))
    if not solutions:
        raise ValueError(f"object hypothesis decoder: cluster {sorted(cluster)} has no valid global covering combination")
    return solutions


def enumerate_object_hypotheses(graph):
    """
    Every FULL request-fact hypothesis consistent with the object graph -- decision
    2's "where several hypotheses fit" enumeration, exposed (not just the single
    chosen one) so tests can check the true facts are always among them. This is NOT
    what decode_object calls (see its own docstring for why: the full cross-product
    over independent clusters this function builds can explode combinatorially even
    though each cluster's own ambiguity stays tiny -- see _object_graph_clusters).
    Intended for testing/introspection on realistic states (at most 1-2 simultaneous
    ambiguous clusters -- see _object_graph_clusters' measurements), not as a
    per-step planning call.

    :param graph: object-convention Data (x [n_obj, 3], edge_index [2, E], edge_attr
        [E, 10] -- OBJECT_EDGE_LABELS). Tensors on any device.
    :return: list of facts-lists, each a full hypothesis (identical to each other
        except for their "request" facts); never empty for a graph that was itself
        produced by facts_to_object_graph on real facts (the true facts are always
        one of them -- see this module's own tests)
    :raises ValueError: see _object_graph_clusters, plus if the TOTAL cross-cluster
        product exceeds MAX_CLUSTER_COMBINATIONS (distinct from, and in addition to,
        _object_graph_clusters' own per-cluster guard)
    """
    type_atoms, base_relational, clusters, local_matchings, cluster_edges = _object_graph_clusters(graph)

    per_cluster_solutions = [
        _cluster_solutions(cluster, local_matchings, cluster_edges[idx])
        for idx, cluster in enumerate(clusters)
    ]

    total = 1
    for solutions in per_cluster_solutions:
        total *= len(solutions)
    if total > MAX_CLUSTER_COMBINATIONS:
        raise ValueError(
            f"enumerate_object_hypotheses: {len(clusters)} independent ambiguous clusters "
            f"combine to {total} total hypotheses, exceeding the {MAX_CLUSTER_COMBINATIONS} "
            f"limit -- use decode_object instead, which resolves each cluster independently "
            f"and never materialises this cross-product"
        )

    hypotheses = []
    for combo in itertools.product(*per_cluster_solutions):
        request_facts = []
        for cluster_solution in combo:
            for p, pairs in cluster_solution.items():
                for o, d in pairs:
                    request_facts.append(("request", (int(p), int(o), int(d))))
        relational = _canonical_sort_relational(base_relational + request_facts)
        hypotheses.append(type_atoms + relational)
    return hypotheses


def decode_object(graph):
    """
    The object convention's decoder, and the one the real planner calls. Resolves
    EACH cluster's ambiguity INDEPENDENTLY (never materialising the cross-cluster
    product enumerate_object_hypotheses does): clusters are independent by
    construction (_object_graph_clusters), so there is no need to consider them
    jointly. A cluster with exactly one valid local-matching combination uses it
    directly; a cluster with several uses _tie_break_index, salted with that
    cluster's own sorted passenger ids so that two different simultaneously-ambiguous
    clusters in the SAME observation get independently-resolved (but each
    individually still deterministic and reproducible) choices, rather than all
    ambiguous clusters happening to move together. This is the ONLY convention whose
    decode is not exact -- see this module's own docstring.
    """
    type_atoms, base_relational, clusters, local_matchings, cluster_edges = _object_graph_clusters(graph)

    request_facts = []
    for idx, cluster in enumerate(clusters):
        solutions = _cluster_solutions(cluster, local_matchings, cluster_edges[idx])
        if len(solutions) == 1:
            chosen = solutions[0]
        else:
            salt = repr(sorted(cluster)).encode("utf-8")
            choice = _tie_break_index(graph, len(solutions), salt=salt)
            chosen = solutions[choice]
        for p, pairs in chosen.items():
            for o, d in pairs:
                request_facts.append(("request", (int(p), int(o), int(d))))

    relational = _canonical_sort_relational(base_relational + request_facts)
    return type_atoms + relational


def _canonical_sort_relational(relational_facts):
    order = {predicate: i for i, predicate in enumerate(RELATIONAL_PREDICATES)}
    return sorted(relational_facts, key=lambda fact: (order[fact[0]], fact[1]))


# ==========================================================================================
# Decision 3: a stable (not Python hash()), device/dtype-independent tie-break
# ==========================================================================================

def _stable_hash_bytes(graph, salt=b""):
    """
    hashlib (not Python's hash(), which is randomised per process by default) over
    x/edge_index/edge_attr/global_features (the last of which carries the time
    feature -- see decision 3's own rationale: consecutive steps must differ, so a
    failed dropoff can be retried with a fresh guess). Every tensor is moved .cpu(),
    cast to a SINGLE fixed dtype (float64 -- lossless for int64 node/edge indices at
    any realistic graph size), and made .contiguous() before extracting bytes, so the
    same logical observation hashes identically regardless of source device, dtype,
    or memory layout -- see this module's own tests (same hash for two independent
    JSON conversions of the same observation; same hash for CPU and CUDA copies of
    the same Data).
    """
    hasher = hashlib.sha256()
    hasher.update(str(_TIE_BREAK_SEED).encode("utf-8"))
    hasher.update(salt)
    for name in ("x", "edge_index", "edge_attr", "global_features"):
        tensor = getattr(graph, name)
        if not isinstance(tensor, th.Tensor):
            tensor = th.as_tensor(tensor)
        tensor = tensor.detach().cpu().to(th.float64).contiguous()
        hasher.update(tensor.numpy().tobytes())
    return hasher.digest()


def _tie_break_index(graph, n_choices, salt=b""):
    """
    :param salt: mixed into the hash ahead of the tensors -- used by decode_object to
        give independently-ambiguous clusters within the SAME observation
        independently-resolved choices (each still individually deterministic and
        reproducible: the same (graph, salt) always gives the same choice). Empty by
        default (the plain per-observation tie-break decision 3 describes).
    """
    digest = _stable_hash_bytes(graph, salt=salt)
    return int.from_bytes(digest, "big") % n_choices


# ==========================================================================================
# plan_on_facts: SHARED across every convention -- decision 4
# ==========================================================================================

def _move_taxi_facts(facts, new_location):
    """Pure (non-mutating): a new fact list with the taxi's in(0, *) fact redirected
    to new_location."""
    new_facts = []
    for predicate, args in facts:
        if predicate == "in" and args[0] == 0:
            new_facts.append(("in", (0, new_location)))
        else:
            new_facts.append((predicate, args))
    return new_facts


def _deliver_facts(facts, passenger_id, destination):
    """
    Pure (non-mutating): the fact-level equivalent of _deliver_atoms -- moves the taxi
    to `destination` (the projection jumps straight to the post-delivery state, same
    as every existing convention), drops every fact mentioning passenger_id (its type
    atom, its in/picked_up_at/request facts), and renumbers the surviving object ids
    contiguously -- exactly TernaryTaxiWorldSimulator._renumber_passengers' own
    scheme (locations and the taxi never move; every passenger id above the removed
    one shifts down by one).
    """
    moved = _move_taxi_facts(facts, destination)
    kept = [(predicate, args) for predicate, args in moved if passenger_id not in args]

    n_obj_old = sum(1 for predicate, _args in facts if predicate in TYPE_PREDICATES)
    surviving_objects = [obj for obj in range(n_obj_old) if obj != passenger_id]
    remap = {old: new for new, old in enumerate(surviving_objects)}

    type_facts = []
    relational_facts = []
    for predicate, args in kept:
        new_args = tuple(remap[a] for a in args)
        if predicate in TYPE_PREDICATES:
            type_facts.append((new_args[0], (predicate, new_args)))
        else:
            relational_facts.append((predicate, new_args))
    type_facts.sort(key=lambda item: item[0])

    return [fact for _new_id, fact in type_facts] + _canonical_sort_relational(relational_facts)


def plan_on_facts(facts, goal):
    """
    The shared planning logic -- decision 4, corrected against the OLD code's ACTUAL
    behaviour (not the shorthand description in the original decision text; see the
    ternary-gnn conversation record for the settling quotes):

      - goal == taxi's own node (id 0): ALWAYS a no-op. graph_to_networkx/
        graph_to_networkx_vilg/graph_to_state_atom all hardcode
        Taxi(..., passenger=None) regardless of what's actually being carried, so
        Planner.plan's "if state.taxi.passenger is not None" check is unconditionally
        False in every existing convention -- selecting the taxi's own node NEVER
        delivers in the old domain. Mirrored exactly here, not "fixed".
      - goal == a passenger's own node: deliver them. If they are ALREADY aboard
        (detected via in(p, 0), the ternary equivalent of oracle_sage's
        `passenger.location == state.taxi.node` check -- NOT via any
        "state.taxi.passenger" concept, which doesn't exist at the fact level).
        Otherwise: move to their waiting location, pick up (implicitly, by planning
        straight through to delivery -- no intermediate "aboard" state is ever
        represented, matching every existing convention), then to their destination.
      - goal == a location: move there.
      - an empty action list becomes [taxi's current location] (a no-op action),
        matching move()/deliver_current_passenger()/plan_atom's own convention.

    Destination is ALWAYS resolved by joining picked_up_at (if already picked up) or
    the passenger's current waiting location (if not) against their own request
    facts on origin == that location -- never from any other source (decision 4).

    :param facts: canonical (predicate, args) list, from ANY decoder (decode_object/
        _vilg/_atom) -- this function has no convention-specific logic at all
    :param goal: node id the policy selected
    :return: (actions, projected_facts)
    """
    n_obj = _validate_facts(facts)
    taxi_node = 0

    road_graph = nx.Graph()
    for predicate, args in facts[:n_obj]:
        if predicate == "location":
            road_graph.add_node(args[0])
    for predicate, args in facts[n_obj:]:
        if predicate == "adjacent":
            road_graph.add_edge(args[0], args[1])

    passenger_ids = [args[0] for predicate, args in facts[:n_obj] if predicate == "passenger"]

    taxi_location = None
    waiting_location = {}
    aboard = None
    picked_up_at = {}
    requests = defaultdict(list)

    for predicate, args in facts[n_obj:]:
        if predicate == "in":
            subject, container = args
            if subject == taxi_node:
                taxi_location = container
            elif container == taxi_node:
                aboard = subject
            else:
                waiting_location[subject] = container
        elif predicate == "picked_up_at":
            p, l = args
            picked_up_at[p] = l
        elif predicate == "request":
            p, o, d = args
            requests[p].append((o, d))

    def destination_of(pid):
        origin = picked_up_at.get(pid, waiting_location.get(pid))
        matches = [d for o, d in requests[pid] if o == origin]
        if len(matches) != 1:
            raise ValueError(
                f"passenger {pid}: expected exactly one request matching origin {origin}, "
                f"got {matches}"
            )
        return matches[0]

    def find_path_to(start, end):
        return nx.shortest_path(road_graph, start, end)[1:]

    if goal == taxi_node:
        actions = []
        projected_facts = facts
    elif goal in passenger_ids:
        if goal == aboard:
            destination = destination_of(goal)
            move = find_path_to(taxi_location, destination)
            projected_facts = _deliver_facts(facts, goal, destination)
            actions = move + [taxi_node]
        else:
            origin = waiting_location[goal]
            destination = destination_of(goal)
            move1 = find_path_to(taxi_location, origin)
            move2 = find_path_to(origin, destination)
            projected_facts = _deliver_facts(facts, goal, destination)
            actions = move1 + [goal] + move2 + [taxi_node]
    else:
        actions = find_path_to(taxi_location, goal)
        projected_facts = _move_taxi_facts(facts, goal)

    if actions == []:
        actions = [taxi_location]

    return actions, projected_facts


# ==========================================================================================
# Top-level dispatch
# ==========================================================================================

_DECODERS = {"oracle_sage": decode_object, "vilg": decode_vilg, "atom": decode_atom}
_ENCODERS = {"oracle_sage": facts_to_object_graph, "vilg": facts_to_vilg_graph, "atom": facts_to_atom_graph}


def _reencode(facts, encode_fn, reference_graph):
    """
    Re-encodes `facts` via `encode_fn` (facts_to_object_graph/_vilg/_atom -- the SAME
    functions the live env uses), then attaches mask/global_features and casts to the
    dtypes/device json_to_ternary_graph_*'s policy-side wrappers use (float32/long),
    mirroring _atoms_to_projection's exact approach: global_features is a CLONE of
    reference_graph's own (increment_timer decrements it in place; cloning keeps the
    caller's input graph untouched), on whichever device reference_graph lives on.
    `meta`'s time/timeout are placeholders -- global_features is overwritten
    immediately after, so only meta["planning"]=True (required by facts_to_atom_graph)
    matters here.
    """
    placeholder_meta = {"time": 0, "timeout": 1, "planning": True}
    node_feats, edge_feats, edge_index, mask, _placeholder_global_feats = encode_fn(facts, placeholder_meta)

    device = reference_graph.x.device
    projection = Data(
        x=th.as_tensor(node_feats, dtype=th.float32, device=device),
        edge_index=th.as_tensor(edge_index, dtype=th.long, device=device),
        edge_attr=th.as_tensor(edge_feats, dtype=th.float32, device=device),
    )
    projection.mask = th.as_tensor(mask, dtype=th.bool, device=device)
    projection.global_features = reference_graph.global_features.clone()
    return projection


def plan_ternary(graph, goal, graph_convention):
    """
    Planner.plan's ternary-domain implementation, for all three conventions -- decode
    -> plan_on_facts -> re-encode -> increment_timer. Never reads anything on `graph`
    beyond x/edge_index/edge_attr/global_features (decision 5); never mutates `graph`
    (every decoder/transform below is pure; the projection is always freshly built).

    :param graph: a ternary Data (any device) for `graph_convention`
    :param goal: node id the policy selected
    :param graph_convention: "oracle_sage" | "vilg" | "atom"
    :return: (projection, actions) -- same shape as the old Planner.plan/plan_atom
    """
    decode = _DECODERS.get(graph_convention)
    if decode is None:
        raise ValueError(f"plan_ternary: unrecognised graph_convention {graph_convention!r}")
    encode = _ENCODERS[graph_convention]

    facts = decode(graph)
    actions, projected_facts = plan_on_facts(facts, goal)
    projection = _reencode(projected_facts, encode, graph)
    return increment_timer(projection, actions)
