"""
.. module:: indistinguishability
   :synopsis: Offline, analysis-only measurement of graph-encoding indistinguishability
   for the Taxi domain's three graph "conventions" (Horcik et al., AAAI-25):
   oracle_sage (object encoding), vilg (object-atom encoding), atom (atom encoding).

   NO training, NO GPU, NO changes to the simulator/env/planner/any convention's
   behaviour -- this script only IMPORTS (read-only) the real graph converters in
   sage/domains/gym_taxi/utils/representations.py and the real WL machinery in
   sage/domains/utils/wl_colours.py, and otherwise stands entirely on its own.

   PART 1 (current Taxi domain, arity <= 2 everywhere): samples ~5000 reachable states
   from a live "city" GraphTaxiEnv (5 seeds, random valid actions), builds each state's
   graph under all three real conventions, and runs EXACT Weisfeiler-Leman colour
   refinement (a fresh, growing vocabulary -- NOT the trained/frozen vocab the real
   training pipeline uses) for L = 1..5, counting how many pairs of DIFFERENT states
   (by ground-atom set) collide (same WL colour-multiset hash) and how many of those
   collisions are "bad" (their approximated optimal action differs). Hypothesis: since
   Taxi has only unary/binary predicates, the three conventions are equally expressive
   here, so collisions -- especially bad ones -- should be near zero at every L.

   "Optimal action" is APPROXIMATED (there is no tractable exact optimal-action oracle
   for this domain) as: if the taxi is carrying a passenger, head for (or, if already
   there, attempt dropoff at) that passenger's destination; otherwise head for (or
   attempt pickup at) the nearest still-waiting passenger by ROAD-GRAPH HOP DISTANCE.
   This is "the best subgoal by shortest-path distance to the nearest deliverable
   passenger", exactly as specified, and ignores full multi-passenger routing/lookahead
   -- see approx_optimal_action()'s docstring. The action is reported as a
   (kind, hop_distance) pair rather than a raw graph-node id, since node ids are
   specific to one randomly generated maze and are not comparable across different
   seeds/mazes.

   PART 2 (prototype of an EXTENDED Taxi domain, STATE GENERATOR ONLY -- no simulator/
   env/planner touched or even imported here): a standalone synthetic generator for a
   hypothetical Taxi variant with one ternary predicate, request(passenger, origin,
   destination), replacing the old binary destination(p, l) -- destination is now only
   derivable by JOINING a dynamic binary picked_up_at(passenger, location) fact against
   whichever request tuple has origin == that location. The three conventions are
   reimplemented HERE (not imported -- the real converters are hardcoded to arity <= 2
   and must not be modified for this task), generalising each one's own real rule
   exactly as specified:
     - object encoding: an edge between every pair of OBJECTS co-occurring in an atom,
       typed by predicate -- this is what loses argument POSITION at arity 3.
     - atom encoding: one node per ATOM, edges labelled (i, j) whenever argument i of one
       atom equals argument j of another -- 9 labels now (positions range over 1..3).
     - object-atom encoding: one node per atom plus one node per object, edges labelled
       by argument POSITION (3 positions now).

   Locations also sit in a small ADJACENT(l1, l2) road network (a maze, exactly like the
   current domain's binary "adjacent" predicate) -- necessary, not decorative: without it,
   "swap two crossed passengers' request tables" is indistinguishable from "rename those
   two passengers", which is invisible to EVERY encoding (not just object encoding), so no
   valid demonstration is possible without it. See the PART 2 section's own correctness
   note for the full derivation (verified empirically, including an exhaustive search that
   confirmed the un-anchored construction fails) and verify_anchor_case() for a second,
   maze-free construction (3 passengers) that isolates the higher-arity effect on its own.

   The key empirical question (per the task): how high must `crossing_rate` (the
   fraction of passengers paired with a "buddy" who shares their origins but has
   swapped destinations) be before object encoding shows MANY bad collisions while atom/
   object-atom show none? See run_part2_sweep()/print_part2_report() for the answer this
   script actually measures.

Run from the repo root with:
    python analysis/indistinguishability.py            # full run (~5000 Part 1 states)
    python analysis/indistinguishability.py --quick    # small, fast smoke-test run
"""
import argparse
import hashlib
import itertools
import time
from collections import defaultdict

import numpy as np

# --- sandbox-vs-RCP compat shims (same as sage/domains/utils/build_wl_vocab.py; this
# script does not touch, or need to touch, any of the files these work around) ---------
if not hasattr(np, "float"):
    np.float = float

import gym.utils.seeding as _seeding

if not hasattr(np.random.Generator, "randint"):
    class _RandintCompatGenerator:
        def __init__(self, generator):
            self._generator = generator

        def randint(self, low, high=None):
            if hasattr(self._generator, "randint"):
                return self._generator.randint(low, high) if high is not None else self._generator.randint(low)
            return self._generator.integers(low, high)

        def __getattr__(self, name):
            return getattr(self._generator, name)

    _original_np_random = _seeding.np_random

    def _np_random_with_randint(seed=None):
        generator, seed = _original_np_random(seed)
        return _RandintCompatGenerator(generator), seed

    _seeding.np_random = _np_random_with_randint

import networkx as nx
import scipy.sparse as sp
import torch as th

import sage.domains.gym_taxi  # noqa: F401  (registers gym env ids)
from sage.domains.gym_taxi import REWARDS
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.utils.representations import (
    atoms_to_graph,
    env_to_atoms,
    env_to_graph,
    env_to_vilg_graph,
)
from sage.domains.utils.wl_colours import (
    _resolve,
    edge_labels,
    edge_labels_atom,
    edge_labels_vilg,
    initial_colours,
    initial_colours_atom,
    initial_colours_vilg,
    refine,
)

MAX_L = 5
PART1_CONVENTIONS = ("oracle_sage", "vilg", "atom")
PART2_CONVENTIONS = ("object", "atom", "object_atom")


# ==========================================================================================
# Shared exact-WL machinery: a FRESH, growing vocabulary per (convention, run) -- never the
# trained/frozen vocab the real pipeline uses. One shared vocab across every state compared
# within a run guarantees identical WL signatures get identical colour ids, which is exactly
# what makes raw-colour-id multiset equality a valid, exact (not approximate) isomorphism
# check at that L -- see collision_table()'s docstring.
# ==========================================================================================

def wl_colours_per_level(x, edge_index, edge_attr, init_fn, edge_fn, vocab, max_l=MAX_L):
    """
    Runs `max_l` rounds of WL colour refinement (sage.domains.utils.wl_colours.refine,
    imported unmodified), recording the per-node colour LIST after each round -- L=1..L.
    `vocab` is mutated in place (growing mode, frozen=False) and should be shared across
    every graph being compared in one run (see module docstring).

    :param init_fn: e.g. initial_colours / initial_colours_vilg / initial_colours_atom,
        or one of this file's ext_initial_colours_* for Part 2
    :param edge_fn: e.g. edge_labels / edge_labels_vilg / edge_labels_atom, or one of
        this file's ext_edge_labels_* for Part 2
    :return: {L: colours_list} for L in 1..max_l
    """
    type_colours = init_fn(x).tolist()
    colours = th.empty(x.shape[0], dtype=th.long)
    for v, type_id in enumerate(type_colours):
        colours[v] = _resolve(("init", type_id), vocab, False)
    labels = edge_fn(edge_attr)
    per_level = {}
    for level in range(1, max_l + 1):
        colours = refine(colours, edge_index, labels, vocab, frozen=False)
        per_level[level] = colours.tolist()
    return per_level


def colour_multiset_hash(colours):
    """
    Canonical hash of the MULTISET of final node colours (order-invariant: colours is
    sorted first) -- a real hash (sha256 of the sorted list's repr), per the task's
    "compute a canonical hash", not just the sorted tuple itself. Collisions of this hash
    at the data scale used here are practically impossible, so hash equality is, for
    every practical purpose, exact multiset equality.
    """
    return hashlib.sha256(repr(sorted(colours)).encode()).hexdigest()


def collision_table(states, convention, init_fn, edge_fn, max_l=MAX_L, same_world_only=False):
    """
    Dedupes `states` by ground-atom set (only DISTINCT states are compared -- see module
    docstring), then for L = 1..max_l builds one shared growing vocab and computes each
    distinct state's exact WL colour-multiset hash, then counts:
      - distinct_states: number of distinct ground-atom-set states (constant across L)
      - distinct_hashes: number of distinct WL hashes at this L
      - collisions: pairs of DIFFERENT distinct states sharing a hash at this L
      - bad_collisions: of those, pairs whose "answer" differs (see each caller for what
        "answer" means -- Part 1's approximated optimal action, or Part 2's exact true
        destination mapping)

    :param states: list of dicts with "atoms" (a hashable, e.g. frozenset, identifying
        the ground-atom-set state), "graphs" (dict: convention -> (x, edge_index,
        edge_attr) tensors), "answer" (anything supporting != for bad-collision
        detection), and, if `same_world_only`, "world_id"
    :param same_world_only: if True, a colliding pair is only classified bad/not-bad when
        both states share the same "world_id" (see Part 2: raw object ids are only
        comparable WITHIN one generated world, not across two different ones) -- pairs
        from different worlds are still counted as collisions but reported separately
        under "cross_world_incomparable", never silently misclassified
    :return: {L: {"distinct_states", "distinct_hashes", "collisions", "bad_collisions",
        "cross_world_incomparable"}}
    """
    seen = {}
    for s in states:
        seen.setdefault(s["atoms"], s)
    distinct = list(seen.values())
    n_distinct = len(distinct)

    vocab = {}
    per_state_levels = [
        wl_colours_per_level(*s["graphs"][convention], init_fn, edge_fn, vocab, max_l)
        for s in distinct
    ]

    table = {}
    for level in range(1, max_l + 1):
        buckets = defaultdict(list)
        for i, levels in enumerate(per_state_levels):
            buckets[colour_multiset_hash(levels[level])].append(i)

        collisions = 0
        bad = 0
        cross_world_incomparable = 0
        for idxs in buckets.values():
            if len(idxs) < 2:
                continue
            for i, j in itertools.combinations(idxs, 2):
                collisions += 1
                if same_world_only and distinct[i].get("world_id") != distinct[j].get("world_id"):
                    cross_world_incomparable += 1
                    continue
                if distinct[i]["answer"] != distinct[j]["answer"]:
                    bad += 1

        table[level] = {
            "distinct_states": n_distinct,
            "distinct_hashes": len(buckets),
            "collisions": collisions,
            "bad_collisions": bad,
            "cross_world_incomparable": cross_world_incomparable,
        }
    return table


# ==========================================================================================
# PART 1 -- current Taxi domain
# ==========================================================================================

def _t3(seven_or_three_tuple):
    """(x, edge_index, edge_attr) torch tensors from either env_to_graph/env_to_vilg_graph's
    7-tuple (node_feats, edge_feats, edge_index, mask, global_feats, wl_ids, wl_hist -- the
    trailing WL fields are the TRAINED/frozen vocab's output and are discarded here, per
    the task: this script computes its own, unrelated, exact WL) or atoms_to_graph's plain
    3-tuple (node_feats, edge_feats, edge_index)."""
    node_feats, edge_feats, edge_index = seven_or_three_tuple[0], seven_or_three_tuple[1], seven_or_three_tuple[2]
    return (
        th.as_tensor(node_feats, dtype=th.float),
        th.as_tensor(edge_index, dtype=th.long),
        th.as_tensor(edge_feats, dtype=th.float),
    )


def sample_action(sim):
    """Uniformly random VALID action -- verbatim algorithm from
    sage/domains/utils/build_wl_vocab.py's sample_action, reproduced here (not imported)
    so this script controls its own shims independently; see that module's docstring for
    why non-adjacent location actions are avoided."""
    taxi_location = sim.taxi.location
    candidates = {
        c for c in sim.graph.successors(taxi_location)
        if sim.graph.nodes[c]["attr"] != [0, 0, 1] or c in sim.passengers
    }
    candidates.add(taxi_location)
    candidates.add(0)
    return int(np.random.choice(list(candidates)))


def road_graph(sim):
    """Undirected road-only nx.Graph read off sim.graph -- same edge_attr[0]==1 filter
    build_wl_vocab.py's road_graph and the oracle_sage planner's graph_to_networkx use."""
    G = nx.Graph()
    for u, v, attr in sim.graph.edges(data=True):
        if attr["attr"][0] == 1:
            G.add_edge(u, v)
    return G


def approx_optimal_action(sim, G):
    """
    APPROXIMATED optimal action (see module docstring): the best subgoal by shortest-path
    distance to the nearest deliverable passenger. Returned as a maze-INDEPENDENT
    (kind, hop_distance) pair -- not a raw node id, which would only be meaningful within
    the one maze it came from:
      - carrying a passenger: ("dropoff", 0) if already at their destination, else
        ("move_to_dropoff", hop distance to it)
      - not carrying, but a passenger is waiting: ("pickup", 0) if already at the nearest
        one's location, else ("move_to_pickup", hop distance to it)
      - no passengers exist at all: ("idle", 0)
    """
    if sim.taxi.passenger is not None:
        pid = sim.taxi.passenger
        dest = sim.passengers[pid].destination
        if sim.taxi.location == dest:
            return ("dropoff", 0)
        return ("move_to_dropoff", nx.shortest_path_length(G, sim.taxi.location, dest))
    if sim.passengers:
        pid = min(
            sim.passengers,
            key=lambda p: nx.shortest_path_length(G, sim.taxi.location, sim.passengers[p].location),
        )
        loc = sim.passengers[pid].location
        if sim.taxi.location == loc:
            return ("pickup", 0)
        return ("move_to_pickup", nx.shortest_path_length(G, sim.taxi.location, loc))
    return ("idle", 0)


def generate_part1_states(seeds=(0, 300, 600, 900, 1200), sample_every=5, target_total=5000, verbose=True):
    """
    For each seed: a fresh "city" GraphTaxiEnv (the real Oracle-SAGE Taxi training
    config), stepped with uniformly-random VALID actions, snapshotting the live
    simulator state every `sample_every` steps until this seed has contributed its share
    of `target_total`.

    :return: list of dicts: {"seed", "step", "atoms" (frozenset, for state identity/
        dedup), "graphs" (dict: convention -> (x, edge_index, edge_attr)), "answer"
        (approx_optimal_action(...))}
    """
    per_seed = max(1, target_total // len(seeds))
    states = []
    for seed in seeds:
        t0 = time.time()
        env = GraphTaxiEnv(representation="graph", scenario="city", mask=False, rewards=REWARDS["v1"], graph_convention="oracle_sage")
        env.seed(seed)
        env.reset()
        sim = env.sim
        G = road_graph(sim)
        step = 0
        collected = 0
        while collected < per_seed:
            if step % sample_every == 0:
                atoms = env_to_atoms(sim)
                graphs = {
                    "oracle_sage": _t3(env_to_graph(sim)),
                    "vilg": _t3(env_to_vilg_graph(sim)),
                    "atom": _t3(atoms_to_graph(atoms)),
                }
                states.append({
                    "seed": seed, "step": step,
                    "atoms": frozenset(atoms),
                    "graphs": graphs,
                    "answer": approx_optimal_action(sim, G),
                })
                collected += 1
            _, _, done, _ = env.step(sample_action(sim))
            step += 1
            if done:
                env.reset()
                sim = env.sim
                G = road_graph(sim)
        env.close()
        if verbose:
            print(f"  seed={seed}: {collected} states sampled over {step} steps ({time.time() - t0:.1f}s)")
    return states


def run_part1(seeds=(0, 300, 600, 900, 1200), sample_every=5, target_total=5000, max_l=MAX_L, verbose=True):
    if verbose:
        print(f"PART 1: sampling ~{target_total} states from scenario='city', seeds={seeds} ...")
    states = generate_part1_states(seeds=seeds, sample_every=sample_every, target_total=target_total, verbose=verbose)
    if verbose:
        print(f"  total snapshots: {len(states)}")

    decoders = {
        "oracle_sage": (initial_colours, edge_labels),
        "vilg": (initial_colours_vilg, edge_labels_vilg),
        "atom": (initial_colours_atom, edge_labels_atom),
    }
    report = {}
    for convention in PART1_CONVENTIONS:
        if verbose:
            print(f"  running exact WL (L=1..{max_l}) for convention={convention!r} ...")
        t0 = time.time()
        init_fn, edge_fn = decoders[convention]
        report[convention] = collision_table(states, convention, init_fn, edge_fn, max_l=max_l)
        if verbose:
            print(f"    done ({time.time() - t0:.1f}s)")
    return states, report


def print_part1_report(report, max_l=MAX_L):
    print()
    print("=" * 88)
    print("PART 1 report -- current Taxi domain (city scenario), 5 seeds, exact WL")
    print("=" * 88)
    print(
        "'Optimal action' is APPROXIMATED as the best subgoal by shortest-path distance "
        "to the nearest deliverable passenger (see approx_optimal_action() docstring) -- "
        "not a true optimal-action oracle."
    )
    print()
    header = f"{'convention':<12} {'L':>2} {'distinct_states':>15} {'distinct_hashes':>15} {'collisions':>10} {'bad_collisions':>14}"
    print(header)
    print("-" * len(header))
    for convention in PART1_CONVENTIONS:
        for level in range(1, max_l + 1):
            row = report[convention][level]
            print(
                f"{convention:<12} {level:>2} {row['distinct_states']:>15} {row['distinct_hashes']:>15} "
                f"{row['collisions']:>10} {row['bad_collisions']:>14}"
            )
    print()


# ==========================================================================================
# PART 2 -- prototype of the extended domain (state generator + reimplemented encodings)
#
# IMPORTANT CORRECTNESS NOTE (discovered while building the hand-check, fixed before this
# script's first real run -- see analysis/README.md for the full writeup): exchanging two
# crossed passengers' ENTIRE request tables is, all by itself, just renaming those two
# passengers -- a pure relabeling, which is INVISIBLE to every graph encoding (object,
# atom, object-atom alike), since none of them bake in absolute object identity beyond
# structure. An earlier version of this script's "flip" construction had BOTH members of
# a crossed pair boarded (picked_up_at) at once, which made the two compared states
# related by exactly that trivial passenger-swap -- so atom/object-atom "not colliding"
# would have been impossible to demonstrate honestly, no matter the scale (verified by
# exhaustive search over the full 2x2x2 configuration space: zero pairs showed the
# intended split). Two independent fixes make the comparison meaningful:
#   1. Only ONE member of a crossed pair boards (picked_up_at) in the compared snapshot --
#      the buddy is present (so its request table still creates the crossing ambiguity)
#      but hasn't been picked up yet. This alone breaks the passenger-relabeling escape.
#   2. Locations are placed in a real ADJACENT(l1, l2) road network (a maze, exactly like
#      the current domain's binary "adjacent" predicate) with a distinctive local
#      structure per location, so a hypothetical destination-relabeling (C<->D) is no
#      longer a symmetry either -- swapping two locations with different neighbours does
#      not preserve the maze's own edge set.
# Both were verified empirically (not just derived by hand) before being adopted here.
# ==========================================================================================

LOCATION, PASSENGER, ADJACENT, REQUEST, PICKED_UP_AT = "location", "passenger", "adjacent", "request", "picked_up_at"
EXT_PREDICATES = [LOCATION, PASSENGER, ADJACENT, REQUEST, PICKED_UP_AT]
_REL_PREDICATES = [REQUEST, PICKED_UP_AT, ADJACENT]


def build_path_location_graph(n_locations):
    """Deterministic, asymmetric location graph for the hand-checkable case: a simple
    path 0-1-2-...-(n-1). Any two interior-vs-endpoint locations have different degree,
    so (per the specific pair used in verify_hand_checkable_case, checked there) they
    are not interchangeable by any relabeling that also has to preserve this graph."""
    return [(i, i + 1) for i in range(n_locations - 1)]


def build_random_location_graph(n_locations, rng, extra_edge_prob=0.3):
    """A random connected graph over n_locations locations (a random spanning path, plus
    a few extra edges) -- used for the sweep. A random graph is, generically, fully
    asymmetric (no nontrivial automorphism), which is what breaks the destination-
    relabeling escape described above; unlike the hand-check, this is not proven for
    every draw, just overwhelmingly likely, which is acceptable for an aggregate sweep."""
    order = list(range(n_locations))
    rng.shuffle(order)
    edges = {frozenset((order[i], order[i + 1])) for i in range(n_locations - 1)}
    for i in range(n_locations):
        for j in range(i + 1, n_locations):
            if frozenset((i, j)) not in edges and rng.random_sample() < extra_edge_prob:
                edges.add(frozenset((i, j)))
    return [tuple(e) for e in edges]


def generate_extended_world(n_passengers, n_origins, n_destinations, crossing_rate, k, rng, location_graph=None):
    """
    Standalone synthetic ground-atom-set generator for the extended domain (see module
    docstring) -- does NOT import or touch the simulator/env/planner.

    Objects: origins 0..n_origins-1 (LOCATION), destinations
    n_origins..n_origins+n_destinations-1 (LOCATION), passengers after that (PASSENGER)
    -- contiguous 0..n_obj-1 for locations, exactly the invariant env_to_atoms relies on
    for the real domain, kept here too so atom-row k really is object k.

    round(crossing_rate * n_passengers / 2) passenger PAIRS are "buddies": both draw the
    SAME random k-subset of origins, but buddy_b's destinations are buddy_a's destination
    list REVERSED (generalises "swapped" beyond k=2). Every other passenger gets its own,
    independently-sampled, k-subset of origins and of destinations (not intentionally
    shared with anyone).

    Boarding: every INDEPENDENT passenger boards at their own first-listed origin
    (picked_up_at(p, origins[0])), same as before. Within a crossed pair, ONLY the first
    member boards in this snapshot -- the second is present (its request table still
    creates the crossing ambiguity) but has not been picked up yet. See the module-level
    note above for why this asymmetry is necessary, not cosmetic.

    :param k: requests_per_passenger, >= 2, with distinct origins (and, here, distinct
        destinations too) per passenger -- requires n_origins >= k, n_destinations >= k
    :param rng: a numpy RandomState
    :param location_graph: list of (u, v) ADJACENT edges over the n_origins+n_destinations
        locations; build_random_location_graph(...) is used if None
    :return: (atoms, tables, crossed_pairs, boarded, true_destination, location_graph)
        atoms: list of (predicate, args_tuple)
        tables: dict passenger_id -> list[(origin, destination)], length k
        crossed_pairs: list of (passenger_a, passenger_b) object ids
        boarded: dict passenger_id -> boarding location, for passengers who HAVE boarded
        true_destination: dict passenger_id -> destination object id (only for `boarded`)
    """
    assert k >= 2, "requests_per_passenger must be >= 2"
    assert n_origins >= k, "n_origins must be >= requests_per_passenger (distinct origins)"
    assert n_destinations >= k, "n_destinations must be >= requests_per_passenger (distinct destinations)"

    n_locations = n_origins + n_destinations
    if location_graph is None:
        location_graph = build_random_location_graph(n_locations, rng)

    origin_pool = list(range(n_origins))
    dest_pool = list(range(n_origins, n_origins + n_destinations))
    passenger_ids = list(range(n_locations, n_locations + n_passengers))

    n_pairs = int(round(crossing_rate * n_passengers / 2))
    n_pairs = max(0, min(n_pairs, n_passengers // 2))

    tables = {}
    boarded = {}
    crossed_pairs = []
    idx = 0
    for _ in range(n_pairs):
        a, b = passenger_ids[idx], passenger_ids[idx + 1]
        origins = list(rng.choice(origin_pool, size=k, replace=False))
        dests = list(rng.choice(dest_pool, size=k, replace=False))
        tables[a] = list(zip(origins, dests))
        tables[b] = list(zip(origins, list(reversed(dests))))
        boarded[a] = origins[0]  # only a boards; b (the buddy) has not been picked up yet
        crossed_pairs.append((a, b))
        idx += 2
    for pid in passenger_ids[idx:]:
        origins = list(rng.choice(origin_pool, size=k, replace=False))
        dests = list(rng.choice(dest_pool, size=k, replace=False))
        tables[pid] = list(zip(origins, dests))
        boarded[pid] = origins[0]

    atoms, true_destination = _atoms_from_tables(origin_pool, dest_pool, passenger_ids, tables, boarded, location_graph)
    return atoms, tables, crossed_pairs, boarded, true_destination, location_graph


def _atoms_from_tables(origin_pool, dest_pool, passenger_ids, tables, boarded, location_graph):
    atoms = []
    for o in origin_pool:
        atoms.append((LOCATION, (o,)))
    for d in dest_pool:
        atoms.append((LOCATION, (d,)))
    for pid in passenger_ids:
        atoms.append((PASSENGER, (pid,)))
    for (u, v) in location_graph:
        atoms.append((ADJACENT, (u, v)))
        atoms.append((ADJACENT, (v, u)))
    true_destination = {}
    for pid in passenger_ids:
        for (o, d) in tables[pid]:
            atoms.append((REQUEST, (pid, o, d)))
        if pid in boarded:
            board = boarded[pid]
            atoms.append((PICKED_UP_AT, (pid, board)))
            true_destination[pid] = dict(tables[pid])[board]
    return atoms, true_destination


def flip_crossed_pairs(tables, crossed_pairs, boarded, origin_pool, dest_pool, passenger_ids, location_graph):
    """
    The counterfactual "State B": for every crossed pair (a, b), EXCHANGE their entire
    request tables. `boarded` is passed through UNCHANGED (a crossed pair shares its
    origin set, so the boarding LOCATION value stays valid for whichever object was
    already marked boarded -- see generate_extended_world: only one member of a pair
    ever boards, and that does not change under the exchange). Independent passengers
    and the location graph are untouched. Whenever >=1 crossed pair exists, this is a
    genuinely DIFFERENT ground-atom set (some request atoms differ) with a DIFFERENT
    true destination for the boarded member of each crossed pair.

    :return: (atoms, tables, crossed_pairs, boarded, true_destination) -- same shape as
        generate_extended_world's return, for the flipped state
    """
    new_tables = dict(tables)
    for a, b in crossed_pairs:
        new_tables[a], new_tables[b] = tables[b], tables[a]
    atoms, true_destination = _atoms_from_tables(origin_pool, dest_pool, passenger_ids, new_tables, boarded, location_graph)
    return atoms, new_tables, crossed_pairs, boarded, true_destination


# --- generalised OBJECT encoding (oracle_sage-style) -------------------------------------

def atoms_to_object_graph_ext(atoms):
    """
    One node per OBJECT (the LOCATION/PASSENGER type atoms -- exactly env_to_atoms' first
    n_obj rows). An UNDIRECTED edge (materialised as a symmetric forward/backward pair,
    matching the real converters' own bidirectional convention) between every PAIR of
    objects co-occurring in a relational atom (request/picked_up_at/adjacent), typed by
    that atom's predicate -- discarding which argument POSITION each object filled,
    exactly the information the task specifies object encoding loses at arity >= 3. If
    two different atoms both connect the same object pair with the same predicate, both
    edges are kept (a literal reading of "an edge between every pair... in AN atom" --
    one edge per (atom, pair) instance, not deduplicated across atoms).

    :return: (node_feats [n_obj, 2], edge_feats [E, 3], edge_index [2, E])
    """
    n_obj = sum(1 for pred, _args in atoms if pred in (LOCATION, PASSENGER))
    node_feats = np.zeros((n_obj, 2), dtype=np.float64)
    for i in range(n_obj):
        node_feats[i, 0 if atoms[i][0] == LOCATION else 1] = 1

    rows, cols, label_ids = [], [], []
    for pred, args in atoms:
        if pred not in _REL_PREDICATES:
            continue
        label = _REL_PREDICATES.index(pred)
        for u, v in itertools.combinations(args, 2):
            rows += [u, v]
            cols += [v, u]
            label_ids += [label, label]

    edge_index = np.array([rows, cols], dtype=np.int64) if rows else np.zeros((2, 0), dtype=np.int64)
    edge_feats = np.zeros((len(label_ids), len(_REL_PREDICATES)), dtype=np.float64)
    for e, lab in enumerate(label_ids):
        edge_feats[e, lab] = 1
    return node_feats, edge_feats, edge_index


def ext_initial_colours_object(x):
    return th.argmax(x, dim=1).long()


def ext_edge_labels_object(edge_attr):
    return th.argmax(edge_attr, dim=1).long()


# --- generalised ATOM encoding (Horcik et al. Def. 2, generalised to max arity 3) --------

def atoms_to_graph_ext(atoms, max_arity=3):
    """
    Generalises representations.py's atoms_to_graph to max_arity=3 (9 labels, (i, j) for
    i, j in {1,2,3} -- see the task's own point that arity 3 needs 3x3 position pairs):
    SAME algorithm (a scipy.sparse argument-position incidence matrix P_p per position p,
    edge a->b labelled (i, j) whenever a's position-i argument equals b's position-j
    argument, merged via np.maximum.at when several (i, j) co-occur for the same ordered
    pair), just with p ranging over 1..max_arity instead of a hardcoded {1, 2}. Written
    independently here -- atoms_to_graph itself is hardcoded to arity <= 2 and must not
    be modified for this analysis-only task. adjacent(l1, l2) atoms (arity 2) fit this
    same machinery without any special-casing.

    :return: (node_feats [N, 5], edge_feats [E, 9], edge_index [2, E])
    """
    n = len(atoms)
    node_feats = np.zeros((n, len(EXT_PREDICATES)), dtype=np.float64)
    for i, (predicate, _args) in enumerate(atoms):
        node_feats[i, EXT_PREDICATES.index(predicate)] = 1

    P = {}
    for p in range(1, max_arity + 1):
        objs, ats = [], []
        for i, (_predicate, args) in enumerate(atoms):
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


def ext_initial_colours_atom(x):
    return th.argmax(x, dim=1).long()


def ext_edge_labels_atom(edge_attr):
    """9-bit multi-hot -> integer bitmask, generalising edge_labels_atom's 4-bit version."""
    bits = th.as_tensor([2 ** k for k in range(edge_attr.shape[1])], dtype=edge_attr.dtype)
    return (edge_attr * bits).sum(dim=1).long()


# --- generalised OBJECT-ATOM encoding (vilg-style, generalised to 3 positions) -----------

def atoms_to_object_atom_graph_ext(atoms):
    """
    One node per OBJECT plus one node per relational atom (request/picked_up_at/adjacent
    -- unary LOCATION/PASSENGER atoms fold into the object nodes' own features directly,
    exactly as env_to_vilg_graph does for location/taxi/passenger). UNDIRECTED position-
    labelled edges (materialised as a symmetric forward/backward pair, matching
    env_to_vilg_graph) from each relational-atom node to each of its argument objects,
    labelled by argument POSITION -- 3 positions now (task spec), instead of vilg's real
    2.

    Node feats: [is_location, is_passenger, is_request, is_picked_up_at, is_adjacent]
    (5-dim, block-padded exactly like env_to_vilg_graph's object/proposition rows -- no
    "goal achieved" status block, since Part 2's states are static one-shot ground-atom
    sets with no achieved/unachieved concept).

    :return: (node_feats [N, 5], edge_feats [E, 3], edge_index [2, E])
    """
    n_obj = sum(1 for pred, _args in atoms if pred in (LOCATION, PASSENGER))
    node_feats = []
    for i in range(n_obj):
        row = [0.0, 0.0, 0.0, 0.0, 0.0]
        row[0 if atoms[i][0] == LOCATION else 1] = 1.0
        node_feats.append(row)

    rows, cols, positions = [], [], []
    next_idx = n_obj
    for pred, args in atoms:
        if pred not in _REL_PREDICATES:
            continue
        row = [0.0, 0.0, 0.0, 0.0, 0.0]
        row[2 + _REL_PREDICATES.index(pred)] = 1.0
        node_feats.append(row)
        for pos, obj in enumerate(args, start=1):
            rows += [next_idx, obj]
            cols += [obj, next_idx]
            positions += [pos, pos]
        next_idx += 1

    node_feats = np.array(node_feats, dtype=np.float64)
    edge_index = np.array([rows, cols], dtype=np.int64) if rows else np.zeros((2, 0), dtype=np.int64)
    edge_feats = np.zeros((len(positions), 3), dtype=np.float64)
    for e, pos in enumerate(positions):
        edge_feats[e, pos - 1] = 1
    return node_feats, edge_feats, edge_index


def ext_initial_colours_object_atom(x):
    """Disjoint-block decode (object type block vs proposition-predicate block), mirroring
    initial_colours_vilg's own disjoint-block handling -- a blind whole-row argmax would
    silently collapse the two blocks together."""
    is_object = x[:, 0:2].sum(dim=1) > 0
    obj_type = x[:, 0:2].argmax(dim=1)
    prop_type = 2 + x[:, 2:5].argmax(dim=1)
    return th.where(is_object, obj_type, prop_type).long()


def ext_edge_labels_object_atom(edge_attr):
    return th.argmax(edge_attr, dim=1).long()


PART2_ENCODERS = {
    "object": (atoms_to_object_graph_ext, ext_initial_colours_object, ext_edge_labels_object),
    "atom": (atoms_to_graph_ext, ext_initial_colours_atom, ext_edge_labels_atom),
    "object_atom": (atoms_to_object_atom_graph_ext, ext_initial_colours_object_atom, ext_edge_labels_object_atom),
}


def _build_part2_graphs(atoms):
    graphs = {}
    for name, (builder, _init_fn, _edge_fn) in PART2_ENCODERS.items():
        node_feats, edge_feats, edge_index = builder(atoms)
        graphs[name] = (
            th.as_tensor(node_feats, dtype=th.float),
            th.as_tensor(edge_index, dtype=th.long),
            th.as_tensor(edge_feats, dtype=th.float),
        )
    return graphs


# --- hop-distance diagnostic --------------------------------------------------------------

def build_undirected_nx(edge_index, n):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    ei = edge_index.numpy() if hasattr(edge_index, "numpy") else np.asarray(edge_index)
    for u, v in zip(ei[0].tolist(), ei[1].tolist()):
        G.add_edge(u, v)
    return G


def compute_hop_distances(atoms, passenger_id):
    """
    For each convention, the shortest-path hop distance from "the node the actor scores"
    (the CORRECT destination location for `passenger_id`, per this state's own
    picked_up_at/request join) to "the picked_up_at fact":
      - object encoding has no fact-nodes at all (facts ARE edges) -- "the picked_up_at
        fact" is represented by the boarding-location OBJECT node it connects to.
      - atom / object-atom encoding both have a dedicated NODE for the picked_up_at
        proposition itself -- used directly.
    This determines the minimum number of GNN message-passing rounds needed to even have
    a CHANCE of combining these two facts, before asking whether the encoding retains
    enough information to do so correctly at all (a separate question -- see the
    collision tables).

    :return: (hop_distances: dict convention -> int, info: dict with "board_loc","dest")
    """
    n_obj = sum(1 for pred, _args in atoms if pred in (LOCATION, PASSENGER))
    picked_up_at_atom = next(a for a in atoms if a[0] == PICKED_UP_AT and a[1][0] == passenger_id)
    board_loc = picked_up_at_atom[1][1]
    matching_request = next(
        a for a in atoms if a[0] == REQUEST and a[1][0] == passenger_id and a[1][1] == board_loc
    )
    dest = matching_request[1][2]

    hop_distances = {}

    _, _, ei_obj = atoms_to_object_graph_ext(atoms)
    G = build_undirected_nx(ei_obj, n_obj)
    hop_distances["object"] = nx.shortest_path_length(G, dest, board_loc)

    _, _, ei_atom = atoms_to_graph_ext(atoms)
    G = build_undirected_nx(ei_atom, len(atoms))
    dest_atom_idx = dest  # a LOCATION atom's own row == its object id -- same invariant as env_to_atoms
    picked_up_at_idx = atoms.index(picked_up_at_atom)
    hop_distances["atom"] = nx.shortest_path_length(G, dest_atom_idx, picked_up_at_idx)

    nf_oa, _, ei_oa = atoms_to_object_atom_graph_ext(atoms)
    rel_atoms = [a for a in atoms if a[0] in _REL_PREDICATES]
    picked_up_at_node = n_obj + rel_atoms.index(picked_up_at_atom)
    G = build_undirected_nx(ei_oa, nf_oa.shape[0])
    hop_distances["object_atom"] = nx.shortest_path_length(G, dest, picked_up_at_node)

    return hop_distances, {"board_loc": board_loc, "dest": dest}


# --- hand-checkable verification, case 1: 2 passengers + a maze -------------------------

def verify_hand_checkable_case(verbose=True):
    """
    2 passengers, 2 origins, 2 destinations, requests_per_passenger=2, crossing_rate=1.0
    (fully crossed -- the ONE possible pair), on a 4-location PATH maze 0-1-2-3 (origins
    {0,1}, destinations {2,3} -- degrees 1,2,2,1, so 2 and 3 are NOT interchangeable: no
    relabeling fixing 0 and 1 in place can swap them). Passenger p: (0->2, 1->3); buddy q
    (swapped): (0->3, 1->2). ONLY p boards, at origin 0 (true dest 2); q is present
    (its table still creates the crossing) but has not been picked up in this snapshot.
    State B exchanges p's/q's request tables (p now has q's old table: 0->3, 1->2; true
    dest becomes 3), holding p's picked_up_at(p, 0) fact and the maze fixed.

    EXPECTED RESULT (written before the asserts below, per the task):
      - under OBJECT encoding, State A and State B are a LITERALLY IDENTICAL graph: p's/
        q's request-typed edges reach the same 4 objects {0,1,2,3} regardless of pairing
        (collectively p+q's four request atoms cover all 4 origin-destination
        combinations either way), the maze's adjacent edges don't change at all, and
        picked_up_at(p,0) is the same single edge in both -- so it collides at every L.
      - under ATOM encoding, State A and State B are NOT isomorphic -- picked_up_at(p,0)
        forces p's boarding fact to align with exactly ONE request atom (State A:
        request(p,0,2); State B: request(p,0,3)) via a DIRECT atom-atom edge (they share
        BOTH position-1 (p) and position-2 (0), a distinguishing multi-hot label), so it
        must NOT collide, already at L=1.
      - under OBJECT-ATOM encoding, the same non-isomorphism holds, but relational-atom
        nodes only ever connect to their OWN argument OBJECTS, never directly to each
        other -- picked_up_at(p,0) and its matching request atom are 2 hops apart (via
        object 0), not 1 -- so this encoding DOES still collide at L=1 (not enough
        rounds for the distinguishing signal to arrive), and only stops colliding from
        L=2 onward. This is itself a result, not a bug: object-atom needs strictly more
        message-passing depth than atom encoding to resolve the exact same join, exactly
        matching compute_hop_distances()'s independently-computed hop counts.
    """
    location_graph = build_path_location_graph(4)  # 0-1-2-3; origins {0,1}, dests {2,3}
    rng = np.random.RandomState(0)
    atoms_a, tables, pairs, boarded, true_dest_a, _lg = generate_extended_world(
        n_passengers=2, n_origins=2, n_destinations=2, crossing_rate=1.0, k=2, rng=rng, location_graph=location_graph,
    )
    assert len(pairs) == 1, "2 passengers at crossing_rate=1.0 must form exactly one crossed pair"
    p, q = pairs[0]
    assert list(boarded.keys()) == [p], "only the pair's first member should have boarded in this snapshot"
    atoms_b, _tables_b, _pairs_b, _boarded_b, true_dest_b = flip_crossed_pairs(
        tables, pairs, boarded, origin_pool=[0, 1], dest_pool=[2, 3], passenger_ids=[p, q], location_graph=location_graph,
    )
    assert frozenset(atoms_a) != frozenset(atoms_b), "flip must produce a genuinely different ground-atom set"
    assert true_dest_a[p] != true_dest_b[p], "the boarded passenger's true destination must actually change under the flip"

    graphs_a = _build_part2_graphs(atoms_a)
    graphs_b = _build_part2_graphs(atoms_b)

    # --- object encoding: literal edge-multiset identity, not just no-collision-at-some-L ---
    obj_a = atoms_to_object_graph_ext(atoms_a)
    obj_b = atoms_to_object_graph_ext(atoms_b)
    edges_a = sorted(zip(obj_a[2][0].tolist(), obj_a[2][1].tolist(), np.argmax(obj_a[1], axis=1).tolist()))
    edges_b = sorted(zip(obj_b[2][0].tolist(), obj_b[2][1].tolist(), np.argmax(obj_b[1], axis=1).tolist()))
    assert obj_a[0].tolist() == obj_b[0].tolist(), "object encoding: node features must be identical"
    assert edges_a == edges_b, "object encoding: edge multiset must be IDENTICAL between state A and state B"

    for level in range(1, MAX_L + 1):
        vocab = {}
        colours_a = wl_colours_per_level(*graphs_a["object"], ext_initial_colours_object, ext_edge_labels_object, vocab, level)[level]
        colours_b = wl_colours_per_level(*graphs_b["object"], ext_initial_colours_object, ext_edge_labels_object, vocab, level)[level]
        assert colour_multiset_hash(colours_a) == colour_multiset_hash(colours_b), f"object encoding must collide at L={level}"

    # --- atom encoding: must NOT collide, already at L=1 (direct atom-atom edge) ---
    init_fn, edge_fn = PART2_ENCODERS["atom"][1], PART2_ENCODERS["atom"][2]
    vocab = {}
    colours_a = wl_colours_per_level(*graphs_a["atom"], init_fn, edge_fn, vocab, 1)[1]
    colours_b = wl_colours_per_level(*graphs_b["atom"], init_fn, edge_fn, vocab, 1)[1]
    assert colour_multiset_hash(colours_a) != colour_multiset_hash(colours_b), "atom encoding must NOT collide, even at L=1"

    # --- object-atom encoding: DOES collide at L=1 (2 hops, not 1 -- see docstring), but
    # must stop colliding from L=2 onward ---
    init_fn, edge_fn = PART2_ENCODERS["object_atom"][1], PART2_ENCODERS["object_atom"][2]
    vocab = {}
    colours_a_by_l = wl_colours_per_level(*graphs_a["object_atom"], init_fn, edge_fn, vocab, MAX_L)
    colours_b_by_l = wl_colours_per_level(*graphs_b["object_atom"], init_fn, edge_fn, vocab, MAX_L)
    assert colour_multiset_hash(colours_a_by_l[1]) == colour_multiset_hash(colours_b_by_l[1]), (
        "object_atom encoding was expected to still collide at L=1 (2 hops between the "
        "picked_up_at fact and its matching request atom, not 1) -- if this now fails, "
        "the hop-distance claim in this function's docstring needs re-checking, not the assert"
    )
    for level in range(2, MAX_L + 1):
        assert colour_multiset_hash(colours_a_by_l[level]) != colour_multiset_hash(colours_b_by_l[level]), (
            f"object_atom encoding must NOT collide from L=2 onward (L={level})"
        )

    if verbose:
        print("Hand-checkable case 1 (2 passengers + maze, fully crossed, only one boards): PASSED")
        print("  object encoding: state A and flipped state B are a literally identical graph -> collides at every L")
        print("  atom encoding: state A and flipped state B are NOT isomorphic -> no collision, even at L=1")
        print("  object_atom encoding: same non-isomorphism, but 2 hops apart -> still collides at L=1, not from L=2 on")


# --- hand-checkable verification, case 2: 3 passengers, NO maze --------------------------

def verify_anchor_case(verbose=True):
    """
    Isolates the higher-arity effect from the maze's contribution (per request): 3
    passengers, NO location graph / adjacent atoms at all. p, q are a crossed pair
    sharing origins {0,1}: p=(0->3,1->4), q=(0->4,1->3); ONLY p boards (at 0, true dest
    3). A third passenger r has its OWN, disjoint origin {2} and references destination 3
    directly -- request(r,2,3) -- anchoring "3" to r's context, which p<->q relabeling
    cannot touch (r never changes) and which a hypothetical combined (p<->q, 3<->4)
    relabeling also cannot preserve (it would need to rewrite r's own atom too, which
    does not change under the flip below).

    EXPECTED RESULT (written before the asserts, per the task): exactly the same
    qualitative split as case 1, achieved without any maze/adjacency atoms at all --
    object encoding collides state A with the flipped state B at every L; atom does not
    collide even at L=1 (direct atom-atom edge); object_atom still collides at L=1 (2
    hops, same reason as case 1) but not from L=2 onward.
    """
    origin_pool = [0, 1, 2]
    dest_pool = [3, 4]
    p, q, r = 5, 6, 7
    passenger_ids = [p, q, r]
    tables = {p: [(0, 3), (1, 4)], q: [(0, 4), (1, 3)], r: [(2, 3)]}
    boarded = {p: 0, r: 2}  # q (the buddy) has not boarded
    pairs = [(p, q)]

    atoms_a, true_dest_a = _atoms_from_tables(origin_pool, dest_pool, passenger_ids, tables, boarded, location_graph=[])
    atoms_b, _tables_b, _pairs_b, _boarded_b, true_dest_b = flip_crossed_pairs(
        tables, pairs, boarded, origin_pool, dest_pool, passenger_ids, location_graph=[],
    )
    assert frozenset(atoms_a) != frozenset(atoms_b)
    assert true_dest_a[p] != true_dest_b[p]

    graphs_a = _build_part2_graphs(atoms_a)
    graphs_b = _build_part2_graphs(atoms_b)

    for level in range(1, MAX_L + 1):
        vocab = {}
        colours_a = wl_colours_per_level(*graphs_a["object"], ext_initial_colours_object, ext_edge_labels_object, vocab, level)[level]
        colours_b = wl_colours_per_level(*graphs_b["object"], ext_initial_colours_object, ext_edge_labels_object, vocab, level)[level]
        assert colour_multiset_hash(colours_a) == colour_multiset_hash(colours_b), f"object encoding must collide at L={level}"

    init_fn, edge_fn = PART2_ENCODERS["atom"][1], PART2_ENCODERS["atom"][2]
    vocab = {}
    colours_a = wl_colours_per_level(*graphs_a["atom"], init_fn, edge_fn, vocab, 1)[1]
    colours_b = wl_colours_per_level(*graphs_b["atom"], init_fn, edge_fn, vocab, 1)[1]
    assert colour_multiset_hash(colours_a) != colour_multiset_hash(colours_b), "atom encoding must NOT collide, even at L=1"

    init_fn, edge_fn = PART2_ENCODERS["object_atom"][1], PART2_ENCODERS["object_atom"][2]
    vocab = {}
    colours_a_by_l = wl_colours_per_level(*graphs_a["object_atom"], init_fn, edge_fn, vocab, MAX_L)
    colours_b_by_l = wl_colours_per_level(*graphs_b["object_atom"], init_fn, edge_fn, vocab, MAX_L)
    assert colour_multiset_hash(colours_a_by_l[1]) == colour_multiset_hash(colours_b_by_l[1])
    for level in range(2, MAX_L + 1):
        assert colour_multiset_hash(colours_a_by_l[level]) != colour_multiset_hash(colours_b_by_l[level]), (
            f"object_atom encoding must NOT collide from L=2 onward (L={level})"
        )

    if verbose:
        print("Hand-checkable case 2 (3 passengers, anchor, NO maze): PASSED")
        print("  same split as case 1, confirming the effect is the higher-arity join, not the maze")


# --- sweep ---------------------------------------------------------------------------------

def run_part2_world_pair(n_passengers, n_origins, n_destinations, crossing_rate, k, rng, world_id, location_graph):
    """Builds one world's State A (+ its flipped State B, if it has >=1 crossed pair) as
    collision_table()-ready state dicts, both tagged with `world_id` (see
    collision_table's `same_world_only`)."""
    atoms_a, tables, pairs, boarded, true_dest_a, _lg = generate_extended_world(
        n_passengers, n_origins, n_destinations, crossing_rate, k, rng, location_graph=location_graph,
    )
    origin_pool = list(range(n_origins))
    dest_pool = list(range(n_origins, n_origins + n_destinations))
    passenger_ids = list(range(n_origins + n_destinations, n_origins + n_destinations + n_passengers))

    out = [{
        "world_id": world_id, "atoms": frozenset(atoms_a),
        "graphs": _build_part2_graphs(atoms_a), "answer": frozenset(true_dest_a.items()),
    }]
    if pairs:
        atoms_b, _tables_b, _pairs_b, _boarded_b, true_dest_b = flip_crossed_pairs(
            tables, pairs, boarded, origin_pool, dest_pool, passenger_ids, location_graph,
        )
        out.append({
            "world_id": world_id, "atoms": frozenset(atoms_b),
            "graphs": _build_part2_graphs(atoms_b), "answer": frozenset(true_dest_b.items()),
        })
    return out, pairs


def run_part2_sweep(crossing_rates, n_passengers=10, n_origins=6, n_destinations=6, k=2, worlds_per_point=30, seed=0, max_l=MAX_L, verbose=True):
    """
    For each crossing_rate: generates `worlds_per_point` independent worlds (each with
    its OWN fresh random location graph/maze, per build_random_location_graph -- see
    this module's correctness note above for why a maze is needed at all), + each
    world's flipped counterpart when it has >=1 crossed pair, and reports, per
    (convention, L):
      - pair_collision_rate: PRIMARY metric -- the fraction of worlds (that had >=1
        crossed pair) whose State A directly collides with its own flipped State B
      - pooled: collision_table() run over the whole 2*worlds_per_point-state corpus
        (same_world_only=True, so a same-world A/B collision is exactly the pair check
        above, and any OTHER, unintended collision is reported honestly rather than
        silently mixed in)
    """
    rng = np.random.RandomState(seed)
    sweep = {}
    for cr in crossing_rates:
        if verbose:
            print(f"  crossing_rate={cr:.2f} ...")
        states = []
        n_worlds_with_pair = 0
        matched_pair_collides = {conv: {level: 0 for level in range(1, max_l + 1)} for conv in PART2_CONVENTIONS}

        for w in range(worlds_per_point):
            location_graph = build_random_location_graph(n_origins + n_destinations, rng)
            world_states, pairs = run_part2_world_pair(n_passengers, n_origins, n_destinations, cr, k, rng, world_id=w, location_graph=location_graph)
            states.extend(world_states)
            if pairs and len(world_states) == 2:
                n_worlds_with_pair += 1
                for convention in PART2_CONVENTIONS:
                    init_fn, edge_fn = PART2_ENCODERS[convention][1], PART2_ENCODERS[convention][2]
                    vocab = {}
                    colours_a = wl_colours_per_level(*world_states[0]["graphs"][convention], init_fn, edge_fn, vocab, max_l)
                    colours_b = wl_colours_per_level(*world_states[1]["graphs"][convention], init_fn, edge_fn, vocab, max_l)
                    for level in range(1, max_l + 1):
                        if colour_multiset_hash(colours_a[level]) == colour_multiset_hash(colours_b[level]):
                            matched_pair_collides[convention][level] += 1

        pooled = {}
        for convention in PART2_CONVENTIONS:
            init_fn, edge_fn = PART2_ENCODERS[convention][1], PART2_ENCODERS[convention][2]
            pooled[convention] = collision_table(states, convention, init_fn, edge_fn, max_l=max_l, same_world_only=True)

        sweep[cr] = {
            "n_worlds_with_pair": n_worlds_with_pair,
            "pair_collision_rate": {
                conv: {level: (matched_pair_collides[conv][level] / n_worlds_with_pair if n_worlds_with_pair else None)
                       for level in range(1, max_l + 1)}
                for conv in PART2_CONVENTIONS
            },
            "pooled": pooled,
        }
    return sweep


def print_part2_sweep(sweep, title, max_l=MAX_L):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    header = f"{'crossing_rate':>13} {'convention':<12} {'L':>2} {'pair_collision_rate':>20} {'pooled_collisions':>18} {'pooled_bad':>10} {'cross_world_incomparable':>24}"
    print(header)
    print("-" * len(header))
    for cr, row in sweep.items():
        for convention in PART2_CONVENTIONS:
            for level in range(1, max_l + 1):
                rate = row["pair_collision_rate"][convention][level]
                rate_str = f"{rate:.2f}" if rate is not None else "n/a"
                pooled = row["pooled"][convention][level]
                print(
                    f"{cr:>13.2f} {convention:<12} {level:>2} {rate_str:>20} {pooled['collisions']:>18} "
                    f"{pooled['bad_collisions']:>10} {pooled['cross_world_incomparable']:>24}"
                )
    print()


def print_hop_distances():
    location_graph = build_path_location_graph(4)
    rng = np.random.RandomState(0)
    atoms, _tables, pairs, _boarded, _true_dest, _lg = generate_extended_world(
        n_passengers=2, n_origins=2, n_destinations=2, crossing_rate=1.0, k=2, rng=rng, location_graph=location_graph,
    )
    p, _q = pairs[0]
    hops, info = compute_hop_distances(atoms, p)
    print()
    print("=" * 88)
    print("Hop distance: destination location node -> picked_up_at fact (hand-checkable case)")
    print("=" * 88)
    print(f"(board_loc={info['board_loc']}, correct dest={info['dest']}, for passenger {p})")
    for convention in PART2_CONVENTIONS:
        print(f"  {convention:<12}: {hops[convention]} hop(s)")
    print()


# ==========================================================================================
# main
# ==========================================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small, fast smoke-test run instead of the full ~5000-state Part 1 run")
    args = parser.parse_args()

    target_total = 200 if args.quick else 5000
    worlds_per_point = 5 if args.quick else 30

    t_start = time.time()

    _states1, report1 = run_part1(target_total=target_total)
    print_part1_report(report1)

    print("Running hand-checkable verification (Part 2) ...")
    verify_hand_checkable_case()
    verify_anchor_case()
    print_hop_distances()

    print("Running Part 2 sweep A: crossing_rate in [0.0 .. 1.0] (n_passengers=10, n_origins=6, n_destinations=6, k=2) ...")
    sweep_a = run_part2_sweep(
        crossing_rates=[round(x * 0.1, 1) for x in range(11)],
        n_passengers=10, n_origins=6, n_destinations=6, k=2, worlds_per_point=worlds_per_point,
    )
    print_part2_sweep(sweep_a, "PART 2 sweep A -- varying crossing_rate (n_passengers=10, n_origins=6, n_destinations=6, k=2)")

    print("Running Part 2 sweep B: requests_per_passenger k in {2,3,4} at crossing_rate=1.0 ...")
    sweep_b = {}
    for k in (2, 3, 4):
        sweep_b[f"k={k}"] = run_part2_sweep(
            crossing_rates=[1.0], n_passengers=10, n_origins=max(6, k), n_destinations=max(6, k), k=k, worlds_per_point=worlds_per_point,
        )[1.0]
    print()
    print("=" * 100)
    print("PART 2 sweep B -- varying requests_per_passenger k, crossing_rate=1.0")
    print("=" * 100)
    for label, row in sweep_b.items():
        print(f"-- {label} --")
        print_part2_sweep({1.0: row}, f"({label})")

    print("Running Part 2 sweep C: n_passengers in {4,10,20} at crossing_rate=0.5, k=2 ...")
    sweep_c = {}
    for n_pass in (4, 10, 20):
        sweep_c[f"n_passengers={n_pass}"] = run_part2_sweep(
            crossing_rates=[0.5], n_passengers=n_pass, n_origins=6, n_destinations=6, k=2, worlds_per_point=worlds_per_point,
        )[0.5]
    print()
    print("=" * 100)
    print("PART 2 sweep C -- varying n_passengers, crossing_rate=0.5, k=2")
    print("=" * 100)
    for label, row in sweep_c.items():
        print(f"-- {label} --")
        print_part2_sweep({0.5: row}, f"({label})")

    print(f"\nTotal elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
