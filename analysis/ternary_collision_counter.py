"""
.. module:: ternary_collision_counter
   :synopsis: Re-runs analysis/indistinguishability.py's PART 2 bad-collision counter,
   but on states sampled from the REAL TernaryTaxiWorldSimulator (this branch's
   ternary_taxi_world.py) instead of that script's standalone synthetic generator
   (generate_extended_world). No simulator/env/planner code is imported or changed --
   this is read-only, analysis-only, exactly like indistinguishability.py itself.

   Reuses, unmodified:
     - analysis/indistinguishability.py's collision_table/PART2_ENCODERS/EXT_PREDICATES
       machinery and its generalised (arity-3) object/atom/object_atom encoders --
       "the counter's own encoders", per the task; the real production converters for
       this domain don't exist yet.
     - tests/test_ternary_world.py's scripted_policy/instrument/crossed_state helpers
       (Part 1's coverage sweep already established these produce real deliveries and
       real crossed states within a couple thousand steps, unlike pure random play).

   New in this file: facts_to_ext_atoms(), the adapter from
   TernaryTaxiWorldSimulator.facts()'s 7-predicate format (taxi, location, passenger,
   adjacent, in, request, picked_up_at) to indistinguishability.py's PART 2 atom format
   (location, passenger, adjacent, request, picked_up_at -- contiguous object ids,
   locations then passengers, NO taxi object) -- see facts_to_ext_atoms's docstring for
   why `taxi`/`in` facts have no equivalent there and are dropped.

   TWO separate experiments are run (see main() -- both matter, and they answer
   different questions):

   1. collision_table over ~2000 INDEPENDENTLY sampled real states (mirrors Part 1's
      own methodology exactly). Result: zero collisions at every L, every convention --
      in sharp contrast to Part 1's huge L=1 collision counts on the CURRENT domain.
      This is a real, explainable finding, not a bug: Part 1's states differ mainly by
      taxi position on an otherwise-static scene (many near-duplicates -> lots of
      low-L collisions); this adapter drops the taxi/`in` facts entirely (the PART 2
      atom format has no taxi concept -- see facts_to_ext_atoms), so two "distinct"
      ternary states here differ only in which specific few (of 400) locations carry
      request/picked_up_at facts -- a huge, sparse combinatorial space extremely
      unlikely to coincide by chance, even at L=1. Independently-sampled real states
      essentially never collide under ANY encoding, so this style of sampling cannot,
      by itself, reproduce or refute the prototype's claim.
   2. flip_pair_collision_rates on REAL crossed states (buddy still present) paired
      with their flipped counterfactual (build_flipped_facts -- the exact same "State
      A vs its own flipped State B" construction indistinguishability.py's PART 2
      prototype uses, sourced from real simulator data instead of the synthetic
      generator). THIS is the direct, faithful comparison against the prototype's
      100%/never/L=1-only claim.

Run from the repo root with:
    PYTHONPATH=. python analysis/ternary_collision_counter.py
"""
import time

import numpy as np
import networkx as nx
import random as pyrandom

from analysis.indistinguishability import (
    LOCATION,
    PASSENGER,
    ADJACENT,
    REQUEST,
    PICKED_UP_AT,
    PART2_CONVENTIONS,
    colour_multiset_hash,
    wl_colours_per_level,
    PART2_ENCODERS,
    _build_part2_graphs,
    collision_table,
)
from sage.domains.gym_taxi.simulator.ternary_taxi_world import TernaryTaxiWorldSimulator
from sage.domains.gym_taxi.utils.config import CITY_TERNARY
from tests.test_ternary_world import scripted_policy, crossed_state


def facts_to_ext_atoms(facts):
    """
    Adapts TernaryTaxiWorldSimulator.facts() into indistinguishability.py's PART 2 atom
    format. Two structural differences to bridge:

    1. facts() ids are NOT contiguous from 0: taxi is id 0, locations are 1..size*size,
       passengers follow after. The PART 2 encoders (atoms_to_object_graph_ext and
       friends) instead assume the FIRST n_obj entries of the atoms list, in order, ARE
       the location/passenger type atoms with ids 0..n_obj-1 (see
       generate_extended_world's own docstring: "contiguous 0..n_obj-1 for locations
       ... so atom-row k really is object k") -- so every object id is remapped here,
       locations first (ascending), then passengers (ascending).
    2. The PART 2 generator has no notion of the taxi as an object, or of a waiting
       passenger's/the taxi's current location as a fact -- boarding status
       (picked_up_at) is the only positional information it tracks. `taxi` and `in`
       facts have no equivalent there and are dropped; every other predicate maps
       straight across.

    :param facts: sim.facts() output (see ternary_taxi_world.py)
    :return: list of (predicate, args_tuple) in the layout
        atoms_to_object_graph_ext/atoms_to_graph_ext/atoms_to_object_atom_graph_ext
        require: locations then passengers first (contiguous ids), relational facts
        after
    """
    locations = sorted(args[0] for predicate, args in facts if predicate == "location")
    passengers = sorted(args[0] for predicate, args in facts if predicate == "passenger")
    mapping = {old: new for new, old in enumerate(locations + passengers)}

    atoms = [(LOCATION, (mapping[l],)) for l in locations]
    atoms += [(PASSENGER, (mapping[p],)) for p in passengers]
    for predicate, args in facts:
        if predicate == "adjacent":
            atoms.append((ADJACENT, tuple(mapping[a] for a in args)))
        elif predicate == "request":
            pid, origin, destination = args
            atoms.append((REQUEST, (mapping[pid], mapping[origin], mapping[destination])))
        elif predicate == "picked_up_at":
            pid, location = args
            atoms.append((PICKED_UP_AT, (mapping[pid], mapping[location])))
        # "taxi" and "in" facts: no equivalent in the PART 2 atom vocabulary -- dropped.
    return atoms


def approx_optimal_action_ternary(sim):
    """
    Same shape and intent as indistinguishability.py's approx_optimal_action, adapted
    for the ternary domain's destination resolution: if carrying a passenger, resolve
    their destination via the picked_up_at/request join (the ONLY way to do it, exactly
    as _attempt_dropoff does) rather than reading a stored .destination; otherwise, head
    for the nearest waiting passenger. Reads sim internals directly (ground truth for
    bad-collision detection is allowed to see what the encodings can't) -- an oracle,
    not a candidate encoding.

    :return: (kind, hop_distance) -- see approx_optimal_action's docstring for the
        four possible `kind` values
    """
    road_graph = nx.Graph()
    road_graph.add_edges_from(sim.roads.edges())

    if sim.taxi.passenger is not None:
        pid = sim.taxi.passenger
        passenger = sim.passengers[pid]
        matches = [d for o, d in passenger.requests if o == passenger.picked_up_at]
        assert len(matches) == 1, f"passenger {pid}: expected exactly one destination match, got {matches}"
        dest = matches[0]
        if sim.taxi.location == dest:
            return ("dropoff", 0)
        return ("move_to_dropoff", nx.shortest_path_length(road_graph, sim.taxi.location, dest))

    waiting = {pid: p.location for pid, p in sim.passengers.items() if p.location is not None}
    if waiting:
        pid = min(waiting, key=lambda p: nx.shortest_path_length(road_graph, sim.taxi.location, waiting[p]))
        loc = waiting[pid]
        if sim.taxi.location == loc:
            return ("pickup", 0)
        return ("move_to_pickup", nx.shortest_path_length(road_graph, sim.taxi.location, loc))
    return ("idle", 0)


def generate_states_from_real_sim(seeds=(0, 1, 2, 3, 4), sample_every=5, target_total=2000, random_prob=0.2, verbose=True):
    """
    Samples states from real TernaryTaxiWorldSimulator episodes, driven by
    scripted_policy (imported from tests/test_ternary_world.py, unmodified) so
    passengers actually get delivered and crossed states actually occur -- Part 1's
    coverage sweep already established pure random play doesn't (0 successful dropoffs
    across 3 seeds x 2000 steps).

    Mirrors generate_part1_states's shape and sampling cadence (sample every
    `sample_every` steps, ~target_total states spread evenly over `seeds`) so the two
    reports are directly comparable.

    :return: list of dicts: {"seed", "step", "atoms" (frozenset of the ADAPTED ext
        atoms, for state identity/dedup), "graphs" (dict: convention -> (x, edge_index,
        edge_attr)), "answer" (approx_optimal_action_ternary(...)), "crossed" (bool,
        crossed_state(sim) at sample time)}
    """
    per_seed = max(1, target_total // len(seeds))
    states = []
    n_crossed_decision_states = 0
    n_ambiguous_object_destination = 0
    n_carried_decision_states = 0
    for seed in seeds:
        t0 = time.time()
        sim = TernaryTaxiWorldSimulator(
            np.random.RandomState(seed), observation_fn=lambda s: None, **CITY_TERNARY
        )
        policy_rng = pyrandom.Random(seed + 5000)
        step = 0
        collected = 0
        while collected < per_seed:
            if step % sample_every == 0:
                facts = sim.facts()
                ext_atoms = facts_to_ext_atoms(facts)
                states.append({
                    "seed": seed, "step": step,
                    "atoms": frozenset(ext_atoms),
                    "graphs": _build_part2_graphs(ext_atoms),
                    "answer": approx_optimal_action_ternary(sim),
                })
                if sim.taxi.passenger is not None:
                    n_carried_decision_states += 1
                    if crossed_state(sim):
                        n_crossed_decision_states += 1
                    if _object_destination_is_ambiguous(facts, sim.taxi.passenger):
                        n_ambiguous_object_destination += 1
                collected += 1
            action = scripted_policy(sim, policy_rng, random_prob=random_prob)
            sim.act(action)
            step += 1
            if sim.done:
                sim = TernaryTaxiWorldSimulator(
                    np.random.RandomState(seed * 100003 + step), observation_fn=lambda s: None, **CITY_TERNARY
                )
        if verbose:
            print(f"  seed={seed}: {collected} states sampled over {step} steps ({time.time() - t0:.1f}s)")

    coverage = {
        "n_carried_decision_states": n_carried_decision_states,
        "n_crossed_decision_states": n_crossed_decision_states,
        "n_ambiguous_object_destination": n_ambiguous_object_destination,
        "fraction_crossed": n_crossed_decision_states / n_carried_decision_states if n_carried_decision_states else float("nan"),
        "fraction_ambiguous": n_ambiguous_object_destination / n_carried_decision_states if n_carried_decision_states else float("nan"),
    }
    return states, coverage


def _object_destination_is_ambiguous(facts, carried_pid):
    """
    From the OBJECT ENCODING's own point of view (per the task): candidate
    destinations for the carried passenger are every location d such that there is a
    req(1,3) edge from carried_pid to d (i.e. SOME request(carried_pid, o, d)) AND a
    req(2,3) edge from picked_up_at to d for the SAME request (i.e. o ==
    picked_up_at(carried_pid)) -- but object encoding only keeps object-object
    co-occurrence, typed by predicate, with NO argument-position information, so it
    cannot tell which request(carried_pid, o, d) triple an edge came from. Concretely:
    object encoding sees "carried_pid is request-connected to d" for EVERY d appearing
    in ANY of carried_pid's own requests, not just the one matching picked_up_at, and
    "picked_up_at's location is request-connected to d" for every d that is the
    destination of ANY passenger's request FROM that origin (not just carried_pid's) --
    so a location is a false candidate if it is graph-reachable via those two request
    hops regardless of whose request supplied them. This reproduces that confusion
    directly from facts() (not from actually building the object graph), which is
    simpler and exactly equivalent since object encoding's edges are just "objects
    co-occurring in a request/picked_up_at atom, predicate-typed, position discarded".

    :return: True if more than one destination candidate survives this join
    """
    board = next(args[1] for predicate, args in facts if predicate == "picked_up_at" and args[0] == carried_pid)
    # req(1,3): destinations of ANY request atom carried_pid appears in as the subject
    pid_linked_destinations = {args[2] for predicate, args in facts if predicate == "request" and args[0] == carried_pid}
    # req(2,3): destinations of ANY request atom whose origin equals picked_up_at,
    # from ANY passenger's table -- object encoding cannot restrict this to
    # carried_pid's own table, since it has no argument-position information at all.
    board_linked_destinations = {args[2] for predicate, args in facts if predicate == "request" and args[1] == board}
    candidates = pid_linked_destinations & board_linked_destinations
    return len(candidates) > 1


def build_flipped_facts(sim):
    """
    The real-data equivalent of indistinguishability.py's flip_crossed_pairs: for the
    taxi's currently-carried passenger p and its buddy q (crossed_state(sim) must hold),
    EXCHANGES p's and q's entire request tables, leaving everything else -- locations,
    adjacency, every passenger/picked_up_at fact including p's own -- unchanged. p's
    picked_up_at fact stays exactly as it is (p is still the one who boarded, at the
    same location); only which destination that boarding location now resolves to (via
    q's swapped-in table) changes. This is a genuinely DIFFERENT ground-atom set
    whenever k >= 2 (some request fact differs), with a DIFFERENT true destination for
    p -- exactly the "State A vs State B" comparison the prototype demonstrates,
    sourced from a real simulator state instead of the synthetic generator.

    Requires p and q to share the same origin set (true by construction -- rule 2 /
    spawn_pair's rotation), so p's own board location remains a valid origin in q's
    (now p's) table after the swap -- the join in _attempt_dropoff-style logic would
    still resolve to SOME destination, just a different one.

    :param sim: a TernaryTaxiWorldSimulator with crossed_state(sim) True
    :return: (p, q, flipped_facts) -- flipped_facts in the same format as sim.facts()
    """
    p = sim.taxi.passenger
    q = sim.passengers[p].buddy
    assert q is not None and q in sim.passengers, "build_flipped_facts requires a crossed state"
    p_requests = sim.passengers[p].requests
    q_requests = sim.passengers[q].requests

    flipped = [
        (predicate, args)
        for predicate, args in sim.facts()
        if not (predicate == "request" and args[0] in (p, q))
    ]
    flipped.extend(("request", (p, o, d)) for o, d in q_requests)
    flipped.extend(("request", (q, o, d)) for o, d in p_requests)
    return p, q, flipped


def sample_real_crossed_flip_pairs(seeds=(0, 1, 2, 3, 4), sample_every=5, per_seed_target=40, random_prob=0.2, verbose=True):
    """
    Samples REAL crossed decision states from live TernaryTaxiWorldSimulator episodes
    and pairs each with its flipped counterfactual (build_flipped_facts) -- the
    real-data analogue of the prototype's State-A/State-B construction.

    :return: list of (ext_atoms_a, ext_atoms_b) pairs, both already run through
        facts_to_ext_atoms
    """
    pairs = []
    for seed in seeds:
        sim = TernaryTaxiWorldSimulator(
            np.random.RandomState(seed), observation_fn=lambda s: None, **CITY_TERNARY
        )
        policy_rng = pyrandom.Random(seed + 9000)
        step = 0
        collected = 0
        while collected < per_seed_target:
            if step % sample_every == 0 and crossed_state(sim):
                _p, _q, flipped_facts = build_flipped_facts(sim)
                pairs.append((facts_to_ext_atoms(sim.facts()), facts_to_ext_atoms(flipped_facts)))
                collected += 1
            action = scripted_policy(sim, policy_rng, random_prob=random_prob)
            sim.act(action)
            step += 1
            if sim.done:
                sim = TernaryTaxiWorldSimulator(
                    np.random.RandomState(seed * 100003 + step), observation_fn=lambda s: None, **CITY_TERNARY
                )
        if verbose:
            print(f"  seed={seed}: {collected} real crossed-state flip pairs sampled over {step} steps")
    return pairs


def flip_pair_collision_rates(pairs, max_l=5):
    """
    For each convention, the fraction of (state, flipped-state) pairs whose WL
    colour-multiset hash MATCHES (i.e. collides) at each L = 1..max_l -- the direct
    measurement to compare against the prototype's "object 100% at every L; atom never;
    object_atom L=1 only" claim. One shared growing vocab per convention across every
    pair (both members), same rationale as collision_table's own shared vocab.

    :param pairs: list of (ext_atoms_a, ext_atoms_b), as sample_real_crossed_flip_pairs
        returns
    :return: {convention: {L: fraction_matching}}
    """
    vocabs = {convention: {} for convention in PART2_CONVENTIONS}
    matches = {convention: {level: 0 for level in range(1, max_l + 1)} for convention in PART2_CONVENTIONS}
    for atoms_a, atoms_b in pairs:
        graphs_a = _build_part2_graphs(atoms_a)
        graphs_b = _build_part2_graphs(atoms_b)
        for convention, (_builder, init_fn, edge_fn) in PART2_ENCODERS.items():
            levels_a = wl_colours_per_level(*graphs_a[convention], init_fn, edge_fn, vocabs[convention], max_l)
            levels_b = wl_colours_per_level(*graphs_b[convention], init_fn, edge_fn, vocabs[convention], max_l)
            for level in range(1, max_l + 1):
                if colour_multiset_hash(levels_a[level]) == colour_multiset_hash(levels_b[level]):
                    matches[convention][level] += 1

    n = len(pairs)
    return {
        convention: {level: (matches[convention][level] / n if n else float("nan")) for level in range(1, max_l + 1)}
        for convention in PART2_CONVENTIONS
    }


def print_report(table, title):
    print(f"\n{title}")
    print(f"{'convention':<12} {'L':>2} {'distinct':>9} {'hashes':>8} {'collisions':>11} {'bad':>6}")
    for convention in PART2_CONVENTIONS:
        for level, row in table[convention].items():
            print(
                f"{convention:<12} {level:>2} {row['distinct_states']:>9} {row['distinct_hashes']:>8} "
                f"{row['collisions']:>11} {row['bad_collisions']:>6}"
            )


def main():
    print("Sampling states from real TernaryTaxiWorldSimulator episodes (CITY_TERNARY, "
          "scripted_policy 80% / random 20%) ...")
    states, coverage = generate_states_from_real_sim()
    print(f"\n{len(states)} raw snapshots -> {len(set(s['atoms'] for s in states))} distinct states")

    table = {
        convention: collision_table(states, convention, init_fn, edge_fn)
        for convention, (_builder, init_fn, edge_fn) in PART2_ENCODERS.items()
    }
    print_report(table, "Bad-collision table -- states sampled from the REAL TernaryTaxiWorldSimulator")

    print("\nCoverage over carried-passenger decision states:")
    print(f"  carried-passenger decision states sampled: {coverage['n_carried_decision_states']}")
    print(f"  of those, buddy still present (crossed):   {coverage['n_crossed_decision_states']} "
          f"({coverage['fraction_crossed']:.1%})")
    print(f"  of those, object-encoding destination is "
          f"ambiguous: {coverage['n_ambiguous_object_destination']} ({coverage['fraction_ambiguous']:.1%})")

    print("\nSampling REAL crossed states + flipped counterfactuals (the prototype's own "
          "State-A/State-B construction, sourced from real simulator data) ...")
    pairs = sample_real_crossed_flip_pairs()
    print(f"\n{len(pairs)} real crossed-state / flipped-counterfactual pairs")

    rates = flip_pair_collision_rates(pairs)
    print("\nFraction of (real state, flipped counterfactual) pairs that COLLIDE, by L "
          "-- compare with the prototype's 100% / never / L=1-only:")
    print(f"{'convention':<12}" + "".join(f"L={level:<8}" for level in range(1, 6)))
    for convention in PART2_CONVENTIONS:
        row = rates[convention]
        print(f"{convention:<12}" + "".join(f"{row[level]:<10.0%}" for level in range(1, 6)))


if __name__ == "__main__":
    main()
