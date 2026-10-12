"""
.. module:: ternary_collision_production
   :synopsis: Buddy-pair WL collisions on the ternary domain, measured with the
   PRODUCTION converters, planner and wl_colours(domain="ternary") -- the follow-up to
   analysis/ternary_collision_counter.py, whose prototype encoders dropped the taxi and
   `in` facts, matched at graph level only, and used a growing vocab only.

States: build_wl_vocab.ternary_corpus (the corpus design the vocabs were built from) on
SEEDS, disjoint from the vocab seed (0) and the held-out OOV seeds (1001-1005): two full
episodes per seed (scripted_policy with random_prob 0.2, then uniform random legal
actions), a live state every SAMPLE_EVERY steps plus 2 planner projections per state
(the corpus' own goals). The trajectory does not depend on the convention, so it is
sampled once and every state is encoded for all three conventions through the policy-side
converters (facts -> facts_to_json -> json_to_ternary_graph_*); projections go through
each convention's own Planner(ternary=True).

Buddy-pair identity is not a fact; it is read from the simulator (analysis only).

Counterfactual of a crossed state (carried p, buddy q still waiting): the same state with
p's and q's request tables exchanged, built by swapping the two TernaryPassenger.requests
on a copy of the simulator and calling its own facts() (canonical order). It is checked
against analysis/ternary_collision_counter.build_flipped_facts (same fact set) and proven
NOT isomorphic to the original by an independent relational invariant (see
nonisomorphism_proof) -- with networkx VF2 as the fallback.

Metrics, per convention x L (growing vocab: L = 1, 2, 3; frozen vocab: L = 1, 2):
  M1 crossed pairs, graph level: the final-colour histograms of the state and its
     counterfactual are identical (what the critic / path_value_net read).
  M2 crossed pairs, node level: the carried passenger p has the same final colour in
     both (what the actor reads for p).
  M3 both buddies waiting: the two passengers have the same colour within the state.
  M4 M1 and M2 restricted to pairs whose correct next delivery destination (p's
     destination by the picked_up_at/request join) differs between the two -- for k = 2
     it always does; the script reports any pair where it does not.
  M5 frozen vocab: collisions under the frozen vocab that the growing vocab does not have
     (caused by OOV).
Projections enter M1/M2 when the projection keeps the pair (goal not p or q): the
projection of the state vs the projection of the counterfactual towards the same goal;
and M3 for every both-waiting pair the projection keeps. Passenger ids are tracked
through a projected delivery's renumbering.

One shared growing vocab per (convention, L) across every graph, as in the prototype, so
equal colours mean WL-equivalent nodes (equal histograms: WL-equivalent graphs).

Checks (run by main, written before the results were seen):
  - object M1 must be 100% at every L: the object encodings of a crossed state and its
    counterfactual are identical tensors (counted and asserted);
  - SPOT_CHECKS random collisions (and as many non-collisions) per reported cell are
    re-derived with the independent reference WL of tests/test_wl_ternary.py over the
    disjoint union of the two graphs; a frozen-only (M5) collision must be NOT equivalent
    under the reference and involve an OOV colour.
tests/test_ternary_collision_production.py runs the hand-built cases of
tests/test_wl_ternary.py through the functions below.

Comparison with the published prototype table (prototype_comparison): M1 with the
prototype's own encoders, with the production converters on facts stripped of the taxi
and `in` facts, and with the full production converters -- on this script's crossed
pairs and on an exact replay of the prototype's own 200-state sample
(published_sample_pairs), so differences can be attributed to the state sample, to
encoder details, or to the taxi/`in` facts.

Run from the repo root with:
    PYTHONPATH=. python analysis/ternary_collision_production.py --out results.json
"""
import argparse
import copy
import json
import random as pyrandom
import time
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import torch as th
from networkx.algorithms.isomorphism import GraphMatcher

from analysis.ternary_collision_counter import (
    build_flipped_facts,
    facts_to_ext_atoms,
    flip_pair_collision_rates,
    sample_real_crossed_flip_pairs,
)
from sage.domains.gym_taxi.simulator.ternary_taxi_world import TernaryTaxiWorldSimulator
from sage.domains.gym_taxi.utils.config import CITY_TERNARY
from sage.agent.wl_plan_feedback_policy import _load_vocab
from sage.domains.gym_taxi.simulator.planner import Planner
from sage.domains.gym_taxi.simulator.ternary_planner import decode_atom, decode_object, decode_vilg
from sage.domains.gym_taxi.utils.ternary_representations import (
    TERNARY_GRAPH_CONVENTION_CONVERTERS,
    facts_to_json,
    facts_to_oracle_facts,
    oracle_facts_to_facts,
)
from sage.domains.utils import build_wl_vocab
from sage.domains.utils.wl_colours import OOV_SIGNATURE, wl_colours
from tests.test_ternary_world import crossed_state, destination_via_join, scripted_policy

CONVENTIONS = ("oracle_sage", "vilg", "atom")
GROWING_LEVELS = (1, 2, 3)
FROZEN_LEVELS = (1, 2)
SEEDS = (2001, 2002, 2003, 2004, 2005)
EPISODES_PER_SEED = 2
SAMPLE_EVERY = 10
SPOT_CHECKS = 5
DECODERS = {"oracle_sage": decode_object, "vilg": decode_vilg, "atom": decode_atom}


# ==========================================================================================
# States
# ==========================================================================================

def buddy_pairs(sim):
    """Live buddy pairs (a, b), a < b, from the simulator's own bookkeeping."""
    pairs = set()
    for pid, passenger in sim.passengers.items():
        if passenger.buddy is not None and passenger.buddy in sim.passengers:
            pairs.add(tuple(sorted((pid, passenger.buddy))))
    return sorted(pairs)


def counterfactual_sim(sim, p, q):
    """A copy of `sim` with p's and q's request tables exchanged -- nothing else changes.
    facts() of the copy is the counterfactual in the simulator's own canonical order."""
    cf = copy.copy(sim)
    cf.passengers = dict(sim.passengers)
    cf.passengers[p] = sim.passengers[p]._replace(requests=sim.passengers[q].requests)
    cf.passengers[q] = sim.passengers[q]._replace(requests=sim.passengers[p].requests)
    return cf


def sample_states(seeds=SEEDS, episodes=EPISODES_PER_SEED, sample_every=SAMPLE_EVERY, max_steps=2500, log=True):
    """
    Plays ternary_corpus once (its convention only decides how it encodes; the trajectory
    and goals are the same for every convention) and records, per live state: its facts
    (compact int32 array), meta, the goals of its projections, the buddy pairs, and -- for
    a crossed state -- the counterfactual's facts, both true destinations, and whether
    build_flipped_facts agrees with the counterfactual.
    """
    states = []
    for seed in seeds:
        start = time.time()
        for sample in build_wl_vocab.ternary_corpus(episodes, sample_every, seed, "vilg", max_steps=max_steps):
            if sample.goal is not None:
                states[-1]["goals"].append(sample.goal)
                continue
            sim = sample.env.sim
            facts = sim.facts()
            carried = sim.taxi.passenger
            pairs = buddy_pairs(sim)
            state = {
                "seed": seed, "episode": sample.episode, "step": sample.step,
                "facts": facts_to_oracle_facts(facts).astype(np.int32),
                "meta": {"time": sim.time, "timeout": sim.timeout, "planning": sim.planning},
                "goals": [],
                "waiting_pairs": [pair for pair in pairs if carried not in pair],
                "crossed": None,
            }
            crossed = [pair for pair in pairs if carried in pair]
            if crossed:
                (a, b), = crossed
                p, q = (a, b) if a == carried else (b, a)
                cf = counterfactual_sim(sim, p, q)
                cf_facts = cf.facts()
                _p, _q, flipped = build_flipped_facts(sim)
                state["crossed"] = {
                    "p": p, "q": q,
                    "pair_id": (seed, sample.episode, sim.passengers[p].requests, sim.passengers[q].requests),
                    "cf_facts": facts_to_oracle_facts(cf_facts).astype(np.int32),
                    "dest": destination_via_join(facts, p),
                    "cf_dest": destination_via_join(cf_facts, p),
                    "flipped_matches": (_p, _q) == (p, q) and sorted(flipped) == sorted(cf_facts),
                    "flipped_canonical_order": flipped == cf_facts,
                    "nonisomorphic": nonisomorphism_proof(facts, cf_facts),
                }
            states.append(state)
        if log:
            print(f"  seed {seed}: {sum(1 for s in states if s['seed'] == seed)} live states ({time.time() - start:.0f}s)",
                  flush=True)
    return states


def as_facts(array):
    return oracle_facts_to_facts(array)


# ==========================================================================================
# Independent non-isomorphism check (no WL, no converter)
# ==========================================================================================

def _role_distances(facts):
    """
    A relational invariant of a crossed state: with T the taxi's location, c the carried
    passenger and D = c's true destination (picked_up_at/request join), the sorted multiset
    of (predicate, argument position, road distance from D) over every location argument
    of every in/request/picked_up_at fact, together with dist(T, D). Built only from the
    facts and the road graph, it is preserved by every isomorphism of fact sets (which must
    map the taxi, the unique carried passenger, and so its destination, to their
    counterparts), so different values PROVE the two states are not isomorphic.
    None when nobody is carried.
    """
    n_obj = 0
    while facts[n_obj][0] in ("taxi", "location", "passenger"):
        n_obj += 1
    roads = nx.Graph()
    roads.add_nodes_from(args[0] for pred, args in facts[:n_obj] if pred == "location")
    roads.add_edges_from(args for pred, args in facts[n_obj:] if pred == "adjacent")
    carried = next((args[0] for pred, args in facts if pred == "in" and args[1] == 0 and args[0] != 0), None)
    if carried is None:
        return None  # no carried passenger: no invariant, VF2 decides
    taxi_location = next(args[1] for pred, args in facts if pred == "in" and args[0] == 0)
    destination = destination_via_join(facts, carried)
    distance = nx.single_source_shortest_path_length(roads, destination)
    locations = set(roads.nodes)
    profile = sorted(
        (pred, position, distance.get(arg, -1))
        for pred, args in facts[n_obj:] if pred in ("in", "request", "picked_up_at")
        for position, arg in enumerate(args) if arg in locations
    )
    return distance.get(taxi_location, -1), profile


def _fact_graph(facts):
    """Facts as a labelled bipartite graph (objects by type, facts by predicate, edges by
    argument position): two fact sets are isomorphic iff these graphs are."""
    g = nx.Graph()
    n_obj = 0
    while facts[n_obj][0] in ("taxi", "location", "passenger"):
        g.add_node(("o", facts[n_obj][1][0]), label=facts[n_obj][0])
        n_obj += 1
    for k, (pred, args) in enumerate(facts[n_obj:]):
        g.add_node(("f", k), label=pred)
        for position, arg in enumerate(args, start=1):
            g.add_edge(("f", k), ("o", arg), position=position)
    return g


def nonisomorphism_proof(facts_a, facts_b):
    """
    :return: "invariant" (proved by _role_distances), "vf2" (networkx VF2 found no
        isomorphism), or "isomorphic" (the counterfactual is a relabelling -- such a pair
        is excluded from M1/M2/M4 and reported).
    """
    if _role_distances(facts_a) != _role_distances(facts_b):
        return "invariant"
    matcher = GraphMatcher(
        _fact_graph(facts_a), _fact_graph(facts_b),
        node_match=lambda x, y: x["label"] == y["label"],
        edge_match=lambda x, y: x["position"] == y["position"],
    )
    return "isomorphic" if matcher.is_isomorphic() else "vf2"


# ==========================================================================================
# Encoding, projection, colouring
# ==========================================================================================

def encode(facts, meta, convention):
    """facts -> the env's JSON -> the policy-side converter: the Data a policy batch holds."""
    return TERNARY_GRAPH_CONVENTION_CONVERTERS[convention]([[facts_to_json(facts, meta)]]).to_data_list()[0]


def passenger_after(pid, goal, passenger_ids):
    """pid's id in the projection towards `goal`: a projected delivery of passenger g
    removes g and renumbers every passenger above it down by one (plan_on_facts'
    _deliver_facts, mirroring TernaryTaxiWorldSimulator._renumber_passengers).
    None if pid itself is delivered."""
    if goal in passenger_ids:
        if pid == goal:
            return None
        if pid > goal:
            return pid - 1
    return pid


def colour(data, convention, level, vocab, frozen):
    colours, histogram = wl_colours(
        data.x, data.edge_index, data.edge_attr, num_iterations=level, vocab=vocab, frozen=frozen,
        graph_convention=convention, domain="ternary",
    )
    nonzero = histogram.nonzero(as_tuple=True)[0]
    return colours, tuple(zip(nonzero.tolist(), histogram[nonzero].tolist()))


def pair_metrics(data_a, data_b, p_a, p_b, convention, level, vocab, frozen=False):
    """(M1, M2) for one pair of graphs: identical final histograms; identical colour of the
    carried passenger (row p_a in data_a, p_b in data_b). Also returns whether any node is
    OOV (frozen mode)."""
    colours_a, hist_a = colour(data_a, convention, level, vocab, frozen)
    colours_b, hist_b = colour(data_b, convention, level, vocab, frozen)
    oov = None
    if frozen:
        oov_id = vocab[OOV_SIGNATURE]
        oov = bool((colours_a == oov_id).any() or (colours_b == oov_id).any())
    return hist_a == hist_b, int(colours_a[p_a]) == int(colours_b[p_b]), oov


def buddies_metric(data, pairs, convention, level, vocab, frozen=False):
    """M3 for every (a, b) in pairs: same final colour within the graph; plus OOV flags."""
    colours, _ = colour(data, convention, level, vocab, frozen)
    oov_id = vocab.get(OOV_SIGNATURE) if frozen else None
    return [
        (int(colours[a]) == int(colours[b]),
         frozen and (int(colours[a]) == oov_id or int(colours[b]) == oov_id))
        for a, b in pairs
    ]


# ==========================================================================================
# The measurement
# ==========================================================================================

def graph_items(state, convention, planner):
    """
    Yields the graph groups of one live state for one convention:
      ("pair", key, data_a, data_b, p_a, p_b, dest_differs)   (M1/M2, crossed only)
      ("single", key, data, waiting_pairs)                     (M3)
    key = (state index set by the caller, goal or None) -- enough to regenerate the graphs.
    """
    facts = as_facts(state["facts"])
    data = encode(facts, state["meta"], convention)
    passenger_ids = {args[0] for pred, args in facts if pred == "passenger"}
    crossed = state["crossed"]
    data_cf = None
    if crossed is not None and crossed["nonisomorphic"] != "isomorphic":
        data_cf = encode(as_facts(crossed["cf_facts"]), state["meta"], convention)
        dest_differs = crossed["dest"] != crossed["cf_dest"]
        yield ("pair", None, data, data_cf, crossed["p"], crossed["p"], dest_differs)
    yield ("single", None, data, state["waiting_pairs"])

    for goal in state["goals"]:
        projection, _ = planner.plan(data, goal)
        if data_cf is not None and goal not in (crossed["p"], crossed["q"]):
            projection_cf, _ = planner.plan(data_cf, goal)
            p_new = passenger_after(crossed["p"], goal, passenger_ids)
            yield ("pair", goal, projection, projection_cf, p_new, p_new, dest_differs)
        kept = []
        for a, b in state["waiting_pairs"]:
            a_new, b_new = passenger_after(a, goal, passenger_ids), passenger_after(b, goal, passenger_ids)
            if a_new is not None and b_new is not None:
                kept.append((a_new, b_new))
        yield ("single", goal, projection, kept)


def measure(states, convention, frozen_vocabs, log=True):
    """
    :param frozen_vocabs: {L: vocab} (may be empty)
    :return: records: {"pairs": [...], "singles": [...]} where each pair record holds
        per-mode results {(mode, L): (m1, m2, oov)} and each single record per-pair
        results {(mode, L): [(m3, oov), ...]}
    """
    growing = {level: {} for level in GROWING_LEVELS}
    planner = Planner(graph_convention=convention, ternary=True)
    pair_records, single_records = [], []
    object_identical = [0, 0]
    start = time.time()
    for index, state in enumerate(states):
        for item in graph_items(state, convention, planner):
            if item[0] == "pair":
                _kind, goal, data_a, data_b, p_a, p_b, dest_differs = item
                if convention == "oracle_sage":
                    object_identical[1] += 1
                    object_identical[0] += all(
                        th.equal(getattr(data_a, n), getattr(data_b, n)) for n in ("x", "edge_index", "edge_attr"))
                results = {}
                for level in GROWING_LEVELS:
                    results[("growing", level)] = pair_metrics(data_a, data_b, p_a, p_b, convention, level, growing[level])
                for level, vocab in frozen_vocabs.items():
                    results[("frozen", level)] = pair_metrics(data_a, data_b, p_a, p_b, convention, level, vocab, True)
                pair_records.append({"state": index, "goal": goal, "p": p_a, "dest_differs": dest_differs,
                                     "results": results})
            else:
                _kind, goal, data, pairs = item
                if not pairs:
                    continue
                results = {}
                for level in GROWING_LEVELS:
                    results[("growing", level)] = buddies_metric(data, pairs, convention, level, growing[level])
                for level, vocab in frozen_vocabs.items():
                    results[("frozen", level)] = buddies_metric(data, pairs, convention, level, vocab, True)
                single_records.append({"state": index, "goal": goal, "pairs": pairs, "results": results})
        if log and (index + 1) % 500 == 0:
            print(f"  {convention}: {index + 1}/{len(states)} states ({time.time() - start:.0f}s)", flush=True)
    return {"pairs": pair_records, "singles": single_records, "object_identical": object_identical}


def summarise(records, frozen_levels):
    """Rates with counts, per (mode, L): M1, M2, M3, M4 (M1/M2 on dest-differs pairs), M5."""
    modes = [("growing", level) for level in GROWING_LEVELS] + [("frozen", level) for level in frozen_levels]
    out = {}
    pairs, singles = records["pairs"], records["singles"]
    for mode in modes:
        m1 = sum(r["results"][mode][0] for r in pairs)
        m2 = sum(r["results"][mode][1] for r in pairs)
        relevant = [r for r in pairs if r["dest_differs"]]
        m3_values = [v for r in singles for v in r["results"][mode]]
        row = {
            "M1": [m1, len(pairs)], "M2": [m2, len(pairs)],
            "M3": [sum(v[0] for v in m3_values), len(m3_values)],
            "M4_M1": [sum(r["results"][mode][0] for r in relevant), len(relevant)],
            "M4_M2": [sum(r["results"][mode][1] for r in relevant), len(relevant)],
            "M1_live": [sum(r["results"][mode][0] for r in pairs if r["goal"] is None),
                        sum(1 for r in pairs if r["goal"] is None)],
            "M2_live": [sum(r["results"][mode][1] for r in pairs if r["goal"] is None),
                        sum(1 for r in pairs if r["goal"] is None)],
            "M3_live": [sum(v[0] for r in singles if r["goal"] is None for v in r["results"][mode]),
                        sum(len(r["results"][mode]) for r in singles if r["goal"] is None)],
        }
        if mode[0] == "frozen":
            g = ("growing", mode[1])
            row["M5_M1"] = [sum(r["results"][mode][0] and not r["results"][g][0] for r in pairs), len(pairs)]
            row["M5_M2"] = [sum(r["results"][mode][1] and not r["results"][g][1] for r in pairs), len(pairs)]
            row["M5_M3"] = [sum(f[0] and not gr[0] for r in singles for f, gr in zip(r["results"][mode], r["results"][g])),
                            len(m3_values)]
            row["pairs_with_oov"] = [sum(bool(r["results"][mode][2]) for r in pairs), len(pairs)]
        out[f"{mode[0]}_L{mode[1]}"] = row
    return out


# ==========================================================================================
# Spot checks against the independent reference WL (tests/test_wl_ternary.py)
# ==========================================================================================

def facts_of(graph, state, goal, convention, cf):
    """The fact set whose encoding IS `graph`: the true facts for a live state, the
    decoded facts for a projection -- verified by re-encoding (exact for vilg/atom; for
    object the decoder's hypothesis reproduces the graph exactly by construction, and the
    re-encoding check enforces it)."""
    if goal is None:
        return as_facts(state["crossed"]["cf_facts"] if cf else state["facts"])
    facts = DECODERS[convention](graph)
    again = encode(facts, state["meta"], convention)
    for name in ("x", "edge_index", "edge_attr"):
        assert th.equal(getattr(again, name), getattr(graph, name)), "decoded facts do not re-encode to the projection"
    return facts


def reference_equivalence(facts_list, convention, level):
    """Reference WL partition at `level` over the disjoint union of the given fact sets'
    reference graphs. Returns (partition, offsets)."""
    from tests.test_wl_ternary import REFERENCE_BUILDERS, reference_wl_partitions
    labels, edges, offsets = [], {}, []
    for facts in facts_list:
        lab, edg = REFERENCE_BUILDERS[convention](facts)
        offset = len(labels)
        offsets.append(offset)
        labels.extend(lab)
        for (a, b), label in edg.items():
            edges[(a + offset, b + offset)] = label
    return reference_wl_partitions(labels, edges, level)[level], offsets, len(labels)


def _same_class(partition, u, v):
    return any(u in c and v in c for c in partition)


def spot_check(states, records, convention, frozen_levels, rng, planner, n=SPOT_CHECKS):
    """For each (mode, L, metric): up to n random collisions and n random non-collisions,
    re-derived with the reference WL. Returns {cell: [checked, agreed]} and a list of
    disagreements (must be empty)."""
    summary, disagreements = defaultdict(lambda: [0, 0]), []
    modes = [("growing", level) for level in GROWING_LEVELS] + [("frozen", level) for level in frozen_levels]

    def regenerate(record):
        state = states[record["state"]]
        facts = as_facts(state["facts"])
        data = encode(facts, state["meta"], convention)
        if record["goal"] is not None:
            data, _ = planner.plan(data, record["goal"])
        return state, data

    def regenerate_cf(record):
        state = states[record["state"]]
        data = encode(as_facts(state["crossed"]["cf_facts"]), state["meta"], convention)
        if record["goal"] is not None:
            data, _ = planner.plan(data, record["goal"])
        return data

    for mode in modes:
        level = mode[1]
        for metric in ("M1", "M2", "M3"):
            if metric == "M3":
                population = [(r, i) for r in records["singles"] for i in range(len(r["pairs"]))]
                value = lambda item: item[0]["results"][mode][item[1]][0]
            else:
                population = [(r, None) for r in records["pairs"]]
                index = 0 if metric == "M1" else 1
                value = lambda item, index=index: item[0]["results"][mode][index]
            for want in (True, False):
                chosen = [item for item in population if value(item) == want]
                for record, i in rng.sample(chosen, min(n, len(chosen))):
                    state, data = regenerate(record)
                    facts = facts_of(data, state, record["goal"], convention, cf=False)
                    if metric == "M3":
                        a, b = record["pairs"][i]
                        partition, _, _ = reference_equivalence([facts], convention, level)
                        equivalent = _same_class(partition, a, b)
                    else:
                        data_cf = regenerate_cf(record)
                        facts_cf = facts_of(data_cf, state, record["goal"], convention, cf=True)
                        partition, (_, offset), total = reference_equivalence([facts, facts_cf], convention, level)
                        if metric == "M2":
                            equivalent = _same_class(partition, record["p"], record["p"] + offset)
                        else:
                            equivalent = all(
                                sum(1 for v in c if v < offset) == sum(1 for v in c if v >= offset) for c in partition)
                    growing_value = (record["results"][("growing", level)][i][0] if metric == "M3"
                                     else record["results"][("growing", level)][0 if metric == "M1" else 1])
                    # The reference decides WL equivalence; the growing vocab must agree with
                    # it exactly. A frozen-mode collision is either a real one (equivalent)
                    # or an OOV merge the growing vocab does not have.
                    agreed = equivalent == growing_value
                    if mode[0] == "frozen" and want and not growing_value:
                        oov = (record["results"][mode][i][1] if metric == "M3" else record["results"][mode][2])
                        agreed = agreed and bool(oov)
                    cell = f"{mode[0]}_L{level}_{metric}_{'collision' if want else 'distinct'}"
                    summary[cell][0] += 1
                    summary[cell][1] += agreed
                    if not agreed:
                        disagreements.append((cell, record["state"], record["goal"], i))
    return dict(summary), disagreements


# ==========================================================================================
# Comparison with the prototype (analysis/ternary_collision_counter.py)
# ==========================================================================================

PROTOTYPE_NAMES = {"oracle_sage": "object", "vilg": "object_atom", "atom": "atom"}


def drop_taxi_and_in(facts):
    """The prototype adapter's information loss applied to a production fact list: the
    taxi object and every `in` fact removed, remaining object ids shifted down by one
    (the taxi is object 0), so the production converters accept it."""
    kept = [(pred, args) for pred, args in facts if pred not in ("taxi", "in")]
    return [(pred, tuple(a - 1 for a in args)) for pred, args in kept]


def published_sample_pairs(seeds=(0, 1, 2, 3, 4), sample_every=5, per_seed_target=40, random_prob=0.2):
    """
    The crossed states behind the published prototype table: a replay of
    ternary_collision_counter.sample_real_crossed_flip_pairs (same seeds, simulators,
    policy RNG and sampling rule) that keeps the full facts. Asserts that its
    facts_to_ext_atoms images equal what that function itself returns, so these are
    exactly the published pairs.

    :return: list of (facts, cf_facts, p, pair_id)
    """
    pairs = []
    for seed in seeds:
        sim = TernaryTaxiWorldSimulator(np.random.RandomState(seed), observation_fn=lambda s: None, **CITY_TERNARY)
        policy_rng = pyrandom.Random(seed + 9000)
        step = collected = 0
        while collected < per_seed_target:
            if step % sample_every == 0 and crossed_state(sim):
                p = sim.taxi.passenger
                q = sim.passengers[p].buddy
                pair_id = (seed, sim.seed_id, sim.passengers[p].requests, sim.passengers[q].requests)
                pairs.append((sim.facts(), counterfactual_sim(sim, p, q).facts(), p, pair_id))
                collected += 1
            sim.act(scripted_policy(sim, policy_rng, random_prob=random_prob))
            step += 1
            if sim.done:
                sim = TernaryTaxiWorldSimulator(
                    np.random.RandomState(seed * 100003 + step), observation_fn=lambda s: None, **CITY_TERNARY)
    published = sample_real_crossed_flip_pairs(seeds, sample_every, per_seed_target, random_prob, verbose=False)
    assert len(published) == len(pairs)
    for (facts, cf_facts, _p, _id), (atoms_a, atoms_b) in zip(pairs, published):
        assert facts_to_ext_atoms(facts) == atoms_a
        assert sorted(facts_to_ext_atoms(cf_facts)) == sorted(atoms_b)
    return pairs


def prototype_comparison(pairs, max_l=3):
    """
    M1 (growing vocab, graph level) on the SAME crossed pairs three ways, to separate
    the two reasons the production numbers can differ from the published prototype table:
      - prototype: the prototype's own encoders and adapter (facts_to_ext_atoms,
        flip_pair_collision_rates), unmodified;
      - production_without_taxi_in: the production converters and wl_colours on
        drop_taxi_and_in of both facts;
      - production: the production converters on the full facts (= M1_live).
    prototype vs the published table isolates the state sample; prototype vs
    production_without_taxi_in isolates encoder details other than taxi/in; that vs
    production isolates the taxi and `in` facts.

    :param pairs: list of (facts, cf_facts, p, pair_id)
    :return: {name: {convention: {L: [collisions, n]}}}, plus "distinct_pairs"
    """
    n = len(pairs)
    rates = flip_pair_collision_rates([(facts_to_ext_atoms(a), facts_to_ext_atoms(b)) for a, b, _p, _id in pairs], max_l)
    out = {"prototype": {}, "production_without_taxi_in": {}, "production": {}}
    for convention in CONVENTIONS:
        out["prototype"][convention] = {
            level: [int(round(rates[PROTOTYPE_NAMES[convention]][level] * n)), n] for level in range(1, max_l + 1)}
        for name, transform in (("production_without_taxi_in", drop_taxi_and_in), ("production", lambda f: f)):
            counts = {}
            for level in range(1, max_l + 1):
                vocab = {}
                hits = 0
                for facts, cf_facts, _p, _id in pairs:
                    data_a = encode(transform(facts), META_PLANNING, convention)
                    data_b = encode(transform(cf_facts), META_PLANNING, convention)
                    hits += pair_metrics(data_a, data_b, 0, 0, convention, level, vocab)[0]
                counts[level] = [hits, n]
            out[name][convention] = counts
    out["distinct_pairs"] = len({pair_id for _a, _b, _p, pair_id in pairs})
    return out


META_PLANNING = {"time": 0, "timeout": 2000, "planning": True}


# ==========================================================================================

def frozen_vocab_paths(convention):
    return {level: str(build_wl_vocab.ternary_default_out_path(convention, level)) for level in FROZEN_LEVELS}


def run(conventions=CONVENTIONS, seeds=SEEDS, episodes=EPISODES_PER_SEED, sample_every=SAMPLE_EVERY,
        max_steps=2500, use_frozen=True, spot_checks=SPOT_CHECKS, log=True, compare_prototype=True):
    t0 = time.time()
    states = sample_states(seeds, episodes, sample_every, max_steps, log)
    sampling_seconds = time.time() - t0
    crossed = [s["crossed"] for s in states if s["crossed"] is not None]
    report = {
        "states": len(states),
        "projections": sum(len(s["goals"]) for s in states),
        "crossed_live_states": len(crossed),
        "distinct_crossed_pairs": len({c["pair_id"] for c in crossed}),
        "crossed_live_states_scripted_episodes": sum(
            1 for s in states if s["crossed"] is not None and s["episode"] < episodes // 2),
        "counterfactual": {
            "flipped_matches": sum(c["flipped_matches"] for c in crossed),
            "flipped_canonical_order": sum(c["flipped_canonical_order"] for c in crossed),
            "nonisomorphic": dict(Counter(c["nonisomorphic"] for c in crossed)),
            "dest_differs": sum(c["dest"] != c["cf_dest"] for c in crossed),
        },
        "sampling_seconds": round(sampling_seconds, 1),
        "conventions": {},
    }
    rng = pyrandom.Random(0)
    for convention in conventions:
        t1 = time.time()
        frozen = {}
        if use_frozen:
            frozen = {level: _load_vocab(path, convention, level, "ternary")
                      for level, path in frozen_vocab_paths(convention).items()}
        records = measure(states, convention, frozen, log)
        summary = summarise(records, sorted(frozen))
        checks, disagreements = spot_check(states, records, convention, sorted(frozen), rng,
                                           Planner(graph_convention=convention, ternary=True), spot_checks)
        report["conventions"][convention] = {
            "summary": summary,
            "object_identical_tensors": records["object_identical"],
            "spot_checks": checks,
            "spot_check_disagreements": disagreements,
            "seconds": round(time.time() - t1, 1),
        }
        if log:
            print(f"  {convention}: done in {time.time() - t1:.0f}s; spot-check disagreements: {len(disagreements)}",
                  flush=True)
    if compare_prototype:
        t2 = time.time()
        live_pairs = [(as_facts(s["facts"]), as_facts(s["crossed"]["cf_facts"]), s["crossed"]["p"], s["crossed"]["pair_id"])
                      for s in states if s["crossed"] is not None and s["crossed"]["nonisomorphic"] != "isomorphic"]
        report["prototype_comparison"] = {
            "this_sample": prototype_comparison(live_pairs),
            "published_sample": prototype_comparison(published_sample_pairs(), max_l=5),
        }
        report["prototype_comparison_seconds"] = round(time.time() - t2, 1)
    report["total_seconds"] = round(time.time() - t0, 1)
    return report


def _rate(cell):
    hits, n = cell
    return f"{100.0 * hits / n:6.1f}% (n={n})" if n else "    -   (n=0)"


def print_report(report):
    print(f"\nstates: {report['states']} live + {report['projections']} projections; "
          f"crossed live states: {report['crossed_live_states']}")
    print(f"counterfactual: {report['counterfactual']}")
    for convention, r in report["conventions"].items():
        print(f"\n{convention}  (object identical tensors: {r['object_identical_tensors']})")
        for mode, row in r["summary"].items():
            line = "  ".join(f"{k}={_rate(v)}" for k, v in row.items())
            print(f"  {mode:<11} {line}")
        print(f"  spot checks: {sum(v[0] for v in r['spot_checks'].values())} checked, "
              f"{sum(v[1] for v in r['spot_checks'].values())} agreed; disagreements: {r['spot_check_disagreements']}")
    for sample_name, comparison in report.get("prototype_comparison", {}).items():
        print(f"\nM1 on the same crossed pairs, {sample_name} ({comparison['distinct_pairs']} distinct buddy pairs):")
        for name, per_convention in comparison.items():
            if name == "distinct_pairs":
                continue
            for convention, per_level in per_convention.items():
                print(f"  {name:<28}{convention:<12}" + "  ".join(f"L{l}={_rate(v)}" for l, v in per_level.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out", default=None)
    parser.add_argument("--conventions", nargs="+", default=list(CONVENTIONS))
    parser.add_argument("--no-frozen", action="store_true", help="growing vocab only (no Part 3 vocab files needed)")
    args = parser.parse_args()
    report = run(conventions=args.conventions, use_frozen=not args.no_frozen)
    print_report(report)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1, default=str)


if __name__ == "__main__":
    main()
