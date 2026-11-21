"""
.. module:: edge_mode_wl
   :synopsis: A5 -- what WL can tell apart under the three edge forms Gr, MGr and EGr
   (Horcik et al. 2025, Sec. 2; sage/agent/edge_modes.py), with multiset WL and a
   count-free (set) WL.

Analysis only. The thesis scripts are imported, never edited:
  (a) standard Taxi bad pairs -- the states, settings and counting of
      analysis/indistinguishability.py PART 1 (5 seeds, ~5000 snapshots, L = 1..5, one
      growing vocab per run, pairs of distinct states with equal colour histograms; bad
      when the approximated optimal action differs);
  (b) ternary crossed buddy pairs -- the states, projections and metrics of
      analysis/ternary_collision_production.py (production converters and planners,
      growing vocab, L = 1..3; M1 graph level, M2 carried passenger), with its
      cluster bootstrap over distinct pairs.

WL variants (one shared growing vocab per variant and run, so equal ids = equal
signatures). Every variant refines over a node's OUTGOING edges, as the thesis refine does
(all our graphs hold both directions of every connection), and reads a graph out as the
histogram of its final node colours:
  gr  : neighbour items (colour, Gr label)            -- the Gr label is the thesis
        decoder's key for the full multi-hot vector
  mgr : neighbour items (colour, type), one per type set on the Gr edge (expand_typed_edges)
  egr : one bag of neighbour colours per type, kept apart: (bag_type_1, ..., bag_type_T)
  multiset: bags keep counts (standard WL); set: duplicates dropped (count-free WL)

Exact relations (proved in A5_RESULTS.md, asserted on every run):
  - gr refines mgr: the mgr signature is a function of the gr one, so a gr collision is
    always an mgr collision (for multiset and for set);
  - mgr and egr give the same partition: the bag of (colour, type) items and the tuple of
    per-type bags determine each other (for multiset and for set);
  - multiset refines set.
gr-multiset is also checked against the thesis code path itself (collision_table /
ternary_collision_production.pair_metrics), item by item.

Run from the repo root:
    PYTHONPATH=. python analysis/edge_mode_wl.py --part a --out a5_a.json
    PYTHONPATH=. python analysis/edge_mode_wl.py --part b --out a5_b.json
    PYTHONPATH=. python analysis/edge_mode_wl.py --part a --quick     # smoke test
"""
import argparse
import hashlib
import itertools
import json
import time
from collections import defaultdict

import numpy as np
import torch as th

from analysis import indistinguishability as ind  # also installs the gym / numpy shims
from analysis import ternary_collision_production as tcp
from sage.agent.edge_modes import edge_type_spec, expand_typed_edges
from sage.domains.gym_taxi.simulator.planner import Planner
from sage.domains.utils import wl_colours as prod_wl

FORMS = ("gr", "mgr", "egr")
AGGS = ("multiset", "set")
VARIANTS = tuple((form, agg) for agg in AGGS for form in FORMS)

# the thesis decoders per (domain, convention): (initial colours, Gr edge label key)
DECODERS = {
    ("standard", "oracle_sage"): (ind.initial_colours, ind.edge_labels),
    ("standard", "vilg"): (ind.initial_colours_vilg, ind.edge_labels_vilg),
    ("standard", "atom"): (ind.initial_colours_atom, ind.edge_labels_atom),
    ("ternary", "oracle_sage"): prod_wl.DECODERS["ternary"]["oracle_sage"],
    ("ternary", "vilg"): prod_wl.DECODERS["ternary"]["vilg"],
    ("ternary", "atom"): prod_wl.DECODERS["ternary"]["atom"],
}


def variant_name(variant):
    return f"{variant[0]}-{variant[1]}"


# ==========================================================================================
# WL under one edge form
# ==========================================================================================

class Prepared:
    """One graph, decoded once: initial type ids and, per form, each node's outgoing
    (key, neighbour) items -- key = Gr label (gr) or type index (mgr / egr)."""

    def __init__(self, x, edge_index, edge_attr, domain, convention):
        init_fn, label_fn = DECODERS[(domain, convention)]
        self.n = x.shape[0]
        self.init = init_fn(x).tolist()
        spec = edge_type_spec(convention, domain == "ternary", "egr")
        self.num_types = spec.num_types
        self.items = {}
        src, dst = edge_index[0].tolist(), edge_index[1].tolist()
        self.items["gr"] = self._by_source(src, dst, label_fn(edge_attr).tolist())
        typed_index, types = expand_typed_edges(edge_index, edge_attr, spec)
        self.items["mgr"] = self._by_source(typed_index[0].tolist(), typed_index[1].tolist(), types.tolist())
        self.items["egr"] = self.items["mgr"]
        self.gr_edges = len(src)
        self.typed_edges = len(types)

    def _by_source(self, src, dst, keys):
        out = [[] for _ in range(self.n)]
        for s, d, k in zip(src, dst, keys):
            out[s].append((k, d))
        return out


def _signature(own, items, colours, form, agg, num_types):
    if form == "egr":
        bags = [[] for _ in range(num_types)]
        for t, d in items:
            bags[t].append(colours[d])
        if agg == "set":
            bags = [set(b) for b in bags]
        return own, tuple(tuple(sorted(b)) for b in bags)
    pairs = [(colours[d], k) for k, d in items]
    if agg == "set":
        pairs = set(pairs)
    return own, tuple(sorted(pairs))


def wl_levels(graph, variant, vocab, max_l):
    """{L: final colour list} for L = 1..max_l, one growing vocab (mutated)."""
    form, agg = variant
    colours = [vocab.setdefault(("init", t), len(vocab)) for t in graph.init]
    items = graph.items[form]
    out = {}
    for level in range(1, max_l + 1):
        colours = [
            vocab.setdefault(sig, len(vocab))
            for sig in (_signature(colours[v], items[v], colours, form, agg, graph.num_types) for v in range(graph.n))
        ]
        out[level] = colours
    return out


def histogram_key(colours):
    return hashlib.sha256(repr(sorted(colours)).encode()).hexdigest()


# ==========================================================================================
# (a) standard Taxi bad pairs
# ==========================================================================================

def count_pairs(distinct, hashes):
    """ind.collision_table's counting, on precomputed per-state hashes."""
    buckets = defaultdict(list)
    for i, h in enumerate(hashes):
        buckets[h].append(i)
    collisions = bad = 0
    for idxs in buckets.values():
        for i, j in itertools.combinations(idxs, 2):
            collisions += 1
            bad += distinct[i]["answer"] != distinct[j]["answer"]
    return {"distinct_hashes": len(buckets), "collisions": collisions, "bad_collisions": bad}


def _refines(fine, coarse):
    """True if equal `fine` keys always mean equal `coarse` keys (same index order)."""
    seen = {}
    return all(seen.setdefault(f, c) == c for f, c in zip(fine, coarse))


def _same_partition(a, b):
    return _refines(a, b) and _refines(b, a)


def hierarchy_checks(per_variant_hashes, levels):
    """The exact relations of the module docstring, on per-item keys, at every L."""
    failures = []
    for level in levels:
        h = {v: per_variant_hashes[v][level] for v in per_variant_hashes}
        for agg in AGGS:
            if not _refines(h[("gr", agg)], h[("mgr", agg)]):
                failures.append(f"L{level}: gr-{agg} does not refine mgr-{agg}")
            if not _same_partition(h[("mgr", agg)], h[("egr", agg)]):
                failures.append(f"L{level}: mgr-{agg} and egr-{agg} differ")
        for form in FORMS:
            if not _refines(h[(form, "multiset")], h[(form, "set")]):
                failures.append(f"L{level}: {form}-multiset does not refine {form}-set")
    return failures


def run_part_a(target_total=5000, max_l=ind.MAX_L, np_seed=0, log=True):
    """np_seed seeds numpy's global generator, which the thesis sampler's random actions
    (ind.sample_action) draw from; the thesis script never seeds it, so its own sample
    (1,738 distinct states) cannot be replayed -- see A5_RESULTS.md."""
    t0 = time.time()
    np.random.seed(np_seed)
    states = ind.generate_part1_states(target_total=target_total, verbose=log)
    seen = {}
    for s in states:
        seen.setdefault(s["atoms"], s)
    distinct = list(seen.values())
    report = {"np_seed": np_seed, "snapshots": len(states), "distinct_states": len(distinct),
              "sampling_seconds": round(time.time() - t0, 1), "conventions": {}}
    for convention in ind.PART1_CONVENTIONS:
        t1 = time.time()
        init_fn, edge_fn = DECODERS[("standard", convention)]
        thesis = ind.collision_table(states, convention, init_fn, edge_fn, max_l=max_l)
        graphs = [Prepared(*s["graphs"][convention], "standard", convention) for s in distinct]
        multi_type = sum(g.typed_edges != g.gr_edges for g in graphs)
        per_variant_hashes, table = {}, {}
        for variant in VARIANTS:
            vocab = {}
            levels = [wl_levels(g, variant, vocab, max_l) for g in graphs]
            per_variant_hashes[variant] = {L: [histogram_key(lv[L]) for lv in levels] for L in range(1, max_l + 1)}
            table[variant_name(variant)] = {L: count_pairs(distinct, per_variant_hashes[variant][L])
                                            for L in range(1, max_l + 1)}
        thesis_match = all(
            table["gr-multiset"][L][k] == thesis[L][k]
            for L in range(1, max_l + 1) for k in ("distinct_hashes", "collisions", "bad_collisions"))
        report["conventions"][convention] = {
            "thesis_code_path": {L: {k: thesis[L][k] for k in ("distinct_hashes", "collisions", "bad_collisions")}
                                 for L in thesis},
            "gr_multiset_matches_thesis_code": thesis_match,
            "states_with_a_multi_type_edge": multi_type,
            "gr_edges_total": sum(g.gr_edges for g in graphs),
            "typed_edges_total": sum(g.typed_edges for g in graphs),
            "variants": table,
            "hierarchy_failures": hierarchy_checks(per_variant_hashes, range(1, max_l + 1)),
            "seconds": round(time.time() - t1, 1),
        }
        if log:
            print(f"  {convention}: {time.time() - t1:.0f}s, thesis match {thesis_match}, "
                  f"states with a multi-type edge {multi_type}/{len(graphs)}", flush=True)
    report["total_seconds"] = round(time.time() - t0, 1)
    return report


def print_part_a(report):
    print(f"\n(a) standard Taxi, np seed {report['np_seed']}: {report['snapshots']} snapshots, {report['distinct_states']} distinct states")
    for convention, r in report["conventions"].items():
        print(f"\n{convention}: thesis code match={r['gr_multiset_matches_thesis_code']}  "
              f"states with a multi-type edge={r['states_with_a_multi_type_edge']}  "
              f"edges Gr={r['gr_edges_total']} typed={r['typed_edges_total']}  "
              f"hierarchy failures={r['hierarchy_failures']}")
        levels = sorted(r["thesis_code_path"])
        print(f"  {'variant':<14}" + "".join(f"{'L' + str(L) + ' coll/bad':>18}" for L in levels))
        for name, per_level in r["variants"].items():
            print(f"  {name:<14}" + "".join(
                f"{per_level[L]['collisions']:>11}/{per_level[L]['bad_collisions']:<6}" for L in levels))


# ==========================================================================================
# (b) ternary crossed buddy pairs
# ==========================================================================================

def run_part_b(levels=tcp.GROWING_LEVELS, min_distinct_pairs=tcp.MIN_DISTINCT_CROSSED_PAIRS, log=True):
    t0 = time.time()
    states, seeds = tcp.sample_until(min_distinct_pairs, log=log)
    crossed = [s["crossed"] for s in states if s["crossed"] is not None]
    report = {"seeds": seeds, "states": len(states), "projections": sum(len(s["goals"]) for s in states),
              "crossed_live_states": len(crossed),
              "distinct_crossed_pairs": len({c["pair_id"] for c in crossed}),
              "sampling_seconds": round(time.time() - t0, 1), "conventions": {}}
    max_l = max(levels)
    for convention in tcp.CONVENTIONS:
        t1 = time.time()
        planner = Planner(graph_convention=convention, ternary=True)
        prod_vocabs = {L: {} for L in levels}
        vocabs = {v: {} for v in VARIANTS}
        obs = []  # one per pair: cluster, production (m1, m2) per L, variant (m1, m2) per L
        multi_type = n_graphs = 0
        for state in states:
            for item in tcp.graph_items(state, convention, planner):
                if item[0] != "pair":
                    continue
                _kind, goal, data_a, data_b, p_a, p_b, _dest_differs, cluster = item
                prod = {L: tcp.pair_metrics(data_a, data_b, p_a, p_b, convention, L, prod_vocabs[L])[:2]
                        for L in levels}
                ga = Prepared(data_a.x, data_a.edge_index, data_a.edge_attr, "ternary", convention)
                gb = Prepared(data_b.x, data_b.edge_index, data_b.edge_attr, "ternary", convention)
                n_graphs += 2
                multi_type += (ga.typed_edges != ga.gr_edges) + (gb.typed_edges != gb.gr_edges)
                res = {}
                for variant in VARIANTS:
                    la = wl_levels(ga, variant, vocabs[variant], max_l)
                    lb = wl_levels(gb, variant, vocabs[variant], max_l)
                    res[variant] = {L: (sorted(la[L]) == sorted(lb[L]), la[L][p_a] == lb[L][p_b]) for L in levels}
                obs.append({"cluster": cluster, "live": goal is None, "prod": prod, "res": res})
        rng = np.random.default_rng(0)

        def cells(hit):
            """M1 / M2 over all observations, and over live states only (no planner projection)."""
            live = [o for o in obs if o["live"]]
            return {"M1": tcp._cell([(o["cluster"], hit(o)[0]) for o in obs], rng),
                    "M2": tcp._cell([(o["cluster"], hit(o)[1]) for o in obs], rng),
                    "M1_live": tcp._cell([(o["cluster"], hit(o)[0]) for o in live], rng),
                    "M2_live": tcp._cell([(o["cluster"], hit(o)[1]) for o in live], rng)}

        table = {variant_name(v): {L: cells(lambda o, v=v, L=L: o["res"][v][L]) for L in levels} for v in VARIANTS}
        thesis_cells = {L: cells(lambda o, L=L: o["prod"][L]) for L in levels}
        match = all(o["prod"][L] == o["res"][("gr", "multiset")][L] for o in obs for L in levels)
        failures = []
        for o in obs:
            for L in levels:
                for m in (0, 1):
                    hit = {v: o["res"][v][L][m] for v in VARIANTS}
                    for agg in AGGS:
                        if hit[("gr", agg)] and not hit[("mgr", agg)]:
                            failures.append(f"L{L} M{m + 1}: gr-{agg} hit, mgr-{agg} not")
                        if hit[("mgr", agg)] != hit[("egr", agg)]:
                            failures.append(f"L{L} M{m + 1}: mgr-{agg} != egr-{agg}")
                    for form in FORMS:
                        if hit[(form, "multiset")] and not hit[(form, "set")]:
                            failures.append(f"L{L} M{m + 1}: {form}-multiset hit, {form}-set not")
        report["conventions"][convention] = {
            "pairs": len(obs), "distinct_pairs": len({o["cluster"] for o in obs}),
            "thesis_code_path": thesis_cells,
            "gr_multiset_matches_thesis_code": match,
            "graphs_with_a_multi_type_edge": [multi_type, n_graphs],
            "variants": table,
            "hierarchy_failures": sorted(set(failures)),
            "seconds": round(time.time() - t1, 1),
        }
        if log:
            print(f"  {convention}: {time.time() - t1:.0f}s, thesis match {match}, "
                  f"graphs with a multi-type edge {multi_type}/{n_graphs}", flush=True)
    report["total_seconds"] = round(time.time() - t0, 1)
    return report


def print_part_b(report):
    print(f"\n(b) ternary: seeds {report['seeds'][0]}-{report['seeds'][-1]}, {report['states']} live states, "
          f"{report['crossed_live_states']} crossed ({report['distinct_crossed_pairs']} distinct pairs)")
    for convention, r in report["conventions"].items():
        print(f"\n{convention}: {r['pairs']} pair observations / {r['distinct_pairs']} pairs; thesis code match="
              f"{r['gr_multiset_matches_thesis_code']}; graphs with a multi-type edge="
              f"{r['graphs_with_a_multi_type_edge']}; hierarchy failures={r['hierarchy_failures']}")
        rows = [("thesis code", r["thesis_code_path"])] + list(r["variants"].items())
        for name, per_level in rows:
            for metric in ("M1", "M2", "M1_live", "M2_live"):
                print(f"  {name:<14}{metric:<8}" + "  ".join(
                    f"L{L}={tcp._rate(cells[metric])}" for L, cells in per_level.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument("--part", choices=("a", "b"), required=True)
    parser.add_argument("--quick", action="store_true", help="small smoke-test run")
    parser.add_argument("--np-seed", type=int, default=0, help="(a) only: numpy seed for the random actions")
    parser.add_argument("--levels", type=int, nargs="+", default=list(tcp.GROWING_LEVELS),
                        help="(b) only: the L values (the thesis used 1 2 3)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.part == "a":
        report = run_part_a(target_total=200 if args.quick else 5000, np_seed=args.np_seed)
        print_part_a(report)
    else:
        report = run_part_b(levels=tuple(args.levels), min_distinct_pairs=5 if args.quick else tcp.MIN_DISTINCT_CROSSED_PAIRS)
        print_part_b(report)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1, default=str)


if __name__ == "__main__":
    main()
