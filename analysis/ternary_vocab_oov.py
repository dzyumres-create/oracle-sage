"""
.. module:: ternary_vocab_oov
   :synopsis: Held-out OOV of the frozen ternary WL vocabs
   (sage/domains/utils/wl_vocab_taxi_city_ternary_{conv}_L{L}.json).

States come from build_wl_vocab.ternary_corpus -- the production path the vocabs were
built from -- on seeds DISJOINT from the vocab's own (seed 0): HELD_OUT_SEEDS, each two
full episodes (one scripted_policy with random_prob 0.2, one uniform random legal), so
late-episode states (up to the 2000-step timeout) and planner projections are included.

Every graph is coloured with the frozen vocab, loaded through the policy's own loader
(_load_vocab, domain="ternary"); a node is OOV when its final colour is the vocab's OOV id.
Reported per convention and L:
  - overall: every node;
  - passenger: passenger object nodes;
  - request: request proposition nodes (vilg), request atom nodes (atom), and for object
    -- which has no request nodes -- the passenger and location nodes, whose edges carry
    the request labels;
  - late: every node of states sampled at step >= LATE_STEP (live and projections);
  - projections: every node of planner projections only;
plus the vocab size and file size.

Run from the repo root with:
    PYTHONPATH=. python analysis/ternary_vocab_oov.py [--out results.json]
"""
import argparse
import json
import os
import time

import numpy as np

from sage.agent.wl_plan_feedback_policy import _load_vocab
from sage.domains.gym_taxi.utils.ternary_representations import (
    ATOM_PREDICATES_TERNARY,
    VILG_PREDICATE_ORDER,
    _OBJECT_TYPE_INDEX,
)
from sage.domains.utils import build_wl_vocab
from sage.domains.utils.wl_colours import OOV_SIGNATURE, wl_colours

CONVENTIONS = ("oracle_sage", "vilg", "atom")
LEVELS = (1, 2)
HELD_OUT_SEEDS = (1001, 1002, 1003, 1004, 1005)
EPISODES_PER_SEED = 2
SAMPLE_EVERY = 10
LATE_STEP = 1000
CATEGORIES = ("overall", "passenger", "request", "late", "projections")

_PASSENGER_COLUMN = {
    "oracle_sage": _OBJECT_TYPE_INDEX["passenger"],
    "vilg": _OBJECT_TYPE_INDEX["passenger"],
    "atom": ATOM_PREDICATES_TERNARY.index("passenger"),
}


def node_masks(x, convention):
    """Boolean masks over the rows of x for the passenger and request categories."""
    x = x.numpy()
    passenger = x[:, _PASSENGER_COLUMN[convention]] == 1
    if convention == "vilg":
        request = x[:, 3 + VILG_PREDICATE_ORDER.index("request")] == 1
    elif convention == "atom":
        request = x[:, ATOM_PREDICATES_TERNARY.index("request")] == 1
    else:
        request = passenger | (x[:, _OBJECT_TYPE_INDEX["location"]] == 1)
    return passenger, request


def held_out_oov(convention, vocab_paths, seeds=HELD_OUT_SEEDS, episodes=EPISODES_PER_SEED, sample_every=SAMPLE_EVERY,
                 max_steps=2500):
    """
    :param vocab_paths: {L: path}
    :return: {L: {category: [oov_nodes, nodes]}}, plus a "graphs" count per L
    """
    vocabs = {level: _load_vocab(path, convention, level, "ternary") for level, path in vocab_paths.items()}
    counts = {level: {c: [0, 0] for c in CATEGORIES} for level in vocabs}
    n_graphs = 0
    for seed in seeds:
        for sample in build_wl_vocab.ternary_corpus(episodes, sample_every, seed, convention, max_steps=max_steps):
            d = sample.data
            n_graphs += 1
            passenger, request = node_masks(d.x, convention)
            for level, vocab in vocabs.items():
                colours, _ = wl_colours(
                    d.x, d.edge_index, d.edge_attr, num_iterations=level, vocab=vocab, frozen=True,
                    graph_convention=convention, domain="ternary",
                )
                oov = (colours == vocab[OOV_SIGNATURE]).numpy()
                c = counts[level]
                selections = {"overall": np.ones_like(oov), "passenger": passenger, "request": request}
                if sample.step >= LATE_STEP:
                    selections["late"] = np.ones_like(oov)
                if sample.goal is not None:
                    selections["projections"] = np.ones_like(oov)
                for category, selected in selections.items():
                    c[category][0] += int(oov[selected].sum())
                    c[category][1] += int(selected.sum())
    for level in counts:
        counts[level]["graphs"] = n_graphs
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out", default=None, help="write the results as JSON here")
    parser.add_argument("--conventions", nargs="+", default=list(CONVENTIONS))
    args = parser.parse_args()

    results = {}
    for convention in args.conventions:
        start = time.time()
        paths = {level: str(build_wl_vocab.ternary_default_out_path(convention, level)) for level in LEVELS}
        counts = held_out_oov(convention, paths)
        results[convention] = {}
        for level, path in paths.items():
            with open(path) as f:
                vocab_size = json.load(f)["vocab_size"]
            results[convention][level] = {
                "vocab_size": vocab_size,
                "file_bytes": os.path.getsize(path),
                "graphs": counts[level]["graphs"],
                **{c: counts[level][c] for c in CATEGORIES},
            }
        print(f"{convention}: {time.time() - start:.0f}s", flush=True)

    print(f"\nHeld-out OOV (seeds {HELD_OUT_SEEDS}, {EPISODES_PER_SEED} full episodes each, "
          f"sample_every={SAMPLE_EVERY}, 2 projections per state)")
    header = f"{'convention':<12}{'L':>2}{'vocab':>9}{'file':>11}{'graphs':>8}" + "".join(f"{c:>22}" for c in CATEGORIES)
    print(header)
    for convention, per_level in results.items():
        for level, r in per_level.items():
            cells = "".join(f"{r[c][0]:>8}/{r[c][1]:<8}{100.0 * r[c][0] / max(r[c][1], 1):>5.2f}%" for c in CATEGORIES)
            print(f"{convention:<12}{level:>2}{r['vocab_size']:>9}{r['file_bytes']:>11}{r['graphs']:>8}{cells}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
