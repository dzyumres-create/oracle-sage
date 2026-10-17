"""
.. module:: ternary_vocab_oov
   :synopsis: Held-out OOV of the frozen ternary WL vocabs
   (sage/domains/utils/wl_vocab_taxi_city_ternary_{conv}_L{L}.json).

States come from build_wl_vocab.ternary_corpus -- the production path the vocabs were
built from -- on seeds DISJOINT from the vocab's own (seed 0): HELD_OUT_SEEDS (1001-1010), each two
full episodes (one scripted_policy with random_prob 0.2, one uniform random legal) -- 20
held-out episodes -- so late-episode states (up to the 2000-step timeout) and planner
projections are included.

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
plus the vocab size and file size. Every rate comes with a 95% percentile bootstrap CI
that resamples held-out EPISODES (the states within one episode are autocorrelated).

Run from the repo root with:
    PYTHONPATH=. python analysis/ternary_vocab_oov.py [--out results.json]
    PYTHONPATH=. python analysis/ternary_vocab_oov.py --conventions atom --vocab some_vocab.json
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
# L=1 only: frozen L=2 vocabs were dropped (held-out OOV 1.9-93%; see analysis/README.md).
LEVELS = (1,)
HELD_OUT_SEEDS = tuple(range(1001, 1011))
BOOTSTRAP_RESAMPLES = 2000
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
    :return: {L: {"graphs": n, "per_episode": {category: {(seed, episode): [oov, nodes]}}}}
    """
    vocabs = {level: _load_vocab(path, convention, level, "ternary") for level, path in vocab_paths.items()}
    per_episode = {level: {c: {} for c in CATEGORIES} for level in vocabs}
    n_graphs = 0
    for seed in seeds:
        for sample in build_wl_vocab.ternary_corpus(episodes, sample_every, seed, convention, max_steps=max_steps):
            d = sample.data
            n_graphs += 1
            episode = (seed, sample.episode)
            passenger, request = node_masks(d.x, convention)
            for level, vocab in vocabs.items():
                colours, _ = wl_colours(
                    d.x, d.edge_index, d.edge_attr, num_iterations=level, vocab=vocab, frozen=True,
                    graph_convention=convention, domain="ternary",
                )
                oov = (colours == vocab[OOV_SIGNATURE]).numpy()
                selections = {"overall": np.ones_like(oov), "passenger": passenger, "request": request}
                if sample.step >= LATE_STEP:
                    selections["late"] = np.ones_like(oov)
                if sample.goal is not None:
                    selections["projections"] = np.ones_like(oov)
                for category, selected in selections.items():
                    cell = per_episode[level][category].setdefault(episode, [0, 0])
                    cell[0] += int(oov[selected].sum())
                    cell[1] += int(selected.sum())
    return {level: {"graphs": n_graphs, "per_episode": per_episode[level]} for level in vocabs}


def summarise(per_episode, rng, resamples=BOOTSTRAP_RESAMPLES):
    """[oov, nodes, rate, [ci_low, ci_high]] from {episode: [oov, nodes]}, bootstrapping
    over episodes (an episode with no nodes of the category counts as 0/0)."""
    values = np.array(list(per_episode.values()), dtype=np.float64).reshape(-1, 2)
    oov, nodes = values[:, 0], values[:, 1]
    total_oov, total_nodes = int(oov.sum()), int(nodes.sum())
    if total_nodes == 0:
        return [0, 0, None, None]
    idx = rng.integers(0, len(oov), size=(resamples, len(oov)))
    sums = nodes[idx].sum(axis=1)
    rates = oov[idx].sum(axis=1) / np.maximum(sums, 1)
    return [total_oov, total_nodes, total_oov / total_nodes,
            [float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))]]


def score(convention, vocab_paths, seed=0):
    """held_out_oov plus summaries: {L: {"vocab_size", "file_bytes", "graphs", "episodes", category: summary}}."""
    raw = held_out_oov(convention, vocab_paths)
    rng = np.random.default_rng(seed)
    out = {}
    for level, path in vocab_paths.items():
        with open(path) as f:
            vocab_size = json.load(f)["vocab_size"]
        out[level] = {
            "vocab_size": vocab_size,
            "file_bytes": os.path.getsize(path),
            "graphs": raw[level]["graphs"],
            "episodes": len(raw[level]["per_episode"]["overall"]),
            **{c: summarise(raw[level]["per_episode"][c], rng) for c in CATEGORIES},
        }
    return out


def _fmt(cell):
    oov, nodes, rate, ci = cell
    if rate is None:
        return "-"
    return f"{100 * rate:.3f}% [{100 * ci[0]:.3f}, {100 * ci[1]:.3f}] ({oov}/{nodes})"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out", default=None, help="write the results as JSON here")
    parser.add_argument("--conventions", nargs="+", default=list(CONVENTIONS))
    parser.add_argument("--vocab", default=None,
                        help="score this L=1 vocab file instead of the committed default (one convention only)")
    args = parser.parse_args()
    if args.vocab is not None and len(args.conventions) != 1:
        parser.error("--vocab needs exactly one --conventions value")

    results = {}
    for convention in args.conventions:
        start = time.time()
        if args.vocab is not None:
            paths = {1: args.vocab}
        else:
            paths = {level: str(build_wl_vocab.ternary_default_out_path(convention, level)) for level in LEVELS}
        results[convention] = score(convention, paths)
        print(f"{convention}: {time.time() - start:.0f}s", flush=True)

    print(f"\nHeld-out OOV (seeds {HELD_OUT_SEEDS[0]}-{HELD_OUT_SEEDS[-1]}, {EPISODES_PER_SEED} full episodes each, "
          f"sample_every={SAMPLE_EVERY}, 2 projections per state); 95% CI over episodes")
    for convention, per_level in results.items():
        for level, r in per_level.items():
            print(f"{convention} L={level}: vocab {r['vocab_size']}, file {r['file_bytes']} bytes, "
                  f"{r['graphs']} graphs, {r['episodes']} episodes")
            for c in CATEGORIES:
                print(f"    {c:<12}{_fmt(r[c])}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
