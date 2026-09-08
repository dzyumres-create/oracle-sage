"""
.. module:: wl_depth_sweep
   :synopsis: Standalone diagnostic script - NOT part of build_wl_vocab.py's
   normal L=5 vocab-building workflow, does not touch it or wl_colours.py,
   and does not freeze/save any vocab to the real wl_vocab_taxi_L5.json path.

   Previous diagnostics (see build_wl_vocab.py's docstring/history) found
   that WL colour vocab growth at L=5 does NOT stabilize over 1,000 sampled
   graphs on either "city" (random maze) or "predictable5" (fixed 5x5
   maze) - ruling out maze randomization as the cause. Working hypothesis:
   at L=5, on a grid this small, WL colours approach encoding the full
   visited (taxi/passenger/destination/holding) state rather than local
   graph structure, and Taxi simply has too many distinct reachable states
   for that to ever stabilize.

   This script checks whether smaller L values behave differently, using
   the predictable5 scenario (fixed maze) - the same one already used for
   the L=5 fixed-maze diagnostic. It samples ONE corpus of 1,000 graphs
   (exactly the same random-action sampling approach as
   build_wl_vocab.sample_graphs/sample_action, reused directly by import),
   then runs wl_colours over that SAME corpus once per L in {1, 2, 3, 4},
   each with its own fresh vocab (vocab dicts are NOT shared across L
   values - colour ids from different L runs are not comparable), logging
   a (graphs, vocab_size, delta) checkpoint every 100 graphs exactly as the
   earlier L=5 diagnostics did. Sampling once and reusing the corpus across
   all four L values (rather than re-sampling per L) removes environment
   randomness as a confound between the four growth curves - the corpus is
   identical for all of them, so only L differs.

   `GraphTaxiEnv`/`MASK`/`REWARDS_VARIANT`/`sample_action` are imported
   directly from build_wl_vocab.py rather than duplicated here - this also
   means build_wl_vocab.py's numpy/gym compat shims (see its docstring)
   run automatically as an import side effect, so this script needs none
   of its own.

Run from the repo root: python -m sage.domains.utils.wl_depth_sweep
"""
import time

import numpy as np

from sage.domains.utils.build_wl_vocab import GraphTaxiEnv, MASK, REWARDS_VARIANT, sample_action, extract_graph_tensors
from sage.domains.utils.wl_colours import wl_colours

SCENARIO = "predictable5"  # fixed maze - same scenario as the earlier L=5 fixed-maze diagnostic
EPISODES = 10
STEPS_PER_EPISODE = 100  # EPISODES * STEPS_PER_EPISODE = 1,000 graphs, matching the earlier diagnostics
SEED = 0
L_VALUES = [1, 2, 3, 4]
LOG_EVERY = 100


def collect_graph_corpus(scenario, episodes, steps_per_episode, seed, graph_convention="oracle_sage"):
    """
    Samples a FIXED corpus of graphs from one Taxi environment, using the
    exact same random-action stepping approach as
    build_wl_vocab.sample_graphs - but does NOT run wl_colours during
    collection, so the resulting corpus can be replayed identically across
    multiple L values.

    :param graph_convention: "oracle_sage" (default, unchanged behaviour) or
        "vilg" - selects both the GraphTaxiEnv construction and the
        translator used to read each sampled graph.
    :return: list of (x, edge_index, edge_attr) torch tensor triples, one per sampled graph
    """
    np.random.seed(seed)
    corpus = []

    env = GraphTaxiEnv(representation="graph", scenario=scenario, mask=MASK, rewards=REWARDS_VARIANT, graph_convention=graph_convention)
    for _ in range(episodes):
        env.reset()
        for _ in range(steps_per_episode):
            corpus.append(extract_graph_tensors(env.sim, graph_convention=graph_convention))

            action = sample_action(env.sim)
            _, _, done, _ = env.step(action)
            if done:
                env.reset()

    return corpus


def run_depth(corpus, num_iterations, log_every=LOG_EVERY, graph_convention="oracle_sage"):
    """
    Runs growing-mode wl_colours over `corpus` at a fixed `num_iterations`
    (L), with a fresh vocab, logging a (graphs, vocab_size, delta)
    checkpoint every `log_every` graphs.

    :param graph_convention: "oracle_sage" (default, unchanged behaviour) or
        "vilg" - selects the wl_colours decoder pair to use, matching
        whatever convention `corpus` was collected with.
    :return: (final vocab dict, list of per-checkpoint deltas)
    """
    vocab = {}
    last_checkpoint_size = 0
    deltas = []

    for i, (x, edge_index, edge_attr) in enumerate(corpus, start=1):
        wl_colours(x, edge_index, edge_attr, num_iterations=num_iterations, vocab=vocab, frozen=False, graph_convention=graph_convention)

        if i % log_every == 0:
            delta = len(vocab) - last_checkpoint_size
            deltas.append(delta)
            print(f"  L={num_iterations}  graphs={i:>6}  vocab_size={len(vocab):>6}  (+{delta} since last checkpoint)")
            last_checkpoint_size = len(vocab)

    return vocab, deltas


def main():
    print(f"Sampling {EPISODES * STEPS_PER_EPISODE} graphs once from scenario={SCENARIO!r} "
          f"({EPISODES} episodes x {STEPS_PER_EPISODE} steps, seed={SEED})...")
    t0 = time.time()
    corpus = collect_graph_corpus(SCENARIO, EPISODES, STEPS_PER_EPISODE, SEED)
    print(f"collected {len(corpus)} graphs in {time.time() - t0:.1f}s\n")

    results = {}
    for L in L_VALUES:
        print(f"=== L={L} ===")
        t0 = time.time()
        vocab, deltas = run_depth(corpus, L)
        elapsed = time.time() - t0
        results[L] = (len(vocab), deltas, elapsed)
        print(f"L={L}: final vocab_size={len(vocab)}, elapsed={elapsed:.1f}s\n")

    print("=== summary ===")
    for L in L_VALUES:
        size, deltas, elapsed = results[L]
        print(f"L={L}: vocab_size={size:>6}  deltas={deltas}  elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
