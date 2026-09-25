"""
.. module:: wl_depth_sweep
   :synopsis: Standalone diagnostic script for choosing L (WL refinement depth) for the
   Taxi domain, by measurable, reproducible criteria - NOT part of build_wl_vocab.py's
   normal vocab-building workflow, does not touch it or wl_colours.py, and does not
   freeze/save any vocab to a real wl_vocab_taxi_*.json path.

   Originally (see git history) a single-purpose vocab-GROWTH sweep: sample one corpus,
   run wl_colours in growing mode once per L, log vocab-size checkpoints. This is now
   generalised into a real "which L" tool with a CLI (--graph-convention/--scenario/
   -L/--seed/corpus-size flags, mirroring build_wl_vocab.py's own flags) and a second
   measurement: HELD-OUT OOV. Vocab growth alone only tells you the corpus hasn't
   converged; held-out OOV against graphs the vocab never saw is the sharper test of
   whether a frozen vocab at a given L will actually generalise during real training.

   Two disjoint held-out corpora, both built from seeds that never overlap corpus A's
   build seed:
     - B-random: same random-action sampling as corpus A (build_wl_vocab.sample_action)
       - directly comparable to A's own growth curve, since it's the same sampling
       process on unseen seeds.
     - B-policy: a nearest-passenger-then-destination greedy policy (build_wl_vocab.
       greedy_action - the same one verification check C used to force real pickups/
       deliveries within a bounded step budget, since pure random action sampling
       rarely delivers on a size-20 maze), PLUS Planner.plan() projections of a few
       random candidate goals sampled at each visited state (build_wl_vocab.
       to_planner_data) - projected graphs are a real, structurally distinct part of
       what a planner-feedback policy actually sees during training (see
       graph_plan_feedback_policy.py's project_actions), and are not guaranteed to have
       the same OOV behaviour as live-stepped states alone.
   Reporting both separately (not pooled) is deliberate: a vocab that generalises to
   B-random but not B-policy (or vice versa) is a real, actionable finding about WHERE
   the frozen vocab's blind spots are, that a single pooled number would hide.

   `GraphTaxiEnv`/`MASK`/`REWARDS_VARIANT`/`sample_action`/`road_graph`/`next_hop`/
   `greedy_action`/`to_planner_data`/`extract_graph_tensors`/`Planner` are imported
   directly from build_wl_vocab.py rather than duplicated here - this also means
   build_wl_vocab.py's numpy/gym compat shims (see its docstring) run automatically as
   an import side effect, so this script needs none of its own.

Run from the repo root: python -m sage.domains.utils.wl_depth_sweep --help
"""
import argparse
import copy
import time

import numpy as np

from sage.domains.utils.build_wl_vocab import (
    GraphTaxiEnv, MASK, REWARDS_VARIANT, sample_action, extract_graph_tensors,
    road_graph, next_hop, greedy_action, to_planner_data, Planner,
)
from sage.domains.utils.wl_colours import OOV_SIGNATURE, freeze_vocab, wl_colours

SCENARIO = "predictable5"
L_VALUES = [1, 2, 3, 4]
LOG_EVERY = 100

# corpus A (vocab-building) defaults - "quick pass" scale; bump via CLI for a full run
EPISODES = 20
STEPS_PER_EPISODE = 50
SEED = 0

# held-out corpora defaults - deliberately far outside corpus A's seed range
HELD_OUT_SEED = 500_000
HELD_OUT_EPISODES = 10
HELD_OUT_STEPS_PER_EPISODE = 50

POLICY_SEED = 900_000
POLICY_EPISODES = 10
POLICY_STEPS_PER_EPISODE = 50
POLICY_GOALS_PER_STATE = 3


def collect_graph_corpus(scenario, episodes, steps_per_episode, seed, graph_convention="oracle_sage"):
    """
    Samples a FIXED corpus of graphs from one Taxi environment, using the
    exact same random-action stepping approach as
    build_wl_vocab.sample_graphs - but does NOT run wl_colours during
    collection, so the resulting corpus can be replayed identically across
    multiple L values.

    :param graph_convention: "oracle_sage" (default, unchanged behaviour),
        "vilg", or "atom" - selects both the GraphTaxiEnv construction and
        the translator used to read each sampled graph.
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


def collect_policy_corpus(scenario, episodes, steps_per_episode, seed, graph_convention="oracle_sage", goals_per_state=POLICY_GOALS_PER_STATE):
    """
    B-policy: greedy_action (nearest-passenger-then-destination) stepping, so pickups
    and deliveries actually happen within the step budget - PLUS Planner.plan()
    projections of `goals_per_state` random legal candidate goals at every visited
    state (mirroring project_actions' own planner.plan(deepcopy(state), goal) call
    pattern exactly - deepcopy is required here too, since the oracle_sage/vilg planner
    branches mutate their input Data in place; see move_taxi/move_taxi_vilg).

    :return: list of (x, edge_index, edge_attr) torch tensor triples - both live-stepped
        states AND planner-projected states, NOT distinguished in the returned list (the
        caller decides how to report them; this function's job is just to sample both).
    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    corpus = []
    planner = Planner(graph_convention=graph_convention)

    env = GraphTaxiEnv(representation="graph", scenario=scenario, mask=MASK, rewards=REWARDS_VARIANT, graph_convention=graph_convention)
    env.seed(seed)
    env.reset()
    G = road_graph(env.sim)

    for _ in range(episodes):
        env.reset()
        G = road_graph(env.sim)
        for _ in range(steps_per_episode):
            corpus.append(extract_graph_tensors(env.sim, graph_convention=graph_convention))

            data = to_planner_data(env.sim, graph_convention=graph_convention)
            selectable = data.mask.nonzero(as_tuple=True)[0].tolist()
            n_goals = min(goals_per_state, len(selectable))
            goals = rng.choice(selectable, size=n_goals, replace=False) if n_goals > 0 else []
            for goal in goals:
                projection, _actions = planner.plan(copy.deepcopy(data), int(goal))
                corpus.append((projection.x, projection.edge_index, projection.edge_attr))

            a = greedy_action(env.sim, G)
            _, _, done, _ = env.step(a)
            if done:
                env.reset()
                G = road_graph(env.sim)

    return corpus


def run_depth(corpus, num_iterations, log_every=LOG_EVERY, graph_convention="oracle_sage"):
    """
    Runs growing-mode wl_colours over `corpus` at a fixed `num_iterations`
    (L), with a fresh vocab, logging a checkpoint every `log_every` graphs.

    :param graph_convention: "oracle_sage" (default, unchanged behaviour),
        "vilg", or "atom" - selects the wl_colours decoder pair to use,
        matching whatever convention `corpus` was collected with.
    :return: (final vocab dict, list of (graphs_seen, vocab_size) checkpoints)
    """
    vocab = {}
    checkpoints = []

    for i, (x, edge_index, edge_attr) in enumerate(corpus, start=1):
        wl_colours(x, edge_index, edge_attr, num_iterations=num_iterations, vocab=vocab, frozen=False, graph_convention=graph_convention)

        if i % log_every == 0:
            checkpoints.append((i, len(vocab)))
            delta = len(vocab) - (checkpoints[-2][1] if len(checkpoints) > 1 else 0)
            print(f"  L={num_iterations}  graphs={i:>6}  vocab_size={len(vocab):>6}  (+{delta} since last checkpoint)")

    return vocab, checkpoints


def measure_held_out_oov(corpus, frozen_vocab, num_iterations, graph_convention="oracle_sage"):
    """
    Runs frozen-mode wl_colours over every graph in `corpus` against `frozen_vocab`,
    counting how many of the corpus's NODES (not graphs) resolve to the OOV id -
    matching how Cell 4's own held-out claims were phrased (node counts, not graph
    counts), since graphs vary widely in node count and a per-graph average would
    silently under-weight large graphs.

    :return: (oov_fraction, total_nodes, oov_nodes)
    """
    oov_id = frozen_vocab[OOV_SIGNATURE]
    total_nodes = 0
    oov_nodes = 0
    for x, edge_index, edge_attr in corpus:
        colours, _hist = wl_colours(
            x, edge_index, edge_attr, num_iterations=num_iterations,
            vocab=frozen_vocab, frozen=True, graph_convention=graph_convention,
        )
        total_nodes += colours.shape[0]
        oov_nodes += int((colours == oov_id).sum().item())
    fraction = oov_nodes / total_nodes if total_nodes > 0 else float("nan")
    return fraction, total_nodes, oov_nodes


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Choose L for a graph_convention by vocab-growth and held-out OOV, "
                     "not just vocab-growth stabilisation alone."
    )
    parser.add_argument("--graph-convention", default="oracle_sage", choices=["oracle_sage", "vilg", "atom"])
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("-L", "--num-iterations", type=int, nargs="+", default=L_VALUES)
    parser.add_argument("--seed", type=int, default=SEED, help="corpus A (vocab-building) seed")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--steps-per-episode", type=int, default=STEPS_PER_EPISODE)
    parser.add_argument("--held-out-seed", type=int, default=HELD_OUT_SEED, help="B-random seed, must be disjoint from --seed")
    parser.add_argument("--held-out-episodes", type=int, default=HELD_OUT_EPISODES)
    parser.add_argument("--held-out-steps-per-episode", type=int, default=HELD_OUT_STEPS_PER_EPISODE)
    parser.add_argument("--policy-seed", type=int, default=POLICY_SEED, help="B-policy seed, must be disjoint from --seed and --held-out-seed")
    parser.add_argument("--policy-episodes", type=int, default=POLICY_EPISODES)
    parser.add_argument("--policy-steps-per-episode", type=int, default=POLICY_STEPS_PER_EPISODE)
    parser.add_argument("--policy-goals-per-state", type=int, default=POLICY_GOALS_PER_STATE)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    args = parser.parse_args(argv)

    assert len({args.seed, args.held_out_seed, args.policy_seed}) == 3, (
        "corpus A / B-random / B-policy seeds must be pairwise disjoint - "
        f"got seed={args.seed}, held_out_seed={args.held_out_seed}, policy_seed={args.policy_seed}"
    )

    conv = args.graph_convention
    print(f"=== graph_convention={conv!r} scenario={args.scenario!r} ===\n")

    print(f"Sampling corpus A: {args.episodes * args.steps_per_episode} graphs "
          f"({args.episodes} episodes x {args.steps_per_episode} steps, seed={args.seed})...")
    t0 = time.time()
    corpus_a = collect_graph_corpus(args.scenario, args.episodes, args.steps_per_episode, args.seed, graph_convention=conv)
    print(f"collected {len(corpus_a)} graphs in {time.time() - t0:.1f}s\n")

    print(f"Sampling B-random (held-out, random-action): "
          f"{args.held_out_episodes * args.held_out_steps_per_episode} graphs "
          f"({args.held_out_episodes} episodes x {args.held_out_steps_per_episode} steps, seed={args.held_out_seed})...")
    t0 = time.time()
    corpus_b_random = collect_graph_corpus(args.scenario, args.held_out_episodes, args.held_out_steps_per_episode, args.held_out_seed, graph_convention=conv)
    print(f"collected {len(corpus_b_random)} graphs in {time.time() - t0:.1f}s\n")

    print(f"Sampling B-policy (held-out, greedy policy + {args.policy_goals_per_state} planner "
          f"projections/state): up to {args.policy_episodes * args.policy_steps_per_episode * (1 + args.policy_goals_per_state)} graphs "
          f"({args.policy_episodes} episodes x {args.policy_steps_per_episode} steps, seed={args.policy_seed})...")
    t0 = time.time()
    corpus_b_policy = collect_policy_corpus(
        args.scenario, args.policy_episodes, args.policy_steps_per_episode, args.policy_seed,
        graph_convention=conv, goals_per_state=args.policy_goals_per_state,
    )
    print(f"collected {len(corpus_b_policy)} graphs in {time.time() - t0:.1f}s\n")

    results = {}
    for L in args.num_iterations:
        print(f"=== L={L} ===")
        t0 = time.time()
        vocab, checkpoints = run_depth(corpus_a, L, log_every=args.log_every, graph_convention=conv)
        grow_elapsed = time.time() - t0
        vocab_size = freeze_vocab(vocab)

        t0 = time.time()
        oov_random, n_random, oov_n_random = measure_held_out_oov(corpus_b_random, vocab, L, graph_convention=conv)
        t0b = time.time()
        oov_policy, n_policy, oov_n_policy = measure_held_out_oov(corpus_b_policy, vocab, L, graph_convention=conv)
        oov_elapsed = time.time() - t0

        results[L] = {
            "vocab_size": vocab_size,
            "checkpoints": checkpoints,
            "oov_random": oov_random, "n_random": n_random, "oov_n_random": oov_n_random,
            "oov_policy": oov_policy, "n_policy": n_policy, "oov_n_policy": oov_n_policy,
            "elapsed": grow_elapsed + oov_elapsed,
        }
        print(f"L={L}: vocab_size(frozen)={vocab_size}  "
              f"OOV[B-random]={oov_random:.4%} ({oov_n_random}/{n_random} nodes)  "
              f"OOV[B-policy]={oov_policy:.4%} ({oov_n_policy}/{n_policy} nodes)  "
              f"elapsed={grow_elapsed + oov_elapsed:.1f}s\n")

    print("=== summary ===")
    print(f"{'L':>3}  {'vocab_size':>10}  {'OOV[B-random]':>14}  {'OOV[B-policy]':>14}  {'elapsed':>8}")
    for L in args.num_iterations:
        r = results[L]
        print(f"{L:>3}  {r['vocab_size']:>10}  {r['oov_random']:>13.4%}  {r['oov_policy']:>13.4%}  {r['elapsed']:>7.1f}s")

    print("\n=== vocab growth (corpus A) ===")
    for L in args.num_iterations:
        cps = results[L]["checkpoints"]
        print(f"L={L}: " + "  ".join(f"{g}:{v}" for g, v in cps))

    return results


if __name__ == "__main__":
    main()
