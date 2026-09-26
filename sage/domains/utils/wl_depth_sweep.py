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

   `GraphTaxiEnv`/`MASK`/`REWARDS_VARIANT`/`sample_action`/`greedy_action`/
   `run_full_episode_states` are imported directly from build_wl_vocab.py rather than
   duplicated here - this also means build_wl_vocab.py's numpy/gym compat shims (see
   its docstring) run automatically as an import side effect, so this script needs
   none of its own. `run_full_episode_states` is also where env.seed()/full-episode/
   Planner-projection logic now lives, shared with build_wl_vocab.sample_graphs - see
   its docstring for why (episode-depth undercounting of real-world OOV).

   Depth-bucketed held-out OOV (BUCKETS below) is the standard report as of Cell 6's
   "near-zero OOV at every depth of full-length training episodes" standard: a flat/
   pooled OOV number hides that OOV climbs steeply with episode depth, since shallow
   states are otherwise over-represented relative to how a real ~2000-step training
   episode actually spends its time.

Run from the repo root: python -m sage.domains.utils.wl_depth_sweep --help
"""
import argparse
import time

import numpy as np

from sage.domains.utils.build_wl_vocab import (
    GraphTaxiEnv, MASK, REWARDS_VARIANT, sample_action, greedy_action, run_full_episode_states,
    load_vocab_with_metadata,
)
from sage.domains.utils.wl_colours import OOV_SIGNATURE, freeze_vocab, wl_colours

SCENARIO = "predictable5"
L_VALUES = [1, 2, 3, 4]
LOG_EVERY = 100

# corpus A (vocab-building) defaults - "quick pass" scale; bump via CLI for a full run
EPISODES = 20
SAMPLE_EVERY = 50
SEED = 0

# held-out corpora defaults - deliberately far outside corpus A's seed range
HELD_OUT_SEED = 500_000
HELD_OUT_EPISODES = 10
HELD_OUT_SAMPLE_EVERY = 50

POLICY_SEED = 900_000
POLICY_EPISODES = 10
POLICY_SAMPLE_EVERY = 50
POLICY_GOALS_PER_STATE = 3

# depth-bucketed held-out OOV: the standard report as of Cell 6's "near-zero OOV at
# every depth of full-length training episodes" standard - a flat/pooled OOV number
# hides the fact that OOV climbs steeply with episode depth (shallow states are
# over-represented relative to how a real ~2000-step training episode actually spends
# its time), so bucketing by simulator-step depth is the sharper, non-misleading report.
BUCKETS = [(0, 60), (60, 250), (250, 1000), (1000, 10 ** 9)]


def bucket_of(step):
    """Returns the (lo, hi) tuple from BUCKETS containing `step`; steps at or beyond
    the last bucket's lo always fall in that last (open-ended) bucket."""
    for lo, hi in BUCKETS:
        if lo <= step < hi:
            return (lo, hi)
    return BUCKETS[-1]


def collect_graph_corpus(scenario, episodes, sample_every, seed, graph_convention="oracle_sage", max_steps=2500):
    """
    Samples a FIXED corpus of graphs from `episodes` FULL episodes (run to natural
    termination via build_wl_vocab.run_full_episode_states - NOT stopped early after a
    fixed step count), sampling a state every `sample_every` simulator steps under
    uniform-random legal actions (build_wl_vocab.sample_action) - the same sampling
    build_wl_vocab.sample_graphs' random-action half uses. Does NOT run wl_colours
    during collection, so the resulting corpus can be replayed identically across
    multiple L values.

    env.seed(seed) is called once before the first episode, so - unlike this
    function's previous version, which only ever seeded numpy's GLOBAL random state
    and left the env's own maze/passenger-spawn generator on unseeded OS entropy at
    construction - the entire sequence of `episodes` mazes/spawns is now genuinely
    reproducible from `seed` (and genuinely disjoint from another call whose `seed`
    differs).

    :param graph_convention: "oracle_sage" (default, unchanged behaviour),
        "vilg", or "atom" - selects both the GraphTaxiEnv construction and
        the translator used to read each sampled graph.
    :return: list of (x, edge_index, edge_attr) torch tensor triples, one per sampled graph
    """
    env = GraphTaxiEnv(representation="graph", scenario=scenario, mask=MASK, rewards=REWARDS_VARIANT, graph_convention=graph_convention)
    env.seed(seed)
    # sample_action reads numpy's GLOBAL random state directly (np.random.choice), not
    # `rng` below - seeding it here too is required for full reproducibility.
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    corpus = []
    for _ in range(episodes):
        env.reset()
        for _step, x, edge_index, edge_attr in run_full_episode_states(
            env, sample_every, graph_convention, lambda sim, G: sample_action(sim), goals_per_state=0, rng=rng, max_steps=max_steps,
        ):
            corpus.append((x, edge_index, edge_attr))

    return corpus


def collect_policy_corpus(scenario, episodes, sample_every, seed, graph_convention="oracle_sage", goals_per_state=POLICY_GOALS_PER_STATE, max_steps=2500):
    """
    B-policy: pure greedy_action (nearest-passenger-then-destination, eps=0) stepping
    through `episodes` FULL episodes, so pickups and deliveries actually happen - PLUS
    Planner.plan() projections of `goals_per_state` random legal candidate goals at
    every sampled state (mirroring project_actions' own planner.plan(deepcopy(state),
    goal) call pattern; see build_wl_vocab.run_full_episode_states for the shared
    implementation).

    :return: list of (x, edge_index, edge_attr) torch tensor triples - both live-stepped
        states AND planner-projected states, NOT distinguished in the returned list (the
        caller decides how to report them; this function's job is just to sample both).
    """
    env = GraphTaxiEnv(representation="graph", scenario=scenario, mask=MASK, rewards=REWARDS_VARIANT, graph_convention=graph_convention)
    env.seed(seed)
    rng = np.random.RandomState(seed)

    corpus = []
    for _ in range(episodes):
        env.reset()
        for _step, x, edge_index, edge_attr in run_full_episode_states(
            env, sample_every, graph_convention, greedy_action, goals_per_state, rng, max_steps=max_steps,
        ):
            corpus.append((x, edge_index, edge_attr))

    return corpus


def collect_bucketed_corpus(scenario, graph_convention, episodes, seed, sample_every, policy, goals_per_state, max_steps=2500):
    """
    Like collect_graph_corpus/collect_policy_corpus, but groups the sampled triples by
    the LIVE state's step-in-episode depth bucket (BUCKETS) instead of returning one
    flat list - a planner projection inherits its SOURCE state's bucket, not a
    hypothetical post-projection depth, since a projection is a one-step lookahead
    FROM that depth, not a state actually reached one step deeper.

    :param policy: callable(sim, road_graph) -> action, e.g. `greedy_action` or
        `lambda sim, G: sample_action(sim)`
    :return: dict[(lo, hi)] -> list of (x, edge_index, edge_attr) triples
    """
    buckets = {b: [] for b in BUCKETS}

    env = GraphTaxiEnv(representation="graph", scenario=scenario, mask=MASK, rewards=REWARDS_VARIANT, graph_convention=graph_convention)
    env.seed(seed)
    # `policy` may be sample_action, which reads numpy's GLOBAL random state directly
    # (np.random.choice), not `rng` below - seeding it here too is required for full
    # reproducibility (a no-op for a purely-deterministic policy like greedy_action).
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    for _ in range(episodes):
        env.reset()
        for step, x, edge_index, edge_attr in run_full_episode_states(
            env, sample_every, graph_convention, policy, goals_per_state, rng, max_steps=max_steps,
        ):
            buckets[bucket_of(step)].append((x, edge_index, edge_attr))

    return buckets


def collect_bucketed_random_corpus(scenario, episodes, seed, sample_every, graph_convention="oracle_sage", max_steps=2500):
    """B-random, bucketed: uniform-random legal actions throughout, no planner projections."""
    return collect_bucketed_corpus(
        scenario, graph_convention, episodes, seed, sample_every,
        policy=lambda sim, G: sample_action(sim), goals_per_state=0, max_steps=max_steps,
    )


def collect_bucketed_policy_corpus(scenario, episodes, seed, sample_every, graph_convention="oracle_sage", goals_per_state=POLICY_GOALS_PER_STATE, max_steps=2500):
    """B-policy, bucketed: pure greedy_action (eps=0) plus `goals_per_state` planner
    projections per sampled state."""
    return collect_bucketed_corpus(
        scenario, graph_convention, episodes, seed, sample_every,
        policy=greedy_action, goals_per_state=goals_per_state, max_steps=max_steps,
    )


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


def measure_held_out_oov_bucketed(bucketed_corpus, frozen_vocab, num_iterations, graph_convention="oracle_sage"):
    """Applies measure_held_out_oov independently within each bucket of a
    collect_bucketed_corpus (or collect_bucketed_random_corpus/collect_bucketed_policy_corpus)
    result - the standard depth-bucketed report (see module docstring).

    :return: dict[(lo, hi)] -> (oov_fraction, total_nodes, oov_nodes, n_graphs)
    """
    results = {}
    for b, corpus in bucketed_corpus.items():
        fraction, total_nodes, oov_nodes = measure_held_out_oov(corpus, frozen_vocab, num_iterations, graph_convention=graph_convention)
        results[b] = (fraction, total_nodes, oov_nodes, len(corpus))
    return results


def print_bucketed_oov_table(results, label):
    """Prints a `results` dict from measure_held_out_oov_bucketed as a table, in BUCKETS order."""
    print(f"\n=== bucketed OOV: {label} ===")
    print(f"{'bucket':>14}  {'graphs':>7}  {'nodes':>9}  {'oov_nodes':>9}  {'oov%':>8}")
    for b in BUCKETS:
        frac, nodes, oov, n_graphs = results[b]
        lo, hi = b
        bucket_label = f"{lo}-{hi if hi < 10 ** 9 else 'inf'}"
        print(f"{bucket_label:>14}  {n_graphs:>7}  {nodes:>9}  {oov:>9}  {frac:>7.4%}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Choose L for a graph_convention by vocab-growth and depth-bucketed "
                     "held-out OOV (the Cell 6 standard - see module docstring); or, with "
                     "--vocab-path, measure an already-saved frozen vocab against that same "
                     "standard without rebuilding it (Task 2/4-style measure-only runs)."
    )
    parser.add_argument("--graph-convention", default="oracle_sage", choices=["oracle_sage", "vilg", "atom"])
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument(
        "-L", "--num-iterations", type=int, nargs="+", default=L_VALUES,
        help="in --vocab-path mode, only used as a fallback for vocabs saved without "
             "recorded num_iterations metadata (first value used); otherwise the L "
             "values to build+measure vocabs at from corpus A.",
    )
    parser.add_argument(
        "--vocab-path", default=None,
        help="measure-only mode: load this already-frozen vocab (build_wl_vocab.save_vocab "
             "format) instead of building one from corpus A via run_depth, and report its "
             "depth-bucketed held-out OOV for both B-random and B-policy.",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="corpus A (vocab-building) seed; unused in --vocab-path mode")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--sample-every", type=int, default=SAMPLE_EVERY)
    parser.add_argument("--held-out-seed", type=int, default=HELD_OUT_SEED, help="B-random seed, must be disjoint from --seed and --policy-seed")
    parser.add_argument("--held-out-episodes", type=int, default=HELD_OUT_EPISODES)
    parser.add_argument("--held-out-sample-every", type=int, default=HELD_OUT_SAMPLE_EVERY)
    parser.add_argument("--policy-seed", type=int, default=POLICY_SEED, help="B-policy seed, must be disjoint from --seed and --held-out-seed")
    parser.add_argument("--policy-episodes", type=int, default=POLICY_EPISODES)
    parser.add_argument("--policy-sample-every", type=int, default=POLICY_SAMPLE_EVERY)
    parser.add_argument("--policy-goals-per-state", type=int, default=POLICY_GOALS_PER_STATE)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    args = parser.parse_args(argv)

    assert len({args.seed, args.held_out_seed, args.policy_seed}) == 3, (
        "corpus A / B-random / B-policy seeds must be pairwise disjoint - "
        f"got seed={args.seed}, held_out_seed={args.held_out_seed}, policy_seed={args.policy_seed}"
    )

    conv = args.graph_convention
    print(f"=== graph_convention={conv!r} scenario={args.scenario!r} ===\n")

    print(f"Sampling B-random (held-out, random-action, bucketed, FULL episodes): "
          f"{args.held_out_episodes} episodes, sample_every={args.held_out_sample_every}, seed={args.held_out_seed}...")
    t0 = time.time()
    buckets_random = collect_bucketed_random_corpus(
        args.scenario, args.held_out_episodes, args.held_out_seed, args.held_out_sample_every, graph_convention=conv,
    )
    n_random = sum(len(v) for v in buckets_random.values())
    print(f"collected {n_random} graphs across {len(BUCKETS)} buckets in {time.time() - t0:.1f}s\n")

    print(f"Sampling B-policy (held-out, greedy + {args.policy_goals_per_state} planner "
          f"projections/state, bucketed, FULL episodes): {args.policy_episodes} episodes, "
          f"sample_every={args.policy_sample_every}, seed={args.policy_seed}...")
    t0 = time.time()
    buckets_policy = collect_bucketed_policy_corpus(
        args.scenario, args.policy_episodes, args.policy_seed, args.policy_sample_every,
        graph_convention=conv, goals_per_state=args.policy_goals_per_state,
    )
    n_policy = sum(len(v) for v in buckets_policy.values())
    print(f"collected {n_policy} graphs across {len(BUCKETS)} buckets in {time.time() - t0:.1f}s\n")

    if args.vocab_path:
        vocab, metadata = load_vocab_with_metadata(args.vocab_path)
        L = metadata.get("num_iterations", args.num_iterations[0])
        print(f"loaded vocab: path={args.vocab_path!r}  size={len(vocab)}  "
              f"graph_convention={metadata.get('graph_convention')!r}  num_iterations={L}\n")

        oov_random = measure_held_out_oov_bucketed(buckets_random, vocab, L, graph_convention=conv)
        oov_policy = measure_held_out_oov_bucketed(buckets_policy, vocab, L, graph_convention=conv)
        print_bucketed_oov_table(oov_random, f"{args.vocab_path} B-random")
        print_bucketed_oov_table(oov_policy, f"{args.vocab_path} B-policy")
        return {"vocab_path": args.vocab_path, "vocab_size": len(vocab), "L": L, "oov_random": oov_random, "oov_policy": oov_policy}

    print(f"Sampling corpus A (vocab-building, FULL episodes): {args.episodes} episodes, "
          f"sample_every={args.sample_every}, seed={args.seed}...")
    t0 = time.time()
    corpus_a = collect_graph_corpus(args.scenario, args.episodes, args.sample_every, args.seed, graph_convention=conv)
    print(f"collected {len(corpus_a)} graphs in {time.time() - t0:.1f}s\n")

    results = {}
    for L in args.num_iterations:
        print(f"=== L={L} ===")
        t0 = time.time()
        vocab, checkpoints = run_depth(corpus_a, L, log_every=args.log_every, graph_convention=conv)
        grow_elapsed = time.time() - t0
        vocab_size = freeze_vocab(vocab)

        t0 = time.time()
        oov_random = measure_held_out_oov_bucketed(buckets_random, vocab, L, graph_convention=conv)
        oov_policy = measure_held_out_oov_bucketed(buckets_policy, vocab, L, graph_convention=conv)
        oov_elapsed = time.time() - t0

        results[L] = {
            "vocab_size": vocab_size,
            "checkpoints": checkpoints,
            "oov_random": oov_random,
            "oov_policy": oov_policy,
            "elapsed": grow_elapsed + oov_elapsed,
        }
        print_bucketed_oov_table(oov_random, f"L={L} B-random")
        print_bucketed_oov_table(oov_policy, f"L={L} B-policy")
        print(f"L={L}: vocab_size(frozen)={vocab_size}  elapsed={grow_elapsed + oov_elapsed:.1f}s\n")

    print("\n=== vocab growth (corpus A) ===")
    for L in args.num_iterations:
        cps = results[L]["checkpoints"]
        print(f"L={L}: " + "  ".join(f"{g}:{v}" for g, v in cps))

    return results


if __name__ == "__main__":
    main()
