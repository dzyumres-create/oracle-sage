"""
Depth-bucketed frozen collision rate for an already-saved WL vocab.

For a frozen vocab file (build_wl_vocab.save_vocab format), runs full episodes of a
greedy policy (build_wl_vocab.greedy_action - nearest passenger, then its destination),
samples the live state every `--sample-every` simulator steps, and projects `--k` random
candidate goals from it with Planner.plan() - the way project_actions does in training.
A pair of candidates COLLIDES when their ground-truth projected states differ
(wl_collision_check.state_key on the planner's own decoders) but their frozen-vocab WL
histograms are identical, i.e. the policy cannot tell them apart. Pairs are grouped by
the live state's step-in-episode depth bucket (wl_depth_sweep.BUCKETS) and by pair type
(wl_collision_check.classify_goal); move-move is the pair type that collides in
practice (see docs/cell6_wl_diagnostics.md).

This is the committed tool behind the depth-bucketed frozen collision tables in
docs/cell4_L1.md. (The atom L2_full table in docs/cell6_wl_diagnostics.md was produced
with the same settings by an uncommitted script whose candidate RNG wiring is not
recorded, so its exact counts are not reproducible with this tool.)

RNG use: env.seed(seed) for mazes/spawns; one numpy RandomState(seed) used only for
candidate selection (wl_collision_check.select_candidates). greedy_action is
deterministic.

Usage (repo root, PYTHONPATH=.):
    python -m sage.domains.utils.wl_collision_depth --graph-convention vilg \
        --vocab-path sage/domains/utils/wl_vocab_taxi_city_vilg_L1_full.json \
        --seed 700001 --episodes 8 --sample-every 20 --k 15
"""
import argparse
import copy
import itertools
import time

import numpy as np
import torch as th

# build_wl_vocab carries the numpy/gym compat shims needed to construct a city env on
# newer stacks (import side effect; no-ops on RCP's pinned stack).
from sage.domains.utils.build_wl_vocab import (
    MASK,
    REWARDS_VARIANT,
    greedy_action,
    load_vocab_with_metadata,
    road_graph,
    to_planner_data,
)
from sage.domains.utils.wl_collision_check import classify_goal, decode_state, select_candidates, state_key
from sage.domains.utils.wl_depth_sweep import BUCKETS, bucket_of
from sage.domains.utils.wl_colours import wl_colours
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.simulator.planner import Planner


def bucket_name(bucket):
    lo, hi = bucket
    return f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"


def measure_collisions_bucketed(graph_convention, vocab, num_iterations, scenario="city", seed=700001,
                                episodes=8, sample_every=20, k=15, max_steps=2500, log=print):
    """
    :return: dict[bucket] -> {"states": int, "pairs": {pair_type: int},
        "collisions": {pair_type: int}}, pair_type a sorted 2-tuple of goal types
    """
    env = GraphTaxiEnv(representation="graph", scenario=scenario, mask=MASK, rewards=REWARDS_VARIANT,
                       graph_convention=graph_convention)
    env.seed(seed)
    rng = np.random.RandomState(seed)
    planner = Planner(graph_convention=graph_convention)
    results = {b: {"states": 0, "pairs": {}, "collisions": {}} for b in BUCKETS}

    t0 = time.time()
    for episode in range(episodes):
        env.reset()
        G = road_graph(env.sim)
        for step in range(max_steps):
            if step % sample_every == 0:
                data = to_planner_data(env.sim, graph_convention=graph_convention)
                r = results[bucket_of(step)]
                r["states"] += 1
                candidates = select_candidates(data.mask, False, k, rng)
                state = decode_state(data.x, data.edge_index, data.edge_attr, graph_convention)
                types, keys, hists = {}, {}, {}
                for goal in candidates:
                    projection, _actions = planner.plan(copy.deepcopy(data), goal)
                    types[goal] = classify_goal(state, goal)
                    keys[goal] = state_key(decode_state(projection.x, projection.edge_index, projection.edge_attr,
                                                        graph_convention))
                    _colours, hists[goal] = wl_colours(
                        projection.x, projection.edge_index, projection.edge_attr,
                        num_iterations=num_iterations, vocab=vocab, frozen=True, graph_convention=graph_convention,
                    )
                for g1, g2 in itertools.combinations(candidates, 2):
                    if keys[g1] == keys[g2]:
                        continue
                    pair_type = tuple(sorted((types[g1], types[g2])))
                    r["pairs"][pair_type] = r["pairs"].get(pair_type, 0) + 1
                    if th.equal(hists[g1], hists[g2]):
                        r["collisions"][pair_type] = r["collisions"].get(pair_type, 0) + 1
            _, _, done, _ = env.step(greedy_action(env.sim, G))
            if done:
                break
        log(f"  episode {episode + 1}/{episodes} done ({time.time() - t0:.0f}s)")
    return results


def print_table(results, pair_type=("move", "move")):
    label = "-".join(pair_type)
    print(f"| bucket | states | {label} pairs | collisions | collision% |")
    print("|---|---:|---:|---:|---:|")
    for bucket, r in results.items():
        pairs = r["pairs"].get(pair_type, 0)
        coll = r["collisions"].get(pair_type, 0)
        pct = 100.0 * coll / pairs if pairs else 0.0
        print(f"| {bucket_name(bucket)} | {r['states']} | {pairs:,} | {coll} | {pct:.4f}% |")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Depth-bucketed frozen WL collision rate for a saved vocab.")
    parser.add_argument("--graph-convention", required=True, choices=["oracle_sage", "vilg", "atom"])
    parser.add_argument("--vocab-path", required=True)
    parser.add_argument("-L", "--num-iterations", type=int, default=None,
                        help="only for vocabs without recorded num_iterations metadata")
    parser.add_argument("--scenario", default="city")
    parser.add_argument("--seed", type=int, default=700001)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args(argv)

    vocab, metadata = load_vocab_with_metadata(args.vocab_path)
    recorded = metadata.get("num_iterations")
    if recorded is not None and args.num_iterations is not None and recorded != args.num_iterations:
        parser.error(f"vocab records L={recorded}, but -L {args.num_iterations} was given")
    num_iterations = recorded if recorded is not None else args.num_iterations
    if num_iterations is None:
        parser.error("vocab has no recorded num_iterations; pass -L")
    if metadata.get("graph_convention", args.graph_convention) != args.graph_convention:
        parser.error(f"vocab was built for graph_convention={metadata['graph_convention']!r}")

    print(f"vocab={args.vocab_path} graph_convention={args.graph_convention} L={num_iterations} "
          f"size={len(vocab)}; seed={args.seed} episodes={args.episodes} sample_every={args.sample_every} k={args.k}")
    results = measure_collisions_bucketed(
        args.graph_convention, vocab, num_iterations, scenario=args.scenario, seed=args.seed,
        episodes=args.episodes, sample_every=args.sample_every, k=args.k,
    )
    print_table(results)
    others = sorted({pt for r in results.values() for pt in r["pairs"]} - {("move", "move")})
    for pt in others:
        total = sum(r["pairs"].get(pt, 0) for r in results.values())
        coll = sum(r["collisions"].get(pt, 0) for r in results.values())
        print(f"{'-'.join(pt)}: {coll}/{total} pairs collide (all buckets)")
    return results


if __name__ == "__main__":
    main()
