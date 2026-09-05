"""
Full-episode smoke test for graph_convention="vilg" -- not testing any new behaviour,
just driving ~100 consecutive steps with multiple passengers ever in flight and
confirming nothing crashes and the graph stays internally consistent at every step.

Run from the repo root: python docs/test_vilg_episode.py
"""
import sys
import traceback

import numpy.random as npr

from sage.domains.gym_taxi.simulator.taxi_world import TaxiWorldSimulator
from sage.domains.gym_taxi.utils.representations import env_to_vilg_graph

N_STEPS = 100
ACHIEVED_GOAL = [1, 0, 0]

env_random = npr.RandomState(42)
action_random = npr.RandomState(99)  # independent stream, only used to sample actions
# (seed chosen so this run actually completes >=1 delivery in ~100 random-walk steps on
# the small grid below, exercising the achieved_propositional_goal count check non-trivially)

env = TaxiWorldSimulator(
    random=env_random,
    size=3,  # small grid so random legal actions plausibly complete a delivery in ~100 steps
    random_walls=False,
    planning=False,  # restricted (SR-DRL-style) mask -- only genuinely legal actions are True
    graph_convention="vilg",
    concurrent_passengers=3,
    passenger_creation_probability=1,
    delivery_limit=1000,  # avoid hitting done mid-run; we want ~100 uninterrupted steps
    timeout=500,
)

delivered_count = 0  # tracked from reward events only, never re-derived from the graph


def classify_nodes(node_feats):
    """object rows have a zeroed predicate/status block; proposition rows have a zeroed
    object-type block -- this is the zero-padding scheme env_to_vilg_graph uses to give
    both kinds a shared 9-dim feature vector (see representations.py docstring)."""
    is_object = node_feats[:, 3:9].sum(axis=1) == 0
    is_proposition = node_feats[:, 0:3].sum(axis=1) == 0
    return is_object, is_proposition


def check_state(step, node_feats, edge_feats, edge_index, mask, global_feats, delivered_count):
    n_nodes = node_feats.shape[0]

    assert node_feats.shape[1] == 9, f"node_feats dim drifted: {node_feats.shape[1]}"
    assert edge_feats.shape[1] == 2, f"edge_feats dim drifted: {edge_feats.shape[1]}"

    if edge_index.size > 0:
        assert edge_index.min() >= 0, f"negative node index in edge_index: {edge_index.min()}"
        assert edge_index.max() < n_nodes, (
            f"edge_index references node {edge_index.max()}, but only {n_nodes} nodes exist"
        )

    is_object, is_proposition = classify_nodes(node_feats)
    assert is_object.sum() + is_proposition.sum() == n_nodes, "a node is classified as neither/both"

    prop_indices = set(i for i in range(n_nodes) if is_proposition[i])
    src_with_edge = set(edge_index[0].tolist()) if edge_index.size > 0 else set()
    orphans = prop_indices - src_with_edge
    assert not orphans, f"orphan proposition node(s) with no edge to an object node: {orphans}"

    achieved_goal_count = sum(
        1 for i in range(n_nodes) if is_proposition[i] and list(node_feats[i][6:9]) == ACHIEVED_GOAL
    )
    assert achieved_goal_count == delivered_count, (
        f"achieved_propositional_goal node count ({achieved_goal_count}) != "
        f"delivered passengers seen so far ({delivered_count})"
    )

    assert mask.shape[0] == n_nodes, "mask length doesn't match node count"


try:
    nf, ef, ei, mask, gf = env_to_vilg_graph(env)
    check_state(-1, nf, ef, ei, mask, gf, delivered_count)
    print(f"step -1 (initial state): {nf.shape[0]} nodes, {ei.shape[1]} edges, mask True count {mask.sum()}")

    last_ok_step = -1
    for step in range(N_STEPS):
        legal = mask.nonzero()[0]
        assert legal.size > 0, f"step {step}: no legal actions in mask"
        action = int(action_random.choice(legal))

        obs, reward, done, info = env.act(action)
        if reward == env.rewards["drop-off"]:
            delivered_count += 1

        nf, ef, ei, mask, gf = env_to_vilg_graph(env)
        check_state(step, nf, ef, ei, mask, gf, delivered_count)
        last_ok_step = step

        if done:
            print(f"step {step}: done=True (delivery_limit exhausted) -- stopping early")
            break

    print()
    print(f"completed {last_ok_step + 1} / {N_STEPS} steps without a crash or failed assertion")
    print(f"deliveries observed: {delivered_count}")
    print(f"final node count: {nf.shape[0]}, final edge count: {ei.shape[1]}")
    print()
    print("ALL CHECKS PASSED")

except Exception:
    print(f"\nCRASHED at step {last_ok_step + 1} (last good step: {last_ok_step})\n", file=sys.stderr)
    print("state at the last good step:", file=sys.stderr)
    print(f"  taxi: {env.taxi}", file=sys.stderr)
    print(f"  passengers: {env.passengers}", file=sys.stderr)
    print(f"  delivered_count so far: {delivered_count}", file=sys.stderr)
    print(f"  node count: {nf.shape[0] if 'nf' in globals() else 'n/a'}", file=sys.stderr)
    print(file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
