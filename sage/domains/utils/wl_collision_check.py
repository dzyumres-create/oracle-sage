"""
.. module:: wl_collision_check
   :synopsis: Standalone diagnostic script measuring how often WL colour refinement
   fails to distinguish two DIFFERENT candidate-goal outcomes from the same state -
   the same question Cell 4's uncommitted vilg diagnostic reportedly answered ("L=1
   collides completely on real move-vs-move candidates; L=2 resolves it", commit
   b57707d), but as a reproducible, committed tool, generalised to oracle_sage/atom too.

   NOT part of build_wl_vocab.py's vocab-building workflow, and unlike
   wl_depth_sweep.py (corpus-sampling/vocab-growth), this drives Planner.plan() the
   same way project_actions (sage/agent/graph_plan_feedback_policy.py) does: for a
   sampled state, project several candidate goals via planner.plan(deepcopy(state),
   goal), then compare pairs of projections.

   A "collision" is a pair of candidates whose GROUND-TRUTH projected logical state
   differs (different taxi location/passenger, or any passenger's location/
   destination - decoded via planner.py's own graph_to_networkx/_vilg/graph_to_state_atom,
   ignoring the road graph itself, which a projection never changes) but whose WL
   HISTOGRAM is identical - i.e. a discriminator reading only the histogram (exactly
   what WLPlanFeedbackPolicy's value/path-value heads do) cannot tell them apart.
   Pairs whose ground truth already matches are excluded from both numerator and
   denominator - a histogram match there is correct, not a collision.

   Three histogram measurements per pair, all against the SAME ground-truth
   differing/matching split:
     - growing (fresh vocab per L, one shared growing vocab across the whole run for
       that L, never frozen so no OOV is possible) - tests the ENCODING's own
       expressiveness at depth L, independent of any specific vocab's coverage.
     - frozen (the vocab wl_depth_sweep.py's tools would build/freeze at that L,
       built inline here from a corpus disjoint from the states under test) - what a
       real deployed policy actually sees, where OOV can itself manufacture spurious
       collisions (two out-of-vocab states both collapsing onto the OOV id looks
       identical to the discriminator, structure aside).
     - floor (run each candidate's OWN colour refinement to a fixed point - the
       partition stops changing between consecutive iterations, capped at
       `--stable-cap`, default 15 - using one shared growing vocab across the whole
       run, not per-L, since it is not parameterised by L at all). Pairs still
       collided at the floor are indistinguishable by WL colour refinement AT ANY
       DEPTH, not just the L values swept - the sharpest of the three numbers.

   Results are broken down by candidate-PAIR TYPE (move-move, move-pickup,
   pickup-pickup, move-dropoff, pickup-dropoff, dropoff-dropoff), classified the same
   way Planner.plan() itself dispatches a goal (planner.py: goal==state.taxi.node ->
   dropoff; goal in passenger nodes -> pickup; else -> move) - Cell 4's claim was
   specifically about move-move pairs.

   Candidate selection is EXHAUSTIVE (every currently-selectable type-atom/object row)
   on small scenarios (predictable5: ~10-20 objects, C(k,2) is cheap and complete) and
   a RANDOM K-SUBSET (default k=15, seeded) on city (where k can exceed 400 and
   exhaustive pairwise projection is infeasible) - `--exhaustive`/`--k` override the
   scenario-based default either way. States are live episodes stepped with
   build_wl_vocab.greedy_action (so pickups/deliveries actually occur, giving
   pickup/dropoff pair types real coverage, not just move-move), snapshotting a state
   every `--sample-every` steps (not every step - consecutive states differ by one
   action and are highly autocorrelated, wasting sample budget).

   `GraphTaxiEnv`/`MASK`/`REWARDS_VARIANT`/`road_graph`/`greedy_action`/
   `to_planner_data`/`Planner` are imported from build_wl_vocab.py;
   `collect_graph_corpus`/`run_depth` (for the inline frozen-vocab build) from
   wl_depth_sweep.py - none of this is duplicated here.

Run from the repo root: python -m sage.domains.utils.wl_collision_check --help
"""
import argparse
import copy
import itertools
import time
from collections import defaultdict

import numpy as np
import torch as th
import networkx as nx

from sage.domains.utils.build_wl_vocab import (
    GraphTaxiEnv, MASK, REWARDS_VARIANT, road_graph, greedy_action, to_planner_data, Planner,
)
from sage.domains.utils.wl_depth_sweep import collect_graph_corpus, run_depth
from sage.domains.utils.wl_colours import (
    freeze_vocab, wl_colours, refine,
    edge_labels, edge_labels_vilg, edge_labels_atom,
)
from sage.domains.gym_taxi.simulator.planner import graph_to_networkx, graph_to_networkx_vilg, graph_to_state_atom

SCENARIO = "predictable5"
L_VALUES = [1, 2]
STABLE_CAP = 15

STATE_SEED = 0
EPISODES = 5
STEPS_PER_EPISODE = 60
SAMPLE_EVERY = 5

VOCAB_SEED = 700_000  # disjoint from STATE_SEED - the frozen vocab must not be built on the exact states under test
VOCAB_EPISODES = 20
VOCAB_STEPS_PER_EPISODE = 50

K_CANDIDATES = 15

PAIR_TYPES = [
    ("dropoff", "dropoff"), ("dropoff", "move"), ("dropoff", "pickup"),
    ("move", "move"), ("move", "pickup"), ("pickup", "pickup"),
]


def classify_goal(state, goal):
    """Mirrors Planner.plan()'s own dispatch exactly (planner.py): "dropoff" (goal is
    the taxi's own node), "pickup" (goal is a passenger's node), else "move"."""
    if goal == state.taxi.node:
        return "dropoff"
    if goal in [p.node for p in state.passengers]:
        return "pickup"
    return "move"


def decode_state(x, edge_index, edge_attr, graph_convention):
    """Decodes a projection back to planner.py's own State(graph, taxi, passengers) -
    the SAME decoders Planner.plan() itself uses (graph_to_networkx/_vilg/
    graph_to_state_atom), so "does the ground truth differ" is judged by the exact
    same logical-state notion the planner operates on, not a bespoke comparator."""
    graph = _Wrapper(x, edge_index, edge_attr)
    if graph_convention == "vilg":
        return graph_to_networkx_vilg(graph)
    if graph_convention == "atom":
        state, _atoms = graph_to_state_atom(graph)
        return state
    return graph_to_networkx(graph)


class _Wrapper:
    """Minimal x/edge_index/edge_attr holder - graph_to_networkx*/graph_to_state_atom
    only ever read these three attributes off their `graph` argument."""
    def __init__(self, x, edge_index, edge_attr):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr


def state_key(state):
    """Ground-truth logical-state key: taxi location/passenger, and every passenger's
    (node, location, destination) as a frozenset (order-independent). The road graph
    itself is excluded - no projection ever changes it, so it can never distinguish
    two candidates from the same state."""
    passengers = frozenset((int(p.node), int(p.location), int(p.destination)) for p in state.passengers)
    taxi_passenger = None if state.taxi.passenger is None else int(state.taxi.passenger)
    return (int(state.taxi.location), taxi_passenger, passengers)


def select_candidates(mask, exhaustive, k, rng):
    selectable = mask.nonzero(as_tuple=True)[0].tolist()
    if exhaustive:
        return selectable
    n = min(k, len(selectable))
    if n <= 0:
        return []
    return [int(g) for g in rng.choice(selectable, size=n, replace=False)]


def _edge_labels(edge_attr, graph_convention):
    if graph_convention == "vilg":
        return edge_labels_vilg(edge_attr)
    if graph_convention == "atom":
        return edge_labels_atom(edge_attr)
    return edge_labels(edge_attr)


def _partition_of(colours):
    groups = defaultdict(list)
    for idx, c in enumerate(colours.tolist()):
        groups[c].append(idx)
    return frozenset(frozenset(v) for v in groups.values())


def wl_to_stable_partition(x, edge_index, edge_attr, vocab, graph_convention, max_iterations=STABLE_CAP):
    """
    Runs this ONE graph's own colour refinement until its partition (which nodes share
    a colour) stops changing between consecutive iterations, capped at
    `max_iterations` - the standard 1-WL fixed-point termination test, not a fixed L.
    Uses `vocab` (growing, shared across the whole run so ids - and therefore
    histograms - stay comparable across different graphs/candidates) via the PUBLIC
    wl_colours()/refine()/edge_labels* API only: num_iterations=0 already runs exactly
    the init-resolution step wl_colours() does internally, with no refine() calls, so
    it is reused here as the L=0 starting point rather than re-deriving that
    resolution logic locally.

    :return: (colours, histogram) at the stable (or capped) iteration
    """
    colours, _hist = wl_colours(x, edge_index, edge_attr, num_iterations=0, vocab=vocab, frozen=False, graph_convention=graph_convention)
    labels = _edge_labels(edge_attr, graph_convention)

    prev_partition = _partition_of(colours)
    for _ in range(max_iterations):
        colours = refine(colours, edge_index, labels, vocab, frozen=False)
        partition = _partition_of(colours)
        if partition == prev_partition:
            break
        prev_partition = partition

    histogram = th.bincount(colours, minlength=len(vocab)).float()
    return colours, histogram


def _hist_equal(h1, h2):
    """Growing-mode vocabs can grow BETWEEN two histogram computations in the same
    run, so the two tensors can differ in length even when semantically comparable -
    pad the shorter with zeros (safe: any id beyond the shorter one's length did not
    exist yet when it was computed, so no node of that graph could have had it)."""
    n = max(h1.shape[0], h2.shape[0])
    if h1.shape[0] < n:
        h1 = th.cat([h1, th.zeros(n - h1.shape[0])])
    if h2.shape[0] < n:
        h2 = th.cat([h2, th.zeros(n - h2.shape[0])])
    return th.equal(h1, h2)


def to_nx_digraph(x, edge_index, edge_attr):
    """A DiGraph with each node/edge's exact one-hot feature vector as an attribute -
    directed, not undirected, since e.g. vilg's proposition->object and
    object->proposition edges carry the SAME label but are structurally distinct
    (source vs target), which only a directed comparison respects."""
    G = nx.DiGraph()
    for i, feat in enumerate(x.tolist()):
        G.add_node(i, feat=tuple(feat))
    ei = edge_index.tolist()
    ea = edge_attr.tolist()
    for e in range(len(ei[0])):
        G.add_edge(ei[0][e], ei[1][e], feat=tuple(ea[e]))
    return G


def graphs_isomorphic(x1, edge_index1, edge_attr1, x2, edge_index2, edge_attr2):
    """True iff the two projections are isomorphic AS ATTRIBUTED GRAPHS (node_match/
    edge_match compare the exact one-hot feature tuples) - the ground-truth structural
    check for "is this WL collision genuine (non-isomorphic graphs WL cannot tell
    apart) or would ANY graph-aware method also see these as the same graph"."""
    G1 = to_nx_digraph(x1, edge_index1, edge_attr1)
    G2 = to_nx_digraph(x2, edge_index2, edge_attr2)
    return nx.is_isomorphic(
        G1, G2,
        node_match=lambda a, b: a["feat"] == b["feat"],
        edge_match=lambda a, b: a["feat"] == b["feat"],
    )


def neighbourhood_within(state, node, radius=2):
    """Everything interesting within `radius` ROAD hops of `node` (state.graph, the
    decoded road graph - not the WL/proposition graph): which nodes in that ball host
    the taxi, a waiting passenger, or a passenger's destination. Used for inspection
    dumps only (item 5) - not part of the collision/isomorphism measurement itself."""
    if node not in state.graph:
        return {"reachable_nodes": [], "taxi_within": False, "passengers_within": [], "destinations_within": []}
    reachable = set(nx.single_source_shortest_path_length(state.graph, node, cutoff=radius).keys())
    return {
        "reachable_nodes": sorted(int(n) for n in reachable),
        "taxi_within": state.taxi.location in reachable,
        "passengers_within": sorted(int(p.node) for p in state.passengers if p.location in reachable),
        "destinations_within": sorted(int(p.node) for p in state.passengers if p.destination in reachable),
    }


def sample_states(scenario, episodes, steps_per_episode, seed, sample_every, graph_convention):
    """Live episodes stepped with build_wl_vocab.greedy_action (real pickups/
    deliveries within the step budget), snapshotting a state every `sample_every`
    steps. Returns a list of (x, edge_index, edge_attr, mask) tuples (mask included -
    unlike collect_graph_corpus, candidate selection needs it)."""
    np.random.seed(seed)
    states = []

    env = GraphTaxiEnv(representation="graph", scenario=scenario, mask=MASK, rewards=REWARDS_VARIANT, graph_convention=graph_convention)
    env.seed(seed)
    env.reset()
    G = road_graph(env.sim)

    step = 0
    for _ in range(episodes):
        env.reset()
        G = road_graph(env.sim)
        for _ in range(steps_per_episode):
            if step % sample_every == 0:
                data = to_planner_data(env.sim, graph_convention=graph_convention)
                states.append((data.x, data.edge_index, data.edge_attr, data.mask))
            step += 1

            a = greedy_action(env.sim, G)
            _, _, done, _ = env.step(a)
            if done:
                env.reset()
                G = road_graph(env.sim)

    return states


def build_frozen_vocabs(scenario, seed, episodes, steps_per_episode, L_values, graph_convention, log_every=10 ** 9):
    """Builds+freezes one vocab per L, inline, from a corpus disjoint from the states
    under test - reuses wl_depth_sweep.py's own corpus-sampling/growing-mode
    machinery directly, not a separate implementation."""
    corpus = collect_graph_corpus(scenario, episodes, steps_per_episode, seed, graph_convention=graph_convention)
    vocabs = {}
    for L in L_values:
        vocab, _checkpoints = run_depth(corpus, L, log_every=log_every, graph_convention=graph_convention)
        freeze_vocab(vocab)
        vocabs[L] = vocab
    return vocabs


def run_collision_check(
    scenario, graph_convention, L_values,
    state_seed=STATE_SEED, episodes=EPISODES, steps_per_episode=STEPS_PER_EPISODE, sample_every=SAMPLE_EVERY,
    vocab_seed=VOCAB_SEED, vocab_episodes=VOCAB_EPISODES, vocab_steps_per_episode=VOCAB_STEPS_PER_EPISODE,
    exhaustive=None, k=K_CANDIDATES, stable_cap=STABLE_CAP, planner_seed=1,
    check_isomorphism=False, iso_example_limit=5,
    inspect_L=None, inspect_pair_type=("move", "move"), inspect_limit=3,
):
    """
    Runs the full collision check and returns a results dict (see the return
    statement below for its exact keys). Per-PAIR collision outcomes (growing at
    every L, frozen at every L, floor) are retained in "pair_records" - not just
    aggregate counts - so the subset-invariant check (item 2) can compare the same
    pair's outcome across consecutive L values directly, and so isomorphism/
    inspection (items 3/5) can be computed inline while `projections` for that
    state is still in scope, without re-running the planner.

    :param check_isomorphism: if True, every pair still colliding at the floor gets
        an nx.is_isomorphic check (graphs_isomorphic) - expensive, so off by default;
        pass True only for scenarios small enough to afford it (predictable5).
    :param inspect_L: if given (e.g. 2), collects up to `inspect_limit` examples of
        `inspect_pair_type` pairs that collide (growing mode) at this L but do NOT
        collide at the floor and are confirmed non-isomorphic - i.e. genuinely
        resolvable-with-more-depth collisions, not floor-level indistinguishability.
    """
    assert state_seed != vocab_seed, "state-sampling seed and vocab-building seed must be disjoint"

    if exhaustive is None:
        exhaustive = (scenario == "predictable5")

    rng = np.random.RandomState(planner_seed)

    print(f"Building frozen vocabs (L={L_values}) from a corpus disjoint from the test states "
          f"(seed={vocab_seed}, {vocab_episodes}x{vocab_steps_per_episode})...")
    t0 = time.time()
    frozen_vocabs = build_frozen_vocabs(scenario, vocab_seed, vocab_episodes, vocab_steps_per_episode, L_values, graph_convention)
    print(f"  done in {time.time() - t0:.1f}s  " + "  ".join(f"L={L}:{len(v)}" for L, v in frozen_vocabs.items()))

    print(f"Sampling states (seed={state_seed}, {episodes}x{steps_per_episode}, every {sample_every} steps, "
          f"exhaustive={exhaustive}, k={k})...")
    t0 = time.time()
    states = sample_states(scenario, episodes, steps_per_episode, state_seed, sample_every, graph_convention)
    print(f"  sampled {len(states)} states in {time.time() - t0:.1f}s")

    growing_vocabs = {L: {} for L in L_values}
    floor_vocab = {}
    planner = Planner(graph_convention=graph_convention)

    pair_records = []
    iso_counts = {"isomorphic": 0, "non_isomorphic": 0}
    iso_examples = []
    inspect_examples = []

    t0 = time.time()
    for state_idx, (x, edge_index, edge_attr, mask) in enumerate(states):
        state = decode_state(x, edge_index, edge_attr, graph_convention)
        candidates = select_candidates(mask, exhaustive, k, rng)
        if len(candidates) < 2:
            continue

        projections = {}
        goal_types = {}
        keys = {}
        for goal in candidates:
            data = _Wrapper(x, edge_index, edge_attr)
            data.mask = mask
            data.global_features = th.zeros(1, 32)  # increment_timer needs SOME global_features to mutate; value irrelevant here
            projection, _actions = planner.plan(copy.deepcopy(data), goal)
            projections[goal] = (projection.x, projection.edge_index, projection.edge_attr)
            goal_types[goal] = classify_goal(state, goal)
            keys[goal] = state_key(decode_state(*projections[goal], graph_convention))

        differing_pairs = [
            (g1, g2, tuple(sorted((goal_types[g1], goal_types[g2]))))
            for g1, g2 in itertools.combinations(candidates, 2)
            if keys[g1] != keys[g2]
        ]
        if not differing_pairs:
            continue
        touched_goals = {g for pair in differing_pairs for g in pair[:2]}

        floor_hist = {}
        for goal in touched_goals:
            xg, eig, eag = projections[goal]
            _c, h = wl_to_stable_partition(xg, eig, eag, floor_vocab, graph_convention, max_iterations=stable_cap)
            floor_hist[goal] = h
        floor_collide = {(g1, g2): _hist_equal(floor_hist[g1], floor_hist[g2]) for g1, g2, _pt in differing_pairs}

        growing_collide = {(g1, g2): {} for g1, g2, _pt in differing_pairs}
        frozen_collide = {(g1, g2): {} for g1, g2, _pt in differing_pairs}
        for L in L_values:
            growing_hist = {}
            frozen_hist = {}
            for goal in touched_goals:
                xg, eig, eag = projections[goal]
                _c1, gh = wl_colours(xg, eig, eag, num_iterations=L, vocab=growing_vocabs[L], frozen=False, graph_convention=graph_convention)
                growing_hist[goal] = gh
                _c2, fh = wl_colours(xg, eig, eag, num_iterations=L, vocab=frozen_vocabs[L], frozen=True, graph_convention=graph_convention)
                frozen_hist[goal] = fh
            for g1, g2, _pt in differing_pairs:
                growing_collide[(g1, g2)][L] = _hist_equal(growing_hist[g1], growing_hist[g2])
                frozen_collide[(g1, g2)][L] = _hist_equal(frozen_hist[g1], frozen_hist[g2])

        for g1, g2, pt in differing_pairs:
            pair_records.append({
                "state_idx": state_idx, "g1": g1, "g2": g2, "pair_type": pt,
                "growing_collide": dict(growing_collide[(g1, g2)]),
                "frozen_collide": dict(frozen_collide[(g1, g2)]),
                "floor_collide": floor_collide[(g1, g2)],
            })

        if check_isomorphism:
            for g1, g2, pt in differing_pairs:
                if not floor_collide[(g1, g2)]:
                    continue
                iso = graphs_isomorphic(*projections[g1], *projections[g2])
                iso_counts["isomorphic" if iso else "non_isomorphic"] += 1
                if not iso and len(iso_examples) < iso_example_limit:
                    iso_examples.append({"state_idx": state_idx, "g1": g1, "g2": g2, "pair_type": pt})

        if inspect_L is not None and len(inspect_examples) < inspect_limit:
            for g1, g2, pt in differing_pairs:
                if pt != inspect_pair_type:
                    continue
                if not growing_collide[(g1, g2)].get(inspect_L, False):
                    continue
                if floor_collide[(g1, g2)]:
                    continue  # only "resolvable with more depth" cases, not floor-level ones
                if graphs_isomorphic(*projections[g1], *projections[g2]):
                    continue
                inspect_examples.append({
                    "state_idx": state_idx, "g1": g1, "g2": g2,
                    "degree_g1": state.graph.degree(g1) if g1 in state.graph else None,
                    "degree_g2": state.graph.degree(g2) if g2 in state.graph else None,
                    "neighbourhood_g1": neighbourhood_within(state, g1, radius=2),
                    "neighbourhood_g2": neighbourhood_within(state, g2, radius=2),
                })
                if len(inspect_examples) >= inspect_limit:
                    break

    elapsed = time.time() - t0
    print(f"processed {len(states)} states, {len(pair_records)} differing pairs, in {elapsed:.1f}s\n")

    # --- aggregate counts (derived from pair_records, not maintained separately) ---
    differing = defaultdict(int)
    collisions_growing = defaultdict(int)
    collisions_frozen = defaultdict(int)
    floor_differing = defaultdict(int)
    collisions_floor = defaultdict(int)
    for rec in pair_records:
        pt = rec["pair_type"]
        floor_differing[pt] += 1
        if rec["floor_collide"]:
            collisions_floor[pt] += 1
        for L in L_values:
            differing[(L, pt)] += 1
            if rec["growing_collide"].get(L):
                collisions_growing[(L, pt)] += 1
            if rec["frozen_collide"].get(L):
                collisions_frozen[(L, pt)] += 1

    # --- subset invariant (item 2): colliding-at-L+1 (growing) must imply colliding-at-L ---
    sorted_L = sorted(L_values)
    subset_violations_growing = []
    subset_violations_frozen = []
    for rec in pair_records:
        for L, L_next in zip(sorted_L, sorted_L[1:]):
            if L_next != L + 1:
                continue
            gc_L, gc_next = rec["growing_collide"].get(L), rec["growing_collide"].get(L_next)
            if gc_L is not None and gc_next is not None and gc_next and not gc_L:
                subset_violations_growing.append({**rec, "L": L, "L_next": L_next})
            fc_L, fc_next = rec["frozen_collide"].get(L), rec["frozen_collide"].get(L_next)
            if fc_L is not None and fc_next is not None and fc_next and not fc_L:
                subset_violations_frozen.append({**rec, "L": L, "L_next": L_next})

    return {
        "L_values": L_values,
        "differing": dict(differing),
        "collisions_growing": dict(collisions_growing),
        "collisions_frozen": dict(collisions_frozen),
        "floor_differing": dict(floor_differing),
        "collisions_floor": dict(collisions_floor),
        "frozen_vocab_sizes": {L: len(v) for L, v in frozen_vocabs.items()},
        "n_states": len(states),
        "pair_records": pair_records,
        "subset_violations_growing": subset_violations_growing,
        "subset_violations_frozen": subset_violations_frozen,
        "iso_counts": iso_counts,
        "iso_examples": iso_examples,
        "inspect_examples": inspect_examples,
    }


def print_report(results, graph_convention, scenario):
    L_values = results["L_values"]
    print(f"=== collision report: graph_convention={graph_convention!r} scenario={scenario!r} ===\n")
    print(f"states={results['n_states']}  frozen_vocab_sizes={results['frozen_vocab_sizes']}\n")

    header = f"{'pair_type':>18}  {'L':>3}  {'differing':>9}  {'growing%':>9}  {'frozen%':>8}  {'floor%':>7}"
    print(header)
    for pt in PAIR_TYPES:
        fd = results["floor_differing"].get(pt, 0)
        cfl = results["collisions_floor"].get(pt, 0)
        floor_pct = f"{cfl/fd:>6.2%}" if fd else f"{'n/a':>6}"
        for L in L_values:
            d = results["differing"].get((L, pt), 0)
            if d == 0:
                continue
            cg = results["collisions_growing"].get((L, pt), 0)
            cf = results["collisions_frozen"].get((L, pt), 0)
            print(f"{'-'.join(pt):>18}  {L:>3}  {d:>9}  {cg/d:>8.2%}  {cf/d:>7.2%}  {floor_pct}")


def print_subset_invariant_report(results):
    gv = results["subset_violations_growing"]
    fv = results["subset_violations_frozen"]
    print(f"\n=== subset invariant (growing mode) ===")
    if not gv:
        print(f"  PASS - no violations across {len(results['pair_records'])} pairs, L={results['L_values']}")
    else:
        print(f"  FAIL - {len(gv)} violation(s):")
        for v in gv[:10]:
            print(f"    state={v['state_idx']} g1={v['g1']} g2={v['g2']} pair_type={'-'.join(v['pair_type'])} "
                  f"collides at L={v['L_next']} but NOT at L={v['L']}")

    print(f"\n=== subset invariant (frozen mode, report-only, OOV can legitimately violate) ===")
    if not fv:
        print(f"  no violations")
    else:
        print(f"  {len(fv)} violation(s) (NOT asserted - OOV merges colours, this can be legitimate):")
        for v in fv[:10]:
            print(f"    state={v['state_idx']} g1={v['g1']} g2={v['g2']} pair_type={'-'.join(v['pair_type'])} "
                  f"collides at L={v['L_next']} but NOT at L={v['L']}")


def print_isomorphism_report(results):
    counts = results["iso_counts"]
    total = counts["isomorphic"] + counts["non_isomorphic"]
    print(f"\n=== isomorphism ground truth (floor-colliding pairs) ===")
    print(f"  {total} floor-colliding pairs checked: {counts['isomorphic']} isomorphic, "
          f"{counts['non_isomorphic']} NON-isomorphic (genuine WL-indistinguishable cases)")
    for ex in results["iso_examples"][:5]:
        print(f"    non-isomorphic: state={ex['state_idx']} g1={ex['g1']} g2={ex['g2']} pair_type={'-'.join(ex['pair_type'])}")


def print_inspection_report(results):
    print(f"\n=== inspection: above-floor, non-isomorphic collisions ===")
    for i, ex in enumerate(results["inspect_examples"]):
        print(f"  example {i+1}: state={ex['state_idx']}")
        print(f"    g1={ex['g1']}  degree={ex['degree_g1']}  neighbourhood(2 roads)={ex['neighbourhood_g1']}")
        print(f"    g2={ex['g2']}  degree={ex['degree_g2']}  neighbourhood(2 roads)={ex['neighbourhood_g2']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure WL histogram collisions between candidate-goal projections.")
    parser.add_argument("--graph-convention", default="oracle_sage", choices=["oracle_sage", "vilg", "atom"])
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("-L", "--num-iterations", type=int, nargs="+", default=L_VALUES)
    parser.add_argument("--state-seed", type=int, default=STATE_SEED)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--steps-per-episode", type=int, default=STEPS_PER_EPISODE)
    parser.add_argument("--sample-every", type=int, default=SAMPLE_EVERY)
    parser.add_argument("--vocab-seed", type=int, default=VOCAB_SEED)
    parser.add_argument("--vocab-episodes", type=int, default=VOCAB_EPISODES)
    parser.add_argument("--vocab-steps-per-episode", type=int, default=VOCAB_STEPS_PER_EPISODE)
    parser.add_argument("--exhaustive", action="store_true", default=None)
    parser.add_argument("--no-exhaustive", dest="exhaustive", action="store_false")
    parser.add_argument("--k", type=int, default=K_CANDIDATES)
    parser.add_argument("--stable-cap", type=int, default=STABLE_CAP)
    parser.add_argument("--planner-seed", type=int, default=1, help="seed for random candidate-subset selection")
    parser.add_argument("--check-isomorphism", action="store_true", default=False,
                         help="nx.is_isomorphic check on every floor-colliding pair (item 3) - expensive, "
                              "only affordable on small scenarios (predictable5)")
    parser.add_argument("--inspect-L", type=int, default=None,
                         help="if given, dump up to --inspect-limit examples of --inspect-pair-type pairs "
                              "colliding (growing) at this L but not at the floor, confirmed non-isomorphic")
    parser.add_argument("--inspect-pair-type", type=str, default="move-move")
    parser.add_argument("--inspect-limit", type=int, default=3)
    args = parser.parse_args(argv)

    inspect_pair_type = tuple(sorted(args.inspect_pair_type.split("-")))
    results = run_collision_check(
        args.scenario, args.graph_convention, args.num_iterations,
        state_seed=args.state_seed, episodes=args.episodes, steps_per_episode=args.steps_per_episode,
        sample_every=args.sample_every,
        vocab_seed=args.vocab_seed, vocab_episodes=args.vocab_episodes, vocab_steps_per_episode=args.vocab_steps_per_episode,
        exhaustive=args.exhaustive, k=args.k, stable_cap=args.stable_cap, planner_seed=args.planner_seed,
        check_isomorphism=args.check_isomorphism,
        inspect_L=args.inspect_L, inspect_pair_type=inspect_pair_type, inspect_limit=args.inspect_limit,
    )
    print_report(results, args.graph_convention, args.scenario)
    print_subset_invariant_report(results)
    if args.check_isomorphism:
        print_isomorphism_report(results)
    if args.inspect_L is not None:
        print_inspection_report(results)
    return results


if __name__ == "__main__":
    main()
