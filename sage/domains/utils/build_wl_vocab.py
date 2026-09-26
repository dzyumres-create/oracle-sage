"""
.. module:: build_wl_vocab
   :synopsis: Standalone, run-once-by-hand script that builds a frozen WL
   colour vocabulary for the Taxi domain, by sampling graphs from ONE live
   Taxi environment and running sage.domains.utils.wl_colours.wl_colours
   in growing mode across all of them, then freezing the result.

   This is NOT wired into env_to_graph, planner.py, or any policy file -
   it just produces a JSON vocab file on disk, named
   `wl_vocab_taxi_{scenario}_L{num_iterations}.json` by default (e.g.
   `wl_vocab_taxi_predictable5_L1.json`), so different scenario/depth
   combinations don't overwrite each other.

   Both the scenario and L (`--num-iterations`) are CLI parameters (see
   `main()`), NOT hardcoded - a depth-sweep diagnostic
   (sage/domains/utils/wl_depth_sweep.py) found that L=5 (Oracle-SAGE's
   own `gnn_steps=5` default for Taxi - see `sage/experiments/
   gnn_global.py`) does not produce a stabilizing vocab even at large
   sample sizes, on either "city" or "predictable5"; L=1 was the only
   depth that clearly stabilized on "predictable5". `--num-iterations`
   still defaults to 5 for backward compatibility with earlier
   invocations of this script, but the real vocab-building run this
   script is meant for right now uses `--scenario predictable5
   --num-iterations 1` explicitly - reconciling that against Oracle-SAGE's
   actual gnn_steps=5 is a separate framing question, out of scope here.

Investigation notes (env construction / graph extraction)
-----------------------------------------------------------
- Confirmed from the repo-root `train` file and sage/experiments/
  gnn_global.py: Oracle-SAGE's real Taxi training command always passes
  `--env-name city-taxi-unmasked-v1` (gnn_global.py exposes no grid-size
  CLI flag at all). That id is registered in
  sage/domains/gym_taxi/utils/config.py via `ENVS["city-taxi-unmasked-"]
  = {"representation": "graph", "scenario": "city", "mask": False}`
  together with `REWARDS["v1"]` (sage/domains/gym_taxi/__init__.py) -
  i.e. it is exactly `GraphTaxiEnv(representation="graph", scenario=
  "city", mask=False, rewards=REWARDS["v1"])`, the single `CITY` config
  (size 20, random_walls=True). This script samples from that ONE config
  only - no multi-config sweep (out of scope here; a prior version of
  this script sampled several scenario variants, which this revision
  deliberately drops in favour of matching the real training config
  exactly). Reward values don't affect graph structure at all (they only
  affect the scalar reward returned by `.step()`), so this choice has no
  bearing on the sampled graphs or the resulting vocab - it's included
  purely for fidelity to the real training env.
- `gym.make("city-taxi-unmasked-v1")` itself does NOT work in this
  environment: gym's `PassiveEnvChecker` wrapper crashes on this
  codebase's custom `JsonGraph` observation space (`AttributeError:
  'JsonGraph' object has no attribute 'low'`) - a separate legacy-gym
  incompatibility from the numpy ones below. So this script constructs
  `GraphTaxiEnv` directly instead (the pattern already proven to work),
  bypassing gym's registry/`make()` machinery entirely.
- A live env's graph tensors are read via
  `sage.domains.gym_taxi.utils.representations.env_to_graph(env.sim)`
  (`env.sim` is the underlying `TaxiWorldSimulator`, not the gym wrapper),
  which returns `(node_feats, edge_feats, edge_index, mask, global_feats)`
  as numpy arrays - exactly wl_colours' (x, edge_index, edge_attr) inputs
  once cast to torch tensors.

Three more things turned up while getting a live env to actually run, worth
recording since they affect this script's design (none of these files is
touched by this script, per this task's constraints):

1. `env_to_graph` (representations.py) builds node/edge features with
   `dtype=np.float`, which numpy >= 1.24 removed entirely - it raises
   `AttributeError: module 'numpy' has no attribute 'float'` under the
   numpy installed in the `sage` conda env (2.2.6) the moment you call
   `env.reset()`. This script works around it locally (see the `np.float`
   shim below) without editing representations.py - numpy's own
   deprecation message confirms `np.float` and the builtin `float` are
   behaviourally identical, so this is a safe, non-invasive patch.
2. `TaxiWorldSimulator.attempt_move` (taxi_world.py) does an unguarded
   `self.graph.edges[(start, action)]` lookup and raises an uncaught
   networkx KeyError if `action` is a location that isn't directly
   adjacent to the taxi. In real Oracle-SAGE usage `env.step()` is only
   ever called with single-hop actions from a planner-expanded path (see
   `Planner.plan()` / `find_path_to` in
   sage/domains/gym_taxi/simulator/planner.py) - never with a distant goal
   node directly - so this never surfaces there. `sample_action` below
   mirrors that contract instead of hitting it.
3. The "city" scenario has `random_walls=True`, so it generates its maze
   via `generate_city_maze`/`try_generate_city_maze`
   (sage/domains/gym_taxi/utils/utils.py), which calls the legacy
   `RandomState.randint(n)` API on `random` - but the `np_random` object
   gym's `seeding.np_random()` now hands back is a numpy `Generator`
   (new numpy random API), which has `.integers` but no `.randint`, so it
   raises `AttributeError` immediately on env construction. This script
   shims it locally (see below) the same non-invasive way as the
   `np.float` fix, rather than editing utils.py.

JSON serialisation scheme
--------------------------
`vocab` maps signatures to integer ids, where a signature is one of:
    - the string OOV_SIGNATURE ("__OOV__")
    - ("init", type_id)                                   (an init signature)
    - (own_colour, ((neighbour_colour, edge_label), ...))  (a refine signature)
None of these are valid JSON object keys, so the vocab is saved as a JSON
object `{"vocab_size": int, "entries": [{"signature": <encoded>, "id": int}, ...]}`,
where `<encoded>` is a small tagged JSON object - see `_encode_signature` /
`_decode_signature` below for the exact (and inverse) encoding.
"""
import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import numpy as np

if not hasattr(np, "float"):
    # See "Investigation notes" above: numpy>=1.24 removed the deprecated
    # `np.float` alias that env_to_graph still uses. This does not touch
    # representations.py (out of scope for this task); it's the same fix
    # numpy's own deprecation message recommends.
    np.float = float

import gym.utils.seeding as _seeding

if not hasattr(np.random.Generator, "randint"):
    # See "Investigation notes" above: try_generate_city_maze (utils.py)
    # calls the legacy RandomState.randint(n) / randint(low, high) API on
    # what gym now hands it as a numpy Generator. `Generator.integers` is
    # the direct, semantically-equivalent replacement (same low-inclusive,
    # high-exclusive behaviour). numpy.random.Generator is a builtin type
    # and can't be monkeypatched directly ("cannot set 'randint' attribute
    # of immutable type"), so instead this wraps the Generator gym hands
    # back from `seeding.np_random()` in a thin adapter that adds
    # `.randint` and forwards everything else unchanged - local to this
    # script's process only, does not touch utils.py or gym itself.
    class _RandintCompatGenerator:
        def __init__(self, generator):
            self._generator = generator

        def randint(self, low, high=None):
            if hasattr(self._generator, "randint"):
                # already a real RandomState (e.g. RCP's pinned gym/numpy) - use its native method
                return self._generator.randint(low, high) if high is not None else self._generator.randint(low)
            # genuinely a modern Generator with no .randint - use the real replacement
            return self._generator.integers(low, high)


        def __getattr__(self, name):
            return getattr(self._generator, name)

    _original_np_random = _seeding.np_random

    def _np_random_with_randint(seed=None):
        generator, seed = _original_np_random(seed)
        return _RandintCompatGenerator(generator), seed

    _seeding.np_random = _np_random_with_randint

import torch as th
import networkx as nx
from torch_geometric.data import Data

import sage.domains.gym_taxi  # noqa: F401  (registers the gym env ids)
from sage.domains.gym_taxi import REWARDS
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.utils.representations import env_to_graph, env_to_vilg_graph, env_to_atom_graph
from sage.domains.gym_taxi.simulator.planner import Planner
from sage.domains.utils.wl_colours import OOV_SIGNATURE, freeze_vocab, wl_colours

# The exact config behind Oracle-SAGE's real Taxi training env-name,
# `city-taxi-unmasked-v1` (see "Investigation notes" above) - the DEFAULT
# scenario this script samples from. `--scenario` can override this (e.g.
# for the fixed-maze diagnostic comparison - see docstring), but "city" is
# what a real vocab-building run should use; the flag exists so switching
# back is a one-line CLI change, not a code edit.
ENV_NAME = "city-taxi-unmasked-v1"
SCENARIO = "city"
MASK = False
REWARDS_VARIANT = REWARDS["v1"]

# Default for `--num-iterations` (L), matching Oracle-SAGE's own
# `gnn_steps=5` default for Taxi (sage/experiments/gnn_global.py,
# `--gnn-steps` default) - kept for backward compatibility with earlier
# invocations of this script. See the module docstring: this default does
# NOT currently produce a stabilizing vocab (see wl_depth_sweep.py), so
# real runs should pass `--num-iterations` explicitly.
NUM_ITERATIONS = 5


def default_out_path(scenario, num_iterations, graph_convention="oracle_sage"):
    # oracle_sage keeps the original, unsuffixed filename exactly - only a
    # non-default convention gets a suffix, so existing oracle_sage vocab
    # filenames/paths are completely unaffected by this generalization.
    suffix = "" if graph_convention == "oracle_sage" else f"_{graph_convention}"
    return Path(__file__).parent / f"wl_vocab_taxi_{scenario}{suffix}_L{num_iterations}.json"


def extract_graph_tensors(sim, graph_convention="oracle_sage"):
    """
    Reads a live sim's current graph as (x, edge_index, edge_attr) torch
    tensors, via whichever translator matches `graph_convention`:
    env_to_graph (oracle_sage) or env_to_vilg_graph (vilg) - both return a
    7-tuple with trailing wl_colours/wl_histogram fields (see
    representations.py), discarded here either way since this module
    recomputes wl_colours itself, in growing mode, below - or
    env_to_atom_graph (atom), which returns a plain 5-tuple (the atom
    convention's GNN path carries no WL fields at all; WL is only ever
    attached decoder-side, see representations.py's attach_wl).

    :param sim: a TaxiWorldSimulator instance (env.sim)
    :param graph_convention: "oracle_sage" (default), "vilg", or "atom"
    :return: (x, edge_index, edge_attr) torch tensors
    """
    if graph_convention == "vilg":
        node_feats, edge_feats, edge_index_np, _, _, _, _ = env_to_vilg_graph(sim)
    elif graph_convention == "atom":
        node_feats, edge_feats, edge_index_np, _, _ = env_to_atom_graph(sim)
    else:
        node_feats, edge_feats, edge_index_np, _, _, _, _ = env_to_graph(sim)
    x = th.as_tensor(node_feats, dtype=th.float)
    edge_attr = th.as_tensor(edge_feats, dtype=th.float)
    edge_index = th.as_tensor(edge_index_np, dtype=th.long)
    return x, edge_index, edge_attr


def sample_action(sim):
    """
    Picks a uniformly random action from the set of actions env.step() can
    safely execute this turn: a single-hop move to a location the taxi's
    current location actually has an outgoing edge to, the taxi's own node
    (a dropoff attempt - always safe, whether or not it succeeds), the
    taxi's current location itself (an explicit no-op move), or any current
    passenger node (a pickup attempt - always safe). See "Investigation
    notes" above for why non-adjacent location actions are avoided.

    Under graph_convention="vilg", a delivered passenger's node (and its
    destination(pid, dest) edge) deliberately survives in sim.graph purely
    to carry goal status (see env_to_vilg_graph/attempt_dropoff's "vilg"
    branch) - so it can still turn up as a successor of whatever location is
    its destination, even though it's no longer in sim.passengers and
    attempt_pickup would KeyError on it. Excluded the same way the real
    action mask already does (env_to_vilg_graph's `selectable` list:
    "(not is_passenger) or (nid in env.passengers)") - this can never fire
    under oracle_sage, since a delivered passenger's node is removed from
    the graph entirely there (attempt_dropoff's non-vilg branch), not kept
    alive, so this check is a no-op for that convention.

    :param sim: a TaxiWorldSimulator instance (env.sim)
    :return: a valid node id to pass to env.step()
    """
    taxi_location = sim.taxi.location
    candidates = {
        c for c in sim.graph.successors(taxi_location)
        if sim.graph.nodes[c]["attr"] != [0, 0, 1] or c in sim.passengers
    }
    candidates.add(taxi_location)
    candidates.add(0)  # taxi's own node -> dropoff attempt
    return int(np.random.choice(list(candidates)))


def road_graph(sim):
    """
    An UNDIRECTED nx.Graph of location<->location road edges read directly off
    `sim.graph` (edge_attr[0]==1, the is_road one-hot column - the same test the
    oracle_sage planner's own graph_to_networkx uses), independent of graph_convention
    (this reads the simulator's own internal graph, not any converted representation).
    Used by greedy_action/next_hop below for shortest-path navigation - built once per
    episode (the road layout is fixed for the episode's lifetime; only taxi/passenger
    state changes step to step) rather than recomputed every step.
    """
    G = nx.Graph()
    for u, v, attr in sim.graph.edges(data=True):
        if attr["attr"][0] == 1:
            G.add_edge(u, v)
    return G


def next_hop(G, src, dst):
    """One step along nx.shortest_path(G, src, dst) - None if src==dst already."""
    if src == dst:
        return None
    path = nx.shortest_path(G, src, dst)
    return path[1]


def greedy_action(sim, G):
    """
    A simple nearest-passenger-then-destination greedy policy (the same one
    verification check C used to force real pickups/deliveries within a bounded step
    budget - pure random action sampling rarely delivers on a size-20 maze): if
    carrying a passenger, head for their destination (dropoff-attempt once there); else
    head for the nearest still-waiting passenger by road-graph hop distance
    (pickup-attempt once there); if no passengers exist yet, hold position. Unlike
    sample_action, this is deliberately NOT uniformly random - it exists to generate a
    corpus with real pickup/delivery structure, not to explore broadly.

    :param sim: a TaxiWorldSimulator instance (env.sim)
    :param G: this episode's road_graph(sim) (passed in, not recomputed every call)
    :return: a valid node id to pass to env.step()
    """
    if sim.taxi.passenger is not None:
        pid = sim.taxi.passenger
        dest = sim.passengers[pid].destination
        if sim.taxi.location == dest:
            return 0
        return next_hop(G, sim.taxi.location, dest)
    if sim.passengers:
        pid = min(
            sim.passengers,
            key=lambda p: nx.shortest_path_length(G, sim.taxi.location, sim.passengers[p].location),
        )
        loc = sim.passengers[pid].location
        if sim.taxi.location == loc:
            return pid
        return next_hop(G, sim.taxi.location, loc)
    return sim.taxi.location


def to_planner_data(sim, graph_convention="oracle_sage"):
    """
    Builds a full torch_geometric Data (x/edge_index/edge_attr/mask/global_features)
    from a live sim, suitable as input to Planner.plan() - unlike extract_graph_tensors
    (which only returns the (x, edge_index, edge_attr) triple wl_colours needs),
    Planner.plan()'s branches (increment_timer in particular) also read mask/
    global_features off the input graph. Dtypes match json_to_graph's/
    json_to_atom_graph's real output exactly (x/edge_attr float32, edge_index int64,
    mask bool, global_features float32 unsqueezed to a leading batch-of-1 dim) - the
    same convention project_actions' real symbolic_batch.to_data_list() elements have.

    :param sim: a TaxiWorldSimulator instance (env.sim)
    :param graph_convention: "oracle_sage" (default), "vilg", or "atom"
    :return: a single (unbatched) Data, ready for Planner(graph_convention=...).plan(data, goal)
    """
    if graph_convention == "vilg":
        node_feats, edge_feats, edge_index, mask, global_feats, _wl, _hist = env_to_vilg_graph(sim)
    elif graph_convention == "atom":
        node_feats, edge_feats, edge_index, mask, global_feats = env_to_atom_graph(sim)
    else:
        node_feats, edge_feats, edge_index, mask, global_feats, _wl, _hist = env_to_graph(sim)

    d = Data(
        x=th.as_tensor(node_feats, dtype=th.float32),
        edge_index=th.as_tensor(edge_index, dtype=th.long),
        edge_attr=th.as_tensor(edge_feats, dtype=th.float32),
    )
    d.mask = th.as_tensor(mask, dtype=th.bool)
    d.global_features = th.as_tensor(global_feats, dtype=th.float32).unsqueeze(0)
    return d


def epsilon_greedy_action(sim, G, eps, rng):
    """
    greedy_action, but with probability `eps` a uniformly-random legal action
    (sample_action) is taken instead - so a corpus built from this policy isn't
    entirely confined to the greedy policy's own on-path states (which are a narrow,
    systematically-biased slice of the reachable state space: always making progress
    towards a delivery, never "wasted" moves, never far from a passenger/destination).

    :param sim: a TaxiWorldSimulator instance (env.sim)
    :param G: this episode's road_graph(sim)
    :param eps: exploration probability, in [0, 1]
    :param rng: a numpy RandomState, used ONLY for the eps coin-flip (kept separate
        from sample_action's own bare `np.random.choice` calls, which read the global
        numpy random state - so this doesn't perturb sample_action's own reproducibility
        contract for callers that also seed the global state directly)
    :return: a valid node id to pass to env.step()
    """
    if rng.random_sample() < eps:
        return sample_action(sim)
    return greedy_action(sim, G)


def run_full_episode_states(env, sample_every, graph_convention, policy, goals_per_state, rng, max_steps=2500):
    """
    Runs ONE episode from env's CURRENT state (caller is responsible for env.reset()
    and any env.seed() beforehand - this does not seed or reset anything itself, so a
    caller can run many consecutive episodes off one continuously-advancing seeded
    env), to its NATURAL end (city's own timeout=2000 or delivery_limit, surfaced as
    `done=True` from env.step()) - capped at `max_steps` as a safety net only, not the
    normal stopping condition. Yields a live state every `sample_every` simulator
    steps (not every step - consecutive states differ by one action and are highly
    autocorrelated), tagged with its step-in-episode depth, followed by up to
    `goals_per_state` Planner.plan() projections from that SAME state (same depth tag)
    - real, structurally distinct states a planner-feedback policy actually
    encounters, not just live-stepped ones (goals_per_state=0 disables this cleanly).

    :param env: a live GraphTaxiEnv, already reset (and seeded, if reproducibility is
        wanted) by the caller
    :param sample_every: yield a state every this many simulator steps
    :param graph_convention: "oracle_sage", "vilg", or "atom"
    :param policy: callable(sim, road_graph) -> action (e.g. sample_action wrapped to
        take the unused road_graph arg, greedy_action, or epsilon_greedy_action
        partially applied)
    :param goals_per_state: number of random legal candidate goals to also project via
        Planner.plan() at each sampled state (0 disables projections entirely)
    :param rng: a numpy RandomState, used for goal selection (kept separate from the
        `policy`'s own randomness source, whatever that is)
    :param max_steps: hard cap, in case `done` is somehow never returned
    :yield: (step, x, edge_index, edge_attr) - one for the live state, then one per
        projected goal, all sharing the same `step` (the live state's depth, not the
        hypothetical post-projection depth - a projection is a one-step lookahead FROM
        this depth, not a state actually reached at a deeper step)
    """
    G = road_graph(env.sim)
    planner = Planner(graph_convention=graph_convention) if goals_per_state > 0 else None

    for step in range(max_steps):
        if step % sample_every == 0:
            data = to_planner_data(env.sim, graph_convention=graph_convention)
            yield step, data.x, data.edge_index, data.edge_attr

            if goals_per_state > 0:
                selectable = data.mask.nonzero(as_tuple=True)[0].tolist()
                n_goals = min(goals_per_state, len(selectable))
                if n_goals > 0:
                    for goal in rng.choice(selectable, size=n_goals, replace=False):
                        projection, _actions = planner.plan(copy.deepcopy(data), int(goal))
                        yield step, projection.x, projection.edge_index, projection.edge_attr

        a = policy(env.sim, G)
        _, _, done, _ = env.step(a)
        if done:
            break


def sample_graphs(
    vocab, episodes, sample_every, num_iterations=NUM_ITERATIONS, seed=0, log_every=100,
    scenario=SCENARIO, graph_convention="oracle_sage", eps=0.2, goals_per_state=2, max_steps=2500,
):
    """
    Builds a vocab from `episodes` FULL episodes (run to natural termination - city's
    own timeout/delivery_limit, NOT stopped early after a fixed step count), sampling a
    state every `sample_every` simulator steps and running growing-mode wl_colours on
    it - plus `goals_per_state` Planner.plan() projections per sampled state - to
    accumulate into the shared `vocab`. Replaces the old short-episode-reset design
    (every episode capped at a fixed step count, uniform-random actions only, no
    projections): that design only ever sampled the first ~50-60 steps of any episode
    and never the deep states a real, long-running training episode (city's episodes
    run to their ~2000-step timeout - see docs/cell6_wl_diagnostics.md) actually
    reaches, which measurably undercounted real-world OOV by an order of magnitude at
    depth.

    Policy mix: the first half of `episodes` use epsilon_greedy_action (mostly the
    nearest-passenger-then-destination greedy policy, with probability `eps` a random
    legal action instead, so sampled states aren't confined to the greedy policy's own
    narrow on-path slice); the second half use sample_action (uniformly random legal
    actions) throughout - deliberately not epsilon-mixed itself, so the corpus also
    contains genuinely exploratory trajectories the greedy-biased half systematically
    avoids (e.g. states reached by "wasted" moves, or far from any passenger).

    env.seed(seed) is called ONCE, before the first episode - env.reset() (called once
    per episode by this function) then advances the SAME seeded generator, so the
    entire sequence of `episodes` mazes/passenger-spawns is deterministic and
    reproducible from `seed` alone (the old version never called env.seed() at all,
    so - despite accepting a `seed` argument - its mazes were driven by whatever
    unseeded OS entropy BaseTaxiEnv.__init__ auto-seeds itself with at construction,
    not by that argument).

    :param vocab: signature -> colour id, mutated in place (growing mode)
    :param episodes: number of FULL episodes (not env.reset() calls capped at a fixed
        step count)
    :param sample_every: sample (and grow the vocab from) a state every this many
        simulator steps within each episode
    :param num_iterations: WL refinement iterations per sampled graph
    :param seed: seeds BOTH env.seed(seed) (mazes/spawns) and this function's own
        goal-selection RandomState - genuinely reproducible and, when disjoint from
        another call's seed, genuinely disjoint (unlike the old version)
    :param log_every: print a (graph count, vocab size) checkpoint every this many
        TOTAL sampled graphs (live states + projections combined)
    :param scenario: `GraphTaxiEnv` scenario key (default SCENARIO="city")
    :param graph_convention: "oracle_sage" (default), "vilg", or "atom"
    :param eps: exploration probability for the greedy half's epsilon_greedy_action
    :param goals_per_state: planner-projected candidate goals sampled per visited state
    :param max_steps: safety cap per episode (city's own timeout=2000 already ends
        episodes via `done=True` well before this at the default)
    :return: total number of graphs sampled (and folded into `vocab`)
    """
    env = GraphTaxiEnv(representation="graph", scenario=scenario, mask=MASK, rewards=REWARDS_VARIANT, graph_convention=graph_convention)
    env.seed(seed)
    # sample_action (used by both the random-action half AND epsilon_greedy_action's
    # eps branch) reads numpy's GLOBAL random state directly (np.random.choice), not
    # `rng` below - seeding it here too is required for full reproducibility, on top of
    # env.seed(seed) for maze/spawn generation and `rng` for goal-projection selection.
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    total_graphs = 0
    last_checkpoint_size = len(vocab)
    half = episodes // 2

    for ep in range(episodes):
        env.reset()
        if ep < half:
            policy = lambda sim, G: epsilon_greedy_action(sim, G, eps, rng)
        else:
            policy = lambda sim, G: sample_action(sim)

        for step, x, edge_index, edge_attr in run_full_episode_states(
            env, sample_every, graph_convention, policy, goals_per_state, rng, max_steps=max_steps,
        ):
            wl_colours(
                x, edge_index, edge_attr,
                num_iterations=num_iterations, vocab=vocab, frozen=False,
                graph_convention=graph_convention,
            )
            total_graphs += 1

            if total_graphs % log_every == 0:
                delta = len(vocab) - last_checkpoint_size
                print(f"  graphs={total_graphs:>6}  vocab_size={len(vocab):>6}  (+{delta} since last checkpoint)")
                last_checkpoint_size = len(vocab)

    return total_graphs


def _encode_signature(signature):
    if signature == OOV_SIGNATURE:
        return {"kind": "oov"}
    first, second = signature
    if first == "init":
        return {"kind": "init", "type_id": second}
    return {
        "kind": "refine",
        "own_colour": first,
        "neighbours": [[colour, label] for colour, label in second],
    }


def _decode_signature(encoded):
    kind = encoded["kind"]
    if kind == "oov":
        return OOV_SIGNATURE
    if kind == "init":
        return ("init", encoded["type_id"])
    if kind == "refine":
        neighbours = tuple((colour, label) for colour, label in encoded["neighbours"])
        return (encoded["own_colour"], neighbours)
    raise ValueError(f"unknown encoded signature kind: {kind!r}")


def _git_commit():
    """Best-effort `git rev-parse HEAD`; returns None (not a placeholder string) if git
    is unavailable or this isn't a git checkout, so metadata never claims a fake commit."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def save_vocab(vocab, path, graph_convention, num_iterations, build_procedure=None):
    """
    Saves `vocab` to `path` as JSON - see the module docstring for the encoding scheme.
    Also records `graph_convention`/`num_iterations` (L) as top-level metadata: a
    vocab's colour ids are only meaningful for the exact convention+L it was built
    with, and wl_vocab_cache.validate_wl_vocab_metadata (used by both
    configure_wl_vocab_override and WLPlanFeedbackPolicy._load_vocab) checks this
    metadata on load, so a mismatched vocab raises loudly instead of silently
    producing wrong-but-valid embedding lookups. Vocab files saved before this
    metadata existed have neither key - validate_wl_vocab_metadata tolerates that for
    oracle_sage/vilg (never for "atom", which is new enough to always require it).

    :param build_procedure: optional dict describing HOW this vocab's corpus was
        built (episodes, sample_every, eps, seed, goals_per_state, policy mix, git
        commit) - recorded as-is under the "build_procedure" key, purely for
        provenance/reproducibility (never read back by validate_wl_vocab_metadata or
        any loader); omitted entirely when None, so old callers/readers are unaffected.
    """
    entries = [
        {"signature": _encode_signature(signature), "id": colour_id}
        for signature, colour_id in vocab.items()
    ]
    payload = {
        "vocab_size": len(vocab),
        "graph_convention": graph_convention,
        "num_iterations": num_iterations,
        "entries": entries,
    }
    if build_procedure is not None:
        payload["build_procedure"] = build_procedure
    with open(path, "w") as f:
        json.dump(payload, f)


def load_vocab(path):
    """Loads a vocab previously saved with `save_vocab` back into a signature -> id dict."""
    with open(path) as f:
        payload = json.load(f)
    vocab = {}
    for entry in payload["entries"]:
        vocab[_decode_signature(entry["signature"])] = entry["id"]
    return vocab


def load_vocab_with_metadata(path):
    """Like `load_vocab`, but also returns the raw metadata dict (graph_convention,
    num_iterations, vocab_size, build_procedure if present) alongside it - useful for
    measure-only tooling that needs to know what convention/L a saved vocab was built
    for, rather than having the caller re-supply (and risk mismatching) that by hand."""
    with open(path) as f:
        payload = json.load(f)
    vocab = {}
    for entry in payload["entries"]:
        vocab[_decode_signature(entry["signature"])] = entry["id"]
    metadata = {k: v for k, v in payload.items() if k != "entries"}
    return vocab, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Build a frozen WL-colour vocab for the Taxi domain by sampling a live GraphTaxiEnv."
    )
    parser.add_argument("--episodes", type=int, default=20, help="number of FULL episodes to run (not step-capped)")
    parser.add_argument("--sample-every", type=int, default=50, help="sample a state every this many simulator steps within each episode")
    parser.add_argument("--seed", type=int, default=0, help="seeds env.seed() (mazes/spawns) and goal selection")
    parser.add_argument("--eps", type=float, default=0.2, help="exploration probability for the greedy half of episodes")
    parser.add_argument("--goals-per-state", type=int, default=2, help="planner-projected candidate goals sampled per visited state (0 disables projections)")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--scenario", default=SCENARIO,
        help=f"GraphTaxiEnv scenario key (default {SCENARIO!r}, matching {ENV_NAME} - "
             f"the real Oracle-SAGE Taxi training config). Override for diagnostics.",
    )
    parser.add_argument(
        "-L", "--num-iterations", type=int, default=NUM_ITERATIONS,
        help=f"WL refinement iterations, i.e. depth L (default {NUM_ITERATIONS}, matching "
             f"Oracle-SAGE's gnn_steps=5 default - but see the module docstring: this default "
             f"does not currently produce a stabilizing vocab; L=1 was the only depth found to "
             f"stabilize on scenario='predictable5').",
    )
    parser.add_argument(
        "--out", default=None,
        help="output path for the frozen vocab JSON "
             "(default: wl_vocab_taxi_{scenario}_L{num_iterations}.json alongside this script, "
             "or wl_vocab_taxi_{scenario}_{graph_convention}_L{num_iterations}.json for a "
             "non-oracle_sage --graph-convention)",
    )
    parser.add_argument(
        "--graph-convention", default="oracle_sage", choices=["oracle_sage", "vilg", "atom"],
        help="graph construction convention to sample from (default: oracle_sage, unchanged "
             "behaviour). 'vilg' constructs GraphTaxiEnv(graph_convention='vilg') and reads "
             "graphs via env_to_vilg_graph instead of env_to_graph; 'atom' likewise via "
             "env_to_atom_graph.",
    )
    args = parser.parse_args()
    out_path = args.out if args.out is not None else str(default_out_path(args.scenario, args.num_iterations, args.graph_convention))

    start = time.time()
    vocab = {}
    total_graphs = sample_graphs(
        vocab,
        episodes=args.episodes,
        sample_every=args.sample_every,
        num_iterations=args.num_iterations,
        seed=args.seed,
        log_every=args.log_every,
        scenario=args.scenario,
        graph_convention=args.graph_convention,
        eps=args.eps,
        goals_per_state=args.goals_per_state,
    )
    vocab_size = freeze_vocab(vocab)
    build_procedure = {
        "episodes": args.episodes,
        "sample_every": args.sample_every,
        "eps": args.eps,
        "goals_per_state": args.goals_per_state,
        "seed": args.seed,
        "policy_mix": "first half of episodes: epsilon_greedy_action (eps as above); "
                      "second half: sample_action (uniform random legal actions)",
        "full_episodes": True,
        "git_commit": _git_commit(),
    }
    save_vocab(
        vocab, out_path, graph_convention=args.graph_convention, num_iterations=args.num_iterations,
        build_procedure=build_procedure,
    )
    elapsed = time.time() - start

    print(f"sampled {total_graphs} graphs from scenario={args.scenario!r} graph_convention={args.graph_convention!r} "
          f"({args.episodes} full episodes, sample_every={args.sample_every}), L={args.num_iterations}")
    print(f"vocab_size (frozen, includes OOV) = {vocab_size}")
    print(f"wrote {out_path}")
    print(f"elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
