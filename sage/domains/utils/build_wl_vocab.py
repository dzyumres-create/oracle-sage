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
import json
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
            return self._generator.integers(low, high)

        def __getattr__(self, name):
            return getattr(self._generator, name)

    _original_np_random = _seeding.np_random

    def _np_random_with_randint(seed=None):
        generator, seed = _original_np_random(seed)
        return _RandintCompatGenerator(generator), seed

    _seeding.np_random = _np_random_with_randint

import torch as th

import sage.domains.gym_taxi  # noqa: F401  (registers the gym env ids)
from sage.domains.gym_taxi import REWARDS
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.utils.representations import env_to_graph
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


def default_out_path(scenario, num_iterations):
    return Path(__file__).parent / f"wl_vocab_taxi_{scenario}_L{num_iterations}.json"


def sample_action(sim):
    """
    Picks a uniformly random action from the set of actions env.step() can
    safely execute this turn: a single-hop move to a location the taxi's
    current location actually has an outgoing edge to, the taxi's own node
    (a dropoff attempt - always safe, whether or not it succeeds), the
    taxi's current location itself (an explicit no-op move), or any current
    passenger node (a pickup attempt - always safe). See "Investigation
    notes" above for why non-adjacent location actions are avoided.

    :param sim: a TaxiWorldSimulator instance (env.sim)
    :return: a valid node id to pass to env.step()
    """
    taxi_location = sim.taxi.location
    candidates = set(sim.graph.successors(taxi_location))
    candidates.add(taxi_location)
    candidates.add(0)  # taxi's own node -> dropoff attempt
    return int(np.random.choice(list(candidates)))


def sample_graphs(vocab, episodes, steps_per_episode, num_iterations=NUM_ITERATIONS, seed=0, log_every=100, scenario=SCENARIO):
    """
    Resets/steps through the Taxi environment (see MASK/REWARDS_VARIANT
    above, and `scenario` below), running growing-mode wl_colours on the
    graph sampled after every step, accumulating into the shared `vocab`.

    :param vocab: signature -> colour id, mutated in place (growing mode)
    :param episodes: number of env.reset() episodes
    :param steps_per_episode: number of random-action steps per episode
    :param num_iterations: WL refinement iterations per sampled graph
    :param seed: seed for the random action sampling
    :param log_every: print a (graph count, vocab size) checkpoint every
        this many sampled graphs, so growth-rate trends are visible in the
        console output rather than only a single before/after number
    :param scenario: `GraphTaxiEnv` scenario key (default SCENARIO="city",
        the real training config - see module docstring for why other
        values might be used for a diagnostic comparison)
    :return: total number of graphs sampled (and folded into `vocab`)
    """
    np.random.seed(seed)
    total_graphs = 0
    last_checkpoint_size = len(vocab)

    env = GraphTaxiEnv(representation="graph", scenario=scenario, mask=MASK, rewards=REWARDS_VARIANT)
    for _ in range(episodes):
        env.reset()
        for _ in range(steps_per_episode):
            # env_to_graph now also returns wl_colours/wl_histogram (this
            # script's own WL wiring is unrelated - it recomputes wl_colours
            # itself, in growing mode, below - so those two extra values are
            # discarded here).
            node_feats, edge_feats, edge_index_np, _, _, _, _ = env_to_graph(env.sim)
            x = th.as_tensor(node_feats, dtype=th.float)
            edge_attr = th.as_tensor(edge_feats, dtype=th.float)
            edge_index = th.as_tensor(edge_index_np, dtype=th.long)

            wl_colours(
                x, edge_index, edge_attr,
                num_iterations=num_iterations, vocab=vocab, frozen=False,
            )
            total_graphs += 1

            if total_graphs % log_every == 0:
                delta = len(vocab) - last_checkpoint_size
                print(f"  graphs={total_graphs:>6}  vocab_size={len(vocab):>6}  (+{delta} since last checkpoint)")
                last_checkpoint_size = len(vocab)

            action = sample_action(env.sim)
            _, _, done, _ = env.step(action)
            if done:
                env.reset()

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


def save_vocab(vocab, path):
    """Saves `vocab` to `path` as JSON - see the module docstring for the encoding scheme."""
    entries = [
        {"signature": _encode_signature(signature), "id": colour_id}
        for signature, colour_id in vocab.items()
    ]
    payload = {"vocab_size": len(vocab), "entries": entries}
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


def main():
    parser = argparse.ArgumentParser(
        description="Build a frozen WL-colour vocab for the Taxi domain by sampling a live GraphTaxiEnv."
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps-per-episode", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
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
             "(default: wl_vocab_taxi_{scenario}_L{num_iterations}.json alongside this script)",
    )
    args = parser.parse_args()
    out_path = args.out if args.out is not None else str(default_out_path(args.scenario, args.num_iterations))

    start = time.time()
    vocab = {}
    total_graphs = sample_graphs(
        vocab,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        num_iterations=args.num_iterations,
        seed=args.seed,
        log_every=args.log_every,
        scenario=args.scenario,
    )
    vocab_size = freeze_vocab(vocab)
    save_vocab(vocab, out_path)
    elapsed = time.time() - start

    print(f"sampled {total_graphs} graphs from scenario={args.scenario!r} "
          f"({args.episodes} episodes x {args.steps_per_episode} steps each), L={args.num_iterations}")
    print(f"vocab_size (frozen, includes OOV) = {vocab_size}")
    print(f"wrote {out_path}")
    print(f"elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
