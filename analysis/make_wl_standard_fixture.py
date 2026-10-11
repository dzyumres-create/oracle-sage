"""
Records the standard-domain WL ground truth that tests/test_wl_ternary.py's regression test
compares against: wl_colours() outputs for all three conventions on real old-domain graphs
(live states and planner projections), at commit ebbf25b -- the code the standard-Taxi L=1
WL baselines ran on.

The fixture stores the INPUT tensors as well as the outputs, so the regression test feeds
the exact same graphs to wl_colours on any stack: env.seed() gives a numpy Generator under
gym 0.26 but a RandomState under gym 0.18, so regenerating the graphs from seeds would give
different worlds on the Mac and on RCP.

Recorded per convention:
  - growing mode, one vocab shared across the graph sequence (in order), at L = 1, 2, 3:
    colours and histogram per graph, and the vocab's signatures in insertion order;
  - frozen mode with that convention's production L=1 vocab: colours and histogram.

Imports only modules that exist at ebbf25b. Run against a checkout of ebbf25b:

    PYTHONPATH=<ebbf25b checkout> python analysis/make_wl_standard_fixture.py <out.json.gz>
"""
import copy
import gzip
import json
import subprocess
import sys

import numpy as np

import sage.domains.utils.build_wl_vocab as bwv  # noqa: F401  (gym 0.26 Generator.randint shim, as a side effect)
from sage.domains.gym_taxi import REWARDS
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.simulator.planner import Planner
from sage.domains.gym_taxi.utils import wl_vocab_cache
from sage.domains.utils.wl_colours import wl_colours

CONVENTIONS = ("oracle_sage", "vilg", "atom")
SEEDS = (0, 1)
SAMPLE_STEPS = (0, 40, 120)
GROWING_LEVELS = (1, 2, 3)
FROZEN_VOCAB = {
    "oracle_sage": "wl_vocab_taxi_city_L1_edgefixed.json",
    "vilg": "wl_vocab_taxi_city_vilg_L1_full.json",
    "atom": "wl_vocab_taxi_city_atom_L1_full.json",
}
FROZEN_L = 1


def _legal_actions(sim):
    """Same candidate set as build_wl_vocab.sample_action, sorted (no RNG)."""
    taxi_location = sim.taxi.location
    candidates = {
        c for c in sim.graph.successors(taxi_location)
        if sim.graph.nodes[c]["attr"] != [0, 0, 1] or c in sim.passengers
    }
    candidates.add(taxi_location)
    candidates.add(0)
    return sorted(candidates)


def _int_list(tensor):
    """Every value in these tensors is an integer (one-hots, 0/1 bits, +-1 directions,
    indices); stored as ints to keep the fixture small, and checked to be exact."""
    values = tensor.detach().cpu().numpy()
    as_int = values.astype(np.int64)
    assert np.array_equal(as_int, values), "non-integer value in a graph tensor"
    return as_int.tolist()


def _graph_record(data, kind, seed, step, goal):
    return {
        "kind": kind, "seed": seed, "step": step, "goal": goal,
        "x": _int_list(data.x),
        "edge_index": _int_list(data.edge_index),
        "edge_attr": _int_list(data.edge_attr),
    }


def collect_graphs(convention):
    """Live states at SAMPLE_STEPS of one episode per seed, each followed by two planner
    projections (the last and the middle selectable node as goals)."""
    planner = Planner(graph_convention=convention)
    records = []
    for seed in SEEDS:
        env = GraphTaxiEnv(representation="graph", scenario="city", mask=False,
                           rewards=REWARDS["v1"], graph_convention=convention)
        env.seed(seed)
        env.reset()
        rng = np.random.RandomState(seed)
        for step in range(max(SAMPLE_STEPS) + 1):
            if step in SAMPLE_STEPS:
                data = bwv.to_planner_data(env.sim, graph_convention=convention)
                records.append(_graph_record(data, "live", seed, step, None))
                selectable = data.mask.nonzero(as_tuple=True)[0].tolist()
                for goal in (selectable[-1], selectable[len(selectable) // 2]):
                    projection, _actions = planner.plan(copy.deepcopy(data), int(goal))
                    records.append(_graph_record(projection, "projection", seed, step, int(goal)))
            actions = _legal_actions(env.sim)
            _obs, _reward, done, _info = env.step(actions[rng.randint(len(actions))])
            if done:
                env.reset()
        env.close()
    return records


def _tensors(record):
    import torch as th
    return (
        th.as_tensor(record["x"], dtype=th.float),
        th.as_tensor(record["edge_index"], dtype=th.long),
        th.as_tensor(record["edge_attr"], dtype=th.float),
    )


def run_wl(convention, records):
    out = {"growing": {}, "frozen": {}}
    for level in GROWING_LEVELS:
        vocab = {}
        colours, histograms = [], []
        for record in records:
            c, h = wl_colours(*_tensors(record), num_iterations=level, vocab=vocab, frozen=False,
                              graph_convention=convention)
            colours.append(c.tolist())
            histograms.append([int(v) for v in h.tolist()])
        out["growing"][str(level)] = {
            "colours": colours,
            "histograms": histograms,
            "vocab_order": [repr(signature) for signature in vocab],
        }
    path = wl_vocab_cache.WL_VOCAB_PATH.parent / FROZEN_VOCAB[convention]
    frozen_vocab = wl_vocab_cache._load_vocab_from_path(str(path))
    colours, histograms = [], []
    for record in records:
        c, h = wl_colours(*_tensors(record), num_iterations=FROZEN_L, vocab=frozen_vocab, frozen=True,
                          graph_convention=convention)
        colours.append(c.tolist())
        histograms.append([int(v) for v in h.tolist()])
    out["frozen"] = {"vocab_file": FROZEN_VOCAB[convention], "L": FROZEN_L,
                     "colours": colours, "histograms": histograms}
    return out


def main(argv):
    out_path = argv[0]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = None
    fixture = {"generated_at_commit": commit, "conventions": {}}
    for convention in CONVENTIONS:
        records = collect_graphs(convention)
        fixture["conventions"][convention] = dict(graphs=records, **run_wl(convention, records))
        sys.stderr.write("{}: {} graphs\n".format(convention, len(records)))
    with gzip.open(out_path, "wt") as f:
        json.dump(fixture, f, sort_keys=True)
    sys.stderr.write("wrote {} (commit {})\n".format(out_path, commit))


if __name__ == "__main__":
    main(sys.argv[1:])
