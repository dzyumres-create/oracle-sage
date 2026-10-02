"""Score the state set: ground truth (fixed simulator) and per-model read-outs.

Stages (driver runs them; the heavy ones sharded over 4 worker processes):
  snapshot  replay each episode on a simulator, save the observation at every
            state frame (head = fixed simulator = Cell 3's view; main = Cell 1's
            stale-edge view). On head also record greedy's target passenger, read
            off the live simulator with greedy_action's own selection rule.
  gt        (head) for every goal of every state: goal type, plan length, and the
            projection's WL histogram key and full discriminator-input key
            (histogram + time_left); for passenger goals the plan split into
            pickup distance and trip length.
  model     for each model on its own view: actor probabilities over all nodes,
            discriminator score of every goal, and K=20 seeded draws of the 3
            candidates with the selection made two ways: native (first argmax, as
            th.max in select_action) and uniform random tie-breaking.

    PYTHONPATH=. python -m analysis.score_states run --cell1-zip ... --cell3-zip ... [--workers 4]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TREES = {"head": ROOT, "main": os.path.join(HERE, ".branch_src", "main")}
STATES = os.path.join(HERE, "results", "states")
OUT = os.path.join(HERE, "results", "scores")
K_DRAWS = 20
MODEL_TREE = {"cell1": "main", "cell3": "head"}


def sub(tree, *args):
    env = dict(os.environ, PYTHONPATH=f"{TREES[tree]}:{ROOT}", PYTHONWARNINGS="ignore",
               ANALYSIS_EXPECT_SAGE=os.path.join(TREES[tree], "sage"))
    r = subprocess.run([sys.executable, "-m", "analysis.score_states", *args], env=env,
                       cwd=TREES[tree], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-4000:])
        raise SystemExit(f"{tree} {args[0]} {args[1:]} failed")
    return r.stdout.rstrip()


def assert_tree():
    import sage
    want = os.environ.get("ANALYSIS_EXPECT_SAGE")
    assert want is None or os.path.dirname(os.path.abspath(sage.__file__)) == want


def load_set():
    return json.load(open(os.path.join(STATES, "state_set.json")))


def shard(states, i, n):
    return [s for s in states if s["id"] % n == i]


# ------------------------------------------------------------------ snapshot
def mode_snapshot(tree, out):
    assert_tree()
    import networkx as nx
    from analysis.envs import make_env, sim_summary
    from analysis.greedy import road_graph
    ss = load_set()
    by_ep = {}
    for s in ss["states"]:
        by_ep.setdefault(f"{s['source']}:{s['seed']}", []).append(s)
    obs_out, meta = {}, {}
    for key, states in by_ep.items():
        seed = int(key.split(":")[1])
        frames = {s["frame"]: s["id"] for s in states}
        env, obs = make_env(seed)
        G = road_graph(env.sim)
        actions = ss["episodes"][key]
        for t in range(max(frames) + 1):
            if t in frames:
                sid = frames[t]
                assert env.sim.taxi.passenger is None, (key, t)
                obs_out[sid] = obs
                m = sim_summary(env)
                if env.sim.passengers:  # greedy_action's own selection rule (analysis/greedy.py)
                    m["greedy_target"] = int(min(env.sim.passengers, key=lambda p: nx.shortest_path_length(
                        G, env.sim.taxi.location, env.sim.passengers[p].location)))
                meta[sid] = m
            if t < len(actions):
                obs, _, _, _ = env.step(actions[t])
    json.dump({"obs": obs_out, "meta": meta}, open(out, "w"))
    print(f"{tree}: {len(obs_out)} observations")


# ------------------------------------------------------------------ ground truth
def _key(t):
    return hashlib.sha1(np.ascontiguousarray(t.cpu().numpy()).tobytes()).hexdigest()[:16]


def mode_gt(i, n, out):
    assert_tree()
    from analysis.envs import obs_to_data, plan
    from analysis.goals import classify_goals
    obs = json.load(open(os.path.join(OUT, "snapshot_head.json")))["obs"]
    res = {}
    for s in shard(load_set()["states"], i, n):
        o = obs[str(s["id"])]
        data = obs_to_data(o)
        types = classify_goals(data)
        plen, pick, trip, hk, fk = [], [], [], [], []
        for g in range(len(types)):
            proj, acts = plan(o, g)
            plen.append(len(acts))
            if types[g] == "pickup_deliver":
                k = acts.index(g)            # move1 + [pid] + move2 + [taxi]
                pick.append(k)
                trip.append(len(acts) - k - 2)
            else:
                pick.append(None); trip.append(None)
            hk.append(_key(proj.wl_histogram))
            fk.append(_key(th_cat(proj.wl_histogram, proj.global_features[:, 0:1])))
        res[s["id"]] = dict(types=types, plan_len=plen, pickup=pick, trip=trip, hist_key=hk, full_key=fk)
    json.dump(res, open(out, "w"))
    print(f"gt shard {i}: {len(res)} states")


def th_cat(a, b):
    import torch as th
    return th.cat([a.float(), b.float()], dim=1)


# ------------------------------------------------------------------ per-model
def mode_model(model, zip_path, i, n, out):
    assert_tree()
    import torch as th
    from analysis.models import load_policy, actor_logits_probs, path_values, obs_array
    tree = MODEL_TREE[model]
    obs = json.load(open(os.path.join(OUT, f"snapshot_{tree}.json")))["obs"]
    pol, _, _ = load_policy(zip_path)
    res = {}
    for s in shard(load_set()["states"], i, n):
        o = obs[str(s["id"])]
        _, probs = actor_logits_probs(pol, o)
        scores = path_values(pol, o, range(len(probs)))
        rng = np.random.RandomState(10_000 + s["id"])
        draws = []
        for k in range(K_DRAWS):
            th.manual_seed(1_000_000 * (1 if model == "cell1" else 3) + 100 * s["id"] + k)
            with th.no_grad():
                batch, _ = pol._get_latent(obs_array(o))
                a, _, _, _ = pol._choose_node(pol.action_net, batch)
            cands = a[:, 0].tolist()
            cs = scores[cands]
            top = cs.max()
            tied = np.flatnonzero(cs == top)
            draws.append(dict(cands=cands, native=cands[int(np.argmax(cs))],
                              pick=cands[int(rng.choice(tied))], n_tied=int(len(tied))))
        res[s["id"]] = dict(probs=probs.tolist(), scores=scores.tolist(), draws=draws)
    json.dump(res, open(out, "w"))
    print(f"{model} shard {i}: {len(res)} states")


# ------------------------------------------------------------------ driver
def run(args):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(2) as ex:
        for r in ex.map(lambda t: sub(t, "snapshot", "--tree", t, "--out", os.path.join(OUT, f"snapshot_{t}.json")),
                        ["head", "main"]):
            print(r)
    print(f"snapshots done in {time.time() - t0:.0f}s")
    W = args.workers
    jobs = [("head", ["gt", "--shard", str(i), "--of", str(W), "--out", os.path.join(OUT, f"gt_{i}.json")]) for i in range(W)]
    for m, z in (("cell1", args.cell1_zip), ("cell3", args.cell3_zip)):
        jobs += [(MODEL_TREE[m], ["model", "--model", m, "--zip", z, "--shard", str(i), "--of", str(W),
                                  "--out", os.path.join(OUT, f"{m}_{i}.json")]) for i in range(W)]
    with ThreadPoolExecutor(W) as ex:
        for r in ex.map(lambda j: sub(j[0], *j[1]), jobs):
            print(r)
    # merge shards
    for name in ("gt", "cell1", "cell3"):
        merged = {}
        for i in range(W):
            merged.update(json.load(open(os.path.join(OUT, f"{name}_{i}.json"))))
        json.dump(merged, open(os.path.join(OUT, f"{name}.json"), "w"))
        print(f"{name}: {len(merged)} states")
    print(f"total {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="mode", required=True)
    r = sp.add_parser("run"); r.add_argument("--cell1-zip", required=True); r.add_argument("--cell3-zip", required=True)
    r.add_argument("--workers", type=int, default=4)
    s = sp.add_parser("snapshot"); s.add_argument("--tree", required=True); s.add_argument("--out", required=True)
    g = sp.add_parser("gt"); g.add_argument("--shard", type=int, required=True); g.add_argument("--of", type=int, required=True)
    g.add_argument("--out", required=True)
    m = sp.add_parser("model"); m.add_argument("--model", required=True); m.add_argument("--zip", required=True)
    m.add_argument("--shard", type=int, required=True); m.add_argument("--of", type=int, required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mode == "run":
        run(a)
    elif a.mode == "snapshot":
        mode_snapshot(a.tree, a.out)
    elif a.mode == "gt":
        mode_gt(a.shard, a.of, a.out)
    else:
        mode_model(a.model, a.zip, a.shard, a.of, a.out)


if __name__ == "__main__":
    main()
