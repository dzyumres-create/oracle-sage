"""Step 2: replay equivalence between Cell 1's simulator and Cell 3's.

Cell 1's simulator is reconstructed from committed code (main, and
cell2-vilg-gnn with its default oracle_sage convention); Cell 3's is this
branch (stale-tether-edge fix 4d6f1ca). Cell 1 actually ran from RCP
~/oracle-sage = main plus uncommitted patches; their diff --stat touches no
simulator/planner/env file (taxi_world.py, planner.py, taxi_env.py, utils.py),
and its line counts match cell2's numpy-alias commit 8058e2e, bfd9426's
model.save, a 2-line make_mask swap, and 5 dtype-alias lines in
gym_taxi/utils/representations.py (full diff not yet inspected).

A. Recorded replay: a 2,000-frame primitive action sequence is recorded on this
   branch (random goals, executed open-loop through the planner, half of them
   passenger goals so deliveries happen), then replayed on every tree. At every
   frame we compare: taxi location and carried passenger, passengers
   (id, location, destination), time, step counter, delivered count, reward,
   done, and the simulator RNG state (so passenger spawns consume the same
   draws). Road edges are compared too; tether edges are expected to differ
   (stale reverse edges on main/cell2) and are characterised, not required equal.
B. Closed loop: each tree generates its own action sequence from the same goal
   RNG through its own planner (on its own, possibly stale-edged, graph). The
   sequences must be identical, i.e. the stale edges never change a plan.

    PYTHONPATH=. python -m analysis.check_step2_replay run [--seeds 0 1 2]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TREES = {
    "head": ROOT,
    "main": os.path.join(HERE, ".branch_src", "main"),
    "cell2": os.path.join(HERE, ".branch_src", "cell2-vilg-gnn"),
}
OUT = os.path.join(HERE, "results", "step2")
FRAMES = 2000


def sub(tree, *args):
    env = dict(os.environ, PYTHONPATH=f"{TREES[tree]}:{ROOT}", PYTHONWARNINGS="ignore",
               ANALYSIS_EXPECT_SAGE=os.path.join(TREES[tree], "sage"))
    r = subprocess.run([sys.executable, "-m", "analysis.check_step2_replay", *args], env=env,
                       cwd=TREES[tree], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-4000:])
        raise SystemExit(f"{tree} {args[0]} failed")
    return r.stdout.strip()


def assert_tree():
    import sage
    want = os.environ.get("ANALYSIS_EXPECT_SAGE")
    got = os.path.dirname(os.path.abspath(sage.__file__))
    assert want is None or got == want, f"imported sage from {got}, expected {want}"


# ------------------------------------------------------------ per-frame trace
def _rng_hash(rng):
    st = rng.get_state()
    return hashlib.sha1(st[1].tobytes() + str(st[2:]).encode()).hexdigest()[:16]


def frame_record(env, reward, done):
    from analysis.envs import sim_summary
    rec = sim_summary(env)
    rec["passengers"] = [list(p) for p in rec["passengers"]]
    road, tether = [], []
    for u, v, d in env.sim.graph.edges(data=True):
        (road if d["attr"][0] == 1 else tether).append((int(u), int(v), tuple(int(x) for x in d["attr"])))
    rec.update(
        reward=float(reward), done=bool(done), rng=_rng_hash(env.sim.random),
        road=hashlib.sha1(str(sorted(road)).encode()).hexdigest()[:16],
        tether=sorted(tether), n_nodes=env.sim.graph.number_of_nodes(),
    )
    return rec


def trace_episode(seed, actions=None, goal_seed=None):
    """Replay `actions`, or (if None) generate them closed-loop from goal_seed."""
    from analysis.envs import make_env, obs_to_data, plan
    env, obs = make_env(seed)
    trace = [frame_record(env, 0.0, False)]
    taken = []
    if actions is not None:
        for a in actions:
            obs, r, done, _ = env.step(int(a))
            taken.append(int(a))
            trace.append(frame_record(env, r, done))
            if done:
                break
        return taken, trace
    rng = np.random.RandomState(goal_seed)
    done = False
    while not done and len(taken) < FRAMES:
        x = obs_to_data(obs).x
        passengers = np.flatnonzero(x[:, 2].numpy() == 1)
        if len(passengers) and rng.uniform() < 0.5:
            goal = int(passengers[rng.randint(len(passengers))])
        else:
            goal = int(rng.randint(len(x)))
        _, acts = plan(obs, goal)
        for a in acts:
            obs, r, done, _ = env.step(int(a))
            taken.append(int(a))
            trace.append(frame_record(env, r, done))
            if done or len(taken) >= FRAMES:
                break
    return taken, trace


def mode_record(seed, out):
    assert_tree()
    actions, trace = trace_episode(seed, goal_seed=1000 + seed)
    json.dump({"actions": actions, "trace": trace}, open(out, "w"))
    print(f"frames={len(actions)} delivered={trace[-1]['delivered']} done={trace[-1]['done']}")


def mode_replay(seed, actions_file, out):
    assert_tree()
    actions = json.load(open(actions_file))["actions"]
    taken, trace = trace_episode(seed, actions=actions)
    json.dump({"actions": taken, "trace": trace}, open(out, "w"))
    print(f"frames={len(taken)} delivered={trace[-1]['delivered']}")


# ------------------------------------------------------------ comparison
DYNAMICS = ["taxi_location", "taxi_passenger", "passengers", "time", "steps", "delivered", "reward", "done", "rng"]


def compare(ref, other):
    """First differing frame per field (None if identical), plus tether-edge diagnostics."""
    n = min(len(ref), len(other))
    first = {}
    for f in DYNAMICS + ["road", "n_nodes"]:
        first[f] = next((i for i in range(n) if ref[i][f] != other[i][f]), None)
    extra_counts, kinds, missing = [], set(), 0
    for r, o in zip(ref, other):
        as_set = lambda es: {(u, v, tuple(a)) for u, v, a in es}  # JSON turns tuples into lists
        rt, ot = as_set(r["tether"]), as_set(o["tether"])
        missing += len(rt - ot)
        extra = ot - rt
        extra_counts.append(len(extra))
        for (u, v, attr) in extra:
            # classify: reverse half (attr[3] == -1) of a location->taxi tether or a
            # location->passenger tether, i.e. what attempt_move/attempt_pickup left behind
            kinds.add(("loc->taxi" if v == 0 else "loc->passenger", tuple(attr)))
    return dict(frames=(len(ref), len(other)), first_diff=first, tether_missing=missing,
                extra_tether_max=max(extra_counts), extra_tether_final=extra_counts[-1],
                extra_tether_frames=sum(c > 0 for c in extra_counts), extra_kinds=sorted(kinds))


def run(args):
    os.makedirs(OUT, exist_ok=True)
    p = lambda name: os.path.join(OUT, name)
    summary = {}
    for seed in args.seeds:
        print(f"== seed {seed}")
        rec = p(f"seed{seed}_record_head.json")
        print(f"  [head] record (closed loop): {sub('head', 'record', '--seed', str(seed), '--out', rec)}")
        traces = {"head": json.load(open(rec))}
        for tree in ("main", "cell2"):
            out = p(f"seed{seed}_replay_{tree}.json")
            print(f"  [{tree}] replay head's actions: {sub(tree, 'replay', '--seed', str(seed), '--actions', rec, '--out', out)}")
            traces[tree] = json.load(open(out))
        for tree in ("main", "cell2"):
            c = compare(traces["head"]["trace"], traces[tree]["trace"])
            same = all(v is None for k, v in c["first_diff"].items())
            print(f"  A head vs {tree:5s}: frames {c['frames']}, dynamics+RNG+road identical at every frame: {same}"
                  + ("" if same else f" first diffs {c['first_diff']}"))
            print(f"      tether edges: head-only {c['tether_missing']}, {tree}-only max {c['extra_tether_max']}"
                  f" / final {c['extra_tether_final']} (present in {c['extra_tether_frames']} frames), kinds {c['extra_kinds']}")
            summary[f"seed{seed}_A_{tree}"] = dict(c, identical=same)
        # B: closed loop on main and cell2 with the same goal RNG
        for tree in ("main", "cell2"):
            out = p(f"seed{seed}_record_{tree}.json")
            sub(tree, "record", "--seed", str(seed), "--out", out)
            own = json.load(open(out))
            same_actions = own["actions"] == traces["head"]["actions"]
            c = compare(traces["head"]["trace"], own["trace"])
            same = all(v is None for v in c["first_diff"].values())
            print(f"  B {tree:5s} closed loop: {len(own['actions'])} actions, identical action sequence to head: "
                  f"{same_actions}, dynamics identical: {same}")
            summary[f"seed{seed}_B_{tree}"] = dict(same_actions=same_actions, identical=same)
    json.dump(summary, open(p("summary.json"), "w"), indent=1, default=str)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="mode", required=True)
    r = sp.add_parser("run"); r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    rc = sp.add_parser("record"); rc.add_argument("--seed", type=int, required=True); rc.add_argument("--out", required=True)
    rp = sp.add_parser("replay"); rp.add_argument("--seed", type=int, required=True)
    rp.add_argument("--actions", required=True); rp.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mode == "run":
        run(a)
    elif a.mode == "record":
        mode_record(a.seed, a.out)
    else:
        mode_replay(a.seed, a.actions, a.out)


if __name__ == "__main__":
    main()
