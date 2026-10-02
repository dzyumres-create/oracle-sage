"""Build the shared state set: empty-taxi decision points from three sources.

  greedy  greedy_action episodes (fixed simulator); decision points = frame 0 and
          the frame right after each drop-off (taxi empty, about to choose).
  cell1   Cell 1 seed 0 episodes on its own stale-edge simulator (main); decision
          points = the start frame of each of its decisions.
  cell3   Cell 3 seed 0 episodes on the fixed simulator; same.

Each state is (source, env seed, frame); the episode's full primitive action
sequence is stored once, so any simulator can replay to any state. Dynamics are
identical across simulators (Step 2), so a state replays to the same taxi /
passengers / time everywhere; only Cell 1's simulator adds stale tether edges.

Stratified: ~55 states per source per depth bucket (early <250, middle 250-1000,
late >=1000). Early has few decision points per episode, so all of them are
kept, middle/late subsampled (seeded).

    PYTHONPATH=. python -m analysis.build_states run --cell1-zip ... --cell3-zip ...
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TREES = {"head": ROOT, "main": os.path.join(HERE, ".branch_src", "main")}
OUT = os.path.join(HERE, "results", "states")
SOURCES = {"greedy": (200, "head"), "cell1": (208, "main"), "cell3": (216, "head")}
EPISODES_PER_SOURCE = 8
PER_BUCKET = 55
BUCKETS = (("early", 0, 250), ("middle", 250, 1000), ("late", 1000, 10 ** 9))


def bucket_of(frame):
    return next(name for name, lo, hi in BUCKETS if lo <= frame < hi)


def sub(tree, *args):
    env = dict(os.environ, PYTHONPATH=f"{TREES[tree]}:{ROOT}", PYTHONWARNINGS="ignore",
               ANALYSIS_EXPECT_SAGE=os.path.join(TREES[tree], "sage"))
    r = subprocess.run([sys.executable, "-m", "analysis.build_states", *args], env=env,
                       cwd=TREES[tree], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-4000:])
        raise SystemExit(f"{tree} {args[0]} failed")
    return r.stdout.rstrip()


def assert_tree():
    import sage
    want = os.environ.get("ANALYSIS_EXPECT_SAGE")
    assert want is None or os.path.dirname(os.path.abspath(sage.__file__)) == want


def greedy_episode(seed):
    from analysis.envs import make_env
    from analysis.greedy import greedy_action, road_graph
    env, _ = make_env(seed)
    G = road_graph(env.sim)
    actions, decisions, done = [], [0], False
    while not done:
        carrying = env.sim.taxi.passenger is not None
        a = int(greedy_action(env.sim, G))
        _, r, done, _ = env.step(a)
        actions.append(a)
        if carrying and r > 0 and not done:
            decisions.append(len(actions))  # right after a drop-off
    return actions, decisions


def model_episode(policy, seed):
    """Same decision loop as analysis.eval_models.run_episode, recording decision frames."""
    import torch as th
    from analysis.envs import make_env, plan
    from analysis.models import obs_array, path_values
    th.manual_seed(seed)
    env, obs = make_env(seed)
    actions, decisions, done = [], [], False
    while not done:
        assert env.sim.taxi.passenger is None
        decisions.append(len(actions))
        with th.no_grad():
            batch, _ = policy._get_latent(obs_array(obs))
            a, _, _, _ = policy._choose_node(policy.action_net, batch)
        cands = a[:, 0].tolist()
        goal = cands[int(np.argmax(path_values(policy, obs, cands)))]
        for act in plan(obs, goal)[1]:
            obs, _, done, _ = env.step(int(act))
            actions.append(int(act))
            if done:
                break
    return actions, decisions


def mode_gen(source, zip_path, out):
    assert_tree()
    base = SOURCES[source][0]
    eps = []
    if source == "greedy":
        for k in range(EPISODES_PER_SOURCE):
            a, d = greedy_episode(base + k)
            eps.append(dict(seed=base + k, actions=a, decisions=d))
    else:
        from analysis.models import load_policy
        pol, _, _ = load_policy(zip_path)
        for k in range(EPISODES_PER_SOURCE):
            a, d = model_episode(pol, base + k)
            eps.append(dict(seed=base + k, actions=a, decisions=d))
    json.dump(eps, open(out, "w"))
    print(f"{source}: {len(eps)} episodes, decision points {[len(e['decisions']) for e in eps]}")


def run(args):
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.RandomState(0)
    episodes, states = {}, []
    for source, (_, tree) in SOURCES.items():
        out = os.path.join(OUT, f"episodes_{source}.json")
        z = {"cell1": args.cell1_zip, "cell3": args.cell3_zip}.get(source, "")
        print(f"[{tree}] " + sub(tree, "gen", "--source", source, "--zip", z, "--out", out))
        eps = json.load(open(out))
        pool = {b[0]: [] for b in BUCKETS}
        for e in eps:
            episodes[f"{source}:{e['seed']}"] = e["actions"]
            for f in e["decisions"]:
                if f < len(e["actions"]):  # a decision point with frames left to act
                    pool[bucket_of(f)].append((e["seed"], f))
        for b, items in pool.items():
            take = items if len(items) <= PER_BUCKET else [items[i] for i in sorted(rng.choice(len(items), PER_BUCKET, replace=False))]
            states += [dict(source=source, seed=s, frame=f, bucket=b) for s, f in take]
            print(f"   {source} {b:6s}: {len(items):4d} decision points, kept {len(take)}")
    for i, s in enumerate(states):
        s["id"] = i
    json.dump({"episodes": episodes, "states": states}, open(os.path.join(OUT, "state_set.json"), "w"))
    print(f"total states: {len(states)}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="mode", required=True)
    r = sp.add_parser("run"); r.add_argument("--cell1-zip", required=True); r.add_argument("--cell3-zip", required=True)
    g = sp.add_parser("gen"); g.add_argument("--source", required=True); g.add_argument("--zip", default="")
    g.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a) if a.mode == "run" else mode_gen(a.source, a.zip, a.out)


if __name__ == "__main__":
    main()
