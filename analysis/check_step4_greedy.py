"""Step 4: greedy reference score. greedy_action (analysis/greedy.py, copied from
cell6-wl-atom ffb504f) for full city-taxi-unmasked-v1 episodes, on the same env
seeds as the model evaluation (100-109). Reports deliveries and env reward per
episode. Run on both simulators (this branch and main); greedy only acts on
env.sim, so the two must agree.

    PYTHONPATH=. python -m analysis.check_step4_greedy run [--episodes 10]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TREES = {"head": ROOT, "main": os.path.join(HERE, ".branch_src", "main")}
SEED_BASE = 100


def sub(tree, *args):
    env = dict(os.environ, PYTHONPATH=f"{TREES[tree]}:{ROOT}", PYTHONWARNINGS="ignore",
               ANALYSIS_EXPECT_SAGE=os.path.join(TREES[tree], "sage"))
    r = subprocess.run([sys.executable, "-m", "analysis.check_step4_greedy", *args], env=env,
                       cwd=TREES[tree], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-4000:])
        raise SystemExit(f"{tree} {args[0]} failed")
    return r.stdout.rstrip()


def greedy_episode(seed):
    from analysis.envs import make_env
    from analysis.greedy import greedy_action, road_graph
    env, _ = make_env(seed)
    G = road_graph(env.sim)  # road layout is fixed for the episode
    done, reward, frames, pickups = False, 0.0, 0, 0
    while not done:
        a = greedy_action(env.sim, G)
        carrying = env.sim.taxi.passenger is not None
        _, r, done, _ = env.step(int(a))
        pickups += int(not carrying and env.sim.taxi.passenger is not None)
        reward += r
        frames += 1
    return dict(seed=seed, delivered=100 - env.sim.delivery_limit, reward=reward, frames=frames, pickups=pickups)


def mode_eval(n, out):
    import sage
    want = os.environ.get("ANALYSIS_EXPECT_SAGE")
    assert want is None or os.path.dirname(os.path.abspath(sage.__file__)) == want
    t0 = time.time()
    res = [greedy_episode(SEED_BASE + k) for k in range(n)]
    json.dump(res, open(out, "w"))
    print(f"{n} episodes in {time.time() - t0:.0f}s")


def run(args):
    out_dir = os.path.join(HERE, "results", "step4")
    os.makedirs(out_dir, exist_ok=True)
    res = {}
    for tree in ("head", "main"):
        out = os.path.join(out_dir, f"greedy_{tree}.json")
        print(f"[{tree}] " + sub(tree, "eval", "--episodes", str(args.episodes), "--out", out))
        res[tree] = json.load(open(out))
    print(f"identical on both simulators: {res['head'] == res['main']}")
    r = res["head"]
    d = np.array([x["delivered"] for x in r]); rw = np.array([x["reward"] for x in r])
    print("seed  delivered  env_reward  pickups  frames")
    for x in r:
        print(f"{x['seed']:4d}  {x['delivered']:9d}  {x['reward']:10.0f}  {x['pickups']:7d}  {x['frames']:6d}")
    print(f"deliveries: mean {d.mean():.1f} sd {d.std(ddof=1):.1f} range {d.min()}-{d.max()}")
    print(f"env reward (city-taxi-unmasked-v1: base 0, failed 0, drop-off +1): mean {rw.mean():.1f} "
          f"sd {rw.std(ddof=1):.1f} range {rw.min():.0f}-{rw.max():.0f}; equals deliveries: {bool((rw == d).all())}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="mode", required=True)
    r = sp.add_parser("run"); r.add_argument("--episodes", type=int, default=10)
    e = sp.add_parser("eval"); e.add_argument("--episodes", type=int, default=10); e.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a) if a.mode == "run" else mode_eval(a.episodes, a.out)


if __name__ == "__main__":
    main()
