"""End-to-end check: run a loaded model for full city episodes and count deliveries.

Each decision follows the training-time logic exactly: _choose_node samples
num_planning_choices distinct candidates (th.multinomial), each is scored by
the discriminator, the first argmax is taken (th.max semantics, as in
select_action), and the chosen plan is executed open-loop to completion (as in
PlanFeedback_A2C.collect_rollouts). Scores come from analysis.models.path_values,
which Step 1 showed reproduces _choose_top_action's choice and value (60/60).

Configurations (same env seeds for all):
  cell1_own   Cell 1 on its own stale-edge simulator (main)
  cell3_fixed Cell 3 on the fixed simulator (this branch)
  cell1_fixed Cell 1 on the fixed simulator (never saw clean graphs in training)

    PYTHONPATH=. python -m analysis.eval_models run --cell1-zip ... --cell3-zip ... [--episodes 10]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TREES = {"head": ROOT, "main": os.path.join(HERE, ".branch_src", "main")}
OUT = os.path.join(HERE, "results", "eval")
SEED_BASE = 100


def sub(tree, *args):
    env = dict(os.environ, PYTHONPATH=f"{TREES[tree]}:{ROOT}", PYTHONWARNINGS="ignore",
               ANALYSIS_EXPECT_SAGE=os.path.join(TREES[tree], "sage"))
    r = subprocess.run([sys.executable, "-m", "analysis.eval_models", *args], env=env,
                       cwd=TREES[tree], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-4000:])
        raise SystemExit(f"{tree} {args[0]} failed")
    return r.stdout.rstrip()


def assert_tree():
    import sage
    want = os.environ.get("ANALYSIS_EXPECT_SAGE")
    got = os.path.dirname(os.path.abspath(sage.__file__))
    assert want is None or got == want, f"imported sage from {got}, expected {want}"
    return os.path.relpath(got, ROOT)


def run_episode(policy, seed, torch_seed):
    import torch as th
    from analysis.envs import make_env, obs_to_data, plan
    from analysis.models import obs_array, path_values
    from analysis.goals import classify_goals
    th.manual_seed(torch_seed)
    env, obs = make_env(seed)
    done, frames, reward = False, 0, 0.0
    decisions, types, plan_lens = 0, Counter(), []
    ties_exact = ties_rel = 0
    while not done:
        with th.no_grad():
            batch, _ = policy._get_latent(obs_array(obs))
            a, _, _, _ = policy._choose_node(policy.action_net, batch)
        cands = a[:, 0].tolist()
        scores = path_values(policy, obs, cands)
        best = int(np.argmax(scores))  # first max, as th.max in select_action
        top = scores.max()
        ties_exact += int((scores == top).sum() > 1)
        ties_rel += int((scores >= top - 1e-6 * max(1.0, abs(top))).sum() > 1)
        goal = cands[best]
        types[classify_goals(obs_to_data(obs), [goal])[0]] += 1
        _, acts = plan(obs, goal)
        plan_lens.append(len(acts))
        decisions += 1
        for act in acts:
            obs, r, done, _ = env.step(int(act))
            frames += 1
            reward += r
            if done:
                break
    delivered = 100 - env.sim.delivery_limit
    return dict(seed=seed, delivered=int(delivered), reward=float(reward), frames=frames, decisions=decisions,
                mean_plan_len=float(np.mean(plan_lens)), types=dict(types),
                tie_exact=ties_exact, tie_rel1e6=ties_rel)


def mode_eval(zip_path, n_episodes, out):
    tree = assert_tree()
    from analysis.models import load_policy
    pol, _, _ = load_policy(zip_path)
    res = []
    t0 = time.time()
    for k in range(n_episodes):
        res.append(run_episode(pol, SEED_BASE + k, torch_seed=SEED_BASE + k))
    json.dump(res, open(out, "w"))
    d = np.array([r["delivered"] for r in res])
    print(f"sage={tree} class={type(pol).__name__} {n_episodes} episodes in {time.time() - t0:.0f}s: "
          f"deliveries mean {d.mean():.1f} sd {d.std(ddof=1):.1f} range {d.min()}-{d.max()} {d.tolist()}")


def summarise(name, res):
    d = np.array([r["delivered"] for r in res])
    rw = np.array([r["reward"] for r in res])
    dec = sum(r["decisions"] for r in res)
    types = Counter()
    for r in res:
        types.update(r["types"])
    te = sum(r["tie_exact"] for r in res)
    tr = sum(r["tie_rel1e6"] for r in res)
    fr = np.array([r["frames"] for r in res])
    print(f"  {name:12s} deliveries mean {d.mean():5.1f}  sd {d.std(ddof=1):4.1f}  range {d.min()}-{d.max()}"
          f"   (reward == deliveries: {bool((rw == d).all())}; frames {fr.min()}-{fr.max()})")
    print(f"  {'':12s} {dec} decisions, {sum(fr) / dec:.1f} frames/decision; chosen goal types "
          + ", ".join(f"{t} {100 * types[t] / dec:.1f}%" for t in ("pickup_deliver", "deliver", "move", "noop", "phantom"))
          + f"; ties among the 3 candidates: exact {100 * te / dec:.1f}%, within 1e-6 rel {100 * tr / dec:.1f}%")


def run(args):
    os.makedirs(OUT, exist_ok=True)
    configs = [("cell1_own", "main", args.cell1_zip), ("cell3_fixed", "head", args.cell3_zip),
               ("cell1_fixed", "head", args.cell1_zip)]
    for name, tree, z in configs:
        out = os.path.join(OUT, f"{name}.json")
        print(f"[{name}] " + sub(tree, "eval", "--zip", z, "--episodes", str(args.episodes), "--out", out))
    print(f"== summary (env seeds {SEED_BASE}-{SEED_BASE + args.episodes - 1}, same for all configs)")
    for name, _, _ in configs:
        summarise(name, json.load(open(os.path.join(OUT, f"{name}.json"))))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="mode", required=True)
    r = sp.add_parser("run")
    r.add_argument("--cell1-zip", required=True); r.add_argument("--cell3-zip", required=True)
    r.add_argument("--episodes", type=int, default=10)
    e = sp.add_parser("eval")
    e.add_argument("--zip", required=True); e.add_argument("--episodes", type=int, default=10)
    e.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a) if a.mode == "run" else mode_eval(a.zip, a.episodes, a.out)


if __name__ == "__main__":
    main()
