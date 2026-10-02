"""Time the per-state work of the sampling stage on one representative state:
an empty-taxi decision point of a greedy trajectory at frame >= 1000 (20 waiting
passengers, ~421 nodes). Cell 1 is scored on main's simulator (stale edges),
Cell 3 on this branch's; ground truth and WL histograms on this branch.

    PYTHONPATH=. python -m analysis.time_one_state run --cell1-zip ... --cell3-zip ...
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
OUT = os.path.join(HERE, "results", "timing")
SEED, MIN_FRAME = 100, 1000


def sub(tree, *args):
    env = dict(os.environ, PYTHONPATH=f"{TREES[tree]}:{ROOT}", PYTHONWARNINGS="ignore",
               ANALYSIS_EXPECT_SAGE=os.path.join(TREES[tree], "sage"))
    r = subprocess.run([sys.executable, "-m", "analysis.time_one_state", *args], env=env,
                       cwd=TREES[tree], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-4000:])
        raise SystemExit(f"{tree} {args[0]} failed")
    return r.stdout.rstrip()


def greedy_prefix():
    """Greedy actions from reset until the first empty-taxi frame at or after MIN_FRAME."""
    from analysis.envs import make_env
    from analysis.greedy import greedy_action, road_graph
    env, obs = make_env(SEED)
    G = road_graph(env.sim)
    actions = []
    while not (env.steps >= MIN_FRAME and env.sim.taxi.passenger is None):
        a = greedy_action(env.sim, G)
        obs, _, _, _ = env.step(int(a))
        actions.append(int(a))
    return actions, obs


def mode_state(out):
    actions, obs = greedy_prefix()
    json.dump({"seed": SEED, "actions": actions}, open(out, "w"))
    print(f"state: seed {SEED}, frame {len(actions)}, nodes {len(json.loads(obs)['node_feats'])}")


def replay(state_file):
    from analysis.envs import make_env
    s = json.load(open(state_file))
    env, obs = make_env(s["seed"])
    for a in s["actions"]:
        obs, _, _, _ = env.step(a)
    return obs


def mode_score(zip_path, state_file, with_gt):
    import torch as th
    from analysis.models import load_policy, actor_logits_probs, path_values
    obs = replay(state_file)
    n = len(json.loads(obs)["node_feats"])
    pol, _, _ = load_policy(zip_path)
    t = {}
    t0 = time.time(); actor_logits_probs(pol, obs); t["actor"] = time.time() - t0
    t0 = time.time(); path_values(pol, obs, range(n)); t["score_all_goals"] = time.time() - t0
    th.manual_seed(0)
    t0 = time.time()
    from analysis.models import obs_array
    for s in range(10):  # 10 candidate draws (reuse of the all-goal scores makes selection free)
        with th.no_grad():
            batch, _ = pol._get_latent(obs_array(obs))
            pol._choose_node(pol.action_net, batch)
    t["10_candidate_draws"] = time.time() - t0
    if with_gt:
        from analysis.envs import obs_to_data, plan
        from analysis.goals import classify_goals
        t0 = time.time()
        hists = [plan(obs, g)[0].wl_histogram for g in range(n)]
        t["plan_all_goals_wl"] = time.time() - t0
        types = classify_goals(obs_to_data(obs))
        n_pass = types.count("pickup_deliver")
        t["n_passenger_goals"] = n_pass
        t["n_distinct_wl_hist_all_goals"] = len({tuple(h.flatten().tolist()) for h in hists})
    t["n_goals"] = n
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in t.items()}))


def run(args):
    os.makedirs(OUT, exist_ok=True)
    sf = os.path.join(OUT, "state.json")
    print("[head] " + sub("head", "state", "--out", sf))
    r1 = json.loads(sub("main", "score", "--zip", args.cell1_zip, "--state", sf))
    r3 = json.loads(sub("head", "score", "--zip", args.cell3_zip, "--state", sf, "--gt"))
    print(f"Cell 1 (main simulator): {r1}")
    print(f"Cell 3 (fixed simulator) + ground truth/WL: {r3}")
    per_state = r1["actor"] + r1["score_all_goals"] + r1["10_candidate_draws"] + \
        r3["actor"] + r3["score_all_goals"] + r3["10_candidate_draws"] + r3["plan_all_goals_wl"]
    print(f"per state, both models + ground truth: {per_state:.1f}s -> 500 states: {500 * per_state / 60:.0f} min "
          f"(single process; replay of trajectories not included)")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="mode", required=True)
    r = sp.add_parser("run"); r.add_argument("--cell1-zip", required=True); r.add_argument("--cell3-zip", required=True)
    s = sp.add_parser("state"); s.add_argument("--out", required=True)
    c = sp.add_parser("score"); c.add_argument("--zip", required=True); c.add_argument("--state", required=True)
    c.add_argument("--gt", action="store_true")
    a = ap.parse_args()
    if a.mode == "run":
        run(a)
    elif a.mode == "state":
        mode_state(a.out)
    else:
        mode_score(a.zip, a.state, a.gt)


if __name__ == "__main__":
    main()
