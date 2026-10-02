"""Step 1 checks: model loading for Cell 1 (GNN) and Cell 3 (WL).

  1. Loaded weights equal the zip's tensors exactly (strict load, torch.equal).
  2. Determinism: actor logits on fixed observations are bit-identical across
     repeated calls and across two independent loads.
  3. Cell 1 code-path equivalence: the same Cell 1 weights give identical actor
     probabilities and discriminator scores under main's policy code (what
     Cell 1 trained with; current-state input = actor's batch.global_features)
     and under this branch's code (_encode_current_state =
     gnn_extractor2(features_extractor(raw state))). Expected equal because
     shared_gnn=True makes gnn_extractor2 the same module as gnn_extractor.
  4. WL loader: a WLPlanFeedbackPolicy is saved into a zip in the SB3 layout and
     loaded back through the same helper (stand-in until Cell 3's zip is on the
     Mac); pass --wl-zip to run the real one.

Each source tree runs in its own subprocess (they all import as `sage`):
    PYTHONPATH=. python -m analysis.check_step1_loading run --cell1-zip ... [--wl-zip ...]
"""
import argparse
import json
import os
import subprocess
import sys
import time
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TREES = {"head": ROOT, "main": os.path.join(HERE, ".branch_src", "main")}
OUT = os.path.join(HERE, "results", "step1")


def sub(tree, *args):
    # cwd = the tree root: `python -m` puts cwd first on sys.path, ahead of PYTHONPATH
    env = dict(os.environ, PYTHONPATH=f"{TREES[tree]}:{ROOT}", PYTHONWARNINGS="ignore",
               ANALYSIS_EXPECT_SAGE=os.path.join(TREES[tree], "sage"))
    cmd = [sys.executable, "-m", "analysis.check_step1_loading", *args]
    r = subprocess.run(cmd, env=env, cwd=TREES[tree], capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines()]
    print("\n".join(f"  [{tree}] {l}" for l in lines))
    if r.returncode != 0:
        print(r.stderr[-4000:])
        raise SystemExit(f"{tree} {args[0]} failed")


# ------------------------------------------------------------ subprocess modes
def assert_tree():
    import sage
    want = os.environ.get("ANALYSIS_EXPECT_SAGE")
    got = os.path.dirname(os.path.abspath(sage.__file__))
    assert want is None or got == want, f"imported sage from {got}, expected {want}"
    return os.path.relpath(got, ROOT)


def gen_obs(out, seed=0, n=12):
    """Fixed observation set from random-goal rollouts on this tree's simulator,
    chosen to include carrying states and many waiting passengers."""
    tree = assert_tree()
    from analysis.envs import make_env, random_goal_rollout, sim_summary
    env, obs = make_env(seed)
    rng = np.random.RandomState(seed)
    states, meta = [obs], [sim_summary(env)]
    while len(states) < n:
        _, obs, done = random_goal_rollout(env, obs, rng, n_decisions=10, p_passenger=0.5,
                                          max_frames=int(rng.randint(15, 60)))
        if done:
            break
        states.append(obs)
        meta.append(sim_summary(env))
    json.dump({"tree": tree, "obs": states, "meta": meta}, open(out, "w"))
    carrying = sum(m["taxi_passenger"] is not None for m in meta)
    print(f"sage={tree} states={len(states)} carrying={carrying} "
          f"passengers={[len(m['passengers']) for m in meta]} time={[m['time'] for m in meta]}")


def probe(zip_path, obs_file, out, n_score_states, wl_vocab=None):
    import torch as th
    tree = assert_tree()
    from analysis.models import load_policy, check_weights_equal, actor_logits_probs, path_values
    th.manual_seed(0)
    obs = json.load(open(obs_file))["obs"]
    pol, data, sd = load_policy(zip_path, wl_vocab)
    bad = check_weights_equal(pol, sd)
    alias = getattr(pol, "gnn_extractor2", None) is getattr(pol, "gnn_extractor", None)
    print(f"sage={tree} class={type(pol).__name__} tensors={len(sd)} "
          f"mismatched={len(bad)} gnn_extractor2_is_gnn_extractor={alias}")

    logits = [actor_logits_probs(pol, o)[0] for o in obs]
    logits_again = [actor_logits_probs(pol, o)[0] for o in obs]
    pol2, _, _ = load_policy(zip_path, wl_vocab)
    logits_reload = [actor_logits_probs(pol2, o)[0] for o in obs]
    same_call = all(np.array_equal(a, b) for a, b in zip(logits, logits_again))
    same_load = all(np.array_equal(a, b) for a, b in zip(logits, logits_reload))
    print(f"logits bit-identical: repeated call={same_call} independent reload={same_load}")

    probs = [actor_logits_probs(pol, o)[1] for o in obs]
    t0 = time.time()
    scores = []
    for o in obs[:n_score_states]:
        n_nodes = len(json.loads(o)["node_feats"])
        scores.append(path_values(pol, o, range(n_nodes)))
    dt = time.time() - t0
    n_proj = sum(len(s) for s in scores)
    print(f"scored all goals in {n_score_states} states: {n_proj} projections in {dt:.1f}s "
          f"({1000 * dt / max(n_proj, 1):.1f} ms/projection)")
    # repeat scoring of state 0: must be identical
    again = path_values(pol, obs[0], range(len(scores[0])))
    print(f"path values bit-identical on repeat: {np.array_equal(again, scores[0])}")

    # training-time decision branch (eval_action=None) vs path_values() on its own 3 candidates
    from analysis.models import obs_array
    ok, n, md = 0, 0, 0.0
    for i, o in enumerate(obs):
        for s in range(5):
            th.manual_seed(1000 * i + s)
            with th.no_grad():
                batch, sym = pol._get_latent(obs_array(o))
                a, pa, ds, ent = pol._choose_node(pol.action_net, batch)
                sel, _, val, _, _, _ = pol._choose_top_action(a, pa, ds, ent, batch, sym, None)
            cands = a[:, 0].tolist()
            pv = path_values(pol, o, cands)
            rel = abs(float(val) - pv.max()) / max(1.0, abs(float(pv.max())))
            md = max(md, rel)
            n += 1
            ok += int(len(set(cands)) == len(cands) and int(sel) == cands[int(np.argmax(pv))]
                      and rel < 1e-6)
    print(f"training decisions consistent with path_values() (distinct candidates, same argmax, "
          f"value within 1e-6 relative): {ok}/{n}; max relative value diff from batch composition = {md:.2g}")
    np.savez(out, **{f"logits_{i}": l for i, l in enumerate(logits)},
             **{f"probs_{i}": p for i, p in enumerate(probs)},
             **{f"scores_{i}": s for i, s in enumerate(scores)})


def make_wl_zip(out):
    """Save a freshly initialised WLPlanFeedbackPolicy in the SB3 zip layout."""
    import io
    import torch as th
    assert_tree()
    from analysis.models import make_spaces
    from sage.agent.wl_plan_feedback_policy import WLPlanFeedbackPolicy
    from sage.domains.gym_taxi.utils.wl_vocab_cache import WL_VOCAB_PATH
    th.manual_seed(123)
    kw = dict(optimizer_class=th.optim.AdamW, optimizer_kwargs={"weight_decay": 1e-4}, ortho_init=False,
              exploration_initial_eps=0, exploration_final_eps=0, exploration_fraction=0.1,
              shared_gnn=False, layer_norm=False, num_planning_choices=3, features_extractor_kwargs={},
              wl_vocab_path=str(WL_VOCAB_PATH))
    o, a = make_spaces()
    pol = WLPlanFeedbackPolicy(o, a, lambda _: 3e-4, **dict(kw, features_extractor_kwargs={}))
    # perturb every tensor so a load that silently kept init values would be caught
    with th.no_grad():
        for p in pol.parameters():
            p.add_(th.randn_like(p))
    data = {"policy_class": {"__module__": "sage.agent.wl_plan_feedback_policy"},
            "policy_kwargs": {k: (str(v) if k == "optimizer_class" else v) for k, v in kw.items()}}
    buf = io.BytesIO()
    th.save(pol.state_dict(), buf)
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("data", json.dumps(data))
        z.writestr("policy.pth", buf.getvalue())
    print(f"wrote synthetic WL zip: vocab={os.path.basename(str(WL_VOCAB_PATH))} "
          f"embedding={tuple(pol.gnn_extractor.embedding.weight.shape)}")


# ------------------------------------------------------------ driver
def run(args):
    os.makedirs(OUT, exist_ok=True)
    p = lambda name: os.path.join(OUT, name)
    print("== fixed observations, generated on Cell 1's simulator (main) and on this branch's")
    sub("main", "gen-obs", "--out", p("obs_main.json"))
    sub("head", "gen-obs", "--out", p("obs_head.json"))

    print("== Cell 1 under main's code and under this branch's code (same obs: main's simulator)")
    for tree in ("main", "head"):
        sub(tree, "probe", "--zip", args.cell1_zip, "--obs", p("obs_main.json"),
            "--out", p(f"cell1_{tree}.npz"), "--n-score-states", str(args.n_score_states))
    a, b = np.load(p("cell1_main.npz")), np.load(p("cell1_head.npz"))
    for kind in ("logits", "probs", "scores"):
        keys = sorted(k for k in a.files if k.startswith(kind + "_"))
        diffs = [float(np.max(np.abs(a[k] - b[k]))) for k in keys]
        exact = all(np.array_equal(a[k], b[k]) for k in keys)
        print(f"  main vs head {kind:6s}: {len(keys)} states, bit-identical={exact}, max|diff|={max(diffs):.3g}")
    ranks_same = all(np.array_equal(np.argsort(-a[k], kind="stable"), np.argsort(-b[k], kind="stable"))
                     for k in a.files if k.startswith("scores_"))
    print(f"  main vs head discriminator ranking of all goals identical: {ranks_same}")

    print("== WL loader")
    wl_zip = args.wl_zip
    if wl_zip is None:
        wl_zip = p("synthetic_wl.zip")
        sub("head", "make-wl-zip", "--out", wl_zip)
    sub("head", "probe", "--zip", wl_zip, "--obs", p("obs_head.json"), "--out", p("wl_head.npz"),
        "--n-score-states", str(args.n_score_states))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="mode", required=True)
    r = sp.add_parser("run")
    r.add_argument("--cell1-zip", required=True)
    r.add_argument("--wl-zip", default=None)
    r.add_argument("--n-score-states", type=int, default=2)
    g = sp.add_parser("gen-obs"); g.add_argument("--out", required=True)
    pr = sp.add_parser("probe")
    pr.add_argument("--zip", required=True); pr.add_argument("--obs", required=True)
    pr.add_argument("--out", required=True); pr.add_argument("--n-score-states", type=int, default=2)
    pr.add_argument("--wl-vocab", default=None)
    w = sp.add_parser("make-wl-zip"); w.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mode == "run":
        run(a)
    elif a.mode == "gen-obs":
        gen_obs(a.out)
    elif a.mode == "probe":
        probe(a.zip, a.obs, a.out, a.n_score_states, a.wl_vocab)
    elif a.mode == "make-wl-zip":
        make_wl_zip(a.out)


if __name__ == "__main__":
    main()
