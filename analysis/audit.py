"""Audit of the goal-selection metrics (pre-meeting). Reads the existing scores; writes only to
analysis/results/audit/.

  attribution  pooled draw counts and rates, so that
               P(proposed) x P(picked | proposed) = P(picked) exactly; compared with the
               published (per-state-mean) rates
  bootstrap    95% CIs (1,000 resamples of states) for the headline metrics, both models,
               plus the paired Cell 1 - Cell 3 difference
  shortest     ground-truth check: a 'shortest-plan' policy (passenger with the shortest
               total plan, else no-op) for 10 episodes, env seeds 100-109
  seeds        headline metrics per training seed (e.g. Cell 1 seed 0 vs seed 300)

Shortest definition (unchanged from metrics.py): S = every passenger whose total plan
length equals the minimum over passengers, so tied shortest passengers all count. A draw
'proposes the shortest' if any member of S is among its 3 candidates, and 'picks the
shortest' if its pick is in S. A pick is always one of the candidates, so picked implies
proposed.

    PYTHONPATH=. python -m analysis.audit attribution|bootstrap|seeds
    PYTHONPATH=. python -m analysis.audit shortest       (runs on this branch's simulator)
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.join(HERE, "results", "scores")
OUT = os.path.join(HERE, "results", "audit")
NAMES = {"cell1": "Cell 1 (GNN) s0", "cell3": "Cell 3 (WL) s0", "cell1fixed": "Cell 1-fixed s0",
         "cell1_s300": "Cell 1 (GNN) s300", "cell3_s300": "Cell 3 (WL) s300"}
MAIN = ("cell1", "cell1fixed", "cell3")
PAIRS = (("cell1", "cell3"), ("cell1fixed", "cell1"), ("cell1fixed", "cell3"))


def model_path(m):
    for d in (SC, OUT):
        p = os.path.join(d, f"{m}.json")
        if os.path.exists(p):
            return p
    return None


def load(models):
    states = json.load(open(os.path.join(HERE, "results", "states", "state_set.json")))["states"]
    gt = json.load(open(os.path.join(SC, "gt.json")))
    return states, gt, {m: json.load(open(model_path(m))) for m in models}


def state_stats(g, mod):
    """Per-state counts and per-state statistics for one model."""
    types = g["types"]
    L = np.array(g["plan_len"])
    P = [i for i, t in enumerate(types) if t == "pickup_deliver"]
    Q = [i for i, t in enumerate(types) if t in ("move", "noop")]
    Lstar = L[P].min()
    S = {i for i in P if L[i] == Lstar}
    draws = mod["draws"]
    picks = [d["pick"] for d in draws]
    prop = [bool(S & set(d["cands"])) for d in draws]
    pk = [p in S for p in picks]
    assert all(p <= q for p, q in zip(pk, prop))          # picked implies proposed
    pass_picks = [p for p in picks if types[p] == "pickup_deliver"]
    sc = np.array(mod["scores"])
    shorter = [i for i in Q if L[i] < Lstar]
    return dict(
        n_draws=len(draws), n_prop=int(sum(prop)), n_pick=int(sum(pk)),
        n_pass_picks=len(pass_picks), sum_delta=float(sum(L[p] - Lstar for p in pass_picks)),
        n_zero=int(sum(L[p] == Lstar for p in pass_picks)), n_nonpass=len(picks) - len(pass_picks),
        n_short=len(S), n_pass=len(P),
        rho_pass=spearmanr(sc[P], -L[P])[0] if len(P) >= 4 else np.nan,
        rho_all=spearmanr(sc, -L)[0],
        disc_best_delta=float(L[max(P, key=lambda i: sc[i])] - Lstar),
        beats=float(np.mean([sc[i] > sc[P].max() for i in shorter])) if shorter else np.nan,
        mass_short=float(np.array(mod["probs"])[list(S)].sum()),
        # published (per-state-mean) versions
        st_prop=float(np.mean(prop)), st_pick=float(np.mean(pk)),
        st_given=float(np.mean([a for a, b in zip(pk, prop) if b])) if any(prop) else np.nan,
    )


def table(states, gt, mod):
    rows = [state_stats(gt[str(s["id"])], mod[str(s["id"])]) for s in states]
    return {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}


HEADLINE = [  # (name, function of the per-state arrays T and index array idx)
    ("M1 mean: selected trip, frames over shortest", lambda T, i: T["sum_delta"][i].sum() / T["n_pass_picks"][i].sum()),
    ("shortest picked, % of all draws", lambda T, i: 100 * T["n_pick"][i].sum() / T["n_draws"][i].sum()),
    ("shortest proposed, % of draws (actor)", lambda T, i: 100 * T["n_prop"][i].sum() / T["n_draws"][i].sum()),
    ("picked if proposed, % (discriminator)", lambda T, i: 100 * T["n_pick"][i].sum() / T["n_prop"][i].sum()),
    ("move/no-op picks, % of draws", lambda T, i: 100 * T["n_nonpass"][i].sum() / T["n_draws"][i].sum()),
    ("disc. Spearman vs plan length, passengers only", lambda T, i: np.nanmean(T["rho_pass"][i])),
    ("disc. Spearman vs plan length, all goals", lambda T, i: np.nanmean(T["rho_all"][i])),
    ("disc. top passenger, frames over shortest", lambda T, i: T["disc_best_delta"][i].mean()),
    ("shorter move/no-op beats every passenger, %", lambda T, i: 100 * np.nanmean(T["beats"][i])),
    ("actor prob on shortest passenger, %", lambda T, i: 100 * T["mass_short"][i].mean()),
]


# ------------------------------------------------------------------ 1. attribution
def attribution(models=MAIN):
    models = [m for m in models if model_path(m)]
    states, gt, mods = load(models)
    lines = ["# 1. Attribution rates, pooled over draws",
             "Shortest = all passengers tied at the minimum total plan length (tied passengers all count).",
             "Published = metrics.py: proposed and picked = mean over states of per-state rates (identical to pooled,",
             "since every state has 20 draws); picked-if-proposed = mean over states of per-state ratios, only over",
             "states where the shortest was proposed at least once (NOT pooled - the source of the mismatch);",
             "'selected is a shortest passenger' = share of PASSENGER picks with zero excess (excludes move/no-op picks).",
             ""]
    for m in models:
        T = table(states, gt, mods[m])
        D, Np, Nk = T["n_draws"].sum(), T["n_prop"].sum(), T["n_pick"].sum()
        tied = int((T["n_short"] > 1).sum())
        lines.append(f"## {NAMES[m]}")
        lines.append(f"draws {int(D)}, shortest proposed {int(Np)}, shortest picked {int(Nk)}; states with tied shortest "
                     f"passengers: {tied}/{len(states)}; passenger picks {int(T['n_pass_picks'].sum())}, "
                     f"move/no-op picks {int(T['n_nonpass'].sum())}")
        lines.append(f"{'group':8s} {'draws':>6s} {'prop':>6s} {'pick':>6s} | {'proposed%':>9s} {'pick|prop%':>10s} "
                     f"{'product':>7s} {'picked%':>7s} | published: {'prop':>5s} {'pick|prop':>9s} {'picked':>6s} "
                     f"| shortest among passenger picks%")
        groups = [("all", np.arange(len(states)))] + [
            (b, np.array([k for k, s in enumerate(states) if s["bucket"] == b])) for b in ("early", "middle", "late")]
        for name, i in groups:
            d, p, k = T["n_draws"][i].sum(), T["n_prop"][i].sum(), T["n_pick"][i].sum()
            prop, given, picked = 100 * p / d, 100 * k / p, 100 * k / d
            assert abs(prop * given / 100 - picked) < 1e-9
            pub_given = 100 * np.nanmean(T["st_given"][i])
            lines.append(f"{name:8s} {int(d):6d} {int(p):6d} {int(k):6d} | {prop:9.1f} {given:10.1f} {prop * given / 100:7.1f} "
                         f"{picked:7.1f} | published: {100 * T['st_prop'][i].mean():5.1f} {pub_given:9.1f} "
                         f"{100 * T['st_pick'][i].mean():6.1f} | {100 * T['n_zero'][i].sum() / T['n_pass_picks'][i].sum():.1f}")
        lines.append("")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "1_attribution.txt"), "w").write(txt)
    print(txt)


# ------------------------------------------------------------------ 2. bootstrap
def bootstrap(models=MAIN, B=1000, seed=0):
    models = [m for m in models if model_path(m)]
    pairs = [p for p in PAIRS if p[0] in models and p[1] in models]
    states, gt, mods = load(models)
    Ts = {m: table(states, gt, mods[m]) for m in models}
    n = len(states)
    rng = np.random.RandomState(seed)
    idx = [rng.randint(n, size=n) for _ in range(B)]
    full = np.arange(n)
    lines = [f"# 2. Bootstrap 95% CIs ({B} resamples of the {n} states, percentile method; pooled ratios recomputed "
             "per resample). Differences are paired: both models on the same resampled states.", ""]
    out = {}
    for name, f in HEADLINE:
        lines.append(name)
        vals = {}
        for m in models:
            est = f(Ts[m], full)
            bs = np.array([f(Ts[m], i) for i in idx])
            vals[m] = bs
            lo, hi = np.percentile(bs, [2.5, 97.5])
            lines.append(f"    {NAMES[m]:28s} {est:8.2f} [{lo:7.2f}, {hi:7.2f}]")
            out.setdefault(name, {})[m] = (est, lo, hi)
        for a, b in pairs:
            dd = vals[a] - vals[b]
            lo, hi = np.percentile(dd, [2.5, 97.5])
            est_d = f(Ts[a], full) - f(Ts[b], full)
            lines.append(f"    diff {NAMES[a]} - {NAMES[b]:16s} {est_d:8.2f} [{lo:7.2f}, {hi:7.2f}]")
            out[name][f"{a}-{b}"] = (est_d, lo, hi)
    # episode-level deliveries (10 episodes each), resampling episodes
    lines += ["", "Deliveries per episode (bootstrap over the 10 episodes, seeds 100-109):"]
    ep = {"Cell 1 own simulator": os.path.join(HERE, "results", "eval", "cell1_own.json"),
          "Cell 3 fixed simulator": os.path.join(HERE, "results", "eval", "cell3_fixed.json"),
          "Cell 1 fixed simulator": os.path.join(HERE, "results", "eval", "cell1_fixed.json"),
          "Cell 1-fixed fixed simulator": os.path.join(HERE, "results", "eval", "cell1fixed_fixed.json"),
          "greedy": os.path.join(HERE, "results", "step4", "greedy_head.json"),
          "shortest-plan": os.path.join(OUT, "3_shortest_plan.json"),
          "best of 3 random passengers": os.path.join(OUT, "5_best_of_3.json"),
          "random passenger": os.path.join(OUT, "5_random_passenger.json")}
    for name, p in ep.items():
        if not os.path.exists(p):
            continue
        d = np.array([r["delivered"] for r in json.load(open(p))], float)
        bs = np.array([d[rng.randint(len(d), size=len(d))].mean() for _ in range(B)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        lines.append(f"  {name:30s} {d.mean():6.1f} [{lo:5.1f}, {hi:5.1f}]  (sd {d.std(ddof=1):.1f}, range {int(d.min())}-{int(d.max())})")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "2_bootstrap.txt"), "w").write(txt)
    json.dump(out, open(os.path.join(OUT, "2_bootstrap.json"), "w"), indent=1)
    print(txt)


# ------------------------------------------------------------------ 3. shortest-plan policy
def reference_episode(seed, rule="shortest"):
    """Plan-level reference policy, plans run open-loop like the models. rule:
       shortest  - passenger with the shortest total plan (ties -> lowest node id)
       best3     - 3 distinct waiting passengers sampled uniformly (all if fewer), shortest of them
       random    - one waiting passenger uniformly
    No waiting passenger -> 1-frame no-op. Sampling RNG: RandomState(seed), separate from the env's."""
    from analysis.envs import make_env, obs_to_data, plan
    from analysis.goals import classify_goals
    env, obs = make_env(seed)
    rng = np.random.RandomState(seed)
    done, decisions, frames, noops = False, 0, 0, 0
    while not done:
        data = obs_to_data(obs)
        types = classify_goals(data)
        P = [i for i, t in enumerate(types) if t == "pickup_deliver"]
        if P:
            if rule == "shortest":
                C = P
            elif rule == "best3":
                C = sorted(rng.choice(P, size=min(3, len(P)), replace=False).tolist())
            else:
                C = [int(rng.choice(P))]
            plans = {i: plan(obs, i)[1] for i in C}
            goal = min(C, key=lambda i: (len(plans[i]), i))   # ties -> lowest node id
            acts = plans[goal]
        else:
            goal = env.sim.taxi.location                      # no-op: 1 frame
            acts = plan(obs, goal)[1]
            noops += 1
        decisions += 1
        for a in acts:
            obs, _, done, _ = env.step(int(a))
            frames += 1
            if done:
                break
    return dict(seed=seed, delivered=100 - env.sim.delivery_limit, decisions=decisions,
                frames_per_decision=frames / decisions, noops=noops)


def shortest():
    import sage
    print(f"sage from {os.path.dirname(sage.__file__)}")
    res = [reference_episode(100 + k, "shortest") for k in range(10)]
    json.dump(res, open(os.path.join(OUT, "3_shortest_plan.json"), "w"))
    greedy = json.load(open(os.path.join(HERE, "results", "step4", "greedy_head.json")))
    gd = {r["seed"]: r["delivered"] for r in greedy}
    lines = ["# 3. Shortest-plan policy (passenger with the shortest total plan, ties -> lowest id; no passenger -> "
             "1-frame no-op), plans executed open-loop like the models; fixed simulator, env seeds 100-109", "",
             "seed  shortest-plan  greedy  diff  decisions  frames/decision"]
    for r in res:
        lines.append(f"{r['seed']:4d}  {r['delivered']:13d}  {gd[r['seed']]:6d}  {r['delivered'] - gd[r['seed']]:+4d}  "
                     f"{r['decisions']:9d}  {r['frames_per_decision']:15.1f}")
    d = np.array([r["delivered"] for r in res]); g = np.array([gd[r["seed"]] for r in res])
    lines.append(f"mean  {d.mean():13.1f}  {g.mean():6.1f}  {np.mean(d - g):+4.1f}   (sd {d.std(ddof=1):.1f} vs "
                 f"{g.std(ddof=1):.1f}; shortest-plan > greedy in {int((d > g).sum())}/10, < in {int((d < g).sum())}/10)")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "3_shortest_plan.txt"), "w").write(txt)
    print(txt)


# ------------------------------------------------------------------ 4. seeds
def seeds(models):
    models = [m for m in models if model_path(m)]
    states, gt, mods = load(models)
    full = np.arange(len(states))
    lines = ["# 4. Headline metrics per training seed (same 495 states, 20 draws each; pooled rates)", "",
             f"{'metric':48s} " + " ".join(f"{NAMES[m]:>18s}" for m in models)]
    Ts = {m: table(states, gt, mods[m]) for m in models}
    for name, f in HEADLINE:
        lines.append(f"{name:48s} " + " ".join(f"{f(Ts[m], full):18.2f}" for m in models))
    txt = "\n".join(lines)
    open(os.path.join(OUT, "4_seeds.txt"), "w").write(txt)
    print(txt)


def references():
    """5. best-of-3-random-passengers and random-passenger policies, next to the other policies."""
    import sage
    print(f"sage from {os.path.dirname(sage.__file__)}")
    for rule, fname in (("best3", "5_best_of_3.json"), ("random", "5_random_passenger.json")):
        res = [reference_episode(100 + k, rule) for k in range(10)]
        json.dump(res, open(os.path.join(OUT, fname), "w"))
        print(rule, [r["delivered"] for r in res])


def clamp(model, zip_path, tree_snapshot):
    """Actor clamp statistics for one model on the 495 states (its own simulator view)."""
    from analysis.models import load_policy, actor_logits_probs
    gt = json.load(open(os.path.join(SC, "gt.json")))
    obs = json.load(open(os.path.join(SC, f"snapshot_{tree_snapshot}.json")))["obs"]
    pol, _, _ = load_policy(zip_path)
    above, short, uni, const, nonp = [], [], [], 0, []
    for sid, g in gt.items():
        P = [i for i, t in enumerate(g["types"]) if t == "pickup_deliver"]
        L = np.array(g["plan_len"])
        S = [i for i in P if L[i] == L[P].min()]
        lg, pr = actor_logits_probs(pol, obs[sid])
        above.append(np.mean(lg[P] >= 30)); short.append(pr[S].sum()); uni.append(len(S) / len(P))
        const += np.ptp(pr[P]) < 1e-12
        nonp.append(np.delete(lg, P).max())
    txt = (f"{NAMES.get(model, model)}: passenger logits >= 30: {100 * np.mean(above):.1f}% | identical prob across all "
           f"passengers in {const}/{len(gt)} states | prob on shortest passenger {100 * np.mean(short):.1f}% (uniform "
           f"share {100 * np.mean(uni):.1f}%) | max non-passenger logit median {np.median(nonp):.1f}")
    open(os.path.join(OUT, f"clamp_{model}.txt"), "w").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["attribution", "bootstrap", "shortest", "seeds", "references", "clamp"])
    ap.add_argument("--model"); ap.add_argument("--zip"); ap.add_argument("--snapshot", default="head")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.mode == "attribution":
        attribution()
    elif a.mode == "bootstrap":
        bootstrap()
    elif a.mode == "shortest":
        shortest()
    elif a.mode == "references":
        references()
    elif a.mode == "clamp":
        clamp(a.model, a.zip, a.snapshot)
    else:
        seeds(["cell1", "cell1_s300", "cell1fixed", "cell3", "cell3_s300"])


if __name__ == "__main__":
    main()
