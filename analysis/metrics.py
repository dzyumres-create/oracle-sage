"""Metrics for the Cell 1 (GNN) vs Cell 3 (WL) goal-selection analysis.

Ground truth = plan length (frames) from the training planner; rollout values
are deferred. Shortest = minimum plan length among passenger goals (the only
goals that deliver); a selected move/no-op goal is its own category.

Per model, over all states and K draws of 3 candidates (ties broken uniformly):
  M1  plan length of the selected goal minus the shortest available
  M2  P(shortest-plan passenger among the 3 candidates)          [actor]
      P(picked | proposed)                                       [discriminator]
      P(picked) = product
M3  (Cell 3) WL collisions among passenger goals: histogram-only and full
      discriminator input (histogram + time_left); is the shortest distinguishable.
Plus: exact score ties (both models), distinct histograms among all goals and
among the 3 candidates, model agreement, and example states for figures.

    PYTHONPATH=. python -m analysis.metrics
"""
import json
import os
from collections import Counter, defaultdict
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.join(HERE, "results", "scores")
OUT = os.path.join(HERE, "results", "metrics")
NAMES = {"cell1": "Cell 1 (GNN)", "cell3": "Cell 3 (WL)", "cell1fixed": "Cell 1-fixed (GNN, both fixes)"}
# models with scores on disk; Cell 1-fixed is included automatically once scored
MODELS = tuple(m for m in NAMES if os.path.exists(os.path.join(SC, f"{m}.json")))


def load():
    states = json.load(open(os.path.join(HERE, "results", "states", "state_set.json")))["states"]
    gt = json.load(open(os.path.join(SC, "gt.json")))
    meta = json.load(open(os.path.join(SC, "snapshot_head.json")))["meta"]
    models = {m: json.load(open(os.path.join(SC, f"{m}.json"))) for m in MODELS}
    return states, gt, meta, models


def per_state(s, g, meta, mod):
    """Everything the tables need for one (state, model)."""
    types = g["types"]
    L = np.array(g["plan_len"])
    P = [i for i, t in enumerate(types) if t == "pickup_deliver"]
    Lstar = int(L[P].min())
    S = {i for i in P if L[i] == Lstar}
    nearest = meta.get("greedy_target")
    scores = np.array(mod["scores"])
    probs = np.array(mod["probs"])
    picks = [d["pick"] for d in mod["draws"]]
    deltas = [int(L[p] - Lstar) if types[p] == "pickup_deliver" else None for p in picks]
    proposed = [bool(S & set(d["cands"])) for d in mod["draws"]]
    picked = [p in S for p in picks]
    disc_best = max(P, key=lambda i: scores[i])
    actor_best = max(P, key=lambda i: probs[i])
    n = len(types)
    return dict(
        n_pass=len(P), Lstar=Lstar, n_short=len(S),
        delta=[d for d in deltas if d is not None],
        nonpass=Counter(types[p] for p in picks if types[p] != "pickup_deliver"),
        proposed=np.mean(proposed),
        picked_given=np.mean([pk for pk, pr in zip(picked, proposed) if pr]) if any(proposed) else None,
        picked=np.mean(picked),
        uniform_actor_proposed=1 - comb(n - len(S), 3) / comb(n, 3),
        greedy_delta=int(L[nearest] - Lstar) if nearest is not None else None,
        random_pass_delta=float(np.mean(L[P] - Lstar)),
        disc_best_delta=int(L[disc_best] - Lstar), actor_best_delta=int(L[actor_best] - Lstar),
        actor_mass_pass=float(probs[P].sum()), actor_mass_short=float(probs[list(S)].sum()),
        ties=np.mean([d["n_tied"] > 1 for d in mod["draws"]]),
        modal_pick=Counter(picks).most_common(1)[0][0],
        picks=picks,
    )


def fmt_block(rows):
    d = np.concatenate([r["delta"] for r in rows]) if rows else np.array([])
    npk = sum(sum(r["nonpass"].values()) for r in rows)
    tot = sum(len(r["picks"]) for r in rows)
    pg = [r["picked_given"] for r in rows if r["picked_given"] is not None]
    return dict(
        states=len(rows), M1_mean=d.mean(), M1_median=float(np.median(d)), M1_zero=100 * np.mean(d == 0),
        nonpass_pct=100 * npk / tot,
        greedy_M1=np.mean([r["greedy_delta"] for r in rows]), random_M1=np.mean([r["random_pass_delta"] for r in rows]),
        disc_best_M1=np.mean([r["disc_best_delta"] for r in rows]),
        proposed=100 * np.mean([r["proposed"] for r in rows]),
        uniform=100 * np.mean([r["uniform_actor_proposed"] for r in rows]),
        picked_given=100 * np.mean(pg), picked=100 * np.mean([r["picked"] for r in rows]),
        mass_pass=100 * np.mean([r["actor_mass_pass"] for r in rows]),
        mass_short=100 * np.mean([r["actor_mass_short"] for r in rows]),
        ties=100 * np.mean([r["ties"] for r in rows]),
    )


def print_table(title, groups, lines):
    lines.append(f"\n### {title}")
    hdr = (f"{'group':16s} {'n':>4s} | {'M1 mean':>7s} {'median':>6s} {'=0 %':>5s} {'non-pass%':>9s} | "
           f"{'greedy':>6s} {'rand-pass':>9s} {'disc-best':>9s} | {'proposed%':>9s} {'(unif)':>6s} "
           f"{'picked|prop%':>12s} {'picked%':>7s} | {'mass pass%':>10s} {'mass short%':>11s} {'ties%':>5s}")
    lines.append(hdr)
    for name, rows in groups:
        b = fmt_block(rows)
        lines.append(
            f"{name:16s} {b['states']:4d} | {b['M1_mean']:7.1f} {b['M1_median']:6.0f} {b['M1_zero']:5.1f} {b['nonpass_pct']:9.1f} | "
            f"{b['greedy_M1']:6.1f} {b['random_M1']:9.1f} {b['disc_best_M1']:9.1f} | {b['proposed']:9.1f} {b['uniform']:6.1f} "
            f"{b['picked_given']:12.1f} {b['picked']:7.1f} | {b['mass_pass']:10.1f} {b['mass_short']:11.1f} {b['ties']:5.1f}")


def wl_collisions(states, gt, models):
    """Cell 3's view: collisions among passenger goals, all goals, and candidates."""
    out = defaultdict(list)
    for s in states:
        g = gt[str(s["id"])]
        types, L = g["types"], np.array(g["plan_len"])
        P = [i for i, t in enumerate(types) if t == "pickup_deliver"]
        Lstar = L[P].min()
        S = [i for i in P if L[i] == Lstar]
        rec = dict(id=s["id"], bucket=s["bucket"], source=s["source"], n_pass=len(P))
        for kind in ("hist_key", "full_key"):
            keys = g[kind]
            c = Counter(keys[i] for i in P)
            rec[f"{kind}_pass_colliding"] = sum(1 for i in P if c[keys[i]] > 1)
            rec[f"{kind}_pass_distinct"] = len(c)
            nonshort_keys = {keys[i] for i in P if i not in S}
            rec[f"{kind}_short_distinct"] = all(keys[i] not in nonshort_keys for i in S)
            rec[f"{kind}_all_distinct"] = len(set(keys))
            rec[f"{kind}_cand_distinct"] = float(np.mean([len({keys[c_] for c_ in d["cands"]})
                                                          for d in models["cell3"][str(s["id"])]["draws"]]))
        rec["n_goals"] = len(types)
        # consistency: identical full input must mean identical Cell 3 score
        sc = np.array(models["cell3"][str(s["id"])]["scores"])
        byk = defaultdict(set)
        for i, k in enumerate(g["full_key"]):
            byk[k].add(round(float(sc[i]), 5))
        rec["full_key_score_inconsistent"] = sum(len(v) > 1 for v in byk.values())
        for m in MODELS:
            scm = np.array(models[m][str(s["id"])]["scores"])
            rec[f"{m}_distinct_scores_all"] = len(set(scm.tolist()))
            rec[f"{m}_distinct_scores_pass"] = len(set(scm[P].tolist()))
        for k, v in rec.items():
            out[k].append(v)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    states, gt, meta, models = load()
    rows = {m: [] for m in MODELS}
    for s in states:
        sid = str(s["id"])
        for m in MODELS:
            r = per_state(s, gt[sid], meta[sid], models[m][sid])
            r.update(id=s["id"], bucket=s["bucket"], source=s["source"])
            rows[m].append(r)
    lines = ["# Goal-selection metrics: Cell 1 (GNN, own stale-edge simulator) vs Cell 3 (WL, fixed simulator)",
             f"{len(states)} states x {len(models['cell1'][str(states[0]['id'])]['draws'])} candidate draws per model; "
             "ties broken uniformly at random. Ground truth = plan length (frames).",
             "Columns: M1 = selected plan length - shortest passenger plan (passenger picks only); non-pass% = picks that are",
             "move/no-op; greedy/rand-pass/disc-best = M1 of greedy's choice, of a uniformly random passenger, and of the",
             "discriminator's top-scored passenger over ALL passengers; proposed% = shortest-plan passenger among the 3",
             "candidates (unif = same for a uniform actor over all nodes); picked|prop% = discriminator picked it when proposed",
             "(random = 33%); mass = actor probability on all passengers / on the shortest one(s)."]
    for m in MODELS:
        R = rows[m]
        groups = [("all", R)] + [(f"bucket {b}", [r for r in R if r["bucket"] == b]) for b in ("early", "middle", "late")] \
            + [(f"source {src}", [r for r in R if r["source"] == src]) for src in ("greedy", "cell1", "cell3")]
        print_table(NAMES[m], groups, lines)
        npc = Counter()
        for r in R:
            npc.update(r["nonpass"])
        lines.append(f"non-passenger picks by type: {dict(npc)}")

    # agreement
    agree = [a["modal_pick"] == b["modal_pick"] for a, b in zip(rows["cell1"], rows["cell3"])]
    both_short = [gt[str(a["id"])]["plan_len"][a["modal_pick"]] == a["Lstar"] and gt[str(b["id"])]["plan_len"][b["modal_pick"]] == b["Lstar"]
                  for a, b in zip(rows["cell1"], rows["cell3"])]
    lines.append(f"\n### Agreement\nmodal pick identical in {100 * np.mean(agree):.1f}% of states; "
                 f"both modal picks are a shortest-plan passenger in {100 * np.mean(both_short):.1f}%")
    for b in ("early", "middle", "late"):
        idx = [i for i, s in enumerate(states) if s["bucket"] == b]
        lines.append(f"  {b:6s}: identical {100 * np.mean([agree[i] for i in idx]):.1f}%")

    # WL collisions
    w = wl_collisions(states, gt, models)
    A = lambda k: np.array(w[k], dtype=float)
    lines.append("\n### WL structural indistinguishability (Cell 3's view, fixed simulator)")
    for kind, label in (("hist_key", "histogram only"), ("full_key", "full input (histogram + time_left)")):
        lines.append(f"{label}:")
        lines.append(f"  passenger goals sharing their key with another passenger: mean {A(kind + '_pass_colliding').mean():.1f} "
                     f"of {A('n_pass').mean():.1f} per state ({100 * A(kind + '_pass_colliding').sum() / A('n_pass').sum():.1f}%); "
                     f"distinct keys among passengers mean {A(kind + '_pass_distinct').mean():.1f}")
        lines.append(f"  shortest passenger distinguishable from all non-shortest passengers: "
                     f"{100 * A(kind + '_short_distinct').mean():.1f}% of states")
        lines.append(f"  distinct keys among ALL goals: mean {A(kind + '_all_distinct').mean():.1f} of {A('n_goals').mean():.1f} goals; "
                     f"among the 3 candidates: mean {A(kind + '_cand_distinct').mean():.2f}")
        for b in ("early", "middle", "late"):
            idx = [i for i, x in enumerate(w["bucket"]) if x == b]
            lines.append(f"    {b:6s}: colliding passengers {100 * A(kind + '_pass_colliding')[idx].sum() / A('n_pass')[idx].sum():5.1f}%, "
                         f"shortest distinguishable {100 * A(kind + '_short_distinct')[idx].mean():5.1f}%, "
                         f"distinct keys among all goals {A(kind + '_all_distinct')[idx].mean():5.1f}")
    lines.append(f"consistency: full-input groups with more than one Cell 3 score: {int(A('full_key_score_inconsistent').sum())}")
    lines.append("\n### Exact score ties")
    for m in MODELS:
        lines.append(f"{NAMES[m]}: distinct scores among all goals mean {A(m + '_distinct_scores_all').mean():.1f} of "
                     f"{A('n_goals').mean():.1f}; among passenger goals {A(m + '_distinct_scores_pass').mean():.2f} of "
                     f"{A('n_pass').mean():.2f}; candidate-level ties {100 * np.mean([r['ties'] for r in rows[m]]):.2f}% of draws")

    # what each component knows about plan length, among passenger goals
    from scipy.stats import spearmanr
    snap = json.load(open(os.path.join(SC, "snapshot_head.json")))["obs"]
    lines.append("\n### Actor vs discriminator: rank agreement with plan length among passenger goals "
                 "(states with >= 4 passengers; Spearman with -plan length, so 1 = shortest ranked first)")
    for m in MODELS:
        rs, ra, nd, const = [], [], [], 0
        for s in states:
            g = gt[str(s["id"])]
            P = [i for i, t in enumerate(g["types"]) if t == "pickup_deliver"]
            if len(P) < 4:
                continue
            L = -np.array(g["plan_len"])[P]
            sc = np.array(models[m][str(s["id"])]["scores"])[P]
            pr = np.array(models[m][str(s["id"])]["probs"])[P]
            rs.append(spearmanr(sc, L)[0])
            if np.ptp(pr) < 1e-12:
                const += 1
            else:
                ra.append(spearmanr(pr, L)[0])
            nd.append(len(set(np.round(pr, 12))))
        lines.append(f"{NAMES[m]}: discriminator {np.mean(rs):.3f} (min {np.min(rs):.2f}); actor "
                     + (f"{np.mean(ra):.3f}" if ra else "undefined") + f"; actor probability identical across all "
                     f"passengers in {const}/{len(rs)} states; distinct actor probabilities among passengers mean {np.mean(nd):.1f}")
    lines.append("\n### MAIN FINDING - discriminator over ALL goals (incl. non-delivering move/no-op)")
    for m in MODELS:
        r_all, beats, n_states_shorter = [], [], 0
        for s in states:
            g = gt[str(s["id"])]
            sc = np.array(models[m][str(s["id"])]["scores"])
            L = np.array(g["plan_len"])
            P = [i for i, t in enumerate(g["types"]) if t == "pickup_deliver"]
            Q = [i for i, t in enumerate(g["types"]) if t in ("move", "noop")]
            r_all.append(spearmanr(sc, -L)[0])
            shorter = [i for i in Q if L[i] < L[P].min()]
            if shorter:
                n_states_shorter += 1
                best_p = sc[P].max()
                beats.append(np.mean([sc[i] > best_p for i in shorter]))
        lines.append(f"{NAMES[m]}: Spearman(score, -plan length) over all goals {np.mean(r_all):.3f}; move/no-op goals "
                     f"with a shorter plan than the shortest passenger that outscore EVERY passenger: "
                     f"{100 * np.mean(beats):.1f}% ({n_states_shorter} states have such goals)")
    lines.append("-> Cell 1's discriminator ranks every goal by plan length whether or not it delivers; it only behaves "
                 "because its actor proposes passengers almost exclusively. Cell 3's never prefers a non-delivering goal.")
    ncol = [len({json.loads(snap[str(s['id'])])['wl_colours'][i] for i, t in enumerate(gt[str(s['id'])]['types'])
                 if t == 'pickup_deliver'}) for s in states]
    lines.append(f"WL colours (L=1) among waiting passengers per state: {sorted(set(ncol))} -> Cell 3's actor cannot "
                 "tell passengers apart; its proposals are uniform over passengers by construction")

    # examples
    ex = pick_examples(states, gt, rows, w)
    lines.append("\n### Example states for figures")
    for e in ex:
        lines.append(f"  state {e['id']:3d} ({e['source']}, {e['bucket']}): {e['why']}")
    json.dump(ex, open(os.path.join(OUT, "examples.json"), "w"), indent=1)
    txt = "\n".join(lines)
    open(os.path.join(OUT, "report.txt"), "w").write(txt)
    print(txt)


def pick_examples(states, gt, rows, w):
    by = {m: {r["id"]: r for r in rows[m]} for m in MODELS}
    L = lambda sid, g: gt[str(sid)]["plan_len"][g]
    T = lambda sid, g: gt[str(sid)]["types"][g]
    chosen, out = set(), []

    def add(sid, why):
        if sid not in chosen:
            chosen.add(sid)
            s = states[sid]
            out.append(dict(id=sid, source=s["source"], bucket=s["bucket"], why=why))

    rng = np.random.RandomState(0)
    ids = list(range(len(states)))
    rng.shuffle(ids)
    # 1. models disagree, one picks the shortest
    for sid in ids:
        a, b = by["cell1"][sid], by["cell3"][sid]
        if a["modal_pick"] != b["modal_pick"] and (L(sid, a["modal_pick"]) == a["Lstar"]) != (L(sid, b["modal_pick"]) == b["Lstar"]) \
                and a["n_pass"] >= 8:
            add(sid, f"models disagree: Cell 1 picks {a['modal_pick']} ({T(sid, a['modal_pick'])}, +{L(sid, a['modal_pick']) - a['Lstar']}), "
                     f"Cell 3 picks {b['modal_pick']} ({T(sid, b['modal_pick'])}, +{L(sid, b['modal_pick']) - b['Lstar']})")
            break
    # 2. Cell 3's shortest passenger collides in histogram with another passenger
    for sid in ids:
        if not w["hist_key_short_distinct"][sid]:
            add(sid, f"Cell 3: shortest passenger shares its WL histogram with a longer passenger "
                     f"(full input distinguishable: {w['full_key_short_distinct'][sid]})")
            break
    # 3. shortest proposed but not picked (most often), one per model
    for m in MODELS:
        cand = sorted(ids, key=lambda sid: -(by[m][sid]["proposed"] * (1 - (by[m][sid]["picked_given"] or 1))))
        sid = cand[0]
        add(sid, f"{NAMES[m]}: shortest proposed in {100 * by[m][sid]['proposed']:.0f}% of draws but picked only "
                 f"{100 * (by[m][sid]['picked_given'] or 0):.0f}% of those")
    # 4. both pick the shortest (a clean case), late bucket
    for sid in ids:
        a, b = by["cell1"][sid], by["cell3"][sid]
        if states[sid]["bucket"] == "late" and a["picked"] > 0.8 and b["picked"] > 0.8 and a["n_pass"] >= 15:
            add(sid, "both models pick the shortest passenger in >80% of draws (late, crowded)")
            break
    # 5. a model picks a move/no-op goal often
    for m in MODELS:
        for sid in ids:
            if sum(by[m][sid]["nonpass"].values()) >= 10:
                add(sid, f"{NAMES[m]} picks non-passenger goals in {sum(by[m][sid]['nonpass'].values())}/20 draws "
                         f"({dict(by[m][sid]['nonpass'])})")
                break
    # 6. largest Cell 3 detour in an early state
    sid = max((i for i in ids if states[i]["bucket"] == "early"), key=lambda i: np.mean(by["cell3"][i]["delta"] or [0]))
    add(sid, f"Cell 3's largest mean detour in an early state: +{np.mean(by['cell3'][sid]['delta']):.0f} frames")
    return out


if __name__ == "__main__":
    main()
