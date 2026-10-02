"""Illustrated example states for the Cell 1 (GNN) vs Cell 3 (WL) write-up.

One PNG per example state (analysis/results/metrics/examples.json):
  left    the 20x20 city: walls, taxi, waiting passengers; the shortest-plan
          passenger (S, ringed) and greedy's choice (G); each model's modal pick
          with its planned route (Cell 1 solid blue, Cell 3 dashed orange).
  right   pick frequency over the 20 candidate draws for every goal either model
          picked, labelled with goal type and plan length (+ frames over shortest),
          plus each model's proposal / selection rates for the shortest passenger.

Location node id = 20 * x + y + 1 (nx.grid_2d_graph order kept by
generate_city_maze and relabelled 1..400 in generate_road_network); y grows
downward (ACTIONS.north = (0, -1)). Asserted below against the road edges.

    PYTHONPATH=. python -m analysis.figures
"""
import json
import os
from collections import Counter

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "reports", "figures")  # tracked; data inputs stay in results/
N = 20

# reference palette (dataviz skill, references/palette.md): categorical slots 1-2, text and surface tokens
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
COL = {"cell1": "#2a78d6", "cell3": "#eb6834"}
STYLE = {"cell1": "-", "cell3": (0, (2.5, 2))}
NAME = {"cell1": "Cell 1 (GNN)", "cell3": "Cell 3 (WL)"}


def xy(node):
    return divmod(node - 1, N)


def load():
    ex = json.load(open(os.path.join(R, "metrics", "examples.json")))
    gt = json.load(open(os.path.join(R, "scores", "gt.json")))
    obs = json.load(open(os.path.join(R, "scores", "snapshot_head.json")))
    models = {m: json.load(open(os.path.join(R, "scores", f"{m}.json"))) for m in COL}
    return ex, gt, obs, models


def road_edges(o):
    js = json.loads(o)
    ei, ea = np.array(js["edge_index"]), np.array(js["edge_feats"])
    roads = {(int(u), int(v)) for (u, v), a in zip(ei.T, ea) if a[0] == 1}
    for u, v in roads:  # coordinate mapping check
        (x1, y1), (x2, y2) = xy(u), xy(v)
        assert abs(x1 - x2) + abs(y1 - y2) == 1, (u, v)
    return roads


def draw_city(ax, o, data, g, sid, picks_by_model, meta):
    from analysis.goals import parse_state
    from analysis.envs import plan
    roads = road_edges(o)
    ax.set_facecolor(SURFACE)
    for x in range(N + 1):
        ax.plot([x - .5, x - .5], [-.5, N - .5], color=GRID, lw=.5, zorder=0)
        ax.plot([-.5, N - .5], [x - .5, x - .5], color=GRID, lw=.5, zorder=0)
    # walls: grid-adjacent cells without a road between them
    for x in range(N):
        for y in range(N):
            a = 20 * x + y + 1
            if x + 1 < N and (a, a + 20) not in roads:
                ax.plot([x + .5, x + .5], [y - .5, y + .5], color=INK2, lw=1.6, solid_capstyle="round", zorder=1)
            if y + 1 < N and (a, a + 1) not in roads:
                ax.plot([x - .5, x + .5], [y + .5, y + .5], color=INK2, lw=1.6, solid_capstyle="round", zorder=1)
    taxi, loc, _, waiting = parse_state(data)
    L = np.array(g["plan_len"])
    P = [i for i, t in enumerate(g["types"]) if t == "pickup_deliver"]
    Lstar = L[P].min()
    from sage.domains.gym_taxi.simulator.planner import graph_to_networkx
    st = graph_to_networkx(data)
    ploc = {p.node: p.location for p in st.passengers}
    pdest = {p.node: p.destination for p in st.passengers}
    # waiting passengers
    for p in waiting:
        px, py = xy(ploc[p])
        ax.scatter(px, py, s=46, color=MUTED, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    obstacles = [xy(ploc[p]) for p in waiting] + [xy(loc)]
    # routes of each model's modal pick: the two models sit on opposite sides of the cell centre, and the
    # trip leg (pickup -> destination) is shifted again so a route that doubles back stays visible
    for k, (m, picks) in enumerate(picks_by_model.items()):
        merged = Counter("noop" if g["types"][p] == "noop" else p for p in picks)
        goal, cnt = merged.most_common(1)[0]
        if goal == "noop":  # modal pick is the (merged) no-op: nothing to route
            ax.plot([], [], color=COL[m], ls=STYLE[m], lw=2, label=f"{NAME[m]} modal pick: no-op ({cnt}/{len(picks)})")
            continue
        _, acts = plan(o, goal)
        off = (-.2, .2)[k]
        split = acts.index(goal) if goal in acts else len(acts)  # pickup action separates the two legs
        leg1 = [xy(loc)] + [xy(a) for a in acts[:split] if 1 <= a <= N * N]
        leg2 = [leg1[-1]] + [xy(a) for a in acts[split:] if 1 <= a <= N * N]
        off2 = off + (-.12, .12)[k]
        for j, (pts, o_) in enumerate(((leg1, off), (leg2, off2))):
            if len(pts) < 2:
                continue
            ax.plot([p[0] + o_ for p in pts], [p[1] + o_ for p in pts], color=COL[m], ls=STYLE[m], lw=2,
                    solid_capstyle="butt", dash_capstyle="butt", zorder=4,
                    label=f"{NAME[m]} modal pick: {goal} ({cnt}/{len(picks)})" if j == 0 else None)
            obstacles += [(p[0] + o_, p[1] + o_) for p in pts]
        if goal in pdest:
            dx, dy = xy(pdest[goal])
            ax.scatter(dx + off2, dy + off2, marker="s", s=60, facecolor=SURFACE, edgecolor=COL[m], linewidth=2, zorder=5)
    # shortest and greedy: ring/marks, then labels placed away from routes, passengers and each other
    labels = {}
    for s_ in [i for i in P if L[i] == Lstar]:
        sx, sy = xy(ploc[s_])
        ax.scatter(sx, sy, s=230, facecolor="none", edgecolor=INK, linewidth=1.6, zorder=6)
        labels.setdefault(s_, []).append("S")
        dx, dy = xy(pdest[s_])
        ax.scatter(dx, dy, marker="s", s=60, facecolor="none", edgecolor=INK, linewidth=1.2, ls=":", zorder=5)
    gtar = meta.get("greedy_target")
    if gtar is not None:
        labels.setdefault(gtar, []).append("G")
    placed = []
    for node, tags in labels.items():
        px, py = xy(ploc[node])
        text = f"{'·'.join(tags)} {node}"
        best = None
        for cx, cy in ((.5, -.6), (.5, .9), (-2.4, -.6), (-2.4, .9), (.6, .15), (-2.5, .15)):
            lx, ly = px + cx, py + cy
            centre = (lx + .9, ly - .1)
            d = min([np.hypot(centre[0] - a, centre[1] - b) for a, b in obstacles + placed] + [9])
            inside = -.3 <= lx and lx + 1.9 <= N - .3 and -.2 <= ly - .5 and ly <= N - .3
            if inside and (best is None or d > best[0]):
                best = (d, lx, ly, centre)
        _, lx, ly, centre = best
        ax.text(lx, ly, text, fontsize=8, color=INK if "S" in tags else INK2, zorder=9, va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=.9))
        placed.append(centre)
    tx, ty = xy(loc)
    ax.scatter(tx, ty, marker="s", s=110, color=INK, edgecolor=SURFACE, linewidth=1.5, zorder=8)
    ax.annotate("taxi", (tx, ty), xytext=(-8, -14), textcoords="offset points", fontsize=8, color=INK, zorder=8)
    ax.set_xlim(-.5, N - .5)
    ax.set_ylim(N - .5, -.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.01), ncol=1, frameon=False, fontsize=8, labelcolor=INK2)


def draw_picks(ax, g, models, sid):
    L = np.array(g["plan_len"])
    types = g["types"]
    P = [i for i, t in enumerate(types) if t == "pickup_deliver"]
    Lstar = L[P].min()
    S = {i for i in P if L[i] == Lstar}
    # merge the two no-op goals (taxi node and the taxi's current location) into one row, keyed "noop"
    key = lambda gg: "noop" if types[gg] == "noop" else gg
    freq = {m: Counter(key(d["pick"]) for d in models[m][sid]["draws"]) for m in COL}
    K = len(models["cell1"][sid]["draws"])
    plen = lambda gg: 1 if gg == "noop" else L[gg]
    picked = set(freq["cell1"]) | set(freq["cell3"])
    shown = set(sorted(picked, key=lambda gg: -(freq["cell1"][gg] + freq["cell3"][gg]))[:7]) | set(S)
    goals = sorted(shown, key=lambda gg: (plen(gg), str(gg)))     # shortest plan first
    folded = picked - shown
    fold_p = {gg for gg in folded if gg != "noop" and types[gg] == "pickup_deliver"}
    fold_n = folded - fold_p
    rows = goals + (["other_p"] if fold_p else []) + (["other_n"] if fold_n else [])
    y = np.arange(len(rows))
    h = .36
    for k, m in enumerate(COL):
        vals = [100 * freq[m][gg] / K for gg in goals]
        for grp in (fold_p, fold_n):
            if grp:
                vals.append(100 * sum(freq[m][gg] for gg in grp) / K)
        assert abs(sum(vals) - 100) < 1e-6, (m, sum(vals))   # every draw accounted for
        ax.barh(y + (k - .5) * (h + .04), vals, height=h, color=COL[m], label=NAME[m],
                hatch=None if m == "cell1" else "////", edgecolor=SURFACE, linewidth=0)
        for yi, v in zip(y, vals):
            if v > 0:
                ax.text(v + 1.5, yi + (k - .5) * (h + .04), f"{v:.0f}%", va="center", fontsize=7.5, color=INK2)
    lab, lab_col = [], []
    for gg in rows:
        if gg == "other_p":
            lab.append(f"other passengers ({len(fold_p)})"); lab_col.append(INK)
        elif gg == "other_n":
            lab.append(f"other move/no-op goals ({len(fold_n)})"); lab_col.append(MUTED)
        elif gg == "noop":
            lab.append("no-op  1 fr (no delivery)"); lab_col.append(MUTED)
        elif types[gg] != "pickup_deliver":
            lab.append(f"{gg}  {types[gg]}  {L[gg]} fr (no delivery)"); lab_col.append(MUTED)
        else:
            lab.append(f"{gg}  passenger  {L[gg]} fr (+{L[gg] - Lstar})" + ("  [S]" if gg in S else ""))
            lab_col.append(INK)
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=8)
    for t, c in zip(ax.get_yticklabels(), lab_col):
        t.set_color(c)
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.set_xlabel("share of 20 draws picking this goal (%) - rows sorted by plan length, grey = no delivery",
                  fontsize=8, color=INK2)
    ax.tick_params(axis="x", labelsize=7.5, colors=INK2)
    ax.grid(axis="x", color=GRID, lw=.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK2)
    # summary lines
    txt = []
    for m in COL:
        d = models[m][sid]["draws"]
        prop = [bool(S & set(x["cands"])) for x in d]
        pk = [x["pick"] in S for x in d]
        pg = np.mean([a for a, b in zip(pk, prop) if b]) if any(prop) else float("nan")
        txt.append(f"{NAME[m]}: shortest proposed {100 * np.mean(prop):.0f}%, picked when proposed "
                   + ("–" if np.isnan(pg) else f"{100 * pg:.0f}%"))
    ax.set_title("\n".join(txt), fontsize=8, color=INK2, loc="left")


def main():
    from analysis.envs import obs_to_data
    os.makedirs(OUT, exist_ok=True)
    ex, gt, snap, models = load()
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    for e in ex:
        sid = str(e["id"])
        o = snap["obs"][sid]
        g = gt[sid]
        meta = snap["meta"][sid]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 6.2), gridspec_kw={"width_ratios": [1, 1.15]})
        picks = {m: [d["pick"] for d in models[m][sid]["draws"]] for m in COL}
        draw_city(a1, o, obs_to_data(o), g, sid, picks, meta)
        draw_picks(a2, g, models, sid)
        n_pass = sum(t == "pickup_deliver" for t in g["types"])
        fig.suptitle(f"State {e['id']}  ·  source {e['source']}, {e['bucket']} (frame {meta['time']}), "
                     f"{n_pass} waiting passengers", x=.02, ha="left", fontsize=11, color=INK, fontweight="bold")
        fig.text(.02, .925, e["why"], fontsize=9, color=INK2, ha="left")
        fig.text(.02, .015, "S = shortest-plan passenger (ring; dotted square = its destination). G = greedy's choice "
                 "(nearest by road). Routes: each model's most frequent pick, square = its destination.\n"
                 "No-op = taxi node or current location (merged). Cell 1 is scored on its stale-edge view; goals and "
                 "plans are identical on both simulators.",
                 fontsize=7.5, color=MUTED, ha="left")
        fig.tight_layout(rect=(0, .05, 1, .91))
        path = os.path.join(OUT, f"state_{e['id']:03d}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
