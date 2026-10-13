"""Greedy reference check (supervisor question; adds outputs only, changes no earlier result).

  1. greedy_action (analysis/greedy.py) is re-run for seeds 100-109 with per-frame checks:
     one env.step per frame; every action legal and effective (moves change the taxi
     location, pickups pick up, drop-offs deliver); greedy_action never touches the
     simulator RNG (future spawns); the road graph built at reset stays the road graph
     of the episode; and an obs-only reimplementation (reads nothing but the observation
     graph the policies get, via the planner's own graph_to_networkx) reproduces the
     target every frame and the move (or an equally short one).
  2. Frames per delivery split into pickup leg and trip for greedy, the three plan-level
     reference policies (shortest-plan, best of 3 random passengers, random passenger),
     Cell 1 (own simulator) and Cell 3 (fixed simulator), same seeds. The existing
     episode functions are re-run unchanged (eval_models.run_episode,
     audit.reference_episode), with env.step wrapped by a recorder; their delivery
     counts are asserted equal to the published ones.
  3. One greedy episode's per-frame trace (seed 100) saved as CSV.

    PYTHONPATH=. python -m analysis.check_greedy_reference run --cell1-zip ... --cell3-zip ...
Outputs: analysis/results/audit/greedy/
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TREES = {"head": ROOT, "main": os.path.join(HERE, ".branch_src", "main")}
OUT = os.path.join(HERE, "results", "audit", "greedy")
SEEDS = list(range(100, 110))
TREE = {"greedy": "head", "shortest": "head", "best3": "head", "random": "head", "cell1": "main", "cell3": "head"}
PUBLISHED = {"greedy": ("step4", "greedy_head.json"), "shortest": ("audit", "3_shortest_plan.json"),
             "best3": ("audit", "5_best_of_3.json"), "random": ("audit", "5_random_passenger.json"),
             "cell1": ("eval", "cell1_own.json"), "cell3": ("eval", "cell3_fixed.json")}


def sub(tree, *args):
    env = dict(os.environ, PYTHONPATH=f"{TREES[tree]}:{ROOT}", PYTHONWARNINGS="ignore",
               ANALYSIS_EXPECT_SAGE=os.path.join(TREES[tree], "sage"))
    r = subprocess.run([sys.executable, "-m", "analysis.check_greedy_reference", *args], env=env,
                       cwd=TREES[tree], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-4000:])
        raise SystemExit(f"{tree} {args[:3]} failed")
    return r.stdout.rstrip()


def assert_tree():
    import sage
    want = os.environ.get("ANALYSIS_EXPECT_SAGE")
    got = os.path.dirname(os.path.abspath(sage.__file__))
    assert want is None or got == want, f"imported sage from {got}, expected {want}"


def xy(node):
    return (int(node) - 1) // 20, (int(node) - 1) % 20


def road_edges(sim):
    return {(int(u), int(v)) for u, v, d in sim.graph.edges(data=True) if d["attr"][0] == 1}


# ------------------------------------------------------------------ recorder
class Recorder:
    """Wraps env.step of every env made through analysis.envs.make_env: one record per
    env.step call (= one frame), with a passenger identity that survives the
    simulator's relabelling after each delivery (resort_passengers)."""

    def __init__(self):
        self.episodes = []

    def install(self):
        from analysis import envs
        orig = envs.make_env

        def make_env(seed):
            env, obs = orig(seed)
            ep = dict(seed=seed, frames=[], roads0=road_edges(env.sim), sid={}, next_sid=0, env=env)
            self._new_ids(ep)
            self.episodes.append(ep)
            step = env.step

            def traced(a):
                self.before(ep, int(a))
                out = step(a)
                self.after(ep, out)
                return out
            env.step = traced
            return env, obs
        envs.make_env = make_env

    def _new_ids(self, ep):
        for p in ep["env"].sim.passengers:
            if p not in ep["sid"]:
                ep["sid"][p] = ep["next_sid"]; ep["next_sid"] += 1

    def before(self, ep, a):
        sim = ep["env"].sim
        ep["cur"] = dict(loc=int(sim.taxi.location),
                         carrying=None if sim.taxi.passenger is None else ep["sid"][sim.taxi.passenger],
                         n_waiting=sum(1 for p in sim.passengers if p != sim.taxi.passenger),
                         action=a, delivered=100 - sim.delivery_limit)

    def after(self, ep, out):
        sim = ep["env"].sim
        f = ep.pop("cur")
        f["frame"] = len(ep["frames"])
        f["loc_after"] = int(sim.taxi.location)
        delivered = 100 - sim.delivery_limit
        f["dropoff"] = delivered > f["delivered"]
        f["delivered"] = delivered
        if f["dropoff"]:  # resort_passengers: node ids above the removed one shift down by one
            removed = next(p for p, s in ep["sid"].items() if s == f["carrying"])
            ep["sid"] = {(p - 1 if p > removed else p): s for p, s in ep["sid"].items() if p != removed}
        f["carrying_after"] = None if sim.taxi.passenger is None else ep["sid"][sim.taxi.passenger]
        f["pickup"] = f["carrying"] is None and f["carrying_after"] is not None
        self._new_ids(ep)
        ep["frames"].append(f)


def decompose(ep):
    """Per-episode accounting of all frames: completed deliveries split into pickup leg
    (frames from the previous drop-off up to and including the pickup action) and trip
    (frames after pickup up to and including the drop-off action); frames after the last
    drop-off are 'unfinished'. Each part split further using road distance on the
    walled road graph."""
    import networkx as nx
    from analysis.greedy import road_graph
    env = ep["env"]
    assert road_edges(env.sim) == ep["roads0"], "road graph changed during the episode"
    G = road_graph(env.sim)
    fr = ep["frames"]
    legs, start, start_loc, pick = [], 0, None, None
    for f in fr:
        if start_loc is None:
            start_loc = f["loc"]
        if f["pickup"]:
            pick = dict(t=f["frame"], loc=f["loc"])
        if f["dropoff"]:
            assert pick is not None
            seg_a, seg_t = fr[start:pick["t"] + 1], fr[pick["t"] + 1:f["frame"] + 1]
            legs.append(dict(
                pickup_frames=len(seg_a), pickup_dist=nx.shortest_path_length(G, start_loc, pick["loc"]),
                pickup_moves=sum(g["loc_after"] != g["loc"] for g in seg_a),
                trip_frames=len(seg_t), trip_dist=nx.shortest_path_length(G, pick["loc"], f["loc"]),
                trip_moves=sum(g["loc_after"] != g["loc"] for g in seg_t)))
            start, start_loc, pick = f["frame"] + 1, None, None
    return dict(seed=ep["seed"], frames=len(fr), delivered=fr[-1]["delivered"], legs=legs,
                unfinished=len(fr) - start)


# ------------------------------------------------------------------ greedy with checks
def greedy_checked(seed, trace_out=None):
    import networkx as nx
    from analysis import envs
    from analysis.goals import parse_state
    from analysis.greedy import greedy_action, road_graph
    from sage.domains.gym_taxi.simulator.planner import graph_to_networkx
    env, obs = envs.make_env(seed)
    ep = REC.episodes[-1]
    G = road_graph(env.sim)
    roads_sym = all((v, u) in ep["roads0"] for u, v in ep["roads0"])
    c = dict(frames=0, rng_touched=0, illegal_or_ineffective=0, obs_target_match=0, obs_action_exact=0,
             obs_action_equal_length=0, empty_frames_with_waiting=0, ties=0, hold_frames=0,
             target_switches=0, legs=0, picked_first_target=0)
    rows, leg_first_target, prev_target = [], None, None
    done = False
    while not done:
        sim = env.sim
        loc, carrying = int(sim.taxi.location), sim.taxi.passenger
        # greedy's own target (same rule as greedy_action)
        if carrying is not None:
            target, kind, tnode = carrying, "destination", int(sim.passengers[carrying].destination)
        elif sim.passengers:
            dist = {p: nx.shortest_path_length(G, loc, sim.passengers[p].location) for p in sim.passengers}
            target = min(sim.passengers, key=lambda p: dist[p])
            kind, tnode = "passenger", int(sim.passengers[target].location)
            dmin = dist[target]
            c["empty_frames_with_waiting"] += 1
            c["ties"] += int(sum(d == dmin for d in dist.values()) > 1)
        else:
            target, kind, tnode = None, "none", None
            c["hold_frames"] += 1
        rng0 = sim.random.get_state()
        a = int(greedy_action(sim, G))
        r1 = sim.random.get_state()
        c["rng_touched"] += int(not (rng0[0] == r1[0] and np.array_equal(rng0[1], r1[1]) and rng0[2:] == r1[2:]))

        # obs-only reimplementation: everything read from the observation graph
        st = graph_to_networkx(envs.obs_to_data(obs))
        Go = st.graph if hasattr(st, "graph") else st[0]
        Go = Go.subgraph([n for n, d in Go.nodes(data=True) if d["type"] == "location"])
        _, oloc, ocarried, owaiting = parse_state(envs.obs_to_data(obs))
        pas = {p.node: p for p in st.passengers}
        if ocarried is not None:
            otarget, onode = ocarried, int(pas[ocarried].destination)
            oa = 0 if oloc == onode else None
        elif owaiting:
            otarget = min(sorted(owaiting), key=lambda p: nx.shortest_path_length(Go, oloc, pas[p].location))
            onode = int(pas[otarget].location)
            oa = otarget if oloc == onode else None
        else:
            otarget, onode, oa = None, None, oloc
        if oa is None:
            oa = nx.shortest_path(Go, oloc, onode)[1]
        c["obs_target_match"] += int(otarget == target and oloc == loc)
        c["obs_action_exact"] += int(oa == a)
        if oa != a and onode is not None and a in G and oa in G:
            c["obs_action_equal_length"] += int(nx.shortest_path_length(G, a, onode) == nx.shortest_path_length(G, oa, onode))

        # stable identity of the target for the trace and switch counting
        sid = None if target is None else ep["sid"][target]
        if kind == "passenger":
            if leg_first_target is None:
                leg_first_target = sid
            if prev_target is not None and sid != prev_target:
                c["target_switches"] += 1
            prev_target = sid
        a_kind = ("dropoff" if a == 0 else "pickup" if a in sim.passengers else
                  "stay" if a == loc else "move")
        obs, r, done, _ = env.step(a)
        f = ep["frames"][-1]
        ok = {"move": f["loc_after"] != loc and (loc, a) in ep["roads0"], "pickup": f["pickup"],
              "dropoff": f["dropoff"], "stay": True}[a_kind]
        c["illegal_or_ineffective"] += int(not ok)
        if f["pickup"]:
            c["legs"] += 1
            c["picked_first_target"] += int(f["carrying_after"] == leg_first_target)
            leg_first_target, prev_target = None, None
        c["frames"] += 1
        rows.append(dict(frame=f["frame"], taxi_node=loc, taxi_x=xy(loc)[0], taxi_y=xy(loc)[1],
                         carrying="" if f["carrying"] is None else f["carrying"],
                         target_passenger="" if sid is None else sid, target_kind=kind,
                         target_node="" if tnode is None else tnode,
                         target_x="" if tnode is None else xy(tnode)[0], target_y="" if tnode is None else xy(tnode)[1],
                         action_node=a, action=a_kind, n_waiting=f["n_waiting"],
                         event="pickup" if f["pickup"] else "dropoff" if f["dropoff"] else "",
                         delivered=f["delivered"]))
    c["roads_symmetric"] = roads_sym
    if trace_out:
        with open(trace_out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    return c


# ------------------------------------------------------------------ modes
REC = Recorder()


def mode_trace(policy, zip_path, out):
    assert_tree()
    REC.install()
    checks = []
    if policy == "greedy":
        for s in SEEDS:
            checks.append(greedy_checked(s, os.path.join(OUT, "greedy_trace_seed100.csv") if s == 100 else None))
    elif policy in ("shortest", "best3", "random"):
        from analysis.audit import reference_episode
        for s in SEEDS:
            reference_episode(s, policy)
    else:
        from analysis.eval_models import run_episode
        from analysis.models import load_policy
        pol, _, _ = load_policy(zip_path)
        for s in SEEDS:
            run_episode(pol, s, torch_seed=s)
    res = dict(episodes=[decompose(ep) for ep in REC.episodes], checks=checks)
    json.dump(res, open(out, "w"))
    print(f"{policy}: {len(res['episodes'])} episodes, deliveries {[e['delivered'] for e in res['episodes']]}")


def summarise(results):
    L = ["# Greedy reference check (env seeds 100-109; fixed simulator, Cell 1 on its own simulator)", ""]
    g = results["greedy"]["checks"]
    tot = {k: sum(c[k] for c in g) for k in g[0] if k != "roads_symmetric"}
    L += ["## Greedy: per-frame checks (10 episodes)",
          f"frames {tot['frames']} (one env.step each); road edges symmetric in every episode: "
          f"{all(c['roads_symmetric'] for c in g)}; road graph unchanged within each episode: True (asserted)",
          f"illegal or ineffective actions: {tot['illegal_or_ineffective']}",
          f"frames where greedy_action changed the simulator RNG state: {tot['rng_touched']}",
          f"obs-only reimplementation: same target {tot['obs_target_match']}/{tot['frames']}, "
          f"same action {tot['obs_action_exact']}/{tot['frames']}, "
          f"different but equally short move {tot['obs_action_equal_length']}",
          f"empty-taxi frames with passengers waiting {tot['empty_frames_with_waiting']}; nearest-distance ties among "
          f"them {tot['ties']} ({100 * tot['ties'] / tot['empty_frames_with_waiting']:.1f}%); frames with no passenger "
          f"at all (hold) {tot['hold_frames']}",
          f"pickup legs {tot['legs']}; target switched mid-leg {tot['target_switches']} times; legs whose picked-up "
          f"passenger was the leg's first target {tot['picked_first_target']} ({100 * tot['picked_first_target'] / tot['legs']:.1f}%)",
          ""]
    L += ["## Frames per delivery, pooled over 10 episodes (sum of frames / sum of deliveries)",
          "pickup leg = previous drop-off -> pickup action (incl.); trip = after pickup -> drop-off action (incl.)",
          "road = shortest road distance on the walled road graph; extra = moving frames beyond it (detours, "
          "move goals); idle = non-moving frames other than the pickup/drop-off action (no-ops, holds)",
          "",
          f"{'policy':16s} {'deliv':>5s} {'fr/del':>6s} | {'pickup leg':>10s} {'road':>5s} {'extra':>5s} {'idle':>5s} "
          f"{'+1':>3s} | {'trip':>5s} {'road':>5s} {'extra':>5s} {'idle':>5s} {'+1':>3s} | {'unfin.':>6s}"]
    names = {"greedy": "greedy", "shortest": "shortest-plan", "best3": "best-of-3 random", "random": "random passenger",
             "cell1": "Cell 1 (GNN)", "cell3": "Cell 3 (WL)"}
    per_seed = []
    for p in TREE:
        eps = results[p]["episodes"]
        D = sum(e["delivered"] for e in eps)
        F = sum(e["frames"] for e in eps)
        legs = [lg for e in eps for lg in e["legs"]]
        s = {k: sum(lg[k] for lg in legs) / D for k in legs[0]}
        un = sum(e["unfinished"] for e in eps) / D
        L.append(f"{names[p]:16s} {D / 10:5.1f} {F / D:6.1f} | {s['pickup_frames']:10.1f} {s['pickup_dist']:5.1f} "
                 f"{s['pickup_moves'] - s['pickup_dist']:5.1f} {s['pickup_frames'] - s['pickup_moves'] - 1:5.1f} "
                 f"{1:3d} | {s['trip_frames']:5.1f} {s['trip_dist']:5.1f} {s['trip_moves'] - s['trip_dist']:5.1f} "
                 f"{s['trip_frames'] - s['trip_moves'] - 1:5.1f} {1:3d} | {un:6.1f}")
        for e in eps:
            n = len(e["legs"])
            per_seed.append(f"{names[p]:16s} {e['seed']:4d} {e['delivered']:5d} {e['frames'] / e['delivered']:6.1f} "
                            f"{sum(lg['pickup_frames'] for lg in e['legs']) / n:6.1f} "
                            f"{sum(lg['trip_frames'] for lg in e['legs']) / n:6.1f} {e['unfinished']:5d}")
    L += ["", "## Per seed: policy, seed, deliveries, frames/delivery, mean pickup leg, mean trip, unfinished frames",
          *per_seed]
    return "\n".join(L)


def run(args):
    os.makedirs(OUT, exist_ok=True)
    zips = {"cell1": args.cell1_zip, "cell3": args.cell3_zip}
    jobs = [(p, TREE[p], ["trace", "--policy", p, "--zip", zips.get(p, "-"), "--out", os.path.join(OUT, f"{p}.json")])
            for p in TREE if args.only is None or p in args.only.split(",")]
    with ThreadPoolExecutor(4) as ex:
        for r in ex.map(lambda j: sub(j[1], *j[2]), jobs):
            print(r)
    results = {p: json.load(open(os.path.join(OUT, f"{p}.json"))) for p in TREE}
    for p, (d, f) in PUBLISHED.items():  # same episodes as published
        pub = [r["delivered"] for r in json.load(open(os.path.join(HERE, "results", d, f)))]
        got = [e["delivered"] for e in results[p]["episodes"]]
        assert got == pub, (p, got, pub)
    txt = summarise(results) + "\n\nDelivery counts equal the published ones for all policies: True\n"
    open(os.path.join(OUT, "greedy_check.txt"), "w").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="mode", required=True)
    r = sp.add_parser("run"); r.add_argument("--cell1-zip", required=True); r.add_argument("--cell3-zip", required=True)
    r.add_argument("--only", default=None, help="comma-separated policies to (re)trace; the rest are read from OUT")
    t = sp.add_parser("trace"); t.add_argument("--policy", required=True, choices=sorted(TREE))
    t.add_argument("--zip", default="-"); t.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a) if a.mode == "run" else mode_trace(a.policy, a.zip, a.out)


if __name__ == "__main__":
    main()
