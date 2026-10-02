"""Step 3: fork-and-rollout validity of fork_env(env), on Cell 3's simulator
(this branch) and Cell 1's (main).

For each seed, drive the env to a mid-episode state E (random planner goals),
then:
  1. Independence: C1 = fork_env(E), C2 = fork_env(E). Step C1 for 300 frames.
     E and C2 must be unchanged (dynamics state, full edge set, RNG state, obs JSON).
  2. Same actions -> same trajectory: step C2 with C1's actions, then continue
     C1 closed-loop to the end of the episode and replay those actions on C2 and
     on E itself. All three must agree at every frame, including passenger
     spawns (counted), RNG state and full edge set.
  3. Common random numbers (diagnostic, not a pass/fail): fork two copies and give
     them different goals. Passenger spawns draw from the shared RNG only while
     fewer than 20 passengers are waiting (try_spawn_passenger), so the streams
     can desynchronise; we report how long the spawn sequences stay identical.

    PYTHONPATH=. python -m analysis.check_step3_fork run [--seeds 0 1 2]
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
TREES = {"head": ROOT, "main": os.path.join(HERE, ".branch_src", "main")}


def sub(tree, *args):
    env = dict(os.environ, PYTHONPATH=f"{TREES[tree]}:{ROOT}", PYTHONWARNINGS="ignore",
               ANALYSIS_EXPECT_SAGE=os.path.join(TREES[tree], "sage"))
    r = subprocess.run([sys.executable, "-m", "analysis.check_step3_fork", *args], env=env,
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


def fingerprint(env, obs):
    """Everything that defines the env's future: dynamics state, edges, RNG, obs."""
    from analysis.envs import sim_summary, graph_edges
    st = env.sim.random.get_state()
    return dict(
        summary=sim_summary(env),
        edges=hashlib.sha1(str(graph_edges(env)).encode()).hexdigest(),
        nodes=tuple(sorted((int(n), tuple(d["attr"])) for n, d in env.sim.graph.nodes(data=True))),
        rng=hashlib.sha1(st[1].tobytes() + str(st[2:]).encode()).hexdigest(),
        obs=hashlib.sha1(obs.encode()).hexdigest(),
        done=bool(env.sim.done), steps=int(env.steps),
    )


def choose_goal(obs, rng):
    from analysis.envs import obs_to_data
    x = obs_to_data(obs).x
    passengers = np.flatnonzero(x[:, 2].numpy() == 1)
    if len(passengers) and rng.uniform() < 0.5:
        return int(passengers[rng.randint(len(passengers))])
    return int(rng.randint(len(x)))


def closed_loop(env, obs, rng, max_frames=None):
    """Random planner goals until done (or max_frames). Returns (actions, obs, done)."""
    from analysis.envs import plan
    actions, done = [], False
    while not done and (max_frames is None or len(actions) < max_frames):
        for a in plan(obs, choose_goal(obs, rng))[1]:
            obs, _, done, _ = env.step(int(a))
            actions.append(int(a))
            if done or (max_frames is not None and len(actions) >= max_frames):
                break
    return actions, obs, done


def spawn_events(env, actions):
    """Step env with actions; return per-frame fingerprints and spawn events (frame, loc, dest)."""
    fps, spawns = [], []
    for t, a in enumerate(actions):
        before = set(env.sim.passengers.values())
        obs, _, done, _ = env.step(int(a))
        # spawn = a (location, destination) not present before; ids are re-keyed on delivery and a
        # picked-up passenger becomes (0, destination), so compare values and skip location 0
        spawns += [(t, int(p.location), int(p.destination)) for p in env.sim.passengers.values()
                   if p not in before and p.location != 0]
        fps.append(fingerprint(env, obs))
        if done:
            break
    return fps, spawns


def mode_check(seed, out):
    assert_tree()
    from analysis.envs import make_env, fork_env
    env, obs = make_env(seed)
    rng = np.random.RandomState(500 + seed)
    _, obs, _ = closed_loop(env, obs, rng, max_frames=700 + 37 * seed)
    E, E_obs = env, obs
    res = {"fork_frame": int(E.steps), "carrying": E.sim.taxi.passenger is not None,
           "waiting": len(E.sim.passengers) - (E.sim.taxi.passenger is not None)}
    fp_E = fingerprint(E, E_obs)

    # 1. independence
    C1, C2 = fork_env(E), fork_env(E)
    res["rng_object_shared_within_copy"] = C1.sim.random is C1.np_random
    res["rng_object_distinct_across_copies"] = C1.sim.random is not E.sim.random and C1.sim.random is not C2.sim.random
    res["copy_equals_original_at_fork"] = fingerprint(C1, E_obs) == fp_E
    a1, c1_obs, _ = closed_loop(C1, E_obs, np.random.RandomState(900 + seed), max_frames=300)
    res["C1_moved"] = fingerprint(C1, c1_obs) != fp_E
    res["E_unchanged_after_stepping_C1"] = fingerprint(E, E_obs) == fp_E
    res["C2_unchanged_after_stepping_C1"] = fingerprint(C2, E_obs) == fp_E

    # 2. same actions -> identical, to the end of the episode
    rest, _, done = closed_loop(C1, c1_obs, np.random.RandomState(901 + seed))
    actions = a1 + rest
    C1b = fork_env(E)  # fresh copy to replay the full sequence and get fingerprints
    fpa, spa = spawn_events(C1b, actions)
    fpb, spb = spawn_events(C2, actions)
    fpe, spe = spawn_events(E, actions)
    first = lambda x, y: next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), None)
    res.update(frames_after_fork=len(actions), reached_episode_end=bool(done and fpa[-1]["steps"] == 2000),  # timeout ends the episode via env.steps; sim.done is only the delivery limit
               copies_first_diff=first(fpa, fpb), copy_vs_original_first_diff=first(fpa, fpe),
               lengths=(len(fpa), len(fpb), len(fpe)), spawns=(len(spa), len(spb), len(spe)),
               spawns_identical=spa == spb == spe, final_delivered=fpa[-1]["summary"]["delivered"])

    # 3. common random numbers across different goals (diagnostic)
    crn = []
    E2_env, E2_obs = make_env(seed)
    _, E2_obs, _ = closed_loop(E2_env, E2_obs, np.random.RandomState(500 + seed), max_frames=700 + 37 * seed)
    for k in range(5):
        A, B = fork_env(E2_env), fork_env(E2_env)  # E2 = same mid-episode state as E, rebuilt (E was stepped in 2.)
        aa, _, _ = closed_loop(fork_env(E2_env), E2_obs, np.random.RandomState(2000 + k), max_frames=300)
        bb, _, _ = closed_loop(fork_env(E2_env), E2_obs, np.random.RandomState(3000 + k), max_frames=300)
        _, sa = spawn_events(A, aa)
        _, sb = spawn_events(B, bb)
        shared = 0
        for u, v in zip(sa, sb):
            if u != v:
                break
            shared += 1
        ld = lambda sp: [(l, d) for _, l, d in sp]
        same_ld = 0
        for u, v in zip(ld(sa), ld(sb)):
            if u != v:
                break
            same_ld += 1
        crn.append(dict(spawns=(len(sa), len(sb)), identical_prefix=shared, same_passengers_prefix=same_ld,
                        first_spawn_mismatch_frame=None if shared == min(len(sa), len(sb)) else min(sa[shared][0], sb[shared][0])))
    res["crn_different_goals_300_frames"] = crn
    json.dump(res, open(out, "w"), default=str)
    ok = all([res["copy_equals_original_at_fork"], res["C1_moved"], res["E_unchanged_after_stepping_C1"],
              res["C2_unchanged_after_stepping_C1"], res["copies_first_diff"] is None,
              res["copy_vs_original_first_diff"] is None, res["spawns_identical"], res["reached_episode_end"]])
    print(f"fork at frame {res['fork_frame']} (carrying={res['carrying']}, waiting={res['waiting']}); "
          f"PASS={ok}")
    print(f"  1. independence: E unchanged={res['E_unchanged_after_stepping_C1']}, C2 unchanged="
          f"{res['C2_unchanged_after_stepping_C1']}, C1 moved={res['C1_moved']}, RNG objects distinct across copies="
          f"{res['rng_object_distinct_across_copies']}, sim.random is env.np_random within a copy={res['rng_object_shared_within_copy']}")
    print(f"  2. same actions for {res['frames_after_fork']} frames to episode end ({res['reached_episode_end']}): "
          f"copy vs copy first diff={res['copies_first_diff']}, copy vs original first diff={res['copy_vs_original_first_diff']}, "
          f"spawns {res['spawns']} identical={res['spawns_identical']}, delivered at end={res['final_delivered']}")
    print("  3. different goals, 300 frames: " + "; ".join(
        f"spawns {c['spawns']}: identical (frame,loc,dest) prefix {c['identical_prefix']}, identical (loc,dest) prefix {c['same_passengers_prefix']}" for c in crn))


def run(args):
    out_dir = os.path.join(HERE, "results", "step3")
    os.makedirs(out_dir, exist_ok=True)
    for tree in ("head", "main"):
        print(f"== {tree} ({'Cell 3 simulator' if tree == 'head' else 'Cell 1 simulator'})")
        for seed in args.seeds:
            print(f"  seed {seed}: " + sub(tree, "check", "--seed", str(seed),
                                          "--out", os.path.join(out_dir, f"{tree}_seed{seed}.json")).replace("\n", "\n    "))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="mode", required=True)
    r = sp.add_parser("run"); r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    c = sp.add_parser("check"); c.add_argument("--seed", type=int, required=True); c.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a) if a.mode == "run" else mode_check(a.seed, a.out)


if __name__ == "__main__":
    main()
