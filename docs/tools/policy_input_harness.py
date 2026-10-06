"""
Byte-identity harness for policy inputs (live observations + planner projections).

Runs with PYTHONPATH pointing at ONE code version (reference worktree or the new branch).
Drives an AsyncVecEnv of city GraphTaxiEnvs (one env per seed, so plan lengths differ
across envs within a decision, as in training) through a decision loop that mirrors
PlanFeedback_A2C.collect_rollouts exactly - including passing obs_mask when the code
version's AsyncVecEnv.step accepts it. At each decision it records, per env, a digest
(dtype, shape, sha256 of bytes) of every tensor the policy reads from the decoded live
observation and from K planner projections, plus the raw JSON's sha256; on every done it
records terminal_observation and whether the episode ended mid-plan.

record mode: goals are drawn from a seeded RNG and saved with the digests.
replay mode: goals are read back from a record file, so both runs execute the same goals.

The harness file itself can live anywhere; only PYTHONPATH selects the code under test.
See docs/cell4_L1.md for the exact commands used.

usage: PYTHONPATH=<worktree> python docs/tools/policy_input_harness.py --convention vilg \
           --vocab P --L 2 --mode record --out REF.json
       PYTHONPATH=<worktree> python docs/tools/policy_input_harness.py ... \
           --mode replay --goals REF.json --out NEW.json
       python docs/tools/policy_input_compare.py REF.json NEW.json
"""
import argparse, hashlib, inspect, json, sys, time
from copy import deepcopy

import numpy as np

# --- Mac-stack shims (gym 0.26 / numpy 2), identical in every run; no-ops on RCP's stack ---
if not hasattr(np, "float"):
    np.float = float
import gym.utils.seeding as _seeding
if not hasattr(np.random.Generator, "randint"):
    class _G:
        def __init__(self, g): self._g = g
        def randint(self, low, high=None): return self._g.integers(low, high)
        def __getattr__(self, n): return getattr(self._g, n)
    _orig = _seeding.np_random
    def _np_random(seed=None):
        g, s = _orig(seed)
        return _G(g), s
    _seeding.np_random = _np_random

import torch as th
import sage.domains.gym_taxi  # noqa: F401
from sage.domains.gym_taxi import REWARDS
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
import sage.domains.gym_taxi.simulator.taxi_world as TW
from sage.domains.gym_taxi.utils import wl_vocab_cache
from sage.agent.async_vec_env import AsyncVecEnv
from sage.forks.stable_baselines3.stable_baselines3.common.monitor import Monitor

FIELDS = ["x", "edge_index", "edge_attr", "mask", "global_features", "wl_colours", "wl_histogram"]


def digest(t):
    if t is None:
        return None
    a = t.detach().cpu().contiguous().numpy()
    return [str(a.dtype), list(a.shape), hashlib.sha256(a.tobytes()).hexdigest()]


def data_digest(d):
    return {f: digest(getattr(d, f, None)) for f in FIELDS}


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--convention", required=True, choices=["vilg", "oracle_sage", "atom"])
    p.add_argument("--vocab", default=None)
    p.add_argument("--L", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 300, 600])
    p.add_argument("--decisions", type=int, default=220)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--goal-seed", type=int, default=12345)
    p.add_argument("--mode", choices=["record", "replay"], required=True)
    p.add_argument("--goals", default=None)
    p.add_argument("--no-obs-mask", action="store_true", help="never pass obs_mask (control run)")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    if a.vocab is not None:
        wl_vocab_cache.configure_wl_vocab_override(a.vocab, a.L)
    else:
        wl_vocab_cache.reset_wl_vocab_override()

    n_built = [0]
    _orig_get = TW.TaxiWorldSimulator._get_state_json
    def _counted(self):
        n_built[0] += 1
        return _orig_get(self)
    TW.TaxiWorldSimulator._get_state_json = _counted

    def make(seed):
        def _f():
            env = GraphTaxiEnv("graph", "city", mask=False, rewards=REWARDS["v1"].copy(),
                               graph_convention=a.convention)
            env.seed(seed)
            env.action_space.seed(seed)
            return Monitor(env, None, info_keywords=("len100", "len200"))
        return _f
    venv = AsyncVecEnv([make(s) for s in a.seeds])  # Monitor-wrapped, as make_vec_env does in training
    obs = venv.reset()
    supports_obs_mask = "obs_mask" in inspect.signature(venv.step).parameters and not a.no_obs_mask
    space = venv.envs[0].observation_space
    converter, planner = space.converter, space.planner

    goals_in = None
    if a.mode == "replay":
        goals_in = json.load(open(a.goals))["goals"]
    rng = np.random.RandomState(a.goal_seed)
    n = len(a.seeds)

    records, goals_out, ends = [], [], []
    stats = {"pickup_plans": 0, "deliveries": 0, "episode_ends": 0, "mid_plan_ends": 0, "sim_steps": 0}
    t0 = time.perf_counter()
    for dec in range(a.decisions):
        batch = converter(obs)
        datas = batch.to_data_list()
        rec = {"live": [], "live_json": [sha(str(o[0])) for o in obs], "proj": [], "plans": []}
        dec_goals = []
        plans = []
        for i, d in enumerate(datas):
            rec["live"].append(data_digest(d))
            x = d.x
            m = d.mask.bool()
            pas = th.nonzero(m & (x[:, 2] == 1)).flatten().tolist()
            loc = th.nonzero(m & (x[:, 0] == 1)).flatten().tolist()
            if goals_in is None:
                cand = []
                for _ in range(a.k):
                    pool = pas if (pas and rng.rand() < 0.5) else loc
                    cand.append(int(pool[rng.randint(len(pool))]))
            else:
                cand = goals_in[dec][i]
            dec_goals.append(cand)
            pr, pl = [], []
            for g in cand:
                proj, acts = planner.plan(deepcopy(d), g)
                pr.append(data_digest(proj))
                pl.append([int(v) for v in acts])
            rec["proj"].append(pr)
            rec["plans"].append(pl)
            plans.append(pl[0])
            if cand[0] in pas and cand[0] in pl[0][:-1]:
                stats["pickup_plans"] += 1
        goals_out.append(dec_goals)

        # --- plan loop: identical to PlanFeedback_A2C.collect_rollouts ---
        plan_lengths = np.array([len(p) for p in plans])
        max_plan_length = max(plan_lengths)
        env_finished = plan_lengths == 0
        grid = -np.ones((n, max_plan_length), dtype=int)
        for i, pp in enumerate(plans):
            grid[i, :plan_lengths[i]] = pp
        plan_step = 0
        while not env_finished.all() and plan_step < max_plan_length:
            acts = grid[:, plan_step]
            env_finished = np.logical_or(env_finished, acts == -1)
            stepped = np.logical_not(env_finished)
            stats["sim_steps"] += int(stepped.sum())
            if supports_obs_mask:
                obs_mask = plan_step == plan_lengths - 1
                new_obs, rew, dones, infos = venv.step(acts, stepped, obs_mask)
            else:
                new_obs, rew, dones, infos = venv.step(acts, stepped)
            for i in range(n):
                if stepped[i]:
                    if rew[i] > 0:
                        stats["deliveries"] += 1
                    if dones[i]:
                        term = infos[i]["terminal_observation"]
                        tb = converter(np.array([[term]], dtype=object)).to_data_list()[0]
                        mid = bool(plan_step < plan_lengths[i] - 1)
                        stats["episode_ends"] += 1
                        stats["mid_plan_ends"] += int(mid)
                        ends.append({"decision": dec, "env": i, "plan_step": plan_step,
                                     "plan_length": int(plan_lengths[i]), "mid_plan": mid,
                                     "terminal_json": sha(term), "terminal": data_digest(tb),
                                     "reset_json": sha(str(new_obs[i][0]))})
            env_finished = np.logical_or(env_finished, dones)
            plan_step += 1
        obs = new_obs
        records.append(rec)
        if dec % 20 == 0:
            print(f"decision {dec} t={time.perf_counter()-t0:.0f}s built={n_built[0]} stats={stats}", flush=True)

    wall = time.perf_counter() - t0
    out = {"args": vars(a), "supports_obs_mask": supports_obs_mask, "goals": goals_out,
           "records": records, "ends": ends, "stats": stats, "json_built": n_built[0], "wall": wall}
    json.dump(out, open(a.out, "w"))
    print("DONE", json.dumps({k: out[k] for k in ["supports_obs_mask", "stats", "json_built", "wall"]}))


if __name__ == "__main__":
    main()
