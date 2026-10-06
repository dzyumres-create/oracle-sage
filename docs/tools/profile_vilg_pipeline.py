"""
Pipeline profiler: runs gnn_global.main for a short vilg training run with timing
wrappers monkeypatched onto every stage of the observation pipeline (env-side WL, vilg
graph construction, json.dumps, decoding, planner and planner-side WL, policy), plus
counters for simulator steps vs observations built. Writes <outdir>/timers.json.
Includes the two Mac-only stack shims (gym>=0.22 Generator.randint, env checker);
they are no-ops on RCP's gym 0.18. See docs/cell4_L1.md for the command used.

usage: PYTHONPATH=. python docs/tools/profile_vilg_pipeline.py <outdir> [--cprofile] -- <gnn_global args...>
"""
import sys, os, time, json, collections, cProfile, pstats, io

outdir = sys.argv[1]
use_cprofile = "--cprofile" in sys.argv[2:sys.argv.index("--")]
gnn_args = sys.argv[sys.argv.index("--") + 1:]
os.makedirs(outdir, exist_ok=True)

T = collections.defaultdict(float)   # seconds per stage (inclusive)
N = collections.Counter()            # call / item counts

def timed(key, fn, count_rows=None):
    def wrapper(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            T[key] += time.perf_counter() - t0
            N[key] += 1
            if count_rows is not None:
                N[key + ":rows"] += count_rows(a, k)
    wrapper.__wrapped__ = fn
    return wrapper

import sage.domains.utils.build_wl_vocab  # noqa: F401  Mac-only gym0.26 Generator.randint shim (import side effect)
import gym as _gym, functools as _ft
if tuple(int(x) for x in _gym.__version__.split(".")[:2]) >= (0, 22):
    _gym.make = _ft.partial(_gym.make, disable_env_checker=True)  # Mac-only: JsonGraph has no .low
import sage.domains.gym_taxi.utils.representations as R
import sage.domains.gym_taxi.simulator.taxi_world as TW
import sage.domains.gym_taxi.simulator.planner as P
import sage.domains.gym_taxi.envs.taxi_env as TE
import sage.agent.graph_plan_feedback_policy as GPFP
import sage.agent.async_vec_env as AVE
from sage.agent.plan_feedback_a2c import PlanFeedback_A2C
from sage.forks.stable_baselines3.stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

# --- env side -------------------------------------------------------------------------
R.wl_colours = timed("env.wl", R.wl_colours)
R.env_to_vilg_graph = timed("env.vilg_graph(incl wl)", R.env_to_vilg_graph)
R.graph_to_json = timed("env.json_dumps", R.graph_to_json)
TW.env_to_vilg_json = timed("env.obs_total", TW.env_to_vilg_json)
TW.TaxiWorldSimulator.act = timed("sim.act(incl obs)", TW.TaxiWorldSimulator.act)
_orig_reset = TE.BaseTaxiEnv.reset
TE.BaseTaxiEnv.reset = timed("env.reset", _orig_reset)
AVE.AsyncVecEnv.step_wait = timed("vec.step_wait", AVE.AsyncVecEnv.step_wait)
DummyVecEnv._obs_from_buf = timed("vec.obs_from_buf_copy", DummyVecEnv._obs_from_buf)
DummyVecEnv._save_obs = timed("vec.save_obs", DummyVecEnv._save_obs)

# --- decoder / planner / policy side ----------------------------------------------------
TE.GRAPH_CONVENTION_CONVERTERS["vilg"] = timed(
    "dec.json_to_graph", TE.GRAPH_CONVENTION_CONVERTERS["vilg"], count_rows=lambda a, k: len(a[0]))
import json as _json
import sage.domains.utils.representations as SR
class _TimedJson:
    def __getattr__(self, name):
        return getattr(_json, name)
    loads = staticmethod(timed("dec.json_loads", _json.loads))
SR.json = _TimedJson()
P.wl_colours = timed("plan.wl", P.wl_colours)
P.Planner.plan = timed("plan.total(incl wl)", P.Planner.plan)
GPFP.project_actions = timed("plan.project_actions(incl deepcopy,Batch)", GPFP.project_actions)
GPFP.GNNPlanFeedbackPolicy.forward = timed("pol.forward", GPFP.GNNPlanFeedbackPolicy.forward)
GPFP.GNNPlanFeedbackPolicy.evaluate_actions = timed("pol.evaluate_actions", GPFP.GNNPlanFeedbackPolicy.evaluate_actions)
PlanFeedback_A2C.train = timed("alg.train", PlanFeedback_A2C.train)
PlanFeedback_A2C.collect_rollouts = timed("alg.collect_rollouts", PlanFeedback_A2C.collect_rollouts)

from sage.experiments import gnn_global

t0 = time.perf_counter()
if use_cprofile:
    prof = cProfile.Profile()
    prof.enable()
gnn_global.main(gnn_args)
if use_cprofile:
    prof.disable()
wall = time.perf_counter() - t0

out = {"wall": wall, "T": dict(T), "N": dict(N), "args": gnn_args}
with open(os.path.join(outdir, "timers.json"), "w") as f:
    json.dump(out, f, indent=1)
if use_cprofile:
    prof.dump_stats(os.path.join(outdir, "profile.pstats"))
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats(45)
    with open(os.path.join(outdir, "profile_tottime.txt"), "w") as f:
        f.write(s.getvalue())
print("WALL", wall)
print(json.dumps(out, indent=1))
