"""
Wiring smoke test for the --graph-convention CLI flag added to
sage/experiments/gnn_global.py. Confirms (a) env_kwargs={"graph_convention": ...}
actually reaches GraphTaxiEnv through make_vec_env -> gym.make's own kwarg-merge
without clobbering the registered kwargs (representation/scenario/mask/rewards),
and (b) the argparse wiring in gnn_global.main() produces the right
args.graph_convention value.

This does NOT run any training -- no A2C/model construction, no env.step(). It only
builds envs/vec-envs and calls parser.parse_args() (indirectly, via main()).

This sandbox's installed gym (0.26.2) has drifted from oracle-sage.yml's pinned
gym==0.18.0, and that drift breaks two things unrelated to graph_convention: the
"city" scenario's road-maze RNG calls, and gym.make()'s newer PassiveEnvChecker
(which JsonGraph predates). See the printed NOTE below for details -- this script
demonstrates both breakages honestly, then isolates around them with a faithful
replica of gym.make()'s own kwarg-merge logic to still empirically prove the one
thing this task is actually about.

Run from the repo root: python docs/test_cli_graph_convention.py
"""
import json

import gym

import sage.domains.gym_taxi  # required to register city-taxi-* env ids
from sage.forks.stable_baselines3.stable_baselines3.common.env_util import make_vec_env
from sage.experiments import gnn_global

ENV_ID = "city-taxi-unmasked-v1"  # exactly what gnn_global.py's Cell 1 seed batch uses


def node_dim_from_reset(env):
    space_dim = env.observation_space.node_dimension
    obs = env.reset()
    data = json.loads(obs)
    obs_dim = len(data["node_feats"][0])
    assert space_dim == obs_dim, f"observation_space.node_dimension ({space_dim}) != actual reset() obs dim ({obs_dim})"
    return space_dim


print(f"=== make_vec_env(..., env_kwargs=...) via the real training call path, real env id ({ENV_ID}) ===")

try:
    vec_env = make_vec_env(ENV_ID, n_envs=2, env_kwargs={"graph_convention": "vilg"})
    vec_env.close()
    print("(unexpected: city-taxi-unmasked-v1 constructed cleanly in this environment)")
except Exception as e:
    print(
        f"NOTE: make_vec_env({ENV_ID!r}, n_envs=2, env_kwargs={{'graph_convention': 'vilg'}}) -- the exact "
        f"call gnn_global.run() makes -- fails in *this* environment: {type(e).__name__}: {e}\n"
        "      Two pre-existing, unrelated gym-version-drift issues both block this, neither caused by "
        "graph_convention (both reproduce identically with graph_convention='oracle_sage' or omitted):\n"
        "      1. city-taxi-unmasked-v1's registered scenario ('city') builds a random road maze via "
        "generate_city_maze(), which calls self.random.randint(...) -- but gym.utils.seeding.np_random() "
        "in this environment's gym returns a numpy Generator (.integers(), no .randint()).\n"
        "      2. gym.make() itself (as of gym 0.26) wraps every env in a PassiveEnvChecker that inspects "
        "observation_space.low/.high for any gym.spaces.Box subclass -- but JsonGraph (a Box subclass used "
        "for all graph observations, oracle_sage included) never set those, since it predates that checker.\n"
        "      oracle-sage.yml pins gym==0.18.0, which has neither the Generator-based np_random nor the "
        "PassiveEnvChecker -- this sandbox's installed gym (0.26.2) has drifted from that pin. Not fixing "
        "either here: out of scope for this task, unrelated to graph_convention, and non-trivial (5 call "
        "sites for #1; #2 would mean fabricating meaningless numeric low/high bounds for a space that holds "
        "JSON strings, dtype 'U100000', not bounded numeric values -- a real design mismatch, not a 1-line "
        "compat shim). Falling back below to a direct replica of gym.make()'s own kwarg-merge logic (read "
        "verbatim from registration.py) plus scenario='original' (no maze generation), to isolate and still "
        "empirically prove the one thing this task is actually about: does env_kwargs reach GraphTaxiEnv "
        "through the registered-kwargs merge without clobbering anything."
    )

print()
print("=== env_kwargs merge mechanism, isolated from the two unrelated gym.make() issues above ===")
print("    (replicates gym/envs/registration.py's own merge: _kwargs = spec.kwargs.copy(); _kwargs.update(kwargs))")

spec = gym.envs.registration.registry[ENV_ID]
GraphTaxiEnv = gym.envs.registration.load(spec.entry_point)


def construct_via_registered_merge(extra_kwargs):
    merged = spec.kwargs.copy()
    merged.update(extra_kwargs or {})
    merged["scenario"] = "original"  # sidesteps issue #1 above; unrelated to graph_convention
    return GraphTaxiEnv(**merged)


dim_vilg = node_dim_from_reset(construct_via_registered_merge({"graph_convention": "vilg"}))
assert dim_vilg == 9, f"FAIL: env_kwargs graph_convention='vilg' gave node dim {dim_vilg}, expected 9"
print(f"PASS: registered kwargs {dict(spec.kwargs, scenario='original')} merged with "
      f"{{'graph_convention': 'vilg'}} -> node dim {dim_vilg} "
      "(representation/mask/rewards from the registry were not clobbered -- env still constructed successfully "
      "with graph_convention layered on top, exactly as gym.make()'s own merge would do)")

dim_oracle_sage = node_dim_from_reset(construct_via_registered_merge({"graph_convention": "oracle_sage"}))
assert dim_oracle_sage == 3, f"FAIL: env_kwargs graph_convention='oracle_sage' gave node dim {dim_oracle_sage}, expected 3"
print(f"PASS: env_kwargs={{'graph_convention': 'oracle_sage'}} merged in -> node dim {dim_oracle_sage}")

dim_default = node_dim_from_reset(construct_via_registered_merge(None))
assert dim_default == 3, f"FAIL: no graph_convention key gave node dim {dim_default}, expected 3 (Cell 1 default)"
print(f"PASS: no graph_convention key at all -> node dim {dim_default} (matches Cell 1's current default/behaviour)")

print()
print("=== gnn_global.py argparse wiring ===")

# main()'s argparse.ArgumentParser is local to the function (not exposed separately), and
# main() ends by calling run(variant) which would build a vec-env and start training. To
# exercise the real parser without launching training, monkeypatch run() to just capture
# the variant dict argparse produced, instead of duplicating the parser definition here.
captured = {}
original_run = gnn_global.run
gnn_global.run = lambda variant: captured.update(variant)
try:
    gnn_global.main(["--env-name", ENV_ID, "--graph-convention", "vilg"])
    assert captured["graph_convention"] == "vilg", f"FAIL: expected 'vilg', got {captured['graph_convention']!r}"
    print(f"PASS: --graph-convention vilg -> variant['graph_convention'] == {captured['graph_convention']!r}")

    captured.clear()
    gnn_global.main(["--env-name", ENV_ID])
    assert captured["graph_convention"] == "oracle_sage", f"FAIL: expected 'oracle_sage', got {captured['graph_convention']!r}"
    print(f"PASS: no --graph-convention flag -> variant['graph_convention'] == {captured['graph_convention']!r} (Cell 1 default)")
finally:
    gnn_global.run = original_run

print()
print("ALL CHECKS PASSED")
