"""
Per-field SHA-256 of the old domain's observation stream: the same stream as
analysis/old_domain_obs_hash.py (city, mask=False, v1 rewards, env.seed(0), 300 steps of a
fixed RNG-free action rule), but hashed separately for each JSON field of the observation.

Why: on the Mac (gym 0.26, with compat's gym-0.18 seeding) the atom stream equals RCP's
exactly, but oracle_sage/vilg differ as whole strings. Comparing this script's output on both
stacks shows which fields differ (expected: only wl_colours / wl_histogram).

Run from the repo root:
    PYTHONPATH=. python analysis/old_domain_obs_field_hashes.py
"""
import hashlib
import json
import os
import sys

import gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from old_domain_obs_hash import NUM_STEPS, SEED, sample_action  # noqa: E402


def field_hashes(convention, seed=SEED, num_steps=NUM_STEPS):
    from sage.domains.gym_taxi import REWARDS
    from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv

    env = GraphTaxiEnv(representation="graph", scenario="city", mask=False,
                       rewards=REWARDS["v1"], graph_convention=convention)
    env.seed(seed)
    observations = [env.reset()]
    for _ in range(num_steps):
        obs, _reward, done, _info = env.step(sample_action(env.sim))
        observations.append(obs)
        if done:
            env.reset()
    env.close()
    parsed = [json.loads(o) for o in observations]
    out = {}
    for key in sorted(parsed[0]):
        blob = "\x00".join(json.dumps(p.get(key), sort_keys=True) for p in parsed)
        out[key] = hashlib.sha256(blob.encode()).hexdigest()
    return out


if __name__ == "__main__":
    print(f"gym {gym.__version__}")
    for convention in ("oracle_sage", "vilg", "atom"):
        for key, digest in field_hashes(convention).items():
            print(convention, key, digest)
