"""
Prints the SHA-256 of the OLD (non-ternary) domain's observation stream for one graph
convention: city scenario, mask=False, v1 rewards, env.seed(0), then 300 steps of a fixed,
RNG-free action rule. This is exactly what tests/test_ternary_wiring.py's
TestOldDomainUnchanged hashes, so its output is a reference value for that test's
OLD_DOMAIN_REFERENCE_SHA256[<stack>][<convention>].

It imports only things that already exist at commit a628f16 (before any ternary wiring),
so the same file can be run against both a628f16 and later commits to check the old
domain is byte-identical. Run from the repo root:

    PYTHONPATH=. python analysis/old_domain_obs_hash.py oracle_sage
    PYTHONPATH=. python analysis/old_domain_obs_hash.py vilg
    PYTHONPATH=. python analysis/old_domain_obs_hash.py atom

The hash is printed alone on stdout; the stack (gym version) it belongs to goes to stderr.
Hashes are only comparable within one stack: env.seed() gives a numpy Generator under
gym 0.26 but a RandomState under gym 0.18, so the same seed produces different worlds.
"""
import argparse
import hashlib
import sys

import gym
import gym.utils.seeding as _seeding

CONVENTIONS = ("oracle_sage", "vilg", "atom")
SEED = 0
NUM_STEPS = 300


class _RandintCompatGenerator(object):
    """gym 0.26's seeding returns a numpy Generator, which has no .randint; the old
    simulator calls .randint. Same shim (and same draws) as tests/test_ternary_wiring.py,
    so hashes computed here match the test's."""

    def __init__(self, generator):
        self._generator = generator

    def randint(self, low, high=None):
        if hasattr(self._generator, "randint"):
            return self._generator.randint(low, high) if high is not None else self._generator.randint(low)
        return self._generator.integers(low, high)

    def __getattr__(self, name):
        return getattr(self._generator, name)


def _install_randint_shim():
    """Wraps seeding.np_random's result only when it lacks .randint (gym 0.26). Under
    gym 0.18 it returns a RandomState, which is left untouched."""
    original_np_random = _seeding.np_random

    def np_random_with_randint(seed=None):
        generator, seed = original_np_random(seed)
        if not hasattr(generator, "randint"):
            generator = _RandintCompatGenerator(generator)
        return generator, seed

    _seeding.np_random = np_random_with_randint


def stack_key():
    return "gym" + ".".join(gym.__version__.split(".")[:2])


def sample_action(sim):
    """Deterministic (no RNG draw): the lowest-id legal action. Identical to
    tests/test_ternary_wiring.py's _old_domain_sample_action."""
    taxi_location = sim.taxi.location
    candidates = {
        c for c in sim.graph.successors(taxi_location)
        if sim.graph.nodes[c]["attr"] != [0, 0, 1] or c in sim.passengers
    }
    candidates.add(taxi_location)
    candidates.add(0)
    return sorted(candidates)[0]


def old_domain_obs_hash(convention, seed=SEED, num_steps=NUM_STEPS):
    from sage.domains.gym_taxi import REWARDS
    from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv

    env = GraphTaxiEnv(
        representation="graph", scenario="city", mask=False,
        rewards=REWARDS["v1"], graph_convention=convention,
    )
    env.seed(seed)
    obs = env.reset()
    observations = [obs]
    for _ in range(num_steps):
        action = sample_action(env.sim)
        obs, _reward, done, _info = env.step(action)
        observations.append(obs)
        if done:
            obs = env.reset()
    env.close()
    blob = "\x00".join(observations)
    return hashlib.sha256(blob.encode()).hexdigest()


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("convention", choices=CONVENTIONS)
    args = parser.parse_args(argv)

    _install_randint_shim()
    sys.stderr.write("stack={} (gym {}) convention={} seed={} steps={}\n".format(
        stack_key(), gym.__version__, args.convention, SEED, NUM_STEPS))
    print(old_domain_obs_hash(args.convention))


if __name__ == "__main__":
    main(sys.argv[1:])
