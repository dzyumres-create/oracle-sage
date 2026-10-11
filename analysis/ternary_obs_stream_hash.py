"""
Prints the SHA-256 of a ternary-domain observation stream for one graph convention, driven
through AsyncVecEnv exactly the way PlanFeedback_A2C.collect_rollouts drives it: each
decision picks one goal per env (an RNG-free rule), Planner.plan turns it into a plan, and
the plans are stepped column by column with the `mask` of still-running envs. Where
AsyncVecEnv.step accepts `obs_mask` (the observation skip, commit 20b3c99 onwards) it is
passed observation_needed(plan_step, plan_lengths), as collect_rollouts does; where it does
not (e.g. ternary-gnn @ e84fb5d), every observation is built.

Hashed: the reset observations, then after every vec step the full observation buffer, the
rewards, the dones and the terminal_observation of every env whose episode ended on that step. The env timeout is shortened so episodes
end inside the run, including mid-plan.

The ternary simulator draws from env.np_random, which is replaced by
np.random.RandomState(seed) before each reset, so the worlds do not depend on gym's
seeding (gym 0.26 hands out a numpy Generator, gym 0.18 a RandomState).

Imports only things that exist at ternary-gnn @ e84fb5d. Run from the repo root:

    PYTHONPATH=. python analysis/ternary_obs_stream_hash.py oracle_sage   # or vilg, atom

The hash is printed alone on stdout; the stack and run counts go to stderr. This is the
reference for tests/test_ternary_wl_obs_skip.py's TERNARY_OBS_STREAM_SHA256[<stack>].
"""
import argparse
import hashlib
import inspect
import sys

import gym
import gym.utils.seeding as _seeding
import numpy as np

CONVENTIONS = ("oracle_sage", "vilg", "atom")
SEEDS = (0, 1, 2)
NUM_DECISIONS = 30
TIMEOUT = 80


class _RandintCompatGenerator(object):
    """gym 0.26's seeding returns a numpy Generator, which has no .randint; the city maze
    generator calls .randint while GraphTaxiEnv is being constructed (before the RNG is
    replaced below). Same shim as analysis/old_domain_obs_hash.py."""

    def __init__(self, generator):
        self._generator = generator

    def randint(self, low, high=None):
        if hasattr(self._generator, "randint"):
            return self._generator.randint(low, high) if high is not None else self._generator.randint(low)
        return self._generator.integers(low, high)

    def __getattr__(self, name):
        return getattr(self._generator, name)


def install_randint_shim():
    """Idempotent: wraps seeding.np_random's result only when it lacks .randint."""
    if getattr(_seeding.np_random, "_randint_shim", False):
        return
    original_np_random = _seeding.np_random

    def np_random_with_randint(seed=None):
        generator, seed = original_np_random(seed)
        if not hasattr(generator, "randint"):
            generator = _RandintCompatGenerator(generator)
        return generator, seed

    np_random_with_randint._randint_shim = True
    _seeding.np_random = np_random_with_randint


def stack_key():
    return "gym" + ".".join(gym.__version__.split(".")[:2])


def _make_env(convention, seed, timeout):
    def _thunk():
        from sage.domains.gym_taxi import REWARDS
        from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv

        env = GraphTaxiEnv(representation="graph", scenario="city_ternary", mask=False,
                           rewards=REWARDS["v1"], graph_convention=convention, ternary=True)
        env.scenario = dict(env.scenario, timeout=timeout)  # a copy: SCENARIOS is untouched
        env.np_random = np.random.RandomState(seed)
        return env
    return _thunk


def _observation_needed(plan_step, plan_lengths):
    """collect_rollouts' obs_mask rule (plan_feedback_a2c.observation_needed, where it
    exists): True exactly at each env's last plan step."""
    return plan_step == np.asarray(plan_lengths) - 1


def ternary_obs_stream_hash(convention, seeds=SEEDS, num_decisions=NUM_DECISIONS, timeout=TIMEOUT,
                            require_obs_mask=False):
    """:return: (sha256 hex digest, stats dict)"""
    from sage.agent.async_vec_env import AsyncVecEnv

    install_randint_shim()
    vec = AsyncVecEnv([_make_env(convention, seed, timeout) for seed in seeds])
    supports_obs_mask = "obs_mask" in inspect.signature(vec.step).parameters
    if require_obs_mask and not supports_obs_mask:
        raise RuntimeError("AsyncVecEnv.step has no obs_mask parameter on this commit")
    converter = vec.envs[0].observation_space.converter
    planner = vec.envs[0].observation_space.planner

    hasher = hashlib.sha256()
    stats = {"vec_steps": 0, "terminal_observations": 0, "obs_mask_false": 0, "supports_obs_mask": supports_obs_mask}

    def feed(text):
        hasher.update(text.encode())
        hasher.update(b"\x00")

    def as_text(observation):
        """A vec-env buffer row is a shape-(1,) string array; a terminal_observation is
        the env's own str. Both are hashed as the JSON string itself."""
        if isinstance(observation, np.ndarray):
            observation = observation.reshape(-1)[0]
        return str(observation)

    obs = vec.reset()
    for o in obs:
        feed(as_text(o))
    for decision in range(num_decisions):
        plans = []
        for i in range(len(seeds)):
            graph = converter([obs[i]]).to_data_list()[0]
            selectable = graph.mask.nonzero(as_tuple=True)[0].tolist()
            goal = selectable[(7 * decision + 3 * i) % len(selectable)]
            _projection, actions = planner.plan(graph, int(goal))
            plans.append([int(a) for a in actions])
        plan_lengths = np.array([len(p) for p in plans])
        plans_grid = -np.ones((len(plans), max(plan_lengths)), dtype=int)
        for i, p in enumerate(plans):
            plans_grid[i, :len(p)] = p
        env_finished = plan_lengths == 0
        plan_step = 0
        while not env_finished.all() and plan_step < max(plan_lengths):
            env_actions = plans_grid[:, plan_step]
            env_finished = np.logical_or(env_finished, env_actions == -1)
            stepped = np.logical_not(env_finished)
            if supports_obs_mask:
                obs_mask = _observation_needed(plan_step, plan_lengths)
                stats["obs_mask_false"] += int(np.logical_and(~env_finished, ~obs_mask).sum())
                obs, rewards, dones, infos = vec.step(env_actions, np.logical_not(env_finished), obs_mask)
            else:
                obs, rewards, dones, infos = vec.step(env_actions, np.logical_not(env_finished))
            stats["vec_steps"] += 1
            for o in obs:
                feed(as_text(o))
            feed(repr([float(r) for r in rewards]))
            feed(repr([bool(d) for d in dones]))
            for i, info in enumerate(infos):
                # only envs stepped this time: a finished env keeps its last done/info
                if stepped[i] and dones[i] and "terminal_observation" in info:
                    stats["terminal_observations"] += 1
                    feed(as_text(info["terminal_observation"]))
            env_finished = np.logical_or(env_finished, dones)
            plan_step += 1
    vec.close()
    return hasher.hexdigest(), stats


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("convention", choices=CONVENTIONS)
    args = parser.parse_args(argv)
    digest, stats = ternary_obs_stream_hash(args.convention)
    sys.stderr.write("stack={} (gym {}) convention={} seeds={} decisions={} timeout={} stats={}\n".format(
        stack_key(), gym.__version__, args.convention, SEEDS, NUM_DECISIONS, TIMEOUT, stats))
    print(digest)


if __name__ == "__main__":
    main(sys.argv[1:])
