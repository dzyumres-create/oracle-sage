"""City-taxi environment helpers for the analysis (tree-agnostic: uses whichever
`sage` is first on PYTHONPATH).

The env is built directly as GraphTaxiEnv with the kwargs gym registers for
`city-taxi-unmasked-v1` (the id Cells 1 and 3 trained on), bypassing gym.make:
gym 0.26's make() adds wrappers that call reset(seed=...)/step() with the new
API, which this env doesn't implement.
"""
from copy import deepcopy

from analysis import compat  # noqa: F401  (must precede sage imports)

import numpy as np

CITY_V1_KWARGS = dict(representation="graph", scenario="city", mask=False,
                      rewards={"base": 0, "failed-action": 0, "drop-off": 1})


def make_env(seed):
    """Seeded city-taxi-unmasked-v1 env, already reset. Returns (env, obs_json)."""
    from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
    env = GraphTaxiEnv(**CITY_V1_KWARGS)
    env.seed(seed)
    obs = env.reset()
    return env, obs


def obs_to_data(obs_json):
    """One observation JSON -> torch_geometric Data (single graph)."""
    from sage.domains.utils.representations import json_to_graph
    return json_to_graph(np.array([[obs_json]], dtype=object)).to_data_list()[0]


def plan(obs_json, goal):
    """(projection, actions) from the training planner for one goal."""
    from sage.domains.gym_taxi.simulator.planner import Planner
    return Planner().plan(deepcopy(obs_to_data(obs_json)), int(goal))


def sim_summary(env):
    """Hashable summary of the dynamics-relevant state: taxi, passengers, time,
    deliveries so far (100 - delivery_limit), episode step counter."""
    sim = env.sim
    passengers = tuple(sorted((int(k), int(v.location), int(v.destination)) for k, v in sim.passengers.items()))
    return dict(
        taxi_location=int(sim.taxi.location),
        taxi_passenger=None if sim.taxi.passenger is None else int(sim.taxi.passenger),
        passengers=passengers,
        time=int(sim.time),
        steps=int(env.steps),
        delivered=100 - int(sim.delivery_limit),
    )


def graph_edges(env):
    """Sorted edge list with attributes (to compare graphs exactly)."""
    return tuple(sorted((int(u), int(v), tuple(d["attr"])) for u, v, d in env.sim.graph.edges(data=True)))


def random_goal_rollout(env, obs, rng, n_decisions, max_frames=None, p_passenger=0.0):
    """Drive env with random goals executed through the planner (open loop, like
    training): with probability p_passenger a uniformly random passenger node,
    otherwise a uniformly random node. Returns (primitive actions, final obs, done)."""
    actions, done = [], False
    for _ in range(n_decisions):
        x = obs_to_data(obs).x
        passengers = np.flatnonzero(x[:, 2].numpy() == 1)
        if len(passengers) and rng.uniform() < p_passenger:
            goal = int(passengers[rng.randint(len(passengers))])
        else:
            goal = int(rng.randint(len(x)))
        _, acts = plan(obs, goal)
        for a in acts:
            obs, r, done, info = env.step(int(a))
            actions.append(int(a))
            if done or (max_frames is not None and len(actions) >= max_frames):
                return actions, obs, done
    return actions, obs, done
