"""
.. module:: compat
   :synopsis: Small shims for library APIs that differ between the RCP stack (torch 1.7.1,
   torch_geometric 1.7.2, gym 0.18.0, networkx 2.6.3) and newer local stacks
   (e.g. torch_geometric 2.x, gym 0.26). Use these instead of the raw attribute anywhere
   the code must run on both.
"""


def data_keys(data):
    """
    Attribute names stored on a torch_geometric Data/Batch.

    PyG 1.7.2 exposes ``Data.keys`` as a property returning a list; PyG 2.x makes it a
    method. Calling ``data.keys()`` therefore fails on 1.7.2, and ``list(data.keys)``
    fails on 2.x.

    :param data: a torch_geometric Data or Batch
    :return: list of attribute names
    """
    keys = data.keys
    return list(keys() if callable(keys) else keys)


def spec_kwargs(spec):
    """
    The constructor kwargs a registered gym EnvSpec was registered with.

    gym 0.26 exposes them as ``spec.kwargs``; gym 0.18 stores them as ``spec._kwargs``.

    :param spec: a gym EnvSpec, e.g. ``gym.spec("city-taxi-ternary-unmasked-v1")``
    :return: a shallow copy of the kwargs dict, so callers can't mutate the registry
    """
    if hasattr(spec, "kwargs"):
        return dict(spec.kwargs)
    return dict(spec._kwargs)


# ==========================================================================================
# gym >= 0.26 (the Macs): run training like gym 0.18 (RCP) does
# ==========================================================================================
#
# On gym 0.18 (RCP) everything below delegates to gym itself, so that path is unchanged.
# On gym >= 0.26:
#   - seeding.np_random returns a numpy Generator, so the same seed gives different episodes
#     than on RCP, and the maze generator (RandomState.randint) fails outright. np_random
#     below reproduces gym 0.18's algorithm exactly instead (a RandomState seeded from a
#     SHA-512 hash of the seed), so --seed gives RCP's episodes;
#   - gym.make wraps the env in PassiveEnvChecker (which fails on JsonGraph, a Box without
#     .low) and OrderEnforcing. gym 0.18's make only constructs the env and attaches its
#     spec (no env here sets max_episode_steps, so no TimeLimit), which make_from_spec does.

import copy
import hashlib
import os
import struct

import gym
import numpy as np

LEGACY_GYM = tuple(int(p) for p in gym.__version__.split(".")[:2]) < (0, 26)


def _bigint_from_bytes(data):
    # gym 0.18.0 gym/utils/seeding.py (MIT licence, OpenAI), unchanged -- including padding
    # a whole extra int when len(data) is already a multiple of 4
    sizeof_int = 4
    padding = sizeof_int - len(data) % sizeof_int
    data += b'\0' * padding
    int_count = int(len(data) / sizeof_int)
    unpacked = struct.unpack("{}I".format(int_count), data)
    accum = 0
    for i, val in enumerate(unpacked):
        accum += 2 ** (sizeof_int * 8 * i) * val
    return accum


def _int_list_from_bigint(bigint):
    # gym 0.18.0, unchanged
    if bigint < 0:
        raise ValueError('Seed must be non-negative, not {}'.format(bigint))
    elif bigint == 0:
        return [0]
    ints = []
    while bigint > 0:
        bigint, mod = divmod(bigint, 2 ** 32)
        ints.append(mod)
    return ints


def _create_seed(a=None, max_bytes=8):
    # gym 0.18.0 create_seed, for the None / int cases (the only ones np_random accepts)
    if a is None:
        a = _bigint_from_bytes(os.urandom(max_bytes))
    else:
        a = a % 2 ** (8 * max_bytes)
    return a


def _hash_seed(seed, max_bytes=8):
    # gym 0.18.0 hash_seed (seed is never None here: np_random creates it first)
    digest = hashlib.sha512(str(seed).encode('utf8')).digest()
    return _bigint_from_bytes(digest[:max_bytes])


def np_random(seed=None):
    """
    (RandomState, seed), as gym 0.18's seeding.np_random returns them. On gym < 0.26 this IS
    gym's own function (looked up at call time); on gym >= 0.26 it is gym 0.18's algorithm,
    so a seed gives the same random stream as on RCP.
    """
    if LEGACY_GYM:
        import gym.utils.seeding as seeding
        return seeding.np_random(seed)
    if seed is not None and not (isinstance(seed, (int, np.integer)) and 0 <= seed):
        raise ValueError('Seed must be a non-negative integer or omitted, not {}'.format(seed))
    seed = _create_seed(None if seed is None else int(seed))
    rng = np.random.RandomState()
    rng.seed(_int_list_from_bigint(_hash_seed(seed)))
    return rng, seed


def make_from_spec(env_id, **kwargs):
    """
    What gym 0.18's gym.make(env_id, **kwargs) returns for these envs: the entry point built
    with the registered kwargs updated by `kwargs`, with a copy of the spec attached -- no
    checker and no wrappers. Refuses specs with max_episode_steps (gym 0.18 would add a
    TimeLimit; no env in this repo sets it).
    """
    spec = gym.spec(env_id)
    if spec.max_episode_steps is not None:
        raise NotImplementedError(f"{env_id}: max_episode_steps is set; add gym 0.18's TimeLimit here first")
    env_kwargs = spec_kwargs(spec)
    env_kwargs.update(kwargs)
    entry_point = spec.entry_point
    if callable(entry_point):
        env = entry_point(**env_kwargs)
    else:
        from gym.envs.registration import load
        env = load(entry_point)(**env_kwargs)
    env_spec = copy.deepcopy(spec)
    if hasattr(env_spec, "kwargs"):
        env_spec.kwargs = env_kwargs
    env.unwrapped.spec = env_spec
    return env


def vec_env_id(env_id):
    """What gnn_global hands make_vec_env as the env: the id itself on gym < 0.26 (so
    make_vec_env calls gym.make, exactly as before), else a constructor that builds the env
    like gym 0.18's make (make_vec_env calls it with env_kwargs)."""
    if LEGACY_GYM:
        return env_id
    return lambda **kwargs: make_from_spec(env_id, **kwargs)


def compat_label():
    """For the startup line: which path builds and seeds the envs."""
    return f"gym={gym.__version__} env_compat={'off' if LEGACY_GYM else 'gym018_seeding+spec_make'}"
