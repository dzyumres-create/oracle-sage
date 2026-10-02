"""Shims so the training code (pinned to torch 1.7 / gym 0.18 / numpy <1.20 on RCP)
runs unmodified in the Mac `sage` env (torch 2.x / gym 0.26 / numpy 2.x).

Import this module BEFORE anything from `sage`. Nothing under sage/ is edited.

Shims:
  1. gym.utils.seeding.np_random -> gym 0.18 behaviour: returns a legacy
     np.random.RandomState (the simulator calls .randint/.choice/.uniform;
     gym 0.26 returns a Generator, which has no .randint and crashes
     generate_city_maze). Seeding replicates gym 0.18's hash_seed scheme so a
     given integer seed gives the same RandomState stream as on RCP.
  2. numpy aliases removed in numpy 1.24 (np.int, np.float, np.bool, np.object),
     still used by main / cell2-vilg-gnn code.
"""
import hashlib
import os
import struct

import numpy as np


# --- 2. numpy aliases ---------------------------------------------------------
for _name, _val in (("int", int), ("float", float), ("bool", bool), ("object", object)):
    if _name not in np.__dict__:
        setattr(np, _name, _val)


# --- 1. gym 0.18 seeding ------------------------------------------------------
def _bigint_from_bytes(b):
    sizeof_int = 4
    padding = sizeof_int - len(b) % sizeof_int
    b += b"\0" * padding
    int_count = len(b) // sizeof_int
    unpacked = struct.unpack("{}I".format(int_count), b)
    accum = 0
    for i, val in enumerate(unpacked):
        accum += 2 ** (sizeof_int * 8 * i) * val
    return accum


def _int_list_from_bigint(bigint):
    if bigint == 0:
        return [0]
    ints = []
    while bigint > 0:
        bigint, mod = divmod(bigint, 2 ** 32)
        ints.append(mod)
    return ints


def create_seed(a=None, max_bytes=8):
    if a is None:
        a = _bigint_from_bytes(os.urandom(max_bytes))
    elif isinstance(a, (int, np.integer)):
        a = int(a) % 2 ** (8 * max_bytes)
    else:
        raise TypeError(f"invalid seed type {type(a)}")
    return a


def hash_seed(seed=None, max_bytes=8):
    if seed is None:
        seed = create_seed(max_bytes=max_bytes)
    h = hashlib.sha512(str(seed).encode("utf8")).digest()
    return _bigint_from_bytes(h[:max_bytes])


def np_random_legacy(seed=None):
    if seed is not None and not (isinstance(seed, (int, np.integer)) and seed >= 0):
        raise ValueError(f"Seed must be a non-negative integer or omitted, not {seed}")
    seed = create_seed(seed)
    rng = np.random.RandomState()
    rng.seed(_int_list_from_bigint(hash_seed(seed)))
    return rng, seed


import gym.utils.seeding as _seeding  # noqa: E402

_seeding.np_random = np_random_legacy

PATCHED = True
