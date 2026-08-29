"""
.. module:: wl_vocab_cache
   :synopsis: Loads and caches the frozen WL-colour vocabulary used to wire
   sage.domains.utils.wl_colours.wl_colours into the Taxi domain's graph
   construction (env_to_graph and Planner.plan()).

   HARDCODED to one specific, already-validated frozen vocab: built from
   scenario="city" (Oracle-SAGE's real Taxi training env, city-taxi-unmasked-v1)
   at L=1, from 18,000 sampled graphs (see sage/domains/utils/build_wl_vocab.py),
   with 0% held-out OOV measured over 121k+ fresh nodes. WL_VOCAB_PATH and
   NUM_ITERATIONS below are two halves of the same fact - a vocab's colour
   ids are only meaningful for the exact L it was built/frozen with - so if
   a different scenario/L vocab is ever adopted for production use, BOTH
   constants must be updated together.

   The vocab is loaded from disk ONCE per process and cached (`lru_cache`):
   this matters because env_to_graph runs on every environment step,
   potentially across many parallel worker processes, so re-parsing the
   vocab JSON on every call would be wasteful.

   Deliberately does NOT import sage.domains.utils.build_wl_vocab (which
   already has its own `load_vocab`): that module runs numpy/gym
   compatibility monkeypatches as an import side effect (needed only for
   its own standalone CLI use against this sandbox's drifted gym/numpy -
   see its docstring), and the live env/policy pipeline importing this
   module (via env_to_graph / Planner.plan()) should not silently inherit
   that as a side effect of loading a vocab file. So the small, pure JSON
   decode logic is reproduced here instead - it must stay in sync with
   build_wl_vocab.py's `_encode_signature`/`save_vocab`, which is what
   actually produced the file on disk.
"""
import json
from functools import lru_cache
from pathlib import Path

from sage.domains.utils.wl_colours import OOV_SIGNATURE

# Tied together - see module docstring. Change both if a different
# scenario/L vocab is ever adopted.
WL_VOCAB_PATH = Path(__file__).resolve().parents[2] / "utils" / "wl_vocab_taxi_city_L1.json"
NUM_ITERATIONS = 1


def _decode_signature(encoded):
    """Inverse of build_wl_vocab.py's `_encode_signature` - must stay in sync with it."""
    kind = encoded["kind"]
    if kind == "oov":
        return OOV_SIGNATURE
    if kind == "init":
        return ("init", encoded["type_id"])
    if kind == "refine":
        neighbours = tuple((colour, label) for colour, label in encoded["neighbours"])
        return (encoded["own_colour"], neighbours)
    raise ValueError(f"unknown encoded signature kind: {kind!r}")


@lru_cache(maxsize=1)
def get_wl_vocab():
    """
    Loads (once per process, then cached) the frozen WL-colour vocab from
    `WL_VOCAB_PATH`, in the signature -> colour id format `wl_colours()`
    expects for its `vocab` argument.
    """
    with open(WL_VOCAB_PATH) as f:
        payload = json.load(f)
    vocab = {}
    for entry in payload["entries"]:
        vocab[_decode_signature(entry["signature"])] = entry["id"]
    return vocab
