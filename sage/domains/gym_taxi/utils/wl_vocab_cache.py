"""
.. module:: wl_vocab_cache
   :synopsis: Loads and caches the frozen WL-colour vocabulary used to wire
   sage.domains.utils.wl_colours.wl_colours into the Taxi domain's graph
   construction (env_to_graph and Planner.plan()).

   DEFAULTS to one specific, already-validated frozen vocab: built from
   scenario="city" (Oracle-SAGE's real Taxi training env, city-taxi-unmasked-v1)
   at L=1, from 18,000 sampled graphs (see sage/domains/utils/build_wl_vocab.py),
   with 0% held-out OOV measured over 121,148 fresh nodes (disjoint seed).
   WL_VOCAB_PATH and NUM_ITERATIONS below are two halves of the same fact -
   a vocab's colour ids are only meaningful for the exact L it was
   built/frozen with.

   These two constants are the graph_convention="oracle_sage" default and
   are NEVER themselves mutated - `get_wl_vocab()`/`get_wl_num_iterations()`
   below fall back to them whenever no override is configured, so oracle_sage
   callers that have never heard of the override mechanism keep working
   exactly as before, unchanged.

   Runtime override (added for graph_convention="vilg"): env_to_graph and
   Planner.plan() both used to import a bare `NUM_ITERATIONS` constant and
   call the bare, path-less `get_wl_vocab()` - meaning BOTH conventions
   always read this one oracle_sage vocab/L, regardless of graph_convention,
   even though WLPlanFeedbackPolicy already loads an arbitrary,
   --wl-vocab-path-configurable vocab independently (see
   sage/agent/wl_plan_feedback_policy.py's own `_load_vocab`). For vilg runs
   that mismatch is silent, not a crash: colour ids from a smaller vocab can
   still be valid (just WRONG) indices into a larger embedding table.
   `configure_wl_vocab_override(path, num_iterations)` lets a caller (e.g.
   gnn_global.py, once it decides how to thread --wl-vocab-path's matching L
   through) point BOTH env_to_graph and Planner.plan() at the SAME vocab
   file+L the policy was already configured with, instead of maintaining a
   second, independently-hardcoded "which vocab for vilg" mapping here that
   could drift out of sync with --wl-vocab-path all over again. Call
   `reset_wl_vocab_override()` to go back to the oracle_sage default.
   Deliberately requires path AND num_iterations together (never one alone)
   - a vocab's colour ids are meaningless without knowing the L they were
   built at, so a partial override would just reintroduce the same silent-
   mismatch bug in a new form.

   The vocab is loaded from disk ONCE per (process, path) and cached
   (`lru_cache`, keyed by path): this matters because env_to_graph runs on
   every environment step, potentially across many parallel worker
   processes, so re-parsing the vocab JSON on every call would be wasteful.
   Keying the cache by path (rather than caching a single result) means the
   default and an override - or several different overrides across a
   process's lifetime - are each parsed at most once, and switching the
   override never needs to invalidate a stale cache entry for the other one.

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

# The graph_convention="oracle_sage" default - see module docstring. Never
# mutated at runtime; get_wl_vocab()/get_wl_num_iterations() fall back to
# these whenever no override is configured.
WL_VOCAB_PATH = Path(__file__).resolve().parents[2] / "utils" / "wl_vocab_taxi_city_L1_edgefixed.json"
NUM_ITERATIONS = 1

# Runtime override state - see module docstring. None/None means "no
# override, use the oracle_sage default above". Always set/cleared together
# via configure_wl_vocab_override()/reset_wl_vocab_override(), never
# partially, so a vocab path and its matching L can never drift apart here.
_override_path = None
_override_num_iterations = None


def configure_wl_vocab_override(path, num_iterations) -> None:
    """
    Points get_wl_vocab()/get_wl_num_iterations() at an arbitrary vocab file
    and its matching L, instead of the oracle_sage default - e.g. to match
    whatever --wl-vocab-path WLPlanFeedbackPolicy was configured with for a
    graph_convention="vilg" run. Both arguments are required together (see
    module docstring for why a partial override isn't offered). Affects the
    whole process until reset_wl_vocab_override() is called.

    :param path: path to a frozen WL-colour vocab JSON (str or Path)
    :param num_iterations: the L that vocab was built/frozen at
    """
    global _override_path, _override_num_iterations
    _override_path = Path(path)
    _override_num_iterations = num_iterations


def reset_wl_vocab_override() -> None:
    """Clears any override set by configure_wl_vocab_override(), reverting
    get_wl_vocab()/get_wl_num_iterations() to the oracle_sage default."""
    global _override_path, _override_num_iterations
    _override_path = None
    _override_num_iterations = None


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


@lru_cache(maxsize=None)
def _load_vocab_from_path(path_str: str) -> dict:
    with open(path_str) as f:
        payload = json.load(f)
    vocab = {}
    for entry in payload["entries"]:
        vocab[_decode_signature(entry["signature"])] = entry["id"]
    return vocab


def get_wl_vocab():
    """
    Loads (once per process per distinct path, then cached) the frozen
    WL-colour vocab from whichever path is active - the one set by
    configure_wl_vocab_override(), or WL_VOCAB_PATH (the oracle_sage
    default) if no override is configured - in the signature -> colour id
    format `wl_colours()` expects for its `vocab` argument.
    """
    path = _override_path if _override_path is not None else WL_VOCAB_PATH
    return _load_vocab_from_path(str(path))


def get_wl_num_iterations():
    """
    Returns whichever L is active - the one set by
    configure_wl_vocab_override(), or NUM_ITERATIONS (the oracle_sage
    default) if no override is configured. A plain module-level constant
    can't be used for this the way NUM_ITERATIONS is: callers that did
    `from wl_vocab_cache import NUM_ITERATIONS` bind that value once, at
    import time, so a later configure_wl_vocab_override() call could never
    reach them - this must be called fresh each time L is needed instead.
    """
    return _override_num_iterations if _override_num_iterations is not None else NUM_ITERATIONS
