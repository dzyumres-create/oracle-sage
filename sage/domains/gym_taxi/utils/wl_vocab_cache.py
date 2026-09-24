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


def configure_wl_vocab_override(path, num_iterations, graph_convention=None) -> None:
    """
    Points get_wl_vocab()/get_wl_num_iterations() at an arbitrary vocab file
    and its matching L, instead of the oracle_sage default - e.g. to match
    whatever --wl-vocab-path WLPlanFeedbackPolicy was configured with for a
    graph_convention="vilg" or "atom" run. `path`/`num_iterations` are
    required together (see module docstring for why a partial override
    isn't offered). Affects the whole process until
    reset_wl_vocab_override() is called.

    `graph_convention`, if given, is validated against the vocab file's own
    recorded metadata (see save_vocab/build_wl_vocab.py) via
    validate_wl_vocab_metadata() BEFORE the override is applied - a failed
    check raises and leaves the override state (and the default vocab)
    untouched, rather than pointing get_wl_vocab() at a vocab that doesn't
    actually match this run. Passing None (the default) skips the
    convention check entirely (num_iterations is still checked whenever the
    vocab has metadata) - existing oracle_sage/vilg callers that have never
    passed this argument, and old vocab files with no metadata at all, are
    completely unaffected.

    :param path: path to a frozen WL-colour vocab JSON (str or Path)
    :param num_iterations: the L that vocab was built/frozen at
    :param graph_convention: expected graph_convention, or None to skip
        that check (see validate_wl_vocab_metadata)
    """
    path = Path(path)
    metadata = _load_metadata_from_path(str(path))
    validate_wl_vocab_metadata(metadata, graph_convention, num_iterations, "configure_wl_vocab_override")

    global _override_path, _override_num_iterations
    _override_path = path
    _override_num_iterations = num_iterations


def reset_wl_vocab_override() -> None:
    """Clears any override set by configure_wl_vocab_override(), reverting
    get_wl_vocab()/get_wl_num_iterations() to the oracle_sage default."""
    global _override_path, _override_num_iterations
    _override_path = None
    _override_num_iterations = None


def is_wl_override_configured() -> bool:
    """
    True iff configure_wl_vocab_override() is currently active (i.e. a
    caller has opted a graph_convention other than the oracle_sage default
    into WL - e.g. --wl-vocab-path/--wl-num-iterations). Distinct from "is
    a vocab available": get_wl_vocab()/get_wl_num_iterations() always
    return SOMETHING (the oracle_sage default, if nothing else) - this is
    for callers (e.g. the atom convention's WL attach helper) that must
    stay a no-op unless WL has been explicitly opted into for their
    convention, since unlike oracle_sage/vilg there is no meaningful atom
    default to silently fall back to.
    """
    return _override_path is not None


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


@lru_cache(maxsize=None)
def _load_metadata_from_path(path_str: str):
    """
    Reads just the "graph_convention"/"num_iterations" top-level keys a vocab file may
    carry (see save_vocab/build_wl_vocab.py), cached by path like
    _load_vocab_from_path (a second, separate cache - both read the same file at most
    once per path, independently of each other).

    :return: {"graph_convention": str, "num_iterations": int}, or None if the file
        predates this metadata (old vocab files - e.g. the oracle_sage/vilg ones built
        before this existed - have neither key).
    """
    with open(path_str) as f:
        payload = json.load(f)
    if "graph_convention" not in payload or "num_iterations" not in payload:
        return None
    return {"graph_convention": payload["graph_convention"], "num_iterations": payload["num_iterations"]}


def validate_wl_vocab_metadata(metadata, expected_graph_convention, expected_num_iterations, source: str) -> None:
    """
    Shared validation used by BOTH configure_wl_vocab_override() (below) and
    WLPlanFeedbackPolicy._load_vocab (sage/agent/wl_plan_feedback_policy.py) - the two
    independent places a vocab file gets loaded for a real run (see that module's
    docstring on why they're two separate loaders, not one) - so the two can't drift
    into checking different things.

    - metadata is None (old vocab file, no recorded graph_convention/num_iterations):
      passes silently UNLESS expected_graph_convention == "atom" - atom is new enough
      that every vocab meant for it MUST have been built by the metadata-writing
      save_vocab, so a metadata-less vocab can only mean "this vocab predates atom
      entirely (or was never meant for it)" - never a legitimate atom vocab that merely
      predates the metadata feature, unlike oracle_sage/vilg's existing vocab files.
    - metadata is present: graph_convention is checked only if
      expected_graph_convention is not None (so oracle_sage/vilg callers that have
      never passed one - see configure_wl_vocab_override's own docstring - still skip
      this check even once metadata exists); num_iterations is always checked once
      metadata is present, since colour ids are meaningless at the wrong L regardless
      of which convention they're for.

    :param metadata: return value of _load_metadata_from_path (a dict or None)
    :param expected_graph_convention: the convention this run needs, or None to skip
        that specific check
    :param expected_num_iterations: the L this run needs
    :param source: caller name, used only in the raised message
    :raises ValueError: on any mismatch, or on missing metadata for an "atom" run
    """
    if metadata is None:
        if expected_graph_convention == "atom":
            raise ValueError(
                f"{source}: vocab has no recorded graph_convention/num_iterations "
                f"metadata, but graph_convention='atom' requires it - old vocab files "
                f"predate the atom convention and cannot be trusted for it."
            )
        return

    if expected_graph_convention is not None and metadata["graph_convention"] != expected_graph_convention:
        raise ValueError(
            f"{source}: vocab was built for graph_convention="
            f"{metadata['graph_convention']!r}, but this run expects "
            f"graph_convention={expected_graph_convention!r}."
        )

    if metadata["num_iterations"] != expected_num_iterations:
        raise ValueError(
            f"{source}: vocab was built at L={metadata['num_iterations']}, but this "
            f"run expects L={expected_num_iterations}."
        )


def get_wl_vocab_metadata():
    """
    Returns the currently active vocab's recorded metadata (see
    _load_metadata_from_path) - the one set by configure_wl_vocab_override(), or
    WL_VOCAB_PATH (the oracle_sage default) if no override is configured - or None if
    that vocab predates metadata.
    """
    path = _override_path if _override_path is not None else WL_VOCAB_PATH
    return _load_metadata_from_path(str(path))


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
