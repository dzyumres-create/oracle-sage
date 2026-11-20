"""
Edge-type variants of the GNN input (Horcik et al. 2025, Sec. 2): Gr, MGr and EGr.

Every Taxi representation builds a **Gr** graph: one edge per ordered node pair, with a
multi-hot label. MGr and EGr are built from it here, inside the network's input step
(GNNExtractor.forward), so the env, the JSON wire format, the planners and WL never see
anything but Gr:

- **MGr**: one parallel edge per edge *type* that is set on a Gr edge, labelled with the
  one-hot code of that type. Read by the usual GraphNet (edge_size = number of types).
- **EGr**: the same typed edges, without labels; each type has its own message function
  (TypedGraphNet in graph_net.py).

A type is one Gr label column, optionally restricted to one value of the Object graph's
`dir` column (+1 forward, -1 reverse). Design choices (W4_NOTES.md, approved 20 Nov 2026):
Object direction is part of the type for the predicates that have one (in / at /
picked_up_at / destination); Object-Atom types are the argument position only; Atom types
are the (i, j) position pairs.

Skipped combinations (MGr would give exactly the Gr graph): Object-Atom in both domains,
and standard Object (no pair is joined by two predicates there).
"""
from typing import NamedTuple, Optional, Tuple

import torch as th

EDGE_MODES = ("gr", "mgr", "egr")
DEFAULT_EDGE_MODE = "gr"


class EdgeType(NamedTuple):
    name: str
    column: int  # the Gr label column that must be set (> 0)
    direction: Optional[int] = None  # None: any; +1 / -1: the Gr `dir` column must equal it


class EdgeTypeSpec(NamedTuple):
    graph_convention: str
    ternary: bool
    types: Tuple[EdgeType, ...]
    dir_column: Optional[int]  # the Gr label column holding dir (Object only)

    @property
    def num_types(self):
        return len(self.types)

    @property
    def names(self):
        return tuple(t.name for t in self.types)


# standard Object: Gr label [adjacent, in, destination, dir]
_STD_OBJECT = (
    EdgeType("adjacent", 0),
    EdgeType("in>", 1, 1), EdgeType("in<", 1, -1),
    EdgeType("destination>", 2, 1), EdgeType("destination<", 2, -1),
)
# ternary Object: Gr label OBJECT_EDGE_LABELS =
# [road, at, picked_up_at, req12, req21, req13, req31, req23, req32, dir]
_TER_OBJECT = (
    EdgeType("road", 0),
    EdgeType("at>", 1, 1), EdgeType("at<", 1, -1),
    EdgeType("picked_up_at>", 2, 1), EdgeType("picked_up_at<", 2, -1),
    EdgeType("req12", 3), EdgeType("req21", 4), EdgeType("req13", 5),
    EdgeType("req31", 6), EdgeType("req23", 7), EdgeType("req32", 8),
)


def _positions(n):
    return tuple(EdgeType(f"pos{p}", p - 1) for p in range(1, n + 1))


def _pairs(m):
    # same column order as ATOM_LABELS / _atoms_to_graph_generic: (i, j) row-major
    pairs = [(i, j) for i in range(1, m + 1) for j in range(1, m + 1)]
    return tuple(EdgeType(f"({i},{j})", k) for k, (i, j) in enumerate(pairs))


_SPECS = {
    # (graph_convention, ternary): (types, dir_column)
    ("oracle_sage", False): (_STD_OBJECT, 3),
    ("oracle_sage", True): (_TER_OBJECT, 9),
    ("vilg", False): (_positions(2), None),
    ("vilg", True): (_positions(3), None),
    ("atom", False): (_pairs(2), None),
    ("atom", True): (_pairs(3), None),
}

# MGr is the Gr graph itself for these (graph_convention, ternary) pairs
MGR_SKIPPED = {("vilg", False), ("vilg", True), ("oracle_sage", False)}


def validate_edge_mode(edge_mode, graph_convention=None, ternary=None):
    """Raises ValueError for an unknown mode, or for a combination this branch does not run.
    With graph_convention None, only the mode name is checked."""
    if edge_mode not in EDGE_MODES:
        raise ValueError(f"edge_mode must be one of {EDGE_MODES}, got {edge_mode!r}")
    if edge_mode == "gr" or graph_convention is None:
        return
    if (graph_convention, bool(ternary)) not in _SPECS:
        raise ValueError(f"edge_mode {edge_mode!r} has no edge types for graph_convention "
                         f"{graph_convention!r} (ternary={bool(ternary)})")
    if edge_mode == "mgr" and (graph_convention, bool(ternary)) in MGR_SKIPPED:
        raise ValueError(
            f"edge_mode 'mgr' is skipped for graph_convention {graph_convention!r} "
            f"(ternary={bool(ternary)}): there MGr gives the same edges as Gr (W4_NOTES.md)"
        )


def edge_type_spec(graph_convention, ternary, edge_mode="egr"):
    """The EdgeTypeSpec for a representation, after validate_edge_mode."""
    validate_edge_mode(edge_mode, graph_convention, ternary)
    types, dir_column = _SPECS[(graph_convention, bool(ternary))]
    return EdgeTypeSpec(graph_convention, bool(ternary), types, dir_column)


def spec_from_observation_space(observation_space, edge_mode):
    """The policy-side lookup: a Taxi JsonGraph space carries its Planner, which knows the
    graph convention and the domain. None for gr (nothing is built)."""
    validate_edge_mode(edge_mode)
    if edge_mode == "gr":
        return None
    planner = getattr(observation_space, "planner", None)
    if planner is None:
        raise ValueError(f"edge_mode {edge_mode!r} needs a Taxi observation space (with a planner)")
    return edge_type_spec(planner.graph_convention, getattr(planner, "ternary", False), edge_mode)


def type_mask(edge_attr, spec):
    """[E, T] bool: which types each Gr edge carries."""
    columns = th.tensor([t.column for t in spec.types], dtype=th.long, device=edge_attr.device)
    mask = edge_attr[:, columns] > 0
    if spec.dir_column is not None:
        directions = th.tensor([0 if t.direction is None else t.direction for t in spec.types],
                               dtype=edge_attr.dtype, device=edge_attr.device)
        dir_ok = (directions == 0) | (edge_attr[:, spec.dir_column:spec.dir_column + 1] == directions)
        mask = mask & dir_ok
    return mask


def expand_typed_edges(edge_index, edge_attr, spec):
    """
    Gr -> typed edges. Returns (typed_edge_index [2, E'], edge_type [E']), one typed edge
    per (Gr edge, type set on it), ordered by Gr edge, then by type (deterministic).
    """
    gr_edge, edge_type = type_mask(edge_attr, spec).nonzero(as_tuple=True)
    return edge_index[:, gr_edge], edge_type


def to_mgr(edge_index, edge_attr, spec):
    """MGr: the typed edges with one-hot labels ([E', T], same dtype as edge_attr)."""
    typed_edge_index, edge_type = expand_typed_edges(edge_index, edge_attr, spec)
    labels = th.nn.functional.one_hot(edge_type, spec.num_types).to(edge_attr.dtype)
    return typed_edge_index, labels


def split_by_type(edge_index, edge_attr, spec):
    """EGr: one (src, dst) pair of index tensors per type, in spec order."""
    mask = type_mask(edge_attr, spec)
    out = []
    for t in range(spec.num_types):
        rows = mask[:, t].nonzero(as_tuple=True)[0]
        out.append((edge_index[0, rows], edge_index[1, rows]))
    return out
