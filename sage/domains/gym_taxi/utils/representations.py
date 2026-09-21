

"""
.. module:: representations
   :synopsis: Contains functions to convert between different representations of the taxi world.
   Functions are of the form x_to_y where x and y are represenation formats, from the list below:
   env - the actual TaxiWorld simulator class itself, holds more than just the current state.
   json - a general purpose JSON serialisation of the current state. All functions should convert to/from this as a common ground.
   image - a four channel image based encoding, designed for input to CNN
   discrete - a discretised encoding only valid for a 5x5 grid, designed for standard Q-agents and compatibility with gym
   pddl - planning domain file designed for input to planner, but lacks a goal.

"""


from math import floor, ceil
import json
import numpy as np
import scipy.sparse as sp
import torch as th
from sage.domains.gym_taxi.utils.config import LOCS, PREDICTABLE5
from sage.domains.utils.representations import graph_to_json, EMB_SIZE
import networkx as nx
import cv2

def env_to_json(env):
    """
    Converts taxi world state from env to json representation

    :param env: taxi world state in env format
    :return: taxi world state in json format
    """
    return graph_to_json(*env_to_graph(env))


def find_new_node(simple_graph,old_node):
    return [x for x,y in simple_graph.nodes if y['old']==old_node ]

def nextwork_to_graph(network,mapping):
    simple_graph = nx.relabel_nodes(network,mapping)
    edges = nx.to_edgelist(simple_graph)
    node_feats = np.array([(1,0,0)]*len(simple_graph.nodes))
    edge_feats = np.array([(1,0,0,1)]*(len(edges)*2))
    start = [x for (x,_,_) in edges]
    end = [y for (_,y,_) in edges]
    edge_index = np.array([start+end,end+start])
    return node_feats,edge_feats,edge_index


def env_to_graph(env):
    node_feats = np.array([v['attr'] for _,v in sorted(env.graph.nodes.items())],dtype=np.float64)
    edges = nx.to_edgelist(env.graph)
    edge_feats = np.array([v['attr'] for (_,_,v) in edges],dtype=np.float64)
    edge_index = np.array([[x,y] for (x,y,_) in edges]).T

    #mask need to be different for SAGE vs SR-DRL. In SAGE, it's probably fine to let mask be true everywhere.
    #In SR-DRL, should only be: the taxis current position and any adjacent positions, any passengers in the taxi's location, and the taxi.
    #Conveniently, this is equal to the taxis location, and all nodes which are adjacent to it.
    #Need to be a bit careful with this because this is masking a number of actions which can be taken in normal taxi: pickup when no passenger present, dropoff when no passenger in taxi, and move into walls.
    if env.planning == False:
        mask = np.zeros(len(node_feats),dtype=bool)
        mask[env.taxi.location]=True
        for x in env.graph[env.taxi.location]:
            mask[x]=True
    else:
        mask = np.ones(len(node_feats),dtype=bool)


    global_feats = np.zeros(EMB_SIZE,dtype=np.float64)
    time_left = (env.timeout-env.time)/env.timeout
    global_feats[0] = time_left

    return node_feats, edge_feats, edge_index, mask, global_feats


def env_to_vilg_json(env):
    """
    Converts taxi world state from env to json representation, using the vILG
    (predicate-instance) graph convention instead of Oracle-SAGE's object-only convention.

    :param env: taxi world state in env format
    :return: taxi world state in json format
    """
    return graph_to_json(*env_to_vilg_graph(env))


def env_to_vilg_graph(env):
    """
    Converts taxi world state from env to a vILG graph (Chen & Thiebaux, Def 3.1): every
    grounded proposition currently true in the state gets its own node, connected by
    position-labeled edges to the object nodes that are its arguments, in addition to the
    object nodes themselves.

    Node features pad the object one-hot (3) and the predicate/status one-hots (3 + 3) out
    to a shared 9-dim vector, since object nodes and proposition nodes use disjoint feature
    blocks but must share one feature vector length.

    destination(pid, loc) is the only propositional goal in Taxi: it's tagged
    unachieved_propositional_goal while pid is still in env.passengers, and flips to
    achieved_propositional_goal once delivered -- see implementation plan Step 4 (this
    relies on attempt_dropoff keeping the passenger node/destination edge alive under the
    "vilg" graph_convention instead of removing it). All other propositions (adjacent, in)
    are tagged achieved_propositional_nongoal, since Taxi has no other goal predicates.

    :param env: taxi world state in env format
    :return: node_feats, edge_feats, edge_index, mask, global_feats
    """
    node_feats = []
    node_kind = []  # parallel list: True = object node, False = proposition node
    selectable = []  # parallel list: True if this node may ever be a legal action target
    edges = []  # list of (src_idx, dst_idx, position_label)

    # --- object nodes: identical source data/order to env_to_graph ---
    sorted_object_nodes = sorted(env.graph.nodes.items())
    obj_index = {}
    for i, (nid, data) in enumerate(sorted_object_nodes):
        obj_index[nid] = i
        node_feats.append(data['attr'] + [0, 0, 0, 0, 0, 0])
        node_kind.append(True)
        # A delivered passenger's node can persist in the graph (Step 4) purely to carry
        # goal status -- it must never be selectable again once popped from
        # env.passengers, since attempt_pickup would KeyError on self.passengers[pid].
        is_passenger = data['attr'] == [0, 0, 1]
        selectable.append((not is_passenger) or (nid in env.passengers))

    # --- proposition nodes: one per forward-direction edge in env.graph ---
    next_idx = len(sorted_object_nodes)
    for (u, v, attr) in env.graph.edges(data=True):
        if attr['attr'][-1] != 1:
            continue  # skip reverse-direction duplicate added for Oracle-SAGE message passing

        pred_onehot = attr['attr'][:-1]
        if pred_onehot == [0, 0, 1]:  # destination(pid, loc) -- u is the passenger
            status_onehot = [0, 1, 0] if u in env.passengers else [1, 0, 0]
        else:
            status_onehot = [0, 0, 1]  # achieved_propositional_nongoal
        node_feats.append([0, 0, 0] + pred_onehot + status_onehot)
        node_kind.append(False)
        selectable.append(False)

        edges.append((next_idx, obj_index[u], 1))
        edges.append((next_idx, obj_index[v], 2))
        next_idx += 1

    node_feats = np.array(node_feats, dtype=np.float64)
    node_kind = np.array(node_kind, dtype=bool)
    selectable = np.array(selectable, dtype=bool)
    edge_index = np.array([[s, d] for (s, d, _) in edges]).T
    # edge label (argument position) replaces predicate-identity edge attr from Oracle-SAGE's convention
    edge_feats = np.array([[1, 0] if pos == 1 else [0, 1] for (_, _, pos) in edges], dtype=np.float64)

    # only currently-live object nodes are selectable actions -- proposition nodes and
    # delivered-but-persisting passenger nodes are always masked out, in both branches below
    mask = node_kind & selectable
    if env.planning == False:
        restricted = np.zeros(len(node_feats), dtype=bool)
        restricted[obj_index[env.taxi.location]] = True
        for x in env.graph[env.taxi.location]:
            restricted[obj_index[x]] = True
        mask = mask & restricted

    global_feats = np.zeros(EMB_SIZE, dtype=np.float64)
    global_feats[0] = (env.timeout - env.time) / env.timeout

    return node_feats, edge_feats, edge_index, mask, global_feats


# --- atom encoding (Horcik et al., AAAI-25, Def. 2) -------------------------------------
#
# Every object AND every true ground proposition gets its own node ("atom"), all sharing
# one node-feature layout: a 6-dim one-hot over ATOM_PREDICATES. An object's own atom is
# an arity-1 "type" atom (location(o)/taxi(o)/passenger(o), one argument: the object
# itself); adjacent/in/destination atoms are arity-2, exactly as in vILG. Unlike vILG,
# there are no atom<->object incidence edges: instead, every pair of atoms that share an
# argument value gets a direct atom<->atom edge, labeled with the (position, position)
# pair at which they share it. This is a fresh implementation of the additive functions
# below (env_to_atoms / atoms_to_graph / graph_to_atoms / env_to_atom_graph) -- not
# wired into taxi_world / taxi_env / planner / the CLI, and no existing function above is
# modified.

ATOM_PREDICATES = ["location", "taxi", "passenger", "adjacent", "in", "destination"]

_TYPE_PREDICATES = {"location", "taxi", "passenger"}

_OBJECT_ATTR_TO_PREDICATE = {
    (1, 0, 0): "location",
    (0, 1, 0): "taxi",
    (0, 0, 1): "passenger",
}

_EDGE_ONEHOT_TO_PREDICATE = {
    (1, 0, 0): "adjacent",
    (0, 1, 0): "in",
    (0, 0, 1): "destination",
}

# edge_attr's 4 columns, in the order Horcik et al.'s Def. 2 uses: a directed edge a->b
# is labeled (i, j) when argument i (1-based) of atom a equals argument j of atom b.
ATOM_LABELS = [(1, 1), (1, 2), (2, 1), (2, 2)]


def env_to_atoms(env):
    """
    Reads a taxi world state directly off env.graph/env.passengers into a flat list of
    ground atoms -- (predicate, args_tuple) pairs -- with NO edge structure yet (see
    atoms_to_graph for that). This is the single source of truth env_to_atom_graph and
    every test in tests/test_atom_encoding.py are built from.

    Atom order (and therefore atom/graph-row index) is:
      1. One type atom per object, in sorted env.graph node order -- node ids are
         asserted to be exactly 0..n_obj-1 (true for every live TaxiWorldSimulator: object
         nodes are added first and never renumbered except by resort_passengers, which
         keeps this invariant), so atom row k IS object k, and env_to_atom_graph's mask
         (type atoms only) is simply "row < n_obj".
      2. One atom per env.graph edge with attr[-1] == 1 -- the SAME forward-direction-only
         filter env_to_vilg_graph uses, so a stale reverse edge left behind by
         TaxiWorldSimulator's un-fixed attempt_move/attempt_pickup (this branch predates
         the stale-edge fix -- see cell4's taxi_world.py) can never produce a spurious
         atom: reverse edges always carry attr[-1] == -1, so they're filtered out here
         exactly as they always were for vILG. These are sorted by (u, v) -- NOT raw
         env.graph.edges() iteration order, which drifts with insertion/removal history
         (pickups and dropoffs mutate the graph) -- so two calls on states that are
         logically identical but reached via different move sequences produce identical
         atom lists.

    :param env: taxi world state in env format
    :return: list of (predicate, args_tuple) -- args_tuple has 1 element for a type atom
        (the object itself), 2 for adjacent/in/destination (their two arguments, in their
        original (u, v) order)
    """
    sorted_object_nodes = sorted(env.graph.nodes.items())
    n_obj = len(sorted_object_nodes)
    node_ids = [nid for nid, _data in sorted_object_nodes]
    assert node_ids == list(range(n_obj)), (
        f"env_to_atoms assumes object node ids are exactly 0..{n_obj - 1} so atom row k "
        f"is object k; got node ids {node_ids}"
    )

    atoms = []
    for nid, data in sorted_object_nodes:
        attr = tuple(data["attr"])
        if attr not in _OBJECT_ATTR_TO_PREDICATE:
            raise ValueError(f"unrecognised object node attr {data['attr']!r} for node {nid}")
        atoms.append((_OBJECT_ATTR_TO_PREDICATE[attr], (nid,)))

    forward_edges = sorted(
        ((u, v, attr) for u, v, attr in env.graph.edges(data=True) if attr["attr"][-1] == 1),
        key=lambda e: (e[0], e[1]),
    )
    for u, v, attr in forward_edges:
        pred_onehot = tuple(attr["attr"][:3])
        if pred_onehot not in _EDGE_ONEHOT_TO_PREDICATE:
            raise ValueError(f"unrecognised edge predicate one-hot {attr['attr'][:3]!r} for edge ({u}, {v})")
        atoms.append((_EDGE_ONEHOT_TO_PREDICATE[pred_onehot], (u, v)))

    return atoms


def atoms_to_graph(atoms):
    """
    Builds the atom-encoding graph (Horcik et al., AAAI-25, Def. 2) from a flat atom list
    (env_to_atoms): node_feats is a 6-dim one-hot per atom over ATOM_PREDICATES; every
    ordered pair of DISTINCT atoms (a, b) that share an argument value gets a directed
    edge a->b, labeled with a 4-dim multi-hot over ATOM_LABELS marking every (i, j) pair
    (1-based argument positions) at which a.args[i] == b.args[j]. Type atoms are ordinary
    arity-1 atoms here, not a special case -- their single argument (position 1) is the
    object itself, exactly like any other atom's arguments.

    Vectorised via scipy.sparse (not torch -- torch 1.7.1, the pinned RCP version, has no
    reliable sparse-sparse matmul): for each argument position p in {1, 2}, build a sparse
    [N, N] "argument-value -> atom-at-position-p" incidence matrix P_p (P_p[v, a] = 1 iff
    atom a's p-th argument equals object value v). Every argument value in this domain is
    an object id, i.e. in [0, n_obj) -- a strict subset of the [0, N) atom-row space (N =
    n_obj + n_propositions) -- so a [N, N] matrix safely holds every possible argument
    value without needing a separate object-id axis. Then for every (i, j) in
    ATOM_LABELS, (P_i.T @ P_j) is an [N, N] atom x atom matrix whose (a, b) entry counts
    argument values shared between a's position i and b's position j -- i.e. exactly the
    (a, b) pairs that get a directed edge labeled (i, j). Diagonal entries (a == b) are
    excluded (no self-loops). No two atoms can share the same argument pair via two
    DIFFERENT values (each atom has each argument slot exactly once), so each (a, b, i, j)
    the matmul finds is already a single instance -- but the four separate (i, j)
    categories can still co-occur for the same ordered (a, b) (e.g. two arity-2 atoms
    sharing both arguments), so those are merged into one 4-dim multi-hot edge_attr row
    per (a, b) via `np.maximum.at` (an OR-reduction over 0/1 values), never dropping a
    label. P_i.T @ P_j and P_j.T @ P_i are exact transposes of each other, so every a->b
    edge is paired with a b->a edge carrying the transposed label by construction.

    :param atoms: list of (predicate, args_tuple), as returned by env_to_atoms
    :return: (node_feats, edge_feats, edge_index)
        node_feats: shape [N, 6], one-hot over ATOM_PREDICATES
        edge_feats: shape [E, 4], multi-hot over ATOM_LABELS
        edge_index: shape [2, E]
    """
    n = len(atoms)
    node_feats = np.zeros((n, len(ATOM_PREDICATES)), dtype=np.float64)
    for i, (predicate, _args) in enumerate(atoms):
        node_feats[i, ATOM_PREDICATES.index(predicate)] = 1

    # incidence: for each position p in {1, 2}, (argument value, atom index) pairs
    obj_p1, atom_p1 = [], []
    obj_p2, atom_p2 = [], []
    for i, (_predicate, args) in enumerate(atoms):
        if len(args) >= 1:
            obj_p1.append(args[0])
            atom_p1.append(i)
        if len(args) >= 2:
            obj_p2.append(args[1])
            atom_p2.append(i)

    P1 = sp.csr_matrix((np.ones(len(obj_p1)), (obj_p1, atom_p1)), shape=(n, n))
    P2 = sp.csr_matrix((np.ones(len(obj_p2)), (obj_p2, atom_p2)), shape=(n, n))
    P = {1: P1, 2: P2}

    rows_list, cols_list, bits_list = [], [], []
    for bit, (i, j) in enumerate(ATOM_LABELS):
        mat = (P[i].T @ P[j]).tocoo()
        keep = (mat.row != mat.col) & (mat.data > 0)
        rows, cols = mat.row[keep], mat.col[keep]
        bits = np.zeros((rows.shape[0], 4), dtype=np.float64)
        bits[:, bit] = 1
        rows_list.append(rows)
        cols_list.append(cols)
        bits_list.append(bits)

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    bits = np.concatenate(bits_list, axis=0)

    if rows.shape[0] == 0:
        return node_feats, np.zeros((0, 4), dtype=np.float64), np.zeros((2, 0), dtype=np.int64)

    pair_ids = rows.astype(np.int64) * n + cols.astype(np.int64)
    unique_pairs, inverse = np.unique(pair_ids, return_inverse=True)
    edge_feats = np.zeros((unique_pairs.shape[0], 4), dtype=np.float64)
    np.maximum.at(edge_feats, inverse, bits)

    edge_index = np.stack([unique_pairs // n, unique_pairs % n]).astype(np.int64)

    return node_feats, edge_feats, edge_index


def _to_numpy(value):
    """
    Accepts a torch.Tensor on ANY device, or anything np.asarray already handles (a numpy
    array, a plain list). A bare `np.asarray` raises `TypeError: can't convert cuda:0
    device type tensor to numpy` on a CUDA tensor -- this is the one call site that needs
    to detach+move it to CPU first; numpy inputs (or CPU tensors) pass through
    `np.asarray` exactly as before, so this is a strict superset of the old behaviour.
    """
    if isinstance(value, th.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def graph_to_atoms(x, edge_index, edge_attr):
    """
    Inverse of atoms_to_graph: decodes an atom-encoding graph back into a flat atom list.
    Used by the planner (sage/domains/gym_taxi/simulator/planner.py's plan_atom), whose
    input graph may live on any device (CPU during evaluation, CUDA during GPU training
    -- see plan_atom's own docstring) -- hence _to_numpy rather than a bare np.asarray.

    A row's predicate is read off its one-hot in `x` (see ATOM_PREDICATES). Type atoms
    (predicate in {location, taxi, passenger}) are exactly rows 0..n_obj-1, and row k's
    single argument is k itself (see env_to_atoms -- "atom row k is object k"). For any
    other atom a, its position-i argument (i in {1, 2}) is the object id of whichever
    type atom b has an edge a->b labeled (i, 1) -- (i, 1) rather than (i, i) because a
    type atom's own sole argument is always at position 1, regardless of which position
    of `a` it fills.

    :param x: node features, shape [N, 6] -- see ATOM_PREDICATES. torch.Tensor (any
        device) or numpy array.
    :param edge_index: shape [2, E]. torch.Tensor (any device) or numpy array.
    :param edge_attr: shape [E, 4], multi-hot over ATOM_LABELS. torch.Tensor (any device)
        or numpy array.
    :return: list of (predicate, args_tuple), in row order -- see atoms_to_graph
    """
    x = _to_numpy(x)
    edge_index = _to_numpy(edge_index)
    edge_attr = _to_numpy(edge_attr)
    n = x.shape[0]

    predicate_idx = np.argmax(x, axis=1)
    predicates = [ATOM_PREDICATES[i] for i in predicate_idx]
    is_type_atom = np.array([p in _TYPE_PREDICATES for p in predicates], dtype=bool)

    src, dst = edge_index[0], edge_index[1]
    dst_is_type = is_type_atom[dst]

    # ATOM_LABELS = [(1,1), (1,2), (2,1), (2,2)] -> bit index 0 and 2 are the (i, 1) labels
    label_pos1 = dst_is_type & (edge_attr[:, 0] > 0)  # (1, 1): a's position-1 arg
    label_pos2 = dst_is_type & (edge_attr[:, 2] > 0)  # (2, 1): a's position-2 arg

    args1 = np.full(n, -1, dtype=np.int64)
    args1[src[label_pos1]] = dst[label_pos1]
    args2 = np.full(n, -1, dtype=np.int64)
    args2[src[label_pos2]] = dst[label_pos2]

    atoms = []
    for i in range(n):
        if is_type_atom[i]:
            atoms.append((predicates[i], (i,)))
            continue
        args = []
        if args1[i] != -1:
            args.append(int(args1[i]))
        if args2[i] != -1:
            args.append(int(args2[i]))
        atoms.append((predicates[i], tuple(args)))

    return atoms


def env_to_atom_graph(env):
    """
    Converts taxi world state from env to the atom-encoding graph convention (Horcik et
    al., AAAI-25, Def. 2) -- see atoms_to_graph for the encoding itself. Returns the same
    5-tuple shape as env_to_graph (node_feats, edge_feats, edge_index, mask,
    global_feats) -- no WL colours (this branch's env_to_graph/env_to_vilg_graph don't
    compute WL either -- see cell4-wl-vilg for that wiring, not present here).

    Only env.planning=True is implemented: mask is True on type-atom rows only (exactly
    rows 0..n_obj-1 -- see env_to_atoms), matching the "every object is a legal action
    target, propositions never are" contract env_to_vilg_graph's planning=True branch
    already has. The env.planning=False (masked-actions) path isn't needed for this
    project and is left unimplemented rather than silently guessed at.

    :param env: taxi world state in env format
    :return: node_feats, edge_feats, edge_index, mask, global_feats
    """
    if not env.planning:
        raise NotImplementedError(
            "env_to_atom_graph only supports env.planning=True; the env.planning=False "
            "(masked action space) path is not needed for this project."
        )

    atoms = env_to_atoms(env)
    node_feats, edge_feats, edge_index = atoms_to_graph(atoms)

    n_obj = len(env.graph.nodes)
    mask = np.zeros(node_feats.shape[0], dtype=bool)
    mask[:n_obj] = True

    global_feats = np.zeros(EMB_SIZE, dtype=np.float64)
    global_feats[0] = (env.timeout - env.time) / env.timeout

    return node_feats, edge_feats, edge_index, mask, global_feats


def env_to_atom_json(env):
    """
    Converts taxi world state from env to json representation, using the atom-encoding
    convention (see env_to_atom_graph) instead of Oracle-SAGE's object-only convention.

    :param env: taxi world state in env format
    :return: taxi world state in json format
    """
    return graph_to_json(*env_to_atom_graph(env))


def env_to_image(env):
    """
    Converts taxi world state from env to image representation

    :param env: taxi world state in env format
    :return: taxi world state in image format
    """

    channels = 4
    image_size = 2 * env.size - 1
    image = np.zeros((channels, image_size, image_size), dtype=np.uint8)
    fill_map(env, image[0], image_size)

    image[1][env.taxi.location] = 1
    for p in env.passengers.values():
        if p.location !=0:
            image[2][p.location] = 1
        image[3][p.destination] = 1

    final =  np.transpose(image, (0, 2, 1))  # swapping x&y

    return resize_image(final,84)

def fill_map(env, image, image_size):
    nodes = [(2 * x, 2 * y) for (x, y) in env.graph.nodes]
    edges = [(x1 + x2, y1 + y2) for ((x1, y1), (x2, y2)) in env.graph.edges]
    for n in nodes:
        image[n] = 1
    for n in edges:
        image[n] = 1
    # for the odd,odd coordinates, need to interpolate values.
    # Will be passable if at least 3 of it's neighbours are passable
    for x in range(1, image_size, 2):
        for y in range(1, image_size, 2):
            passable_neighbours = (
                image[x - 1][y] + image[x + 1][y] + image[x][y - 1] + image[x][y + 1]
            )
            image[x][y] = 1 if passable_neighbours > 2 else 0

def resize_image(img, size):
    """
    Modifies image dimensions . 
    :param img: taxi world state in image format
    :param size: size for converted 
    :return: taxi world state in image format of specified size
    """
    resized = cv2.resize(
        np.transpose(img, (1, 2, 0)), (size, size), interpolation=cv2.INTER_AREA
    )

    if len(resized.shape) == 2:
        return np.expand_dims(resized, axis=0)

    return np.transpose(resized, (2, 0, 1))



# def json_to_image(js):
#     """
#     Converts taxi world state from json to image representation

#     :param js: taxi world state in json format
#     :return: taxi world state in image format
#     """
#     env = json.loads(js)
#     channels = 5 if len(env["fuel_stations"]) > 0 else 4
#     image_size = 2 * env["n"] - 1
#     image = np.zeros((channels, image_size, image_size), dtype=np.uint8)
#     fill_map(env, image[0], image_size)

#     image[1][tuple(2 * i for i in env["taxi"]["location"])] = 1
#     for p in env["passenger"]:
#         if not p["in_taxi"]:
#             image[2][tuple(2 * i for i in p["location"])] = 1
#         image[3][tuple(2 * i for i in p["destination"])] = 1
#     for f in env["fuel_stations"]:
#         image[4][tuple(2 * i for i in f["location"])] = f["price"]

#     return np.transpose(image, (0, 2, 1))  # swapping x&y
