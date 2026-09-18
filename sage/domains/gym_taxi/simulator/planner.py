from os import remove
import numpy as np
import torch as th
import networkx as nx
from torch_geometric.data import Data

from typing import List, NamedTuple

from sage.domains.gym_taxi.utils.representations import ATOM_PREDICATES, graph_to_atoms, atoms_to_graph


class Taxi(NamedTuple):
    node: int
    location: int
    passenger: int

class Passenger(NamedTuple):
    node: int
    location: int
    destination: int

class State(NamedTuple):
    graph: int
    taxi: Taxi
    passengers: List[Passenger]


class Planner:

    def __init__(self, graph_convention="oracle_sage"):
        self.graph_convention = graph_convention

    def plan(self,graph,goal):
        if self.graph_convention == "atom":
            return plan_atom(graph, goal)
        if self.graph_convention == "vilg":
            state = graph_to_networkx_vilg(graph)
        else:
            state = graph_to_networkx(graph)
        if goal==state.taxi.node:
            if state.taxi.passenger is not None:
                projection, actions =  deliver_current_passenger(graph,state,self.graph_convention)
            else:
                projection = graph
                actions = []
        elif goal in [p.node for p in state.passengers]:
            projection, actions = deliver_passenger(graph,state,goal,self.graph_convention)
        else:
            projection, actions = move(graph,state,goal,self.graph_convention)

        if actions == []:
            actions = [state.taxi.location]

        return increment_timer(projection,actions)

def graph_to_networkx(graph):
    nodes = graph.x.cpu().numpy()
    edges = graph.edge_index.cpu().numpy()
    edge_attr = graph.edge_attr.cpu().numpy()

    passengers=[]
    G = nx.Graph()
    for i,(l,t,p) in enumerate(nodes):
        if l:
            node_type = "location"
        elif t:
            node_type = "taxi"

            location = edges[1,np.logical_and(edges[0]==i,edge_attr[:,3]==1)]
            taxi=Taxi(i,location.item(),None)
        elif p:
            node_type = "passenger"
            edge_index = edges[0]==i
            passenger_edges =  edges[:,edge_index]
            passenger_edge_attributes = edge_attr[edge_index,:]
            #quite hacky here. There should always be two edges, one for location and destination, so all we need to do is figure out which one is first.
            if passenger_edge_attributes[0,1] == 1: #location is first
                location = passenger_edges[1,0]
                destination = passenger_edges[1,1]
            else: #destination is first
                location = passenger_edges[1,1]
                destination = passenger_edges[1,0]
            passenger = Passenger(i,location,destination)
            passengers.append(passenger)
        else:
            raise ValueError("Invalid node is neither location, taxi or passenger.")
        G.add_node(i,type=node_type)

    edge_indices = edge_attr[:,0]==1
    G.add_edges_from(edges.T[edge_indices])

    return State(G,taxi,passengers)


# --- vILG equivalents (Cell 2) --------------------------------------------------------
#
# Under the "vilg" graph_convention, every grounded proposition is its own node (see
# env_to_vilg_graph in utils/representations.py), so there is no longer a single direct
# object-object edge to read a relation off. Node feature columns are a fixed 9-dim
# layout: [0:3] object-type one-hot (location/taxi/passenger, same ordering as
# oracle_sage), [3:6] predicate one-hot (adjacent/in/destination), [6:9] goal-status
# one-hot (achieved_goal/unachieved_goal/achieved_nongoal) -- object nodes zero out
# [3:9], proposition nodes zero out [0:3]. Edge feature columns are a 2-dim one-hot for
# argument position: [1,0] = position 1 (the edge's "subject"), [0,1] = position 2 (the
# edge's "value"). See docs/vILG_taxi_translator_spec.md / Step 0 report for the full
# derivation of which predicate puts which argument in which position.

PRED_ADJACENT, PRED_IN, PRED_DESTINATION = 0, 1, 2
STATUS_ACHIEVED_GOAL, STATUS_UNACHIEVED_GOAL, STATUS_ACHIEVED_NONGOAL = 0, 1, 2


def graph_to_networkx_vilg(graph):
    x = graph.x.cpu().numpy()
    edge_index = graph.edge_index.cpu().numpy()
    edge_attr = graph.edge_attr.cpu().numpy()

    n_nodes = x.shape[0]
    is_object = np.all(x[:, 3:9] == 0, axis=1)

    pos1_mask = edge_attr[:, 0] == 1
    pos2_mask = edge_attr[:, 1] == 1

    # for each proposition node: its position-1 target (the "subject") and position-2
    # target (the "value") -- every proposition has exactly one of each.
    pos1_target = {}
    pos2_target = {}
    for e in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, e]), int(edge_index[1, e])
        if pos1_mask[e]:
            pos1_target[src] = dst
        elif pos2_mask[e]:
            pos2_target[src] = dst

    # index "in"/"destination" propositions by their position-1 target (the subject
    # object -- taxi or passenger) for O(1) lookup per object node below.
    in_prop_by_subject = {}
    destination_prop_by_subject = {}
    for i in range(n_nodes):
        if is_object[i]:
            continue
        pred = int(np.argmax(x[i, 3:6]))
        subject = pos1_target.get(i)
        if subject is None:
            continue
        if pred == PRED_IN:
            in_prop_by_subject[subject] = i
        elif pred == PRED_DESTINATION:
            destination_prop_by_subject[subject] = i

    passengers = []
    taxi = None
    G = nx.Graph()
    for i in range(n_nodes):
        if not is_object[i]:
            continue
        obj_type = int(np.argmax(x[i, 0:3]))  # 0=location, 1=taxi, 2=passenger
        if obj_type == 0:
            node_type = "location"
        elif obj_type == 1:
            node_type = "taxi"
            in_prop = in_prop_by_subject[i]
            location = pos2_target[in_prop]
            taxi = Taxi(i, location, None)
        elif obj_type == 2:
            node_type = "passenger"
            dest_prop = destination_prop_by_subject[i]
            status = int(np.argmax(x[dest_prop, 6:9]))
            if status == STATUS_ACHIEVED_GOAL:
                # delivered: excluded entirely, matching oracle_sage's behaviour where a
                # delivered passenger's node is removed and so naturally absent from
                # state.passengers (implementation plan Step 4).
                G.add_node(i, type=node_type)
                continue
            destination = pos2_target[dest_prop]
            in_prop = in_prop_by_subject[i]
            location = pos2_target[in_prop]
            passengers.append(Passenger(i, location, destination))
        else:
            raise ValueError("Invalid node is neither location, taxi or passenger.")
        G.add_node(i, type=node_type)

    # adjacency: each "adjacent" proposition connects two location objects via its
    # position-1/position-2 edges, mirroring the original's edge_attr[:,0]==1 filter.
    for i in range(n_nodes):
        if is_object[i]:
            continue
        pred = int(np.argmax(x[i, 3:6]))
        if pred == PRED_ADJACENT:
            G.add_edge(pos1_target[i], pos2_target[i])

    return State(G, taxi, passengers)


def move_taxi_vilg(graph, taxi, node):
    """Redirects the taxi's single "in" proposition's position-2 edge from its old
    location to the new one. Under vilg there is only one such proposition (the reverse
    direction that oracle_sage keeps for message-passing symmetry is never materialised
    as its own node -- see env_to_vilg_graph), so unlike move_taxi this only ever needs
    to update one edge, not a forward/backward pair."""
    pos1 = graph.edge_attr[:, 0] == 1
    prop_idx = graph.edge_index[0, th.logical_and(pos1, graph.edge_index[1] == taxi)][0]
    pos2 = graph.edge_attr[:, 1] == 1
    graph.edge_index[1, th.logical_and(pos2, graph.edge_index[0] == prop_idx)] = node


def remove_node_from_graph_vilg(graph, node):
    """Planning-time equivalent of a vilg dropoff (mirrors attempt_dropoff's "vilg"
    branch in taxi_world.py, implementation plan Step 4): does NOT delete the
    passenger's object node. Removes its (now-stale) "in" proposition node/edges, and
    flips its destination proposition's status to achieved_propositional_goal -- so
    planning-time lookahead stays consistent with what the real env actually does on
    delivery, rather than the oracle_sage node-deletion semantics."""
    pos1 = graph.edge_attr[:, 0] == 1
    subject_edges = th.logical_and(pos1, graph.edge_index[1] == node)
    candidate_props = graph.edge_index[0, subject_edges]

    is_in = graph.x[candidate_props, 3 + PRED_IN] == 1
    is_destination = graph.x[candidate_props, 3 + PRED_DESTINATION] == 1
    in_prop_idx = candidate_props[is_in][0]
    dest_prop_idx = candidate_props[is_destination][0]

    status = th.zeros(3, dtype=graph.x.dtype, device=graph.x.device)
    status[STATUS_ACHIEVED_GOAL] = 1
    graph.x[dest_prop_idx, 6:9] = status

    keep_nodes = th.ones(graph.x.shape[0], dtype=th.bool)
    keep_nodes[in_prop_idx] = False
    graph.x = graph.x[keep_nodes]

    keep_edges = th.logical_not(
        th.logical_or(graph.edge_index[0] == in_prop_idx, graph.edge_index[1] == in_prop_idx)
    )
    graph.edge_index = graph.edge_index[:, keep_edges]
    graph.edge_attr = graph.edge_attr[keep_edges]
    graph.edge_index = th.where(graph.edge_index > in_prop_idx, graph.edge_index - 1, graph.edge_index)


def deliver_current_passenger(graph,state,graph_convention="oracle_sage"):
    passenger = [p for p in state.passengers if p.location==state.taxi.node][0]
    move = find_path_to(state,state.taxi.location,passenger.destination)
    if graph_convention == "vilg":
        move_taxi_vilg(graph,state.taxi.node,passenger.destination)
        remove_node_from_graph_vilg(graph,passenger.node)
    else:
        move_taxi(graph,state.taxi.node,passenger.destination)
        remove_node_from_graph(graph,passenger.node)
    return graph,move + [state.taxi.node]

def deliver_passenger(graph,state,goal,graph_convention="oracle_sage"):
    passenger = [p for p in state.passengers if p.node==goal][0]
    if passenger.location == state.taxi.node:
        return deliver_current_passenger(graph,state,graph_convention)
    move1 = find_path_to(state,state.taxi.location,passenger.location)
    move2 = find_path_to(state,passenger.location,passenger.destination)
    if graph_convention == "vilg":
        move_taxi_vilg(graph,state.taxi.node,passenger.destination)
        remove_node_from_graph_vilg(graph,passenger.node)
    else:
        move_taxi(graph,state.taxi.node,passenger.destination)
        remove_node_from_graph(graph,passenger.node)
    return  graph,move1 + [passenger.node] + move2 + [state.taxi.node]

def move(graph,state,goal,graph_convention="oracle_sage"):
    actions = find_path_to(state,state.taxi.location,goal)
    if graph_convention == "vilg":
        move_taxi_vilg(graph,state.taxi.node,goal)
    else:
        move_taxi(graph,state.taxi.node,goal)
    return graph, actions

def find_path_to(state,start,end):
    path =  nx.shortest_path(state.graph,start,end)
    return path[1:]


def remove_node_from_graph(graph,node):

    graph.x = graph.x[:-1]
    #remove all incoming/outgoing edges
    edge_index = th.logical_or(graph.edge_index[0]==node,graph.edge_index[1]==node)
    graph.edge_index = graph.edge_index[:,th.logical_not(edge_index)]
    graph.edge_attr = graph.edge_attr[th.logical_not(edge_index)]
    #resort nodes after removed node
    graph.edge_index = th.where(graph.edge_index>node,graph.edge_index-1,graph.edge_index)

def move_taxi(graph,taxi,node):
    graph.edge_index[1,graph.edge_index[0]==taxi] = node
    taxi_edge_backwards_index = th.logical_and(graph.edge_index[1]==taxi,graph.edge_attr[:,3]==-1)
    graph.edge_index[0,taxi_edge_backwards_index] = node

def increment_timer(projection,actions):
    projection.global_features[0,0] = projection.global_features[0,0] - (len(actions)/2000)
    return projection, actions


# --- atom encoding (Step 3, Horcik et al. Def. 2) ---------------------------------------
#
# oracle_sage's move_taxi/remove_node_from_graph mutate graph.x/.edge_index/.edge_attr in
# place, relying on positional invariants that don't hold for the atom encoding (e.g.
# remove_node_from_graph blindly chops the LAST row off x, assuming the removed node is
# always the highest-indexed one; move_taxi identifies the tether's reverse copy via a
# single edge_attr[:,3]==-1 flag that has no atom-encoding analogue). Rather than
# reinventing an equally fragile positional scheme for atom-atom edges, this decodes the
# input graph back to a flat atom list (graph_to_atoms, already proven by Step 1's tests),
# computes the new logical state as a plain Python transformation of that list, then
# re-encodes from scratch via atoms_to_graph -- the exact same function env_to_atom_graph
# uses to build a live env's graph, so a projected graph and a live env_to_atom_graph
# output are constructed by identical code, not two parallel implementations that could
# drift apart. No tensor is ever edited in place; the returned Data is always freshly
# built, and the input `graph` is never touched (see plan_atom's docstring).

_ATOM_TYPE_PREDICATES = set(ATOM_PREDICATES[:3])  # {"location", "taxi", "passenger"}


def graph_to_state_atom(graph):
    """
    Decodes an atom-encoding graph into the same State(graph, taxi, passengers) shape
    graph_to_networkx/graph_to_networkx_vilg produce, so find_path_to and the goal-match
    conditions in plan_atom read identically to the oracle_sage/vilg branches. `state.graph`
    is an UNDIRECTED nx.Graph built directly from `adjacent` atoms (Horcik's atom encoding
    has no direct location-location edge otherwise) -- matching graph_to_networkx's own
    nx.Graph(...) (undirected) semantics, per this task's instruction.

    state.taxi.passenger is always None here -- this is not a simplification, it exactly
    mirrors graph_to_networkx's own real behaviour: Taxi(i, location.item(), None)
    hardcodes None regardless of whether a passenger is actually being carried. plan()'s
    "goal == taxi.node" branch's "if state.taxi.passenger is not None" check is therefore
    already always False for oracle_sage today; a carried passenger is delivered by
    selecting THEIR OWN node as goal instead (deliver_passenger's "already at taxi.node"
    case delegates to deliver_current_passenger regardless). Replicating this exactly
    (rather than "fixing" it) keeps atom's actions list identical to oracle_sage's for
    the same logical state, which is the whole point of this convention existing.

    :param graph: atom-encoding Data (x: [N,6], edge_index: [2,E], edge_attr: [E,4])
    :return: (state, atoms) -- state is the State(...) described above; atoms is the full
        flat atom list (graph_to_atoms' output), which plan_atom needs for re-encoding
    """
    atoms = graph_to_atoms(graph.x, graph.edge_index, graph.edge_attr)

    taxi_row = None
    passenger_rows = set()
    for predicate, args in atoms:
        if predicate == "taxi":
            taxi_row = args[0]
        elif predicate == "passenger":
            passenger_rows.add(args[0])

    road_graph = nx.Graph()
    taxi_location = None
    passenger_location = {}
    passenger_destination = {}
    for predicate, args in atoms:
        if predicate == "location":
            road_graph.add_node(args[0])
        elif predicate == "adjacent":
            road_graph.add_edge(args[0], args[1])
        elif predicate == "in":
            subject, container = args
            if subject == taxi_row:
                taxi_location = container
            elif subject in passenger_rows:
                passenger_location[subject] = container
        elif predicate == "destination":
            subject, dest = args
            if subject in passenger_rows:
                passenger_destination[subject] = dest

    taxi = Taxi(taxi_row, taxi_location, None)
    passengers = [
        Passenger(pid, passenger_location[pid], passenger_destination[pid])
        for pid in sorted(passenger_rows)
    ]
    return State(road_graph, taxi, passengers), atoms


def _move_taxi_atoms(atoms, taxi_row, new_location):
    """Pure (non-mutating): returns a NEW atom list with the taxi's `in` atom's location
    argument replaced by new_location. Node count/numbering is unchanged -- only one
    atom's args differ, mirroring move_taxi's effect without touching any tensor."""
    new_atoms = []
    for predicate, args in atoms:
        if predicate == "in" and args[0] == taxi_row:
            new_atoms.append((predicate, (taxi_row, new_location)))
        else:
            new_atoms.append((predicate, args))
    return new_atoms


def _deliver_atoms(atoms, taxi_row, passenger_row, destination):
    """
    Pure (non-mutating): returns a NEW atom list reflecting a delivery -- the atom-level
    equivalent of move_taxi(..., destination) + remove_node_from_graph(..., passenger).
    The taxi's `in` atom is redirected to `destination` (the projected graph jumps
    straight to the post-delivery state, exactly like oracle_sage -- no intermediate
    "passenger aboard" state is ever represented), and every atom mentioning
    passenger_row as an argument (its own type atom, its `in` atom, its `destination`
    atom) is dropped.

    Object ids are then renumbered contiguously over the survivors, in ascending order --
    exactly mirroring TaxiWorldSimulator.resort_passengers' own
    `{k: v for v, k in enumerate(sorted(self.graph.nodes))}` scheme. Locations and the
    taxi never move (they're always numbered below every passenger id, and removing one
    passenger only ever shifts higher-numbered passengers down by one -- same invariant
    resort_passengers relies on), so this only ever renumbers passenger ids above the
    removed one. Type atoms are placed first, sorted by their new id, so the result's
    row k is object k -- matching env_to_atoms' own convention exactly.
    """
    moved = _move_taxi_atoms(atoms, taxi_row, destination)
    kept = [(predicate, args) for predicate, args in moved if passenger_row not in args]

    n_obj_old = sum(1 for predicate, _args in atoms if predicate in _ATOM_TYPE_PREDICATES)
    surviving_objects = [obj for obj in range(n_obj_old) if obj != passenger_row]
    remap = {old: new for new, old in enumerate(surviving_objects)}

    type_atoms = []
    proposition_atoms = []
    for predicate, args in kept:
        new_args = tuple(remap[a] for a in args)
        if predicate in _ATOM_TYPE_PREDICATES:
            type_atoms.append((new_args[0], (predicate, new_args)))
        else:
            proposition_atoms.append((predicate, new_args))
    type_atoms.sort(key=lambda item: item[0])

    return [atom for _new_row, atom in type_atoms] + proposition_atoms


def _atoms_to_projection(atoms, reference_graph):
    """
    Re-encodes `atoms` into a fresh Data via atoms_to_graph -- the SAME function
    env_to_atom_graph uses -- then attaches mask (True on type-atom rows only, matching
    env_to_atom_graph's planning=True convention) and a CLONE of reference_graph's
    global_features (increment_timer mutates global_features in place; cloning keeps
    `reference_graph`, the caller's input, untouched). dtype/device match json_to_graph's
    real output exactly (x/edge_attr float32, edge_index int64, mask bool), on whichever
    device reference_graph itself lives on.
    """
    node_feats, edge_feats, edge_index = atoms_to_graph(atoms)
    n_obj = sum(1 for predicate, _args in atoms if predicate in _ATOM_TYPE_PREDICATES)
    mask = np.zeros(node_feats.shape[0], dtype=bool)
    mask[:n_obj] = True

    device = reference_graph.x.device
    projection = Data(
        x=th.as_tensor(node_feats, dtype=th.float32, device=device),
        edge_index=th.as_tensor(edge_index, dtype=th.long, device=device),
        edge_attr=th.as_tensor(edge_feats, dtype=th.float32, device=device),
    )
    projection.mask = th.as_tensor(mask, dtype=th.bool, device=device)
    projection.global_features = reference_graph.global_features.clone()
    return projection


def plan_atom(graph, goal):
    """
    The "atom" convention's Planner.plan implementation (Step 3): decode -> plan ->
    re-encode, never in-place tensor edits -- see this section's module-level comment.
    Branch structure and `actions` construction are a direct atom-level mirror of
    oracle_sage's plan()/deliver_current_passenger/deliver_passenger/move (down to the
    state.taxi.passenger-is-always-None quirk -- see graph_to_state_atom), so the
    returned `actions` list is identical to what the oracle_sage branch would return for
    the same logical state.

    :param graph: atom-encoding Data (never mutated)
    :param goal: node id the policy selected
    :return: (projection, actions) -- projection is a freshly-built Data, never `graph` itself
    """
    state, atoms = graph_to_state_atom(graph)
    taxi_row = state.taxi.node

    if goal == state.taxi.node:
        if state.taxi.passenger is not None:
            # Always unreachable today -- state.taxi.passenger is hardcoded None (see
            # graph_to_state_atom) -- kept only so this mirrors oracle_sage's branch
            # structure exactly, in case that upstream quirk is ever fixed.
            passenger = [p for p in state.passengers if p.location == state.taxi.node][0]
            move = find_path_to(state, state.taxi.location, passenger.destination)
            projected_atoms = _deliver_atoms(atoms, taxi_row, passenger.node, passenger.destination)
            actions = move + [state.taxi.node]
        else:
            projected_atoms = atoms
            actions = []
    elif goal in [p.node for p in state.passengers]:
        passenger = [p for p in state.passengers if p.node == goal][0]
        if passenger.location == state.taxi.node:
            move = find_path_to(state, state.taxi.location, passenger.destination)
            projected_atoms = _deliver_atoms(atoms, taxi_row, passenger.node, passenger.destination)
            actions = move + [state.taxi.node]
        else:
            move1 = find_path_to(state, state.taxi.location, passenger.location)
            move2 = find_path_to(state, passenger.location, passenger.destination)
            projected_atoms = _deliver_atoms(atoms, taxi_row, passenger.node, passenger.destination)
            actions = move1 + [passenger.node] + move2 + [state.taxi.node]
    else:
        actions = find_path_to(state, state.taxi.location, goal)
        projected_atoms = _move_taxi_atoms(atoms, taxi_row, goal)

    if actions == []:
        actions = [state.taxi.location]

    projection = _atoms_to_projection(projected_atoms, graph)
    return increment_timer(projection, actions)
