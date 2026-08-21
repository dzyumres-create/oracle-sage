from os import remove
import numpy as np
import torch as th
import networkx as nx

from typing import List, NamedTuple


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
