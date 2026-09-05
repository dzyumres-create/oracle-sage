

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
