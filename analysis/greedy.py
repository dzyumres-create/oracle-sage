"""Greedy reference policy for city taxi, copied verbatim (no edits) from
sage/domains/utils/build_wl_vocab.py on branch cell6-wl-atom, commit ffb504f
(last commit touching that file; branch tip 46260c9):
sample_action, road_graph, next_hop, greedy_action, epsilon_greedy_action.

These act on the simulator directly (env.sim) and return one primitive action
(a node id for env.step). greedy_action is deterministic given the sim state:
carrying -> next hop towards the destination (drop off on arrival); empty ->
next hop towards the nearest waiting passenger by road hops (pick up on
arrival); no passengers -> stay. Note sample_action draws from the GLOBAL numpy
RNG (np.random.choice); seed it explicitly when reproducibility matters.

Wherever rollout values are reported: they continue with greedy_action, so they
measure a goal's value under greedy continuation, not under a model's own policy.
"""
import numpy as np
import networkx as nx


def sample_action(sim):
    """
    Picks a uniformly random action from the set of actions env.step() can
    safely execute this turn: a single-hop move to a location the taxi's
    current location actually has an outgoing edge to, the taxi's own node
    (a dropoff attempt - always safe, whether or not it succeeds), the
    taxi's current location itself (an explicit no-op move), or any current
    passenger node (a pickup attempt - always safe). See "Investigation
    notes" above for why non-adjacent location actions are avoided.

    Under graph_convention="vilg", a delivered passenger's node (and its
    destination(pid, dest) edge) deliberately survives in sim.graph purely
    to carry goal status (see env_to_vilg_graph/attempt_dropoff's "vilg"
    branch) - so it can still turn up as a successor of whatever location is
    its destination, even though it's no longer in sim.passengers and
    attempt_pickup would KeyError on it. Excluded the same way the real
    action mask already does (env_to_vilg_graph's `selectable` list:
    "(not is_passenger) or (nid in env.passengers)") - this can never fire
    under oracle_sage, since a delivered passenger's node is removed from
    the graph entirely there (attempt_dropoff's non-vilg branch), not kept
    alive, so this check is a no-op for that convention.

    :param sim: a TaxiWorldSimulator instance (env.sim)
    :return: a valid node id to pass to env.step()
    """
    taxi_location = sim.taxi.location
    candidates = {
        c for c in sim.graph.successors(taxi_location)
        if sim.graph.nodes[c]["attr"] != [0, 0, 1] or c in sim.passengers
    }
    candidates.add(taxi_location)
    candidates.add(0)  # taxi's own node -> dropoff attempt
    return int(np.random.choice(list(candidates)))


def road_graph(sim):
    """
    An UNDIRECTED nx.Graph of location<->location road edges read directly off
    `sim.graph` (edge_attr[0]==1, the is_road one-hot column - the same test the
    oracle_sage planner's own graph_to_networkx uses), independent of graph_convention
    (this reads the simulator's own internal graph, not any converted representation).
    Used by greedy_action/next_hop below for shortest-path navigation - built once per
    episode (the road layout is fixed for the episode's lifetime; only taxi/passenger
    state changes step to step) rather than recomputed every step.
    """
    G = nx.Graph()
    for u, v, attr in sim.graph.edges(data=True):
        if attr["attr"][0] == 1:
            G.add_edge(u, v)
    return G


def next_hop(G, src, dst):
    """One step along nx.shortest_path(G, src, dst) - None if src==dst already."""
    if src == dst:
        return None
    path = nx.shortest_path(G, src, dst)
    return path[1]


def greedy_action(sim, G):
    """
    A simple nearest-passenger-then-destination greedy policy (the same one
    verification check C used to force real pickups/deliveries within a bounded step
    budget - pure random action sampling rarely delivers on a size-20 maze): if
    carrying a passenger, head for their destination (dropoff-attempt once there); else
    head for the nearest still-waiting passenger by road-graph hop distance
    (pickup-attempt once there); if no passengers exist yet, hold position. Unlike
    sample_action, this is deliberately NOT uniformly random - it exists to generate a
    corpus with real pickup/delivery structure, not to explore broadly.

    :param sim: a TaxiWorldSimulator instance (env.sim)
    :param G: this episode's road_graph(sim) (passed in, not recomputed every call)
    :return: a valid node id to pass to env.step()
    """
    if sim.taxi.passenger is not None:
        pid = sim.taxi.passenger
        dest = sim.passengers[pid].destination
        if sim.taxi.location == dest:
            return 0
        return next_hop(G, sim.taxi.location, dest)
    if sim.passengers:
        pid = min(
            sim.passengers,
            key=lambda p: nx.shortest_path_length(G, sim.taxi.location, sim.passengers[p].location),
        )
        loc = sim.passengers[pid].location
        if sim.taxi.location == loc:
            return pid
        return next_hop(G, sim.taxi.location, loc)
    return sim.taxi.location


def epsilon_greedy_action(sim, G, eps, rng):
    """
    greedy_action, but with probability `eps` a uniformly-random legal action
    (sample_action) is taken instead - so a corpus built from this policy isn't
    entirely confined to the greedy policy's own on-path states (which are a narrow,
    systematically-biased slice of the reachable state space: always making progress
    towards a delivery, never "wasted" moves, never far from a passenger/destination).

    :param sim: a TaxiWorldSimulator instance (env.sim)
    :param G: this episode's road_graph(sim)
    :param eps: exploration probability, in [0, 1]
    :param rng: a numpy RandomState, used ONLY for the eps coin-flip (kept separate
        from sample_action's own bare `np.random.choice` calls, which read the global
        numpy random state - so this doesn't perturb sample_action's own reproducibility
        contract for callers that also seed the global state directly)
    :return: a valid node id to pass to env.step()
    """
    if rng.random_sample() < eps:
        return sample_action(sim)
    return greedy_action(sim, G)

