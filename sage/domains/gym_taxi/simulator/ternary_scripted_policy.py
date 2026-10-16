"""
.. module:: ternary_scripted_policy
   :synopsis: A scripted policy for the ternary-predicate taxi world
   (TernaryTaxiWorldSimulator): nearest waiting passenger, pick up, then the passenger's
   destination by the picked_up_at/request join, drop off -- mixed with uniform-random
   legal actions. Not a training component: it drives the WL-vocab corpus
   (sage/domains/utils/build_wl_vocab.py), the analysis scripts and the tests.

   Moved here unchanged from tests/test_ternary_world.py, which re-exports every name.
"""
import networkx as nx


def legal_actions(sim):
    """Every action guaranteed not to raise KeyError from the current state: a dropoff
    attempt, a pickup attempt on any live passenger, staying put, or moving along any
    real road edge out of the taxi's current location."""
    actions = [sim.taxi.node, sim.taxi.location]
    actions.extend(sim.passengers.keys())
    actions.extend(sim.roads.successors(sim.taxi.location))
    return actions


def _road_graph_from_facts(facts):
    """Builds a plain nx.DiGraph of locations from facts() alone -- deliberately not
    reading sim.roads directly, so this exercises exactly what a real converter/planner
    downstream would have to work with."""
    graph = nx.DiGraph()
    for predicate, args in facts:
        if predicate == "location":
            graph.add_node(args[0])
    for predicate, args in facts:
        if predicate == "adjacent":
            graph.add_edge(args[0], args[1])
    return graph


def _taxi_location_from_facts(facts):
    for predicate, args in facts:
        if predicate == "in" and args[0] == 0:
            return args[1]
    raise ValueError("facts() has no in(0, l) fact for the taxi")


def _carried_passenger_from_facts(facts):
    """Returns the carried passenger's id, or None -- in(p, 0) with p != 0."""
    for predicate, args in facts:
        if predicate == "in" and args[0] != 0 and args[1] == 0:
            return args[0]
    return None


def _waiting_passengers_from_facts(facts):
    """{pid: location} for every passenger currently at a location (not carried)."""
    return {args[0]: args[1] for predicate, args in facts if predicate == "in" and args[0] != 0 and args[1] != 0}


def destination_via_join(facts, pid):
    """
    The only legitimate way to find where a carried passenger is going: join their
    picked_up_at fact against their own request facts on origin == picked_up_at.
    Returns None if pid hasn't been picked up (no picked_up_at fact).
    """
    picked_up_at = None
    for predicate, args in facts:
        if predicate == "picked_up_at" and args[0] == pid:
            picked_up_at = args[1]
            break
    if picked_up_at is None:
        return None
    for predicate, args in facts:
        if predicate == "request" and args[0] == pid and args[1] == picked_up_at:
            return args[2]
    raise ValueError(f"no request({pid}, {picked_up_at}, ?) fact -- picked_up_at with no matching request")


def scripted_policy(sim, rng, random_prob=0.2):
    """
    A simple, reusable scripted policy for exercising realistic play (not just
    uniform-random wandering): plan to the nearest waiting passenger, pick them up,
    plan to their destination via the facts' picked_up_at/request join, drop off --
    mixed with `random_prob` uniform-random legal actions, so the play distribution
    isn't perfectly deterministic and still covers edge cases (failed pickups/dropoffs,
    idle moves) a pure planner would never produce.

    Reused by the Part 2 collision-counter's state sampler, so any change here changes
    what states that counter sees -- keep this in sync with facts()'s contract, not
    with sim internals (this deliberately reads only sim.facts(), sim.taxi.node and
    sim.passengers.keys() -- the id-membership check needed to issue a pickup/dropoff
    action, not to plan one).

    :param sim: a TernaryTaxiWorldSimulator
    :param rng: a random.Random instance (NOT sim.random -- policy randomness must
        stay independent of the simulator's own RNG stream, or determinism tests that
        replay a fixed action sequence would silently start depending on this
        function's internals)
    :param random_prob: probability of ignoring the plan and taking a uniform-random
        legal action instead
    :return: an action (node id) for sim.act()/sim._apply()
    """
    if rng.random() < random_prob:
        return rng.choice(legal_actions(sim))

    facts = sim.facts()
    taxi_location = _taxi_location_from_facts(facts)
    carried_pid = _carried_passenger_from_facts(facts)

    if carried_pid is not None:
        destination = destination_via_join(facts, carried_pid)
        if taxi_location == destination:
            return sim.taxi.node  # attempt dropoff
        road_graph = _road_graph_from_facts(facts)
        path = nx.shortest_path(road_graph, taxi_location, destination)
        return path[1]

    waiting = _waiting_passengers_from_facts(facts)
    if not waiting:
        return rng.choice(legal_actions(sim))

    road_graph = _road_graph_from_facts(facts)
    lengths = nx.single_source_shortest_path_length(road_graph, taxi_location)
    reachable = {pid: loc for pid, loc in waiting.items() if loc in lengths}
    if not reachable:
        return rng.choice(legal_actions(sim))
    nearest_pid = min(reachable, key=lambda pid: lengths[reachable[pid]])
    nearest_loc = reachable[nearest_pid]
    if taxi_location == nearest_loc:
        return nearest_pid  # attempt pickup
    path = nx.shortest_path(road_graph, taxi_location, nearest_loc)
    return path[1]
