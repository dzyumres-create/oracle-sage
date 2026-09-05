"""
Verifies the vilg-aware planner equivalents: graph_to_networkx_vilg, move_taxi_vilg,
remove_node_from_graph_vilg (sage/domains/gym_taxi/simulator/planner.py).

Ground truth is always compared against env.taxi.location / env.passengers directly --
never re-derived from the same graph that graph_to_networkx_vilg itself consumed, to
keep this an independent check.

Run from the repo root: python docs/test_planner_vilg.py
"""
import numpy.random as npr
import torch as th
from torch_geometric.data import Data

from sage.domains.gym_taxi.simulator.taxi_world import TaxiWorldSimulator
from sage.domains.gym_taxi.utils.representations import env_to_vilg_graph
from sage.domains.gym_taxi.simulator.planner import (
    graph_to_networkx_vilg,
    move_taxi_vilg,
    remove_node_from_graph_vilg,
)


def build_data(env):
    nf, ef, ei, mask, gf = env_to_vilg_graph(env)
    data = Data(
        x=th.as_tensor(nf),
        edge_index=th.as_tensor(ei),
        edge_attr=th.as_tensor(ef),
    )
    data.mask = th.as_tensor(mask)
    data.global_features = th.as_tensor(gf).unsqueeze(0)
    return data


def assert_state_matches_env(state, env, expect_passengers=True, note=""):
    assert state.taxi.location == env.taxi.location, (
        f"{note}: state.taxi.location {state.taxi.location} != env.taxi.location {env.taxi.location}"
    )
    # CRITICAL quirk (Step 0): the original graph_to_networkx hardcodes
    # Taxi(i, location, passenger=None) regardless of whether the taxi is actually
    # carrying someone -- downstream code (deliver_current_passenger via
    # passenger.location==state.taxi.node) relies on this exact behaviour, so
    # graph_to_networkx_vilg must reproduce it, not "fix" it. Confirm this explicitly,
    # including at a moment when the taxi genuinely IS carrying a passenger.
    assert state.taxi.passenger is None, (
        f"{note}: state.taxi.passenger should always be hardcoded None (matching the "
        f"original's quirk), got {state.taxi.passenger}"
    )

    if expect_passengers:
        state_by_node = {p.node: p for p in state.passengers}
        assert set(state_by_node.keys()) == set(env.passengers.keys()), (
            f"{note}: passenger node set mismatch -- state has {set(state_by_node.keys())}, "
            f"env.passengers has {set(env.passengers.keys())}"
        )
        for pid, ground_truth in env.passengers.items():
            got = state_by_node[pid]
            assert got.location == ground_truth.location, (
                f"{note}: passenger {pid} location {got.location} != ground truth {ground_truth.location}"
            )
            assert got.destination == ground_truth.destination, (
                f"{note}: passenger {pid} destination {got.destination} != ground truth {ground_truth.destination}"
            )


# =====================================================================================
# Part 1: graph_to_networkx_vilg matches ground truth, including through a pickup
# =====================================================================================
r = npr.RandomState(0)
env = TaxiWorldSimulator(
    random=r, size=3, random_walls=False, planning=True,
    graph_convention="vilg", concurrent_passengers=2, passenger_creation_probability=1,
)

state0 = graph_to_networkx_vilg(build_data(env))
assert_state_matches_env(state0, env, note="initial state")
print(f"PASS: initial state matches ground truth -- taxi@{state0.taxi.location}, "
      f"{len(state0.passengers)} passenger(s): {[(p.node, p.location, p.destination) for p in state0.passengers]}")

# drive one move (also triggers try_spawn_passenger -> a 2nd passenger should appear)
first_passenger_pid, first_passenger = next(iter(env.passengers.items()))
neighbours = list(env.graph[env.taxi.location])
move_target = next(x for x in neighbours if env.graph.nodes[x]['attr'] == [1, 0, 0])
env.act(move_target)
assert len(env.passengers) == 2, "expected a 2nd passenger to have spawned after one step"

state1 = graph_to_networkx_vilg(build_data(env))
assert_state_matches_env(state1, env, note="after one move (2 passengers now present)")
print(f"PASS: state matches ground truth with 2 passengers present -- taxi@{state1.taxi.location}")

# =====================================================================================
# Part 2: deliver one passenger for real, confirm exclusion from graph_to_networkx_vilg
# =====================================================================================
# navigate to first_passenger's location, pick up, navigate to their destination, drop off
def goto(env, target_location):
    while env.taxi.location != target_location:
        path = __import__("networkx").shortest_path(
            __import__("networkx").Graph(
                (u, v) for u, v, d in env.graph.edges(data=True) if d['attr'][:3] == [1, 0, 0]
            ),
            env.taxi.location, target_location,
        )
        env.act(path[1])

goto(env, first_passenger.location)
env.act(first_passenger_pid)  # pickup
assert env.taxi.passenger == first_passenger_pid

# while genuinely carrying, re-confirm the hardcoded-None quirk (per Step 0 / Part 1's helper)
state_carrying = graph_to_networkx_vilg(build_data(env))
assert state_carrying.taxi.passenger is None, "quirk check failed while genuinely carrying a passenger"
carried = [p for p in state_carrying.passengers if p.node == first_passenger_pid][0]
assert carried.location == env.taxi.node, (
    f"carried passenger's reconstructed location should read as the taxi's own node "
    f"({env.taxi.node}), matching oracle_sage's identical convention -- got {carried.location}"
)
print(f"PASS: while carrying, passenger {first_passenger_pid}'s reconstructed location "
      f"== taxi's own node ({carried.location}), matching oracle_sage's convention; "
      "state.taxi.passenger still correctly hardcoded None")

destination = env.passengers[first_passenger_pid].destination
goto(env, destination)
_, reward, _, _ = env.act(0)  # dropoff: node 0 is always the taxi's own node
assert reward == env.rewards["drop-off"]
assert first_passenger_pid not in env.passengers

state_after_delivery = graph_to_networkx_vilg(build_data(env))
delivered_pids = {p.node for p in state_after_delivery.passengers}
assert first_passenger_pid not in delivered_pids, (
    f"FAIL: delivered passenger {first_passenger_pid} still appears in graph_to_networkx_vilg's passengers list"
)
assert_state_matches_env(state_after_delivery, env, note="after real delivery")
print(f"PASS: delivered passenger {first_passenger_pid} correctly excluded from "
      f"graph_to_networkx_vilg's passengers list; remaining passenger(s) still match ground truth")


# =====================================================================================
# Part 3: move_taxi_vilg / remove_node_from_graph_vilg (planning-time projection)
# =====================================================================================
r2 = npr.RandomState(0)
env2 = TaxiWorldSimulator(
    random=r2, size=3, random_walls=False, planning=True,
    graph_convention="vilg", concurrent_passengers=2, passenger_creation_probability=0,
)
pid2, passenger2 = next(iter(env2.passengers.items()))

# --- move_taxi_vilg ---
data = build_data(env2)
old_location = env2.taxi.location
new_location = next(x for x in env2.graph[old_location] if env2.graph.nodes[x]['attr'] == [1, 0, 0])
move_taxi_vilg(data, 0, new_location)
projected_state = graph_to_networkx_vilg(data)
assert projected_state.taxi.location == new_location, (
    f"FAIL: move_taxi_vilg projected taxi to {projected_state.taxi.location}, expected {new_location}"
)
print(f"PASS: move_taxi_vilg correctly moved projected taxi location {old_location} -> {new_location}")

# --- remove_node_from_graph_vilg (mirrors a planning-time dropoff) ---
# set up a scenario where env2's actual taxi is already carrying the passenger, matching
# the precondition deliver_current_passenger/deliver_passenger call these functions under.
r3 = npr.RandomState(0)
env3 = TaxiWorldSimulator(
    random=r3, size=3, random_walls=False, planning=True,
    graph_convention="vilg", concurrent_passengers=2, passenger_creation_probability=0,
)
pid3, passenger3 = next(iter(env3.passengers.items()))
goto(env3, passenger3.location)
env3.act(pid3)  # pickup -- env3.taxi.passenger == pid3 now
assert env3.taxi.passenger == pid3

data3 = build_data(env3)
n_nodes_before = data3.x.shape[0]
move_taxi_vilg(data3, 0, passenger3.destination)
remove_node_from_graph_vilg(data3, pid3)

projected_state3 = graph_to_networkx_vilg(data3)
assert projected_state3.taxi.location == passenger3.destination, (
    f"FAIL: projected taxi location {projected_state3.taxi.location} != expected destination {passenger3.destination}"
)
projected_pids = {p.node for p in projected_state3.passengers}
assert pid3 not in projected_pids, (
    f"FAIL: remove_node_from_graph_vilg did not exclude passenger {pid3} from the projected state"
)
assert data3.x.shape[0] == n_nodes_before - 1, (
    f"FAIL: expected exactly one node removed (the stale 'in' proposition), "
    f"{n_nodes_before} -> {data3.x.shape[0]}"
)
print(f"PASS: remove_node_from_graph_vilg projected passenger {pid3} as delivered "
      f"(excluded from passengers list, node count {n_nodes_before} -> {data3.x.shape[0]}, "
      f"taxi projected to destination {passenger3.destination})")

# sanity: real env3 (untouched by the projection above, which only mutated the detached
# `data3` snapshot) still has the passenger live -- confirms projection didn't mutate env3
assert pid3 in env3.passengers
print("PASS: planning-time projection did not mutate the real env's own state")

print()
print("ALL CHECKS PASSED")
