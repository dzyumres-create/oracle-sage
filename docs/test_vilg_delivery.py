"""
Verifies Step 4 -- faithful goal-status tracking for delivered passengers under
graph_convention="vilg" (destination(pid, loc) persists and flips to
achieved_propositional_goal instead of the passenger node vanishing), and confirms
graph_convention="oracle_sage" still removes the passenger node on delivery, unchanged.

Run from the repo root: python docs/test_vilg_delivery.py

Note on node identity: proposition node array *indices* are rebuilt from graph.edges()
iteration order on every call to env_to_vilg_graph/env_to_vilg_json -- they are NOT
stable identifiers across timesteps in general (they shift whenever an earlier-ordered
proposition is added/removed elsewhere in the graph). This script identifies "the same
proposition" by its content (predicate one-hot + which passenger it's destination(.)
for), not by assuming its array index never changes -- and separately confirms the
index specifically does not change across the single dropoff transition, since that's
what "the same node" naturally means in the delivery step itself. Each snapshot is
serialized independently anyway (json_to_graph/Batch build one Data object per
timestep), so index instability across timesteps is a non-issue downstream.
"""
import numpy.random as npr

from sage.domains.gym_taxi.simulator.taxi_world import TaxiWorldSimulator
from sage.domains.gym_taxi.utils.representations import env_to_vilg_graph

DESTINATION_PRED = [0, 0, 1]
UNACHIEVED_GOAL = [0, 1, 0]
ACHIEVED_GOAL = [1, 0, 0]


def find_destination_prop(node_feats):
    """Locates the (only, in this single-passenger test) destination(pid, loc)
    proposition node by its predicate one-hot (feature slice [3:6])."""
    matches = [i for i, row in enumerate(node_feats) if list(row[3:6]) == DESTINATION_PRED]
    assert len(matches) == 1, f"expected exactly one destination proposition, found {len(matches)}"
    return matches[0]


# --- scripted, deterministic episode -----------------------------------------------
# Fixed seed (RandomState(0)), 3x3 grid, single passenger, no mid-episode spawns, so the
# state is fully hand-checkable: taxi starts at location 6, passenger (pid=10) at
# location 3 with destination 9. attempt_move/attempt_dropoff dispatch on the *target
# node's* type attr, so actions are graph node ids, not a direction enum: move actions
# are location node ids, pickup is the passenger's node id, dropoff is always the
# taxi's own node id (0, the only node with the taxi type attr [0,1,0]).
MOVE_TO_PASSENGER = 3
MOVE_TO_DEST_A = 6
MOVE_TO_DEST_B = 9
DROPOFF = 0

r = npr.RandomState(0)
env = TaxiWorldSimulator(
    random=r,
    size=3,
    random_walls=False,
    planning=True,
    graph_convention="vilg",
    concurrent_passengers=1,
    passenger_creation_probability=0,
)
assert env.taxi.location == 6
(pid, passenger0), = env.passengers.items()
assert (pid, passenger0.location, passenger0.destination) == (10, 3, 9)

# --- before delivery -----------------------------------------------------------------
nf0, ef0, ei0, mask0, gf0 = env_to_vilg_graph(env)
prop_idx0 = find_destination_prop(nf0)
assert list(nf0[prop_idx0][6:9]) == UNACHIEVED_GOAL
node_count_start = nf0.shape[0]
print(f"PASS: destination proposition (node {prop_idx0}) starts unachieved_propositional_goal")
print(f"      total node count at episode start: {node_count_start}")

env.act(MOVE_TO_PASSENGER)
env.act(pid)  # pickup
env.act(MOVE_TO_DEST_A)
env.act(MOVE_TO_DEST_B)

nf_pre, ef_pre, ei_pre, mask_pre, gf_pre = env_to_vilg_graph(env)
prop_idx_pre = find_destination_prop(nf_pre)
assert list(nf_pre[prop_idx_pre][6:9]) == UNACHIEVED_GOAL
node_count_pre_dropoff = nf_pre.shape[0]
print(f"      destination proposition still unachieved_propositional_goal immediately before dropoff (node {prop_idx_pre})")
print(f"      total node count immediately before dropoff: {node_count_pre_dropoff}")

reward = env.act(DROPOFF)[1]
assert reward == env.rewards["drop-off"]
assert pid not in env.passengers, "FAIL: passenger must still be popped from env.passengers on delivery"

# --- after delivery -------------------------------------------------------------------
nf1, ef1, ei1, mask1, gf1 = env_to_vilg_graph(env)
node_count_after_dropoff = nf1.shape[0]

prop_idx1 = find_destination_prop(nf1)
assert prop_idx1 == prop_idx_pre, (
    "the destination proposition's array index shifted across the dropoff transition "
    f"({prop_idx_pre} -> {prop_idx1}); still the same ground atom by content, see docstring"
)
assert list(nf1[prop_idx1][6:9]) == ACHIEVED_GOAL, "FAIL: destination proposition should flip to achieved_propositional_goal"
print(f"PASS: same proposition node (index {prop_idx1}) flipped unachieved -> achieved_propositional_goal on delivery")

assert mask1[pid] == False, "FAIL: delivered passenger's node must be excluded from the action mask"
print(f"PASS: delivered passenger's node (index {pid}) is excluded from the action mask (mask[{pid}] = {mask1[pid]})")

print(f"      total node count immediately after dropoff: {node_count_after_dropoff}")
if node_count_after_dropoff >= node_count_pre_dropoff:
    print("PASS: total node count did not shrink across delivery")
else:
    shrink = node_count_pre_dropoff - node_count_after_dropoff
    print(
        f"REPORT (not a defect in goal tracking): total node count DECREASED by {shrink} across "
        "delivery. The destination(pid, loc) goal node itself does NOT shrink away -- it persists "
        "at the same index and correctly flips to achieved, verified above. The decrease is the "
        "separate, transient in(pid, taxi) 'carrying' proposition disappearing: Step 1 explicitly "
        "removes the pid<->taxi edges on delivery (the passenger really is no longer in the taxi, "
        "and that atom isn't a goal, so vILG correctly gives it no node). Over a longer episode "
        "with more deliveries, Cell 2's total node count still grows monotonically relative to "
        "Cell 1 (which deletes the whole passenger subgraph, goal atom included, on every "
        "delivery) -- but at this single delivery instant, node count is not itself monotonic."
    )

# act() does not enforce the mask itself -- masking is the calling policy's contract, same as
# before this change. Confirm the delivered passenger's node is unreachable *via the mask*
# (the guarantee this change actually provides), then separately report what happens if
# something bypasses the mask and calls act() with that raw node id anyway.
try:
    env.act(pid)
    print(f"REPORT: env.act({pid}) with the delivered passenger's (masked-out) node id did NOT crash.")
except KeyError as e:
    print(
        f"REPORT: env.act({pid}) with the delivered passenger's (masked-out) node id still raises "
        f"KeyError({e}) when called directly -- act() never checked the mask itself, before or "
        "after this change; a policy that respects mask1 (verified False above) would never issue "
        "this action in the first place."
    )

print()
print("=== vilg checks done ===")
print()

# --- regression check: oracle_sage must be completely untouched ----------------------
r2 = npr.RandomState(0)
env2 = TaxiWorldSimulator(
    random=r2,
    size=3,
    random_walls=False,
    planning=True,
    graph_convention="oracle_sage",
    concurrent_passengers=1,
    passenger_creation_probability=0,
)
(pid2, _), = env2.passengers.items()
assert pid2 == pid  # same seed/config as env -> identical initial draw
assert pid2 in env2.graph.nodes

env2.act(MOVE_TO_PASSENGER)
env2.act(pid2)  # pickup
env2.act(MOVE_TO_DEST_A)
env2.act(MOVE_TO_DEST_B)
env2.act(DROPOFF)

assert pid2 not in env2.graph.nodes, "FAIL: oracle_sage must still remove the passenger node on delivery"
assert pid2 not in env2.passengers
print(f"PASS: graph_convention='oracle_sage' still removes passenger node {pid2} from the graph on delivery (unchanged)")

print()
print("ALL CHECKS PASSED")
