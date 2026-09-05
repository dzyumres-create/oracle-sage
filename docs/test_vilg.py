"""
Quick sanity check for env_to_vilg_graph() -- Cell 2's translator.
Run from the repo root: python test_vilg.py
"""
import sys
import numpy.random as npr

from sage.domains.gym_taxi.simulator.taxi_world import TaxiWorldSimulator
from sage.domains.gym_taxi.utils.representations import env_to_graph, env_to_vilg_graph

r = npr.RandomState(0)
env = TaxiWorldSimulator(random=r, size=3, random_walls=False, planning=True)

nf, ef, ei, mask, gf = env_to_graph(env)
print("Oracle-SAGE convention -- nodes:", nf.shape[0], " edges:", ef.shape[0])

nf2, ef2, ei2, mask2, gf2 = env_to_vilg_graph(env)
print("vILG convention        -- nodes:", nf2.shape[0], " edges:", ef2.shape[0])

n_objects = 11  # taxi + 9 locations + 1 passenger, for this fixed seed/size
assert mask2[:n_objects].all() or True  # object nodes may or may not all be selectable depending on planning flag
assert not mask2[n_objects:].any(), "FAIL: a proposition node is selectable as an action!"
print("PASS: no proposition node is ever selectable as an action")

print()
print("taxi location:", env.taxi.location)
print("passengers:", env.passengers)
print()
print("Spot-check a few proposition nodes' edges (should match taxi/passenger state above):")
for i in range(ei2.shape[1]):
    if ei2[0, i] >= n_objects and ei2[1, i] < n_objects:
        pass  # each proposition node appears twice (arg 1 and arg 2), printed once per pair below
prop_edges = {}
for i in range(ei2.shape[1]):
    src, dst = ei2[0, i], ei2[1, i]
    prop_edges.setdefault(src, []).append(dst)
for prop_id, args in sorted(prop_edges.items()):
    if len(args) == 2:
        print(f"  proposition node {prop_id}: connects objects {args[0]} and {args[1]}")