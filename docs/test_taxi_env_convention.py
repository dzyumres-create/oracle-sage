"""
Sanity check for Step 3: GraphTaxiEnv's graph_convention flag.
Run from the repo root: python docs/test_taxi_env_convention.py
"""
import json

from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv

# default (oracle_sage) convention -- node feature dimension should be 3
env_default = GraphTaxiEnv(representation="graph", scenario="original")
obs_default = json.loads(env_default.reset())
node_dim_default = len(obs_default["node_feats"][0])
assert node_dim_default == 3, f"FAIL: default convention node dim {node_dim_default}, expected 3"
print(f"PASS: default (graph_convention omitted) node feature dim == {node_dim_default}")

# explicit oracle_sage convention -- node feature dimension should be 3
env_oracle_sage = GraphTaxiEnv(representation="graph", scenario="original", graph_convention="oracle_sage")
obs_oracle_sage = json.loads(env_oracle_sage.reset())
node_dim_oracle_sage = len(obs_oracle_sage["node_feats"][0])
assert node_dim_oracle_sage == 3, f"FAIL: oracle_sage convention node dim {node_dim_oracle_sage}, expected 3"
print(f"PASS: graph_convention='oracle_sage' node feature dim == {node_dim_oracle_sage}")

# vilg convention -- node feature dimension should be 9
env_vilg = GraphTaxiEnv(representation="graph", scenario="original", graph_convention="vilg")
obs_vilg = json.loads(env_vilg.reset())
node_dim_vilg = len(obs_vilg["node_feats"][0])
assert node_dim_vilg == 9, f"FAIL: vilg convention node dim {node_dim_vilg}, expected 9"
print(f"PASS: graph_convention='vilg' node feature dim == {node_dim_vilg}")

# invalid convention -- should raise ValueError
try:
    GraphTaxiEnv(representation="graph", scenario="original", graph_convention="bogus")
    raise AssertionError("FAIL: invalid graph_convention did not raise ValueError")
except ValueError as e:
    print(f"PASS: invalid graph_convention raised ValueError: {e}")

print()
print("ALL CHECKS PASSED")
