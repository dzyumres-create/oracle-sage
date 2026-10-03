"""
Is the GNN's output on CUDA affected by oracle_facts, or only by GPU run-to-run noise?

Uses tests/test_oracle_decoder.py's own setup (build_model: an oracle-decoder ternary env
and a PlanFeedback_A2C GNNPlanFeedbackPolicy). For several real states and each output
(latent node embeddings, latent globals, values) it prints the max absolute difference for

  (i)  control: two forwards WITHOUT oracle_facts on the same observation;
  (ii) oracle:  a forward WITH oracle_facts vs one without, same observation;

each repeated --repeats times. If the GPU were deterministic, (i) would be exactly 0; if
oracle_facts reached the GNN, (ii) would exceed (i).

Exit status: 1 if (i) is exactly 0 everywhere while (ii) is non-zero somewhere (the GNN
is deterministic and oracle_facts changes its output); 0 otherwise.

Run from the repo root:

    PYTHONPATH=. python analysis/oracle_cuda_determinism.py --device cuda
"""
import argparse
import random as pyrandom
import sys

import torch as th
import torch_geometric

# The test module installs the gym-0.26 randint shim and provides the setup reused here.
from tests.test_oracle_decoder import build_model, sim_json
from tests.test_ternary_planner import make_ternary_sim
from tests.test_ternary_world import scripted_policy
from sage.domains.gym_taxi.utils.ternary_representations import (
    json_to_ternary_graph_object,
    json_to_ternary_graph_object_oracle,
)

OUTPUTS = ("latent nodes", "latent globals", "values")


def real_observations(n_states):
    """One observation per state, each a different seed played for a different number
    of scripted steps -- in the vec-env shape the converter expects ([[json]])."""
    observations = []
    for i in range(n_states):
        sim = make_ternary_sim(i)
        rng = pyrandom.Random(1000 + i)
        for _ in range(60 * (i + 1)):
            sim.act(scripted_policy(sim, rng, random_prob=0.2))
        observations.append([[sim_json(sim)]])
    return observations


def forward(policy, obs, oracle):
    policy.observation_space.converter = (
        json_to_ternary_graph_object_oracle if oracle else json_to_ternary_graph_object
    )
    with th.no_grad():
        batch, _symbolic = policy._get_latent(obs)
        values = policy.value_net(batch.global_features)
    return {"latent nodes": batch.x, "latent globals": batch.global_features, "values": values}


def max_abs_diff(a, b):
    return (a - b).abs().max().item() if a.numel() else 0.0


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)

    print(f"device={args.device} torch={th.__version__} torch_geometric={torch_geometric.__version__} "
          f"states={args.states} repeats={args.repeats}")
    env, model = build_model(oracle_decoder=True, device=args.device)
    policy = model.policy
    original_converter = policy.observation_space.converter
    control_max = {name: 0.0 for name in OUTPUTS}
    oracle_max = {name: 0.0 for name in OUTPUTS}
    try:
        print(f"{'state':>5} {'repeat':>6}  {'output':<15} {'(i) control':>12} {'(ii) oracle':>12}")
        for s, obs in enumerate(real_observations(args.states)):
            for r in range(args.repeats):
                without_a = forward(policy, obs, oracle=False)
                without_b = forward(policy, obs, oracle=False)
                with_oracle = forward(policy, obs, oracle=True)
                for name in OUTPUTS:
                    control = max_abs_diff(without_a[name], without_b[name])
                    oracle = max_abs_diff(with_oracle[name], without_a[name])
                    control_max[name] = max(control_max[name], control)
                    oracle_max[name] = max(oracle_max[name], oracle)
                    print(f"{s:>5} {r:>6}  {name:<15} {control:>12.3e} {oracle:>12.3e}")
    finally:
        policy.observation_space.converter = original_converter
        env.close()

    print("\nmax over all states and repeats:")
    for name in OUTPUTS:
        print(f"  {name:<15} (i) control {control_max[name]:.3e}   (ii) oracle {oracle_max[name]:.3e}")

    control_zero = all(v == 0.0 for v in control_max.values())
    oracle_nonzero = any(v != 0.0 for v in oracle_max.values())
    if control_zero and oracle_nonzero:
        print("\nVERDICT: forwards without oracle_facts are bitwise identical, but adding oracle_facts "
              "changes the output -- oracle_facts reaches the GNN.")
        return 1
    if control_zero:
        print("\nVERDICT: deterministic on this device, and oracle_facts changes nothing.")
    else:
        worse = [name for name in OUTPUTS if oracle_max[name] > control_max[name]]
        print("\nVERDICT: run-to-run noise on this device (control is non-zero); "
              + ("oracle differences stay within it." if not worse else
                 f"oracle differences exceed the control maximum for {worse} -- compare magnitudes "
                 f"over more repeats before concluding."))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
