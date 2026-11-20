"""Size of the --edge-mode variants (W4): edges, update memory and update time, against gr.

For each (domain, representation) and each edge mode run on this branch, builds
PlanFeedback_A2C as gnn_global does (--planner --feedback --shared-gnn, k = 3, 5 GNN steps;
CPU), collects real rollouts, and measures one training update (model.train()):

- edges: typed edges / Gr edges, over every GNN pass of the update;
- act_MB: activation memory of the update, i.e. the bytes autograd saves for the backward
  pass (each storage counted once), taken per backward call, max over calls -- so with
  --update-chunks 2 it is the peak of one chunk. Weights and optimizer state are not
  included; on the GPU they are small next to activations;
- s/update: wall time of model.train(), median over the measured updates (after one
  warm-up update).

The ratios against gr of the same row are what carry over to RCP; absolute numbers depend
on --num-processes / --num-steps (set below) and the machine.

Run from the repo root (not while other heavy jobs run, or the times are noisy):
    PYTHONPATH=. python analysis/edge_mode_size.py [--n-envs 4] [--n-steps 5] [--updates 3]
"""
import argparse
import statistics
import sys
import time
from unittest import mock

import gym as gym_module
import torch as th

sys.path.insert(0, ".")
import tests.test_ternary_planner  # noqa: F401,E402  installs the gym-0.26 randint shim
import sage.agent.graph_policy as graph_policy  # noqa: E402
from sage.agent.async_vec_env import AsyncVecEnv  # noqa: E402
from sage.agent.edge_modes import MGR_SKIPPED, type_mask  # noqa: E402
from sage.agent.graph_plan_feedback_policy import GNNPlanFeedbackPolicy  # noqa: E402
from sage.agent.plan_feedback_a2c import PlanFeedback_A2C  # noqa: E402
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv  # noqa: E402
from sage.domains.gym_taxi.utils.compat import spec_kwargs  # noqa: E402
from sage.domains.utils import spaces as sage_spaces  # noqa: E402
from sage.forks.stable_baselines3.stable_baselines3.common.env_util import make_vec_env  # noqa: E402

ENVS = {"std": "city-taxi-unmasked-v1", "ter": "city-taxi-ternary-unmasked-v1"}
POLICY_KWARGS = {"optimizer_class": th.optim.AdamW, "optimizer_kwargs": {"weight_decay": 0.0001},
                 "ortho_init": False, "exploration_initial_eps": 0.0, "exploration_final_eps": 0.0,
                 "exploration_fraction": 0.1, "shared_gnn": True, "layer_norm": False,
                 "num_planning_choices": 3, "features_extractor_kwargs": {"gnn_steps": 5}}
SUPPORTED = (sage_spaces.BinaryAction, gym_module.spaces.Discrete, sage_spaces.Autoregressive)


def variants():
    for dom in ("std", "ter"):
        for conv in ("oracle_sage", "vilg", "atom"):
            for mode in ("gr", "mgr", "egr"):
                if mode == "mgr" and (conv, dom == "ter") in MGR_SKIPPED:
                    continue
                for chunks in ((1, 2) if conv == "atom" and mode != "gr" else (1,)):
                    yield dom, conv, mode, chunks
            if conv == "atom":
                yield dom, conv, "gr", 2


def measure(dom, conv, mode, chunks, n_envs, n_steps, updates, seed=0):
    policy_kwargs = dict(POLICY_KWARGS, features_extractor_kwargs={"gnn_steps": 5})
    if mode != "gr":
        policy_kwargs["edge_mode"] = mode
    kwargs = dict(spec_kwargs(gym_module.spec(ENVS[dom])), graph_convention=conv)
    env = make_vec_env(lambda: GraphTaxiEnv(**kwargs), n_envs=n_envs, seed=seed,
                       monitor_kwargs={"info_keywords": ("len100", "len200")}, vec_env_cls=AsyncVecEnv)
    algo_kwargs = {"update_chunks": chunks} if chunks != 1 else {}
    model = PlanFeedback_A2C(GNNPlanFeedbackPolicy, env, verbose=0, device="cpu", seed=seed, n_steps=n_steps,
                             supported_action_spaces=SUPPORTED, policy_kwargs=policy_kwargs, **algo_kwargs)
    spec = model.policy.edge_spec

    state = {"measure": False, "saved": {}, "peaks": [], "gr_edges": 0, "typed_edges": 0, "times": []}
    params = {id(p) for p in model.policy.parameters()}

    def pack(t):
        if state["measure"] and id(t) not in params:
            storage = t.untyped_storage()
            state["saved"][storage.data_ptr()] = storage.nbytes()
        return t

    real_backward = th.Tensor.backward

    def backward_spy(self, *args, **kw):
        if state["measure"]:
            state["peaks"].append(sum(state["saved"].values()))
            state["saved"] = {}
        return real_backward(self, *args, **kw)

    real_forward = graph_policy.GNNExtractor.forward

    def forward_spy(self_ext, x, xg, edge_attr, edge_index, batch):
        if state["measure"]:
            state["gr_edges"] += edge_index.shape[1]
            state["typed_edges"] += int(type_mask(edge_attr, spec).sum()) if spec is not None else edge_index.shape[1]
        return real_forward(self_ext, x, xg, edge_attr, edge_index, batch)

    real_train = model.train

    def train_spy():
        state["measure"] = len(state["times"]) > 0 or state.get("warm", False)
        start = time.perf_counter()
        with th.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            real_train()
        elapsed = time.perf_counter() - start
        if state.get("warm", False):
            state["times"].append(elapsed)
        state["warm"] = True
        state["measure"] = False

    with mock.patch.object(model, "train", train_spy), mock.patch.object(th.Tensor, "backward", backward_spy), \
            mock.patch.object(graph_policy.GNNExtractor, "forward", forward_spy):
        model.learn(total_timesteps=(updates + 1) * n_envs * n_steps)
    env.close()
    n_params = sum(p.numel() for n, p in model.policy.named_parameters() if n.startswith("gnn_extractor."))
    return {
        "edges": state["typed_edges"] / state["gr_edges"],
        "act_MB": max(state["peaks"]) / 2 ** 20,
        "s_update": statistics.median(state["times"]),
        "gnn_params": n_params,
    }


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--updates", type=int, default=3)
    args = parser.parse_args(argv)
    print(f"torch {th.__version__}, threads {th.get_num_threads()}, n_envs {args.n_envs}, n_steps {args.n_steps}, "
          f"updates {args.updates} (+1 warm-up)", flush=True)
    rows, gr = [], {}
    for dom, conv, mode, chunks in variants():
        r = measure(dom, conv, mode, chunks, args.n_envs, args.n_steps, args.updates)
        if mode == "gr" and chunks == 1:
            gr[(dom, conv)] = r
        base = gr[(dom, conv)]
        rows.append((dom, conv, mode, chunks, r, base))
        print(f"| {dom} | {conv} | {mode} | {chunks} | {r['edges']:.3f} | {r['act_MB']:.0f} | "
              f"{r['act_MB'] / base['act_MB']:.2f} | {r['s_update']:.2f} | {r['s_update'] / base['s_update']:.2f} | "
              f"{r['gnn_params']} |", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
