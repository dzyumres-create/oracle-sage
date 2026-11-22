"""Record policy outputs on fixed inputs at the current commit (run at ternary-wl 5547edb).

Run from the repo root at the base commit:
    PYTHONPATH=. python analysis/make_policy_reference.py tests/fixtures/policy_reference_<commit>.json.gz

Saves, per case: the observations (JSON strings), the policy weights, the actions/values/
log-probs/plans of a forward() under a fixed torch seed, and evaluate_actions() outputs on
those actions. tests/test_paper_unified_base.py replays them on the new code.

The file is gzipped JSON, not a torch or numpy pickle, so any torch/numpy version reads it
(a torch 2.x pickle failed to load on RCP's torch 1.7). A tensor is stored as
{"__tensor__": numpy dtype string (little-endian), "shape": [...], "data": base64 of its raw
bytes}, so values survive the round trip bit for bit.
"""
import base64
import gzip
import json
import sys

import gym
import numpy as np
import torch as th

import sage.domains.gym_taxi  # noqa: F401  registers envs
import sage.experiments.gnn_global as gnn_global
from sage.agent.async_vec_env import AsyncVecEnv
from sage.agent.graph_plan_feedback_policy import GNNPlanFeedbackPolicy
from sage.agent.wl_plan_feedback_policy import WLPlanFeedbackPolicy
from sage.agent.graph_policy import make_mask
from sage.agent.plan_feedback_a2c import PlanFeedback_A2C
from sage.domains.gym_taxi.utils.wl_vocab_cache import reset_wl_vocab_override
from sage.domains.utils import spaces as sage_spaces
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.utils.compat import spec_kwargs
from sage.forks.stable_baselines3.stable_baselines3.common.env_util import make_vec_env
from tests.test_ternary_wl_training_path import committed_vocab_path, wl_args

POLICY_KWARGS = {"optimizer_class": th.optim.AdamW, "optimizer_kwargs": {"weight_decay": 0.0001},
                 "ortho_init": False, "exploration_initial_eps": 0.0, "exploration_final_eps": 0.0,
                 "exploration_fraction": 0.1, "shared_gnn": True, "layer_norm": False,
                 "num_planning_choices": 3, "features_extractor_kwargs": {"gnn_steps": 5}}
CASES = [
    ("std_object_gnn", "city-taxi-unmasked-v1", "oracle_sage", False),
    ("std_objatom_gnn", "city-taxi-unmasked-v1", "vilg", False),
    ("std_atom_gnn", "city-taxi-unmasked-v1", "atom", False),
    ("ter_object_gnn", "city-taxi-ternary-unmasked-v1", "oracle_sage", False),
    ("ter_objatom_gnn", "city-taxi-ternary-unmasked-v1", "vilg", False),
    ("ter_atom_gnn", "city-taxi-ternary-unmasked-v1", "atom", False),
    ("ter_object_wl", "city-taxi-ternary-unmasked-v1", "oracle_sage", True),
]
N_ENVS = 2


def build(env_id, convention, wl):
    policy_kwargs = dict(POLICY_KWARGS)
    policy_cls = GNNPlanFeedbackPolicy
    if wl:
        gnn_global.configure_wl(wl_args(graph_convention=convention, wl_vocab_path=committed_vocab_path(convention)),
                                policy_kwargs)
        policy_cls = WLPlanFeedbackPolicy
    env_kwargs = dict(spec_kwargs(gym.spec(env_id)), graph_convention=convention)
    env = make_vec_env(lambda: GraphTaxiEnv(**env_kwargs), n_envs=N_ENVS, seed=0, monitor_kwargs={"info_keywords": ("len100", "len200")},
                       vec_env_cls=AsyncVecEnv)
    th.manual_seed(0)
    model = PlanFeedback_A2C(policy_cls, env, verbose=0, device="cpu",
                             supported_action_spaces=(sage_spaces.BinaryAction, gym.spaces.Discrete,
                                                      sage_spaces.Autoregressive),
                             n_steps=2, policy_kwargs=policy_kwargs)
    return env, model


def record(case):
    name, env_id, convention, wl = case
    env, model = build(env_id, convention, wl)
    obs = env.reset()
    th.manual_seed(123)
    with th.no_grad():
        actions, values, log_prob, explored, plans = model.policy.forward(obs)
        ev_values, ev_log_prob, ev_entropy, ev_projected = model.policy.evaluate_actions(obs, actions)
        # extra goals: the taxi (node 0) and each graph's last valid node (a passenger)
        batch, _sym = model.policy._get_latent(obs)
        mask, splits, starts = make_mask(batch)
        last = []
        for g in range(len(obs)):
            local = mask[batch.batch == g].nonzero().flatten()
            last.append(int(local[-1]))
        extra = {}
        for label, acts in (("taxi", [0] * len(obs)), ("last_valid", last)):
            acts = th.as_tensor(acts)
            v, lp, ent, pv = model.policy.evaluate_actions(obs, acts)
            extra[label] = dict(actions=acts, values=v.clone(), log_prob=lp.clone(), entropy=ent.clone(),
                                projected=None if pv is None else pv.clone())
    out = dict(
        extra=extra,
        env_id=env_id, convention=convention, wl=wl,
        obs=[[str(x) for x in row] for row in obs],  # plain strings: no numpy pickle (numpy 1 vs 2)
        state_dict={k: v.clone() for k, v in model.policy.state_dict().items()},
        actions=actions.clone(), values=values.clone(), log_prob=log_prob.clone(),
        plans=[list(map(int, p)) for p in plans],
        ev_values=ev_values.clone(), ev_log_prob=ev_log_prob.clone(), ev_entropy=ev_entropy.clone(),
        ev_projected=None if ev_projected is None else ev_projected.clone(),
    )
    env.close()
    if wl:
        reset_wl_vocab_override()
    return out


def to_jsonable(obj):
    """Tensors -> {"__tensor__", "shape", "data"} (see the module docstring); dicts, lists and tuples recursively; the
    rest (str, int, float, bool, None) unchanged."""
    if isinstance(obj, th.Tensor):
        a = obj.detach().cpu().numpy()
        a = a.astype(a.dtype.newbyteorder("<"))
        return {"__tensor__": a.dtype.str, "shape": list(a.shape),
                "data": base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


if __name__ == "__main__":
    th.set_num_threads(1)
    ref = {case[0]: record(case) for case in CASES}
    # forward() samples with th.multinomial: other torch versions may draw differently
    for v in ref.values():
        v["torch_version"] = str(th.__version__)  # a plain string (torch 2.x returns a TorchVersion)
    with gzip.open(sys.argv[1], "wt") as f:
        json.dump(to_jsonable(ref), f)
    for k, v in ref.items():
        print(k, v["actions"].tolist(), {l: (e["actions"].tolist(), e["log_prob"].flatten().tolist()) for l, e in v["extra"].items()})
