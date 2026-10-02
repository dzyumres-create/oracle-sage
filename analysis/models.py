"""Load trained GNNPlanFeedbackPolicy (Cell 1) / WLPlanFeedbackPolicy (Cell 3) models
from an SB3 final_model.zip, without SB3's .load() (which fails on JsonGraph).

The policy is rebuilt from the zip's policy_kwargs with a fresh observation and
action space, then load_state_dict(policy.pth, strict=True).

Works against whichever `sage` is first on PYTHONPATH (this branch, or an
extracted copy of main / cell2-vilg-gnn under analysis/.branch_src/), so the
same helper can run a model under its own training code.

Also provides the two read-outs the analysis needs, both through the policy's
own methods (no re-implementation of the decision logic):
  actor_probs(policy, obs)               -> masked softmax over all nodes
  path_values(policy, obs, goals)        -> discriminator score per goal, via
                                            _choose_top_action(eval_action=...)
"""
import io
import json
import os
import re
import zipfile

from analysis import compat  # noqa: F401  (must precede sage imports)

import numpy as np
import torch as th
import gym

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB_DIR = os.path.join(REPO_ROOT, "sage", "domains", "utils")


def _readable(v):
    if isinstance(v, dict):
        return {k: _readable(x) for k, x in v.items() if k not in (":serialized:", ":type:")}
    return v


def read_zip(path):
    """Return (data dict from the JSON `data` entry, policy state_dict on CPU)."""
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read("data").decode())
        state_dict = th.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=False)
    return data, state_dict


def _infer_gnn_steps(state_dict, prefix):
    steps = {int(m.group(1)) for k in state_dict for m in [re.match(rf"{prefix}\.gnn\.gnns\.(\d+)\.", k)] if m}
    return max(steps) + 1 if steps else None


def resolve_vocab_path(path):
    """Map the training-time (RCP) vocab path to this checkout's copy, by basename."""
    if path is None:
        return None
    if os.path.exists(path):
        return path
    local = os.path.join(VOCAB_DIR, os.path.basename(path))
    if not os.path.exists(local):
        raise FileNotFoundError(f"WL vocab {path!r} not found locally (looked for {local})")
    return local


def make_spaces():
    from sage.domains.utils.spaces import JsonGraph
    from sage.domains.utils.representations import json_to_graph
    from sage.domains.gym_taxi.simulator.planner import Planner
    obs_space = JsonGraph(converter=json_to_graph, node_dimension=3, edge_dimension=4, planner=Planner())
    return obs_space, gym.spaces.Discrete(6)


def build_policy(data, state_dict, wl_vocab_path=None):
    pk = _readable(data["policy_kwargs"])
    kwargs = dict(
        optimizer_class=th.optim.AdamW,
        optimizer_kwargs=pk.get("optimizer_kwargs", {}),
        ortho_init=pk["ortho_init"],
        exploration_initial_eps=pk["exploration_initial_eps"],
        exploration_final_eps=pk["exploration_final_eps"],
        exploration_fraction=pk["exploration_fraction"],
        shared_gnn=pk["shared_gnn"],
        layer_norm=pk["layer_norm"],
        num_planning_choices=pk["num_planning_choices"],
    )
    # features_extractor_kwargs is saved as {} because GNNPolicy pops gnn_steps
    # out of the dict before saving; recover it from the weights instead.
    fe_kwargs = dict(pk.get("features_extractor_kwargs") or {})
    steps = _infer_gnn_steps(state_dict, "gnn_extractor") or _infer_gnn_steps(state_dict, "gnn_extractor2")
    if steps is not None:
        fe_kwargs["gnn_steps"] = steps
    kwargs["features_extractor_kwargs"] = fe_kwargs

    vocab = wl_vocab_path or pk.get("wl_vocab_path")
    if vocab is not None:
        from sage.agent.wl_plan_feedback_policy import WLPlanFeedbackPolicy as cls
        kwargs["wl_vocab_path"] = resolve_vocab_path(vocab)
    else:
        from sage.agent.graph_plan_feedback_policy import GNNPlanFeedbackPolicy as cls

    obs_space, act_space = make_spaces()
    policy = cls(obs_space, act_space, lambda _: 3e-4, **kwargs)
    missing, unexpected = policy.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected
    policy.eval()
    policy.exploration_rate = 0.0
    return policy


def load_policy(zip_path, wl_vocab_path=None):
    data, state_dict = read_zip(zip_path)
    return build_policy(data, state_dict, wl_vocab_path), data, state_dict


def check_weights_equal(policy, state_dict):
    """Every tensor in the zip equals the loaded policy's tensor exactly."""
    own = policy.state_dict()
    bad = [k for k, v in state_dict.items() if k not in own or not th.equal(own[k].cpu(), v)]
    return bad


# ---------------------------------------------------------------- read-outs
def obs_array(json_strs):
    """Observation array in the shape the converter expects: [[json], [json], ...]."""
    if isinstance(json_strs, str):
        json_strs = [json_strs]
    return np.array([[s] for s in json_strs], dtype=object)


@th.no_grad()
def actor_logits_probs(policy, obs_json):
    """Raw actor logits and masked softmax over all nodes of one state."""
    from sage.agent.graph_policy import make_mask, masked_segmented_softmax
    batch, _ = policy._get_latent(obs_array(obs_json))
    logits = policy.action_net(batch.x).flatten()
    mask, _, _ = make_mask(batch)
    probs = masked_segmented_softmax(logits.clone(), mask, batch.batch)
    return logits.cpu().numpy(), probs.cpu().numpy()


@th.no_grad()
def path_values(policy, obs_json, goals, chunk=64):
    """Discriminator score for each goal in one state, through the policy's own
    eval_action branch of _choose_top_action (project -> encode -> path_value_net).
    The state is replicated once per goal so all goals are scored in one batch."""
    goals = list(goals)
    out = []
    for i in range(0, len(goals), chunk):
        g = goals[i:i + chunk]
        batch, symbolic = policy._get_latent(obs_array([obs_json] * len(g)))
        a, pa, data_starts, entropy = policy._choose_node(policy.action_net, batch)
        _, _, values, _, _, _ = policy._choose_top_action(
            a, pa, data_starts, entropy, batch, symbolic, th.as_tensor(g, dtype=th.long))
        out.append(values.flatten().cpu().numpy())
    return np.concatenate(out)
