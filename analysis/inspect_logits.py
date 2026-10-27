"""
Cell 1 logits, embeddings and Adam state, read from saved models (read-only analysis, CPU).

Question: do Cell 1's action logits and GNN embeddings grow into masked_segmented_softmax's
[-30, 30] clamp (sage/agent/graph_policy.py:45, where a logit's gradient is exactly 0 strictly
outside the range), and does that differ between stable, declining and dead Cell 1 models and the
oracle-decoder models? And were the final updates consistent drift or noise (Adam's moments)?

Every model is evaluated on ONE fixed set of decision states (built once from a rollout of a
reference model, saved, and reloaded on later runs), split by whether the taxi was occupied.

Models are loaded without unpickling the zip's JsonGraph: the policy is built with Cell 1's
constructor and flags (CELL1_POLICY_KWARGS; checked against the zip's own policy_kwargs), then
policy.pth and policy.optimizer.pth are loaded with load_state_dict. The zip's "data" JSON is read
only through a restricted unpickler (collections.deque, plus the AdamW class for policy_kwargs).

Run from the repo root (CPU):
    PYTHONPATH=. python analysis/inspect_logits.py
    PYTHONPATH=. python analysis/inspect_logits.py --model oracle_s0 oracle /path/to/model.zip ...
Writes analysis/inspect_logits_results.csv; the state set lives in
analysis/data/inspect_logits_states.json.gz (built on first use, or with --rebuild-states).
"""
import argparse
import base64
import copy
import csv
import gzip
import io
import json
import math
import os
import pickle
import sys
import zipfile

from unittest import mock

import gym
import numpy as np
from gym.utils import seeding
import torch as th

import sage.domains.gym_taxi  # registers the ternary env
from sage.agent.diag_occupancy import log_policy as _log_policy
from sage.agent.graph_plan_feedback_policy import GNNPlanFeedbackPolicy
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.utils.compat import spec_kwargs

ENV_ID = "city-taxi-ternary-unmasked-v1"
LOGIT_CLAMP = 30.0  # masked_segmented_softmax's clamp
LEARNING_RATE = 3e-4  # Cell 1's --learning-rate
# Cell 1's policy as gnn_global builds it for: --planner --feedback --shared-gnn (eps 0, no
# layer norm, no ortho init, 3 planning choices, 5 GNN steps, AdamW with weight decay 1e-4).
CELL1_POLICY_KWARGS = {
    "optimizer_class": th.optim.AdamW, "optimizer_kwargs": {"weight_decay": 0.0001},
    "ortho_init": False,
    "exploration_initial_eps": 0.0, "exploration_final_eps": 0.0, "exploration_fraction": 0.1,
    "shared_gnn": True, "layer_norm": False, "num_planning_choices": 3,
    "features_extractor_kwargs": {"gnn_steps": 5},
}
NODE_TYPES = ("taxi", "location", "waiting_passenger", "carried_passenger")
GROUP_ORDER = ("stable", "declining", "dead", "oracle")
ADAM_GROUPS = ("gnn_extractor", "features_extractor", "action_net", "value_net")
SENSITIVITY_MODULES = ("features_extractor", "gnn_extractor", "action_net")  # "the GNN" + action_net

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STATES = os.path.join(HERE, "data", "inspect_logits_states.json.gz")
DEFAULT_OUT = os.path.join(HERE, "inspect_logits_results.csv")
DEFAULT_CLAMP_OUT = os.path.join(HERE, "inspect_logits_clamp_compare.csv")
N_STATES, MIN_OCCUPIED = 300, 100
STATE_ENV_SEED, STATE_TORCH_SEED = 2026, 0

_MODELS_ROOT = os.path.expanduser("~/Documents/research_project/train/ternary_domain/GNN_ternary")
DEFAULT_MODELS = [
    # (label, group, zip) -- groups as read from the training curves
    ("new_s0", "stable", f"{_MODELS_ROOT}/new_tcell1_models/final_model_tcell1_seed0_new.zip"),
    ("new_s600", "stable", f"{_MODELS_ROOT}/new_tcell1_models/final_model_tcell1_seed600_new.zip"),
    ("old_s600", "stable", f"{_MODELS_ROOT}/old_tcell1_models/final_model_seed600_old.zip"),
    ("new_s1200", "declining", f"{_MODELS_ROOT}/new_tcell1_models/final_model_tcell1_seed1200_new.zip"),
    ("new_s300", "declining", f"{_MODELS_ROOT}/new_tcell1_models/final_model_tcell1_seed300_new.zip"),
    ("old_s300", "declining", f"{_MODELS_ROOT}/old_tcell1_models/final_model_tcell1_seed300_old.zip"),
    ("old_s1200", "declining", f"{_MODELS_ROOT}/old_tcell1_models/final_model_tcell1_seed1200_old.zip"),
    ("old_s0", "declining", f"{_MODELS_ROOT}/old_tcell1_models/final_model_tcell1_seed0_old.zip"),
    ("new_s900", "dead", f"{_MODELS_ROOT}/new_tcell1_models/final_model_tcell1_seed900_new.zip"),
    ("old_s900", "dead", f"{_MODELS_ROOT}/old_tcell1_models/final_model_tcell1_seed900_old.zip"),
] + [(f"oracle_s{s}", "oracle", f"{_MODELS_ROOT}/oracle_tcell1_models/final_model_tcell1_oracle_seed{s}.zip")
     for s in (0, 300, 600, 900, 1200)]
STATE_MODEL = DEFAULT_MODELS[0][2]  # the new seed-0 model


# ==========================================================================================
# Loading
# ==========================================================================================

class RestrictedUnpickler(pickle.Unpickler):
    """Unpickles only the globals in `allowed` ({(module, name)}); anything else raises."""

    def __init__(self, data, allowed):
        super(RestrictedUnpickler, self).__init__(io.BytesIO(data))
        self.allowed = allowed

    def find_class(self, module, name):
        if (module, name) in self.allowed:
            return super(RestrictedUnpickler, self).find_class(module, name)
        raise pickle.UnpicklingError(f"restricted unpickler: {module}.{name} is not allowed")


def decode_serialized(entry, allowed):
    """One SB3 "data" JSON entry ({":type:", ":serialized:" base64 pickle}) through the restricted
    unpickler."""
    return RestrictedUnpickler(base64.b64decode(entry[":serialized:"]), allowed).load()


def read_zip(zip_path):
    """(data JSON, policy state_dict, optimizer state_dict) -- tensors mapped to CPU."""
    with zipfile.ZipFile(zip_path) as z:
        data = json.loads(z.read("data").decode())
        policy_state = th.load(io.BytesIO(z.read("policy.pth")), map_location="cpu")
        optimizer_state = th.load(io.BytesIO(z.read("policy.optimizer.pth")), map_location="cpu")
    return data, policy_state, optimizer_state


def make_env():
    """The ternary env. GraphTaxiEnv builds its first maze inside __init__ from seeding.np_random,
    which on gym 0.26 is a numpy Generator the maze generator can't use (no randint); construct it
    with a RandomState instead, as gym 0.18 does. Callers that need a specific episode set
    env.np_random themselves before reset()."""
    if isinstance(seeding.np_random(0)[0], np.random.RandomState):
        return GraphTaxiEnv(**spec_kwargs(gym.spec(ENV_ID)))
    with mock.patch.object(seeding, "np_random", lambda seed=None: (np.random.RandomState(seed), seed)):
        return GraphTaxiEnv(**spec_kwargs(gym.spec(ENV_ID)))


def build_policy(env=None, clamp_mode="absolute"):
    """A fresh Cell 1 policy (CPU), built the way PlanFeedback_A2C._setup_model builds it, with the
    given --clamp-mode (absolute builds exactly the original policy: no clamp_mode kwarg)."""
    env = env or make_env()
    kwargs = copy.deepcopy(CELL1_POLICY_KWARGS)
    if clamp_mode != "absolute":
        kwargs["clamp_mode"] = clamp_mode
    return GNNPlanFeedbackPolicy(env.observation_space, env.action_space, lambda _progress: LEARNING_RATE, **kwargs)


def saved_policy_kwargs(data):
    return decode_serialized(data["policy_kwargs"], {("torch.optim.adamw", "AdamW")})


def load_policy(zip_path, env=None, clamp_mode=None):
    """(policy with the zip's weights and AdamW state, the zip's data JSON). clamp_mode None uses
    the one the model was trained with (its saved policy_kwargs); a mode overrides it."""
    data, policy_state, optimizer_state = read_zip(zip_path)
    if clamp_mode is None:
        clamp_mode = saved_policy_kwargs(data).get("clamp_mode", "absolute")
    policy = build_policy(env, clamp_mode)
    policy.load_state_dict(policy_state)
    policy.optimizer.load_state_dict(optimizer_state)
    params = [p for group in policy.optimizer.param_groups for p in group["params"]]
    for p in params:  # load_state_dict checks group sizes, not shapes
        state = policy.optimizer.state.get(p, {})
        if "exp_avg" in state and state["exp_avg"].shape != p.shape:
            raise ValueError(f"{zip_path}: optimizer state shape {tuple(state['exp_avg'].shape)} != {tuple(p.shape)}")
    policy.eval()
    return policy, data


def provenance(data):
    """Training facts recorded in the zip: the last 100 episodes (Monitor), and the settings."""
    buffer = data.get("ep_info_buffer")  # null for a model that never ran learn()
    episodes = [] if buffer is None else list(decode_serialized(buffer, {("collections", "deque")}))
    kwargs = saved_policy_kwargs(data)
    # GNNPolicy.__init__ pops gnn_steps out of features_extractor_kwargs in place, so a saved
    # model's policy_kwargs never carries it; clamp_mode is saved only when not "absolute"
    expected = copy.deepcopy(CELL1_POLICY_KWARGS)
    expected["features_extractor_kwargs"].pop("gnn_steps", None)
    if kwargs.get("clamp_mode", "absolute") != "absolute":
        expected["clamp_mode"] = kwargs["clamp_mode"]
    mismatched = sorted(k for k in set(kwargs) | set(expected) if kwargs.get(k) != expected.get(k))

    def mean(key):
        values = [ep[key] for ep in episodes if key in ep]
        return float(np.mean(values)) if values else float("nan")
    return {
        "episodes": len(episodes), "mean_reward": mean("r"), "mean_fails": mean("dropoff/fails"),
        "mean_attempts": mean("dropoff/attempts"), "mean_fail_own": mean("dropoff/fail_own"),
        "n_updates": data.get("_n_updates"), "num_timesteps": data.get("num_timesteps"),
        "learning_rate": data.get("learning_rate"), "ent_coef": data.get("ent_coef"),
        "kwargs_mismatch": ";".join(mismatched), "trained_clamp_mode": kwargs.get("clamp_mode", "absolute"),
    }


# ==========================================================================================
# The fixed state set
# ==========================================================================================

def build_state_set(zip_path=STATE_MODEL, n=N_STATES, env_seed=STATE_ENV_SEED, torch_seed=STATE_TORCH_SEED):
    """n consecutive decision states from one env rolled out by the model at zip_path (its own
    sampling, selector and object-decoder planner), with the simulator's ground truth for each."""
    env = make_env()
    policy, _data = load_policy(zip_path, env)
    env.np_random = np.random.RandomState(env_seed)  # same maze/passengers on any gym version
    obs = env.reset()
    th.manual_seed(torch_seed)
    states = []
    while len(states) < n:
        sim = env.sim
        carried = -1 if sim.taxi.passenger is None else int(sim.taxi.passenger)
        states.append({"obs": obs, "occupied": int(carried >= 0), "carried": carried,
                       "taxi_loc": int(sim.taxi.location), "n_locations": int(sim.size * sim.size),
                       "time": int(sim.time)})
        with th.no_grad():
            _actions, _values, _log_probs, _explored, plans = policy.forward(np.array([[obs]]))
        for action in plans[0]:
            obs, _reward, done, _info = env.step(int(action))
            if done:
                obs = env.reset()
                break
    n_occ = sum(s["occupied"] for s in states)
    if n_occ < MIN_OCCUPIED:
        raise RuntimeError(f"only {n_occ} occupied states of {n}; script failed dropoffs on a crossed pair instead")
    return {"model": os.path.basename(zip_path), "env_seed": env_seed, "torch_seed": torch_seed, "states": states}


def load_or_build_states(path=DEFAULT_STATES, rebuild=False):
    if rebuild or not os.path.exists(path):
        state_set = build_state_set()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with gzip.open(path, "wt") as f:
            json.dump(state_set, f)
        print(f"built {len(state_set['states'])} states from {state_set['model']} -> {path}")
    with gzip.open(path, "rt") as f:
        return json.load(f)


# ==========================================================================================
# Forward pass and metrics
# ==========================================================================================

def forward_states(policy, states, chunk=25):
    """Per state: raw pre-clamp logits, the valid mask, node types, node-embedding norms (the
    GNN's output, i.e. what action_net reads) and the global feature norm. No sampling."""
    out = []
    with th.no_grad():
        for start in range(0, len(states), chunk):
            part = states[start:start + chunk]
            batch, symbolic = policy.extract_features(np.array([[s["obs"]] for s in part]), policy.device)
            nodes, glob = policy.gnn_extractor(batch.x, batch.global_features, batch.edge_attr,
                                               batch.edge_index, batch.batch)
            logits = policy.action_net(nodes).flatten()
            index = batch.batch.cpu().numpy()
            kind = symbolic.x.cpu().numpy().argmax(1)  # one-hot [location, taxi, passenger]
            node_norm = nodes.norm(dim=1).cpu().numpy()
            logits, mask = logits.cpu().numpy().astype(np.float64), batch.mask.cpu().numpy().astype(bool).reshape(-1)
            glob_norm = glob.norm(dim=1).cpu().numpy()
            for g, state in enumerate(part):
                sel = index == g
                types = np.array([("location", "taxi", "waiting_passenger")[k] for k in kind[sel]], dtype=object)
                if state["carried"] >= 0:
                    types[state["carried"]] = "carried_passenger"
                out.append({"logits": logits[sel], "mask": mask[sel], "types": types,
                            "node_norm": node_norm[sel], "glob_norm": float(glob_norm[g])})
    return out


def log_policy(logits, mask, clamp_mode="absolute"):
    """float64 log pi as masked_segmented_softmax builds it under clamp_mode (the clamp, then
    softmax over valid nodes) -- the --diag-occupancy diagnostics' own implementation."""
    return _log_policy(logits, mask, clamp_mode)


def clamp_stats(logits, mask):
    """For one state's valid nodes: counts strictly above +30 / below -30 (where the clamp's
    gradient is 0), whether EVERY valid logit is clamped, and whether all post-clamp logits are equal."""
    v = logits[mask]
    above, below = int((v > LOGIT_CLAMP).sum()), int((v < -LOGIT_CLAMP).sum())
    clamped = np.clip(v, -LOGIT_CLAMP, LOGIT_CLAMP)
    return {"n_valid": int(v.size), "above": above, "below": below,
            "all_clamped": bool(above + below == v.size), "all_equal": bool(clamped.max() == clamped.min())}


def entropy(logits, mask, clamp_mode="absolute"):
    logp = log_policy(logits, mask, clamp_mode)
    return float(-(np.exp(logp[mask]) * logp[mask]).sum())


def state_group_metrics(results, states):
    """Aggregates over a group of states (empty or occupied)."""
    if not results:
        return {}
    per = []
    above = below = n_valid = 0
    margins = []
    norms = {t: [] for t in NODE_TYPES}
    for r, s in zip(results, states):
        v = r["logits"][r["mask"]]
        c = clamp_stats(r["logits"], r["mask"])
        above, below, n_valid = above + c["above"], below + c["below"], n_valid + c["n_valid"]
        per.append((v.max(), v.min(), v.max() - v.min(), v.std(), entropy(r["logits"], r["mask"]),
                    c["all_clamped"], c["all_equal"], r["glob_norm"]))
        if s["carried"] >= 0:
            others = np.delete(r["logits"], s["carried"])[np.delete(r["mask"], s["carried"])]
            margins.append(r["logits"][s["carried"]] - others.max())
        for t in NODE_TYPES:
            norms[t].extend(r["node_norm"][r["types"] == t].tolist())
    per = np.array(per, dtype=np.float64)
    m = {
        "n_states": len(results),
        "logit_max": per[:, 0].mean(), "logit_min": per[:, 1].mean(), "logit_range": per[:, 2].mean(),
        "logit_std": per[:, 3].mean(), "logit_max_overall": per[:, 0].max(), "logit_min_overall": per[:, 1].min(),
        "frac_above_30": above / n_valid, "frac_below_m30": below / n_valid,
        "entropy": per[:, 4].mean(), "frac_all_clamped": per[:, 5].mean(), "frac_all_equal": per[:, 6].mean(),
        "glob_norm": per[:, 7].mean(),
        "carried_margin": float(np.mean(margins)) if margins else float("nan"),
        "carried_margin_median": float(np.median(margins)) if margins else float("nan"),
    }
    for t in NODE_TYPES:
        m[f"emb_{t}"] = float(np.mean(norms[t])) if norms[t] else float("nan")
    return m


def sensitivity_kl(policy, states, step=LEARNING_RATE, n_directions=10, seed=0, env=None, clamp_mode=None):
    """Mean over states of KL(pi || pi') after one random sign step of `step` per parameter of
    features_extractor, gnn_extractor and action_net, averaged over n_directions directions
    (torch.Generator seeded from `seed`; the global RNG is untouched). Computed on a separate
    copy of the policy; `policy` itself is never modified. pi under clamp_mode (default: the
    policy's own)."""
    clamp_mode = clamp_mode or getattr(policy, "clamp_mode", "absolute")
    base = forward_states(policy, states)
    base_logp = [log_policy(r["logits"], r["mask"], clamp_mode) for r in base]
    clone = build_policy(env, clamp_mode)
    original = {k: v.clone() for k, v in policy.state_dict().items()}
    kls = []
    for d in range(n_directions):
        clone.load_state_dict(original)
        generator = th.Generator().manual_seed(seed + d)
        with th.no_grad():
            for name, p in clone.named_parameters():
                if name.split(".")[0] in SENSITIVITY_MODULES:
                    sign = th.randint(0, 2, p.shape, generator=generator).to(p.dtype) * 2 - 1
                    p.add_(step * sign)
        perturbed = forward_states(clone, states)
        per_state = []
        for lp, r in zip(base_logp, perturbed):
            lq = log_policy(r["logits"], r["mask"], clamp_mode)
            m = r["mask"]
            per_state.append(float((np.exp(lp[m]) * (lp[m] - lq[m])).sum()))
        kls.append(np.mean(per_state))
    return float(np.mean(kls))


def adam_ratios(policy):
    """Per parameter group, the median over elements of |m_hat| / (sqrt(v_hat) + eps) -- the
    size of AdamW's last step in units of lr -- with bias correction as AdamW applies it, and the
    raw |exp_avg| / (sqrt(exp_avg_sq) + eps). ~1: consistent drift; pure zero-mean noise gives
    ~0.155 at beta1 = 0.9 (|m| ~ |N(0, 0.229 sigma)|). Elements with exp_avg_sq == 0 (no gradient
    ever) are left out."""
    out = {}
    group = policy.optimizer.param_groups[0]
    beta1, beta2 = group["betas"]
    eps = group["eps"]
    by_group = {g: ([], []) for g in ADAM_GROUPS}
    for name, p in policy.named_parameters():
        top = name.split(".")[0]
        state = policy.optimizer.state.get(p)
        if top not in by_group or not state:
            continue
        step = float(state["step"])
        m, v = state["exp_avg"].double().flatten(), state["exp_avg_sq"].double().flatten()
        keep = v > 0
        m, v = m[keep], v[keep]
        bc1, bc2 = 1 - beta1 ** step, 1 - beta2 ** step
        by_group[top][0].append((m.abs() / bc1) / (v.sqrt() / math.sqrt(bc2) + eps))
        by_group[top][1].append(m.abs() / (v.sqrt() + eps))
        out["adam_step"] = step
    for g, (corrected, raw) in by_group.items():
        out[f"adam_{g}"] = float(th.cat(corrected).median()) if corrected else float("nan")
        out[f"adam_raw_{g}"] = float(th.cat(raw).median()) if raw else float("nan")
    return out


def inspect_model(label, group, zip_path, state_set, env):
    policy, data = load_policy(zip_path, env)
    states = state_set["states"]
    results = forward_states(policy, states)
    empty = [i for i, s in enumerate(states) if not s["occupied"]]
    occ = [i for i, s in enumerate(states) if s["occupied"]]
    row = {"label": label, "group": group, "zip": zip_path}
    row.update(provenance(data))
    row["action_net_w_norm"] = float(policy.action_net.weight.norm())
    row["action_net_b_norm"] = float(policy.action_net.bias.norm())
    for prefix, idx in (("empty", empty), ("occ", occ)):
        for k, v in state_group_metrics([results[i] for i in idx], [states[i] for i in idx]).items():
            row[f"{prefix}_{k}"] = v
    row["empty_sens_kl"] = sensitivity_kl(policy, [states[i] for i in empty], env=env)
    row.update(adam_ratios(policy))
    return row


# ==========================================================================================
# --compare-clamp: the same saved weights under --clamp-mode absolute and relative
# ==========================================================================================

TOP_TYPES = ("waiting_passenger", "carried_passenger", "location", "taxi", "mixed", "uniform")


def top_type(logits, mask, types, clamp_mode):
    """The node type of pi's argmax. Several nodes can share the top probability: if they are all
    of one type (e.g. two crossed buddies whose object-encoding neighbourhoods are identical),
    that type; "uniform" if EVERY valid node ties (every logit clamped to the same bound);
    "mixed" for any other tie across types."""
    z = log_policy(logits, mask, clamp_mode)
    top = z[mask].max()
    winners = np.flatnonzero(mask & (z == top))
    if len(winners) > 1 and len(winners) == int(mask.sum()):
        return "uniform"
    kinds = set(types[w] for w in winners)
    return kinds.pop() if len(kinds) == 1 else "mixed"


def clamp_comparison(label, group, zip_path, state_set, env):
    """One row per model: entropy, top-node type shares and sensitivity KL on empty and occupied
    states under both clamp modes, plus the largest total-variation distance between the two pi."""
    policy, data = load_policy(zip_path, env, clamp_mode="absolute")
    states = state_set["states"]
    results = forward_states(policy, states)  # raw logits do not depend on the clamp mode
    row = {"label": label, "group": group, "mean_reward": provenance(data)["mean_reward"]}
    for prefix, occupied in (("empty", 0), ("occ", 1)):
        idx = [i for i, s in enumerate(states) if s["occupied"] == occupied]
        tv = []
        for i in idx:
            r = results[i]
            pa = np.exp(log_policy(r["logits"], r["mask"], "absolute"))
            pr = np.exp(log_policy(r["logits"], r["mask"], "relative"))
            tv.append(0.5 * np.abs(pa - pr).sum())
        row[f"{prefix}_max_tv"] = float(np.max(tv))
        row[f"{prefix}_frac_changed"] = float(np.mean([t > 1e-6 for t in tv]))
        for mode in ("absolute", "relative"):
            tag = f"{prefix}_{mode[:3]}"
            row[f"{tag}_entropy"] = float(np.mean([entropy(results[i]["logits"], results[i]["mask"], mode) for i in idx]))
            counts = {t: 0 for t in TOP_TYPES}
            for i in idx:
                counts[top_type(results[i]["logits"], results[i]["mask"], results[i]["types"], mode)] += 1
            for t in TOP_TYPES:
                row[f"{tag}_top_{t}"] = counts[t] / len(idx)
            if prefix == "empty":
                row[f"{tag}_sens_kl"] = sensitivity_kl(policy, [states[i] for i in idx], env=env, clamp_mode=mode)
    return row


def format_clamp_table(rows):
    rows = sorted(rows, key=lambda r: (GROUP_ORDER.index(r["group"]) if r["group"] in GROUP_ORDER else 99, r["label"]))

    def shares(r, tag):
        short = {"waiting_passenger": "wait", "carried_passenger": "carr", "location": "loc", "taxi": "taxi",
                 "mixed": "mixed", "uniform": "UNIFORM"}
        parts = [f"{short[t]} {r[f'{tag}_top_{t}']:.2f}" for t in TOP_TYPES if r[f"{tag}_top_{t}"] > 0]
        return ", ".join(parts)
    header = ["model", "group", "rew", "H empty abs->rel", "H occ abs->rel", "top empty abs | rel",
              "top occ abs | rel", "sensKL empty abs->rel", "max TV empty/occ", "changed empty/occ"]
    lines = [header]
    for r in rows:
        lines.append([
            r["label"], r["group"], f"{r['mean_reward']:.1f}",
            f"{r['empty_abs_entropy']:.3f} -> {r['empty_rel_entropy']:.3f}",
            f"{r['occ_abs_entropy']:.3f} -> {r['occ_rel_entropy']:.3f}",
            f"{shares(r, 'empty_abs')} | {shares(r, 'empty_rel')}",
            f"{shares(r, 'occ_abs')} | {shares(r, 'occ_rel')}",
            f"{r['empty_abs_sens_kl']:.2e} -> {r['empty_rel_sens_kl']:.2e}",
            f"{r['empty_max_tv']:.1e} / {r['occ_max_tv']:.1e}",
            f"{r['empty_frac_changed']:.2f} / {r['occ_frac_changed']:.2f}",
        ])
    widths = [max(len(line[i]) for line in lines) for i in range(len(header))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(line, widths)) for line in lines)


# ==========================================================================================
# Output
# ==========================================================================================

TABLE = [  # (header, row key or callable(row, prefix), format); prefix-dependent ones per state group
    ("model", lambda r, p: r["label"], "{:<11}"), ("group", lambda r, p: r["group"], "{:<9}"),
    ("states", lambda r, p: f"{p}:{r[p + '_n_states']}", "{:<9}"),
    ("lmax", "logit_max", "{:>7.1f}"), ("lmin", "logit_min", "{:>7.1f}"), ("lstd", "logit_std", "{:>6.2f}"),
    (">30", "frac_above_30", "{:>7.4f}"), ("<-30", "frac_below_m30", "{:>7.4f}"),
    ("carr-best", "carried_margin", "{:>9.2f}"), ("H", "entropy", "{:>5.2f}"),
    ("allclmp", "frac_all_clamped", "{:>7.3f}"), ("alleq", "frac_all_equal", "{:>6.3f}"),
    ("e_taxi", "emb_taxi", "{:>7.1f}"), ("e_loc", "emb_location", "{:>7.1f}"),
    ("e_wait", "emb_waiting_passenger", "{:>7.1f}"), ("e_carr", "emb_carried_passenger", "{:>7.1f}"),
    ("e_glob", "glob_norm", "{:>7.1f}"),
]
MODEL_COLUMNS = [  # once per model
    ("|W_a|", "action_net_w_norm", "{:>7.2f}"), ("b_a", "action_net_b_norm", "{:>6.2f}"),
    ("sensKL", "empty_sens_kl", "{:>8.2e}"),
    ("A_gnn", "adam_gnn_extractor", "{:>5.2f}"), ("A_feat", "adam_features_extractor", "{:>6.2f}"),
    ("A_act", "adam_action_net", "{:>5.2f}"), ("A_val", "adam_value_net", "{:>5.2f}"),
    ("rew", "mean_reward", "{:>5.1f}"), ("fails", "mean_fails", "{:>6.1f}"),
]


def format_table(rows):
    """One table, two lines per model (empty / occupied states), sorted by GROUP_ORDER; the
    model-level columns are on the first line."""
    rows = sorted(rows, key=lambda r: (GROUP_ORDER.index(r["group"]) if r["group"] in GROUP_ORDER else 99, r["label"]))
    columns = TABLE + MODEL_COLUMNS
    lines = [[h for h, _k, _f in columns]]
    for r in rows:
        for prefix in ("empty", "occ"):
            cells = []
            for _h, key, fmt in TABLE:
                cells.append(fmt.format(key(r, prefix) if callable(key) else r.get(f"{prefix}_{key}", float("nan"))))
            for _h, key, fmt in MODEL_COLUMNS:
                cells.append(fmt.format(r.get(key, float("nan"))) if prefix == "empty" else "")
            lines.append(cells)
    widths = [max(len(line[i].strip()) for line in lines) for i in range(len(columns))]
    return "\n".join("  ".join(cell.strip().rjust(w) if i > 2 else cell.strip().ljust(w)
                               for i, (cell, w) in enumerate(zip(line, widths))) for line in lines)


def write_csv(rows, path):
    columns = []
    for r in rows:
        for k in r:
            if k not in columns:
                columns.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", nargs=3, action="append", metavar=("LABEL", "GROUP", "ZIP"),
                        help="add a model (default: the Cell 1 and oracle models under ~/Documents/research_project)")
    parser.add_argument("--only", nargs=3, action="append", metavar=("LABEL", "GROUP", "ZIP"),
                        help="evaluate only these models")
    parser.add_argument("--states", default=DEFAULT_STATES)
    parser.add_argument("--rebuild-states", action="store_true")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--compare-clamp", action="store_true",
                        help="instead: every model's saved weights under --clamp-mode absolute vs relative "
                             "(entropy, top-node type, sensitivity KL), written to --clamp-out")
    parser.add_argument("--clamp-out", default=DEFAULT_CLAMP_OUT)
    args = parser.parse_args(argv)
    th.set_num_threads(max(1, th.get_num_threads()))
    models = args.only or (DEFAULT_MODELS + (args.model or []))
    state_set = load_or_build_states(args.states, args.rebuild_states)
    n_occ = sum(s["occupied"] for s in state_set["states"])
    print(f"state set: {len(state_set['states'])} states ({n_occ} occupied) from {state_set['model']}, "
          f"env seed {state_set['env_seed']}")
    env = make_env()
    if args.compare_clamp:
        rows = []
        for label, group, zip_path in models:
            if not os.path.exists(zip_path):
                print(f"skip {label}: {zip_path} not found")
                continue
            rows.append(clamp_comparison(label, group, zip_path, state_set, env))
            print(f"{label} done", flush=True)
        print()
        print(format_clamp_table(rows))
        write_csv(rows, args.clamp_out)
        print(f"\nwrote {args.clamp_out}")
        return rows
    rows = []
    for label, group, zip_path in models:
        if not os.path.exists(zip_path):
            print(f"skip {label}: {zip_path} not found")
            continue
        row = inspect_model(label, group, zip_path, state_set, env)
        flag = ""
        if group == "oracle" and row["mean_fails"] != 0:
            flag = "  PROVENANCE: oracle model with dropoff fails"
        if row["kwargs_mismatch"]:
            flag += f"  PROVENANCE: policy_kwargs differ in {row['kwargs_mismatch']}"
        print(f"{label}: {row['episodes']} episodes, mean reward {row['mean_reward']:.1f}, mean fails "
              f"{row['mean_fails']:.1f}, updates {row['n_updates']}{flag}", flush=True)
        rows.append(row)
    print()
    print(format_table(rows))
    write_csv(rows, args.out)
    print(f"\nwrote {args.out}")
    return rows


if __name__ == "__main__":
    main(sys.argv[1:])
