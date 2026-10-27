"""
--diag-occupancy: logging-only diagnostics for PlanFeedback_A2C on the ternary Taxi domain.

Per decision (one row per env per collection forward) it records what the actor proposed (the
candidates th.multinomial drew), what the selector executed (path_value_net's argmax over their
projections), what the executed plan did, and whether the taxi was occupied at decision time.
Per update it records the exact KL(pi_old || pi_new) over valid nodes on the bootstrap states: the
rollout's last forward (for last_values) and the next rollout's first forward run on the same
observations, before and after the one optimizer step in between.

Three CSVs under out_dir, flushed after every rollout:
  <run>_decisions.csv   DECISION_COLUMNS, one row per env per decision
  <run>_bootstrap.csv   BOOTSTRAP_COLUMNS, one row per env per update (the KL states)
  <run>_updates.csv     UPDATE_COLUMNS, one row per update
and diag/* aggregates in the stdout log (STDOUT_KEYS).

Nothing here draws random numbers, runs a forward pass, or touches a tensor that collection or
training uses: it reads the policy's detached stash (GNNFeedbackPolicy.diag_stash), the env's info
fields (taxi_env.DIAG_INFO_KEYS), each env's simulator (read only, through VecEnv.get_attr), the
rollout buffer after compute_returns_and_advantage, and the gradient norms train() hands it.

Conventions:
  - Ground truth comes from the simulator, never the graph. A decision's state is the simulator's
    state when the forward runs: the end of that env's previous plan, or a fresh episode after a
    reset.
  - Node ids are object ids (taxi 0, locations 1..size*size, then passengers), which is how every
    ternary converter orders the selectable nodes.
  - pi probabilities (p_*, cand*_p) are the float32 values the policy used. Entropy, KL and
    pg_proxy are recomputed in float64 from the raw logits exactly as masked_segmented_softmax
    builds pi under the policy's clamp_mode (clamp_logits), invalid nodes excluded.
  - Candidates are ordered by pi probability, highest first, ties kept in draw order;
    cand*_draw is each one's position in th.multinomial's draw (what the logger's
    action_selection/* metrics index by -- at initialisation many locations tie exactly).
"""
import csv
import math
import os

import numpy as np
import torch as th

from sage.forks.stable_baselines3.stable_baselines3.common import logger
from sage.domains.gym_taxi.envs.taxi_env import (
    DIAG_CARRIED_PID,
    DIAG_DROPOFF,
    DIAG_DROPOFF_PID,
    DIAG_OCCUPIED,
    DIAG_PICKUP,
    DIAG_TAXI_LOC,
)

TAXI_NODE = 0
LOGIT_CLAMP = 30.0  # masked_segmented_softmax's clamp
RELATIVE_CLAMP_WINDOW = 2 * LOGIT_CLAMP  # graph_policy.RELATIVE_CLAMP_WINDOW (--clamp-mode relative)

NODE_TYPES = ("taxi", "location", "waiting_passenger", "carried_passenger")
# delivered / fail_*: the plan's dropoff (always its last action) and what it did; fail_empty is a
# dropoff with nobody aboard. no_dropoff: the plan finished without one (a move). noop: the taxi
# node or the taxi's own location was selected, a one-step no-op. truncated: the episode ended
# before the plan finished, without a dropoff.
OUTCOMES = ("delivered", "fail_own", "fail_nondest", "fail_empty", "no_dropoff", "noop", "truncated")
DROPOFF_OUTCOME = {"success": "delivered", "fail_own": "fail_own", "fail_nondest": "fail_nondest", "fail_empty": "fail_empty"}
ATTEMPT_OUTCOMES = ("delivered", "fail_own", "fail_nondest")  # dropoffs with a passenger aboard

DECISION_COLUMNS_BASE = (
    "update_idx", "total_timesteps", "env_idx",
    "occupied", "carried_pid", "selected_node", "selected_type", "selected_is_carried",
    "same_node_as_prev_decision", "n_valid", "p_selected", "p_top", "entropy", "plan_len",
    "outcome", "plan_reward", "advantage", "pg_proxy",
    "rollout_step", "episode_idx", "taxi_loc", "p_cand_max", "executed_rank", "p_carried", "logp_stored",
    "steps_run", "dropoff_pid", "pickup_failed",
)
BOOTSTRAP_COLUMNS_BASE = (
    "update_idx", "env_idx", "occupied", "carried_pid", "n_valid",
    "p_top", "entropy", "p_carried", "selected_node", "executed_rank",
    "p_top_after", "entropy_after", "p_carried_after", "kl",
)
UPDATE_COLUMNS = (
    "update_idx", "total_timesteps", "n_decisions", "n_occ",
    "kl_all", "kl_occ", "kl_empty", "n_boot_occ", "n_boot_empty",
    "p_carried_before", "p_carried_after",
    "ent_occ_before", "ent_occ_after", "ent_empty_before", "ent_empty_after",
    "pg_share_occ", "ploss_occ", "ploss_empty",
    "occ_exec_not_carried", "occ_pv_margin", "occ_abs_logp",
    "grad_norm_head", "grad_norm_train", "grad_norm_frozen", "grad_norm_total", "clip_coef",
    # every loss_grad_every-th update (NaN otherwise): each loss term's gradient norm w.r.t. the
    # shared GNN (gnn_extractor + features_extractor), coefficient included, and the cosine
    # between the policy and path_value gradients
    "gn_pol", "gn_ent", "gn_val", "gn_pv", "cos_pol_pv",
)
LOSS_GRAD_EVERY = 5

# diag/* keys in the stdout log. The fork's table shows a key's part after "diag/" indented by 3
# and truncated to 23 characters, so each must be at most 20 characters (as for progress/dropoff/*).
OUTCOME_SHORT = {"delivered": "deliv", "fail_own": "f_own", "fail_nondest": "f_nondest", "fail_empty": "f_empty",
                 "no_dropoff": "no_drop", "noop": "noop", "truncated": "trunc"}
TYPE_SHORT = {"taxi": "taxi", "location": "loc", "waiting_passenger": "wait"}
STDOUT_KEYS = (
    ("occ_frac", "p_top_occ", "p_top_empty", "ent_occ", "ent_empty", "retry_frac_occ",
     "succ_first", "succ_retry", "kl_all", "kl_occ", "kl_empty", "pg_share_occ",
     "occ_exec_not_carr", "occ_pv_margin", "occ_abs_logp")
    + tuple(f"{group}_t_{short}" for group in ("occ", "emp") for short in TYPE_SHORT.values())
    + tuple(f"{group}_o_{short}" for group in ("occ", "emp") for short in OUTCOME_SHORT.values())
)


def run_paths(out_dir, run_name):
    """The three CSV paths for run_name under out_dir. Never reuses an existing run's files: a
    taken name gets _1, _2, ... appended."""
    suffix, n = "", 0
    while True:
        paths = {kind: os.path.join(out_dir, f"{run_name}{suffix}_{kind}.csv")
                 for kind in ("decisions", "bootstrap", "updates")}
        if not any(os.path.exists(p) for p in paths.values()):
            return paths
        n += 1
        suffix = f"_{n}"


def node_type(node, n_locations, carried):
    if node == TAXI_NODE:
        return "taxi"
    if node <= n_locations:
        return "location"
    return "carried_passenger" if node == carried else "waiting_passenger"


def clamp_logits(logits, mask, clamp_mode="absolute"):
    """float64 logits after masked_segmented_softmax's clamp for this graph: absolute clips to
    [-30, 30]; relative subtracts the max over valid nodes and clips to [-60, 0]."""
    z = logits.astype(np.float64)
    if clamp_mode == "absolute":
        return np.clip(z, -LOGIT_CLAMP, LOGIT_CLAMP)
    if clamp_mode == "relative":
        return np.clip(z - z[mask].max(), -RELATIVE_CLAMP_WINDOW, 0.0)
    raise ValueError(f"unknown clamp_mode {clamp_mode!r}")


def log_policy(logits, mask, clamp_mode="absolute"):
    """float64 log pi over one graph's nodes, as masked_segmented_softmax builds pi; -inf on
    invalid nodes."""
    z = clamp_logits(logits, mask, clamp_mode)
    valid = z[mask]
    top = valid.max()
    log_norm = top + math.log(np.exp(valid - top).sum())
    return np.where(mask, z - log_norm, -np.inf)


def nan_mean(values):
    return float(np.mean(values)) if len(values) else float("nan")


def fmt(value):
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return "nan" if math.isnan(value) else "%.9g" % value
    return value


class _GraphView(object):
    """One graph of one forward, from the stash. All arrays are this graph's nodes only."""

    def __init__(self, logits, mask, probs, candidates, path_values, clamp_mode="absolute"):
        self.mask = mask
        self.logp = log_policy(logits, mask, clamp_mode)
        p64 = np.exp(self.logp)
        self.p64 = p64
        self.p = probs.astype(np.float64)  # the float32 pi the policy used
        self.n_valid = int(mask.sum())
        self.entropy = float(-(p64[mask] * self.logp[mask]).sum())
        self.p_top = float(self.p[mask].max())
        self.sum_p64_sq = float((p64[mask] ** 2).sum())
        cand_p = self.p[candidates]
        order = np.argsort(-cand_p, kind="stable")
        self.cand_draw = [int(j) for j in order]  # position in th.multinomial's draw
        self.cand_nodes = [int(candidates[j]) for j in order]
        self.cand_p = [float(cand_p[j]) for j in order]
        self.cand_pv = [float("nan") if path_values is None else float(path_values[j]) for j in order]

    def rank_of(self, node):
        return self.cand_nodes.index(node) + 1 if node in self.cand_nodes else 0

    def p_of(self, node):
        return float(self.p[node]) if node >= 0 else float("nan")


class OccupancyDiagnostics(object):
    """
    Called by PlanFeedback_A2C.collect_rollouts (begin_rollout, on_forward, on_env_step,
    end_decision, on_bootstrap, end_rollout), by Feedback_A2C.train (on_grad_norms) and when
    learn() returns (close).
    """

    def __init__(self, out_dir, run_name, loss_grad_every=LOSS_GRAD_EVERY):
        self.paths = run_paths(out_dir, run_name)
        self.loss_grad_every = loss_grad_every
        self._written_header = set()
        self.n_envs = None
        self.n_candidates = None
        # cross-check: the last step's DIAG_* info vs the simulator at the next decision
        self.state_checks = 0
        self.state_mismatches = 0
        self._pending = None  # update row (and its bootstrap rows) waiting for the post-update forward

    # -- per-env bookkeeping ------------------------------------------------------------------

    def _init_envs(self, n_envs):
        self.n_envs = n_envs
        self._episode = [0] * n_envs
        self._prev_node = [None] * n_envs  # None: first decision, or a reset/delivery since
        self._last_info = [None] * n_envs  # (occupied, carried, taxi_loc) after the env's last step

    def _decision_states(self, env):
        """Ground truth at decision time, read from each env's simulator."""
        states = []
        for i, sim in enumerate(env.get_attr("sim")):
            carried = -1 if sim.taxi.passenger is None else int(sim.taxi.passenger)
            state = {"occupied": int(carried >= 0), "carried": carried, "taxi_loc": int(sim.taxi.location),
                     "n_locations": int(sim.size * sim.size)}
            if self._last_info[i] is not None:
                self.state_checks += 1
                if self._last_info[i] != (state["occupied"], carried, state["taxi_loc"]):
                    self.state_mismatches += 1
            states.append(state)
        return states

    def _views(self, policy):
        stash = policy.diag_stash
        policy.diag_stash = None
        logits = stash["logits"].cpu().numpy()
        mask = stash["mask"].cpu().numpy().astype(bool)
        batch_index = stash["batch_index"].cpu().numpy()
        probs = stash["probs"].cpu().numpy()
        candidates = stash["candidates"].cpu().numpy()  # [k, n]
        path_values = None if stash["path_values"] is None else stash["path_values"].cpu().numpy()  # [n, k]
        n = candidates.shape[1]
        if self.n_candidates is None:
            self.n_candidates = candidates.shape[0]
        counts = np.bincount(batch_index, minlength=n)
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        views = []
        for g in range(n):
            s, e = starts[g], starts[g] + counts[g]
            views.append(_GraphView(logits[s:e], mask[s:e], probs[s:e], candidates[:, g],
                                    None if path_values is None else path_values[g],
                                    getattr(policy, "clamp_mode", "absolute")))
        return views

    def _candidate_columns(self):
        cols = []
        for k in range(1, self.n_candidates + 1):
            cols += [f"cand{k}_node", f"cand{k}_type", f"cand{k}_p", f"cand{k}_pv", f"cand{k}_draw"]
        return cols

    def _candidate_values(self, view, state):
        values = {}
        for k, (node, p, pv, draw) in enumerate(zip(view.cand_nodes, view.cand_p, view.cand_pv, view.cand_draw), start=1):
            values[f"cand{k}_draw"] = draw
            values[f"cand{k}_node"] = node
            values[f"cand{k}_type"] = node_type(node, state["n_locations"], state["carried"])
            values[f"cand{k}_p"] = p
            values[f"cand{k}_pv"] = pv
        return values

    # -- collection hooks ---------------------------------------------------------------------

    def begin_rollout(self, model):
        self._update_idx = model._n_updates
        self._rows = []
        self._aux = []  # per row: the view's sum p64^2 and p64(executed), for pg_proxy

    def on_forward(self, model, env, rollout_step, actions, log_probs, plans):
        """Right after a collection forward: the decision-time half of each env's row."""
        if self.n_envs is None:
            self._init_envs(env.num_envs)
        states = self._decision_states(env)
        views = self._views(model.policy)
        if rollout_step == 0 and self._pending is not None:
            self._finish_update(views, states)
        actions = np.asarray(actions).reshape(-1)
        log_probs = log_probs.detach().cpu().numpy().reshape(-1)
        self._decision = []
        for i, (view, state) in enumerate(zip(views, states)):
            node = int(actions[i])
            prev = self._prev_node[i]
            row = {
                "update_idx": self._update_idx, "env_idx": i, "rollout_step": rollout_step,
                "episode_idx": self._episode[i], "occupied": state["occupied"], "carried_pid": state["carried"],
                "taxi_loc": state["taxi_loc"], "selected_node": node,
                "selected_type": node_type(node, state["n_locations"], state["carried"]),
                "selected_is_carried": int(node == state["carried"]),
                "same_node_as_prev_decision": -1 if prev is None else int(node == prev),
                "n_valid": view.n_valid, "p_selected": view.p_of(node), "p_top": view.p_top, "entropy": view.entropy,
                "plan_len": len(plans[i]), "p_cand_max": view.cand_p[0], "executed_rank": view.rank_of(node),
                "p_carried": view.p_of(state["carried"]) if state["occupied"] else float("nan"),
                "logp_stored": float(log_probs[i]),
            }
            row.update(self._candidate_values(view, state))
            self._rows.append(row)
            self._aux.append((view.sum_p64_sq, float(view.p64[node])))
            self._decision.append({"dropoff": None, "dropoff_pid": -1, "pickup_failed": 0, "done": False, "steps": 0})

    def on_env_step(self, active, infos, dones):
        """After every env.step of the plan-execution loop; `active` is the mask passed to it."""
        for i in np.flatnonzero(active):
            event, info = self._decision[i], infos[i]
            event["steps"] += 1
            if info[DIAG_DROPOFF] != "none" and event["dropoff"] is None:
                event["dropoff"] = info[DIAG_DROPOFF]
                event["dropoff_pid"] = info[DIAG_DROPOFF_PID]
            if info[DIAG_PICKUP] == "fail":
                event["pickup_failed"] = 1
            if dones[i]:
                event["done"] = True
            self._last_info[i] = (info[DIAG_OCCUPIED], info[DIAG_CARRIED_PID], info[DIAG_TAXI_LOC])

    def end_decision(self, plan_rewards, num_timesteps):
        """After the plan-execution loop: what each env's plan did."""
        rows = self._rows[-self.n_envs:]
        for i, (row, event) in enumerate(zip(rows, self._decision)):
            if event["dropoff"] is not None:
                outcome = DROPOFF_OUTCOME[event["dropoff"]]
            elif event["done"] and event["steps"] < row["plan_len"]:
                outcome = "truncated"
            elif row["selected_node"] in (TAXI_NODE, row["taxi_loc"]):
                outcome = "noop"
            else:
                outcome = "no_dropoff"
            row.update(outcome=outcome, plan_reward=float(plan_rewards[i]), total_timesteps=num_timesteps,
                       steps_run=event["steps"], dropoff_pid=event["dropoff_pid"], pickup_failed=event["pickup_failed"])
            if event["done"]:
                # the vec env reset it: a fresh episode, an empty taxi, a fresh set of passenger ids
                self._episode[i] += 1
                self._prev_node[i] = None
                self._last_info[i] = None
            else:
                # a delivery renumbers the passengers above it, so a node id may name someone else
                self._prev_node[i] = None if outcome == "delivered" else row["selected_node"]

    def on_bootstrap(self, model, env, actions):
        """After the forward that computes last_values: the pi_old half of this update's KL.
        `actions` is that forward's own selection, which collection discards; it is kept only
        so the choice_* metrics (whose logger means include this forward) can be recomputed."""
        states = self._decision_states(env)
        self._boot = (self._views(model.policy), states, np.asarray(actions).reshape(-1))

    def end_rollout(self, rollout_buffer):
        """After compute_returns_and_advantage: advantages (still [n_steps, n_envs], collection
        order), the decision rows, this rollout's aggregates, and the pending update row."""
        advantages = rollout_buffer.advantages
        n = len(self._rows)
        for row, (sum_sq, p_exec) in zip(self._rows, self._aux):
            a = float(advantages[row["rollout_step"], row["env_idx"]])
            row["advantage"] = a
            row["pg_proxy"] = abs(a) * math.sqrt(max(sum_sq - p_exec ** 2 + (1.0 - p_exec) ** 2, 0.0))
        self._write("decisions", DECISION_COLUMNS_BASE + tuple(self._candidate_columns()), self._rows)

        occ = [r for r in self._rows if r["occupied"]]
        emp = [r for r in self._rows if not r["occupied"]]
        pg_total = sum(r["pg_proxy"] for r in self._rows)
        margins = []
        for r in occ:
            nodes = [r[f"cand{k}_node"] for k in range(1, self.n_candidates + 1)]
            if r["carried_pid"] in nodes:
                pvs = [r[f"cand{k}_pv"] for k in range(1, self.n_candidates + 1)]
                carried_pv = pvs[nodes.index(r["carried_pid"])]
                others = [pv for node, pv in zip(nodes, pvs) if node != r["carried_pid"]]
                if others:
                    margins.append(carried_pv - max(others))
        update = {
            "update_idx": self._update_idx, "total_timesteps": self._rows[-1]["total_timesteps"],
            "n_decisions": n, "n_occ": len(occ),
            "pg_share_occ": sum(r["pg_proxy"] for r in occ) / pg_total if pg_total > 0 else float("nan"),
            "ploss_occ": sum(-r["advantage"] * r["logp_stored"] for r in occ) / n,
            "ploss_empty": sum(-r["advantage"] * r["logp_stored"] for r in emp) / n,
            "occ_exec_not_carried": nan_mean([int(r["selected_node"] != r["carried_pid"]) for r in occ]),
            "occ_pv_margin": nan_mean(margins),
            "occ_abs_logp": nan_mean([abs(r["logp_stored"]) for r in occ]),
        }
        self._record_rollout(occ, emp, margins, update["pg_share_occ"])

        boot_views, boot_states, boot_actions = self._boot
        boot_rows = []
        for i, (view, state) in enumerate(zip(boot_views, boot_states)):
            node = int(boot_actions[i])
            boot_rows.append({
                "update_idx": self._update_idx, "env_idx": i, "occupied": state["occupied"],
                "carried_pid": state["carried"], "n_valid": view.n_valid, "p_top": view.p_top,
                "entropy": view.entropy, "p_carried": view.p_of(state["carried"]) if state["occupied"] else float("nan"),
                "selected_node": node, "executed_rank": view.rank_of(node),
            })
            boot_rows[-1].update(self._candidate_values(view, state))
        self._pending = {"update": update, "boot_rows": boot_rows, "boot_views": boot_views, "boot_states": boot_states}

    # -- update hooks -------------------------------------------------------------------------

    def wants_loss_grads(self, update_idx):
        """Whether train() should compute the per-loss-term gradients for this update."""
        return self._pending is not None and update_idx % self.loss_grad_every == 0

    def on_loss_grads(self, grads):
        """From train(), before its backward: {"pol", "ent", "val", "pv"} -> flat gradient over
        the shared GNN (summed over chunks with their n_i/B weights), or None."""
        def norm(g):
            return float("nan") if g is None else float(g.double().norm())
        update = self._pending["update"]
        pol, pv = grads["pol"], grads["pv"]
        update.update(gn_pol=norm(pol), gn_ent=norm(grads["ent"]), gn_val=norm(grads["val"]), gn_pv=norm(pv))
        cos = float("nan")
        if pv is not None and update["gn_pol"] > 0 and update["gn_pv"] > 0:
            cos = float(th.dot(pol.double(), pv.double())) / (update["gn_pol"] * update["gn_pv"])
        update["cos_pol_pv"] = cos

    def on_grad_norms(self, norms):
        """From train(), after backward and before clipping (Feedback_A2C._grad_norms)."""
        if self._pending is not None:
            self._pending["update"].update(norms)

    def _finish_update(self, views_after, states_after):
        """The next rollout's first forward ran on the bootstrap observations with the updated
        policy: complete the pending update row with the exact KL and the after-values."""
        pending, self._pending = self._pending, None
        update = pending["update"]
        kls = {"occ": [], "empty": []}
        p_carried = {"before": [], "after": []}
        ent = {(g, w): [] for g in ("occ", "empty") for w in ("before", "after")}
        for i, row in enumerate(pending["boot_rows"]):
            before, after, state = pending["boot_views"][i], views_after[i], pending["boot_states"][i]
            if (states_after[i]["occupied"], states_after[i]["carried"]) != (state["occupied"], state["carried"]):
                self.state_mismatches += 1
            group = "occ" if state["occupied"] else "empty"
            if before.mask.shape == after.mask.shape and (before.mask == after.mask).all():
                m = before.mask
                kl = float((before.p64[m] * (before.logp[m] - after.logp[m])).sum())
            else:  # cannot happen when both forwards see the same observation
                kl = float("nan")
            kls[group].append(kl)
            ent[(group, "before")].append(before.entropy)
            ent[(group, "after")].append(after.entropy)
            row.update(p_top_after=after.p_top, entropy_after=after.entropy, kl=kl,
                       p_carried_after=after.p_of(state["carried"]) if state["occupied"] else float("nan"))
            if state["occupied"]:
                p_carried["before"].append(row["p_carried"])
                p_carried["after"].append(row["p_carried_after"])
            logger.record_mean("diag/kl_all", kl)
            logger.record_mean("diag/kl_occ" if state["occupied"] else "diag/kl_empty", kl)
        update.update(
            kl_all=nan_mean(kls["occ"] + kls["empty"]), kl_occ=nan_mean(kls["occ"]), kl_empty=nan_mean(kls["empty"]),
            n_boot_occ=len(kls["occ"]), n_boot_empty=len(kls["empty"]),
            p_carried_before=nan_mean(p_carried["before"]), p_carried_after=nan_mean(p_carried["after"]),
            ent_occ_before=nan_mean(ent[("occ", "before")]), ent_occ_after=nan_mean(ent[("occ", "after")]),
            ent_empty_before=nan_mean(ent[("empty", "before")]), ent_empty_after=nan_mean(ent[("empty", "after")]),
        )
        self._write_update(pending)

    def _write_update(self, pending):
        self._write("bootstrap", BOOTSTRAP_COLUMNS_BASE + tuple(self._candidate_columns()), pending["boot_rows"])
        self._write("updates", UPDATE_COLUMNS, [pending["update"]])

    def close(self):
        """learn() returned: write the last update, which has no post-update forward (its KL and
        after-values stay NaN)."""
        if self._pending is not None:
            pending, self._pending = self._pending, None
            for row in pending["boot_rows"]:
                row.update(p_top_after=float("nan"), entropy_after=float("nan"),
                           p_carried_after=float("nan"), kl=float("nan"))
            self._write_update(pending)

    # -- output -------------------------------------------------------------------------------

    def _record_rollout(self, occ, emp, margins, pg_share_occ):
        """diag/* stdout aggregates. record_mean once per decision (or per attempt, state, ...), so
        each key is the pooled mean over every item since the last dump."""
        rec = logger.record_mean
        for r in occ + emp:
            rec("diag/occ_frac", r["occupied"])
        for group, rows in (("occ", occ), ("emp", emp)):
            for r in rows:
                rec("diag/p_top_occ" if group == "occ" else "diag/p_top_empty", r["p_top"])
                rec("diag/ent_occ" if group == "occ" else "diag/ent_empty", r["entropy"])
                for t, short in TYPE_SHORT.items():
                    rec(f"diag/{group}_t_{short}", int(r["selected_type"] == t))
                for o, short in OUTCOME_SHORT.items():
                    rec(f"diag/{group}_o_{short}", int(r["outcome"] == o))
                if r["outcome"] in ATTEMPT_OUTCOMES:
                    rec("diag/succ_retry" if group == "occ" else "diag/succ_first", int(r["outcome"] == "delivered"))
        for r in occ:
            rec("diag/retry_frac_occ", r["selected_is_carried"])
            rec("diag/occ_exec_not_carr", int(r["selected_node"] != r["carried_pid"]))
            rec("diag/occ_abs_logp", abs(r["logp_stored"]))
        for m in margins:
            rec("diag/occ_pv_margin", m)
        if not math.isnan(pg_share_occ):
            rec("diag/pg_share_occ", pg_share_occ)

    def _write(self, kind, columns, rows):
        path = self.paths[kind]
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if kind not in self._written_header:
                writer.writerow(columns)
                self._written_header.add(kind)
            for row in rows:
                writer.writerow([fmt(row.get(c, float("nan"))) for c in columns])
