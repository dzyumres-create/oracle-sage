"""Inspect an SB3 final_model.zip without SB3's .load() (which fails on JsonGraph).

Prints:
  - policy_kwargs (and a few algorithm fields) from the zip's JSON `data` entry
  - every tensor name and shape in policy.pth
  - per param group of policy.optimizer.pth: number of params and elements,
    compared with the policy's total, and which policy tensors the optimizer
    does not cover (flagging path_value_net explicitly).

How optimizer params are matched to names: torch.optim stores params by index,
in the order of policy.parameters() at the time the optimizer was built. That
order is the module registration order, which is also the state_dict order,
with tensors shared between modules (e.g. gnn_extractor2 aliasing
gnn_extractor under shared_gnn=True) listed once. Modules registered after the
optimizer was built come last in the state_dict and so fall past the
optimizer's index range. Shapes are checked position by position (using Adam's
exp_avg where present) to confirm the alignment.

Usage:
    PYTHONPATH=. python analysis/inspect_model_zip.py path/to/final_model.zip
"""
import argparse
import io
import json
import zipfile

import torch as th


ALGO_FIELDS = ["num_timesteps", "env_steps", "n_steps", "learning_rate", "vf_coef", "pvf_coef",
               "policy_coef", "ent_coef", "gamma", "gae_lambda", "max_grad_norm", "seed"]


def readable(v):
    """SB3 stores non-JSON objects as {':type:', ':serialized:', <str fields>}; drop the pickle blob."""
    if isinstance(v, dict):
        return {k: readable(x) for k, x in v.items() if k != ":serialized:"}
    return v


def load_zip(path):
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read("data").decode())
        policy = th.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=False)
        optim = th.load(io.BytesIO(z.read("policy.optimizer.pth")), map_location="cpu", weights_only=False)
    return data, policy, optim


def unique_params(state_dict):
    """state_dict entries with aliases (same storage, offset, shape) removed, in order."""
    seen = {}
    out = []
    for name, t in state_dict.items():
        key = (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape))
        if key in seen:
            seen[key].append(name)
            continue
        seen[key] = [name]
        out.append((name, t, seen[key]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    args = ap.parse_args()
    data, policy, optim = load_zip(args.zip)

    print("== policy_class:", data["policy_class"].get("__module__"), )
    print("== policy_kwargs")
    for k, v in readable(data["policy_kwargs"]).items():
        if k != ":type:":
            print(f"  {k}: {v}")
    print("== algorithm fields")
    for k in ALGO_FIELDS:
        if k in data:
            print(f"  {k}: {readable(data[k])}")
    obs = readable(data.get("observation_space", {}))
    print("== observation_space:", {k: obs.get(k) for k in ["shape", "dtype", "node_dimension", "edge_dimension"]})

    print("== policy.pth tensors")
    total = 0
    for name, t in policy.items():
        print(f"  {name:55s} {str(tuple(t.shape)):18s} {t.numel()}")
    params = unique_params(policy)
    aliased = [al for _, _, al in params if len(al) > 1]
    for name, t, aliases in params:
        total += t.numel()
    if aliased:
        prefixes = sorted({(a[0].split(".")[0], a[1].split(".")[0]) for a in aliased})
        print(f"  aliased (same storage): {len(aliased)} tensors, " + ", ".join(f"{x} == {y}" for x, y in prefixes))
    print(f"  unique tensors: {len(params)}  elements: {total}")

    print("== policy.optimizer.pth")
    state = optim["state"]
    covered = 0
    mismatches = []
    for gi, g in enumerate(optim["param_groups"]):
        idx = g["params"]
        n_el = 0
        for i in idx:
            if i < len(params):
                name, t, _ = params[i]
                n_el += t.numel()
                st = state.get(i)
                if st is not None and "exp_avg" in st and tuple(st["exp_avg"].shape) != tuple(t.shape):
                    mismatches.append((i, name, tuple(t.shape), tuple(st["exp_avg"].shape)))
        hp = {k: v for k, v in g.items() if k != "params"}
        print(f"  group {gi}: {len(idx)} params, {n_el} elements; {hp}")
        covered = max(covered, max(idx) + 1 if idx else 0)
    n_opt = sum(len(g["params"]) for g in optim["param_groups"])
    n_state = len(state)
    print(f"  optimizer params: {n_opt} (with Adam state: {n_state}) vs policy unique tensors: {len(params)}")
    print(f"  shape mismatches (optimizer exp_avg vs aligned policy tensor): {mismatches or 'none'}")

    missing = params[covered:]
    miss_el = sum(t.numel() for _, t, _ in missing)
    print(f"  policy elements in optimizer: {total - miss_el} / {total}")
    print(f"  tensors NOT in optimizer ({len(missing)}, {miss_el} elements):")
    for name, t, aliases in missing:
        print(f"    {' == '.join(aliases)} {tuple(t.shape)}")
    pvn = [n for n, _, al in params if n.startswith("path_value_net")]
    pvn_missing = [n for n, _, _ in missing if n.startswith("path_value_net")]
    if not pvn:
        print("  FLAG: no path_value_net tensors in policy.pth")
    elif pvn_missing:
        print(f"  FLAG: path_value_net is NOT in the optimizer ({len(pvn_missing)}/{len(pvn)} tensors missing)"
              " -> its weights were never updated (frozen at init)")
    else:
        print("  OK: path_value_net is in the optimizer")


if __name__ == "__main__":
    main()
