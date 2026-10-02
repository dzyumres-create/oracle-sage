"""Check: Cell 1's actor logits vs the +-30 clamp in masked_segmented_softmax, for
all 5 Cell 1 seeds on the shared state set (Cell 1's stale-edge view), plus an
autograd check that clamped logits receive zero gradient.

Run under Cell 1's own code (main):
    cd analysis/.branch_src/main && PYTHONPATH=$PWD:<repo root> python -m analysis.check_actor_clamp
"""
import glob
import json
import os

import numpy as np
import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.join(HERE, "results", "scores")
CELL1 = "/Users/liutingniuniu2009/Documents/research_project/train/cell1"


def grad_check():
    import inspect
    from sage.agent.graph_policy import masked_segmented_softmax
    print("masked_segmented_softmax source (Cell 1's code):")
    print("    " + inspect.getsource(masked_segmented_softmax).replace("\n", "\n    "))
    x = th.tensor([35.0, 30.0, 29.0, 0.0, -31.0], requires_grad=True)
    mask = th.ones(5, dtype=th.bool)
    p = masked_segmented_softmax(x, mask, th.zeros(5, dtype=th.long))
    (p * th.tensor([1.0, 2.0, 3.0, 4.0, 5.0])).sum().backward()
    print(f"autograd: logits {x.detach().tolist()} -> d/dlogit {[round(g, 6) for g in x.grad.tolist()]} "
          "(nonzero only inside [-30, 30]; torch.clamp passes gradient at the boundary itself)")


def main():
    from analysis.models import load_policy, actor_logits_probs
    grad_check()
    gt = json.load(open(os.path.join(SC, "gt.json")))
    obs = json.load(open(os.path.join(SC, "snapshot_main.json")))["obs"]
    print(f"\n{len(gt)} states (Cell 1's stale-edge view); passenger goals only")
    for z in sorted(glob.glob(os.path.join(CELL1, "seed*", "final_model.zip")), key=lambda p: int(p.split("seed")[-1].split("/")[0])):
        pol, _, _ = load_policy(z)
        above, mass_short, mass_pass, uniform_short, const, loc_max = [], [], [], [], 0, []
        for sid, g in gt.items():
            P = [i for i, t in enumerate(g["types"]) if t == "pickup_deliver"]
            L = np.array(g["plan_len"])
            S = [i for i in P if L[i] == L[P].min()]
            lg, pr = actor_logits_probs(pol, obs[sid])
            above.append(np.mean(lg[P] >= 30))
            mass_short.append(pr[S].sum())
            mass_pass.append(pr[P].sum())
            uniform_short.append(len(S) / len(P))
            const += np.ptp(pr[P]) < 1e-12
            nonp = [i for i in range(len(lg)) if i not in set(P)]
            loc_max.append(lg[nonp].max())
        seed = z.split("seed")[-1].split("/")[0]
        print(f"seed {seed:>4s}: passenger logits >= 30: {100 * np.mean(above):5.1f}%  | identical prob across all passengers in "
              f"{const:3d}/{len(gt)} states | actor prob on shortest passenger {100 * np.mean(mass_short):5.1f}% "
              f"(uniform over passengers would give {100 * np.mean(uniform_short):5.1f}%) | mass on passengers "
              f"{100 * np.mean(mass_pass):5.1f}% | max non-passenger logit median {np.median(loc_max):6.1f}")


if __name__ == "__main__":
    main()
