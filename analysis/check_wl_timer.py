"""Check: does Cell 3's discriminator use the time-left input, and at what scale?

Cell 3's path_value_net is PathValueNet(layer_norm=False, input_dim=51):
a single Linear(102 -> 1) over [current (wl_histogram[50], time_left),
projected (wl_histogram[50], time_left)]. Within one state the current half is
the same for every goal, so the score difference between two goals is exactly
    w_h . (h_i - h_j)  +  w_t * (t_i - t_j),
with t = projected time_left = current time_left - plan_len / 2000.

Reports, over passenger goals of every state:
  a) within groups that share a projected histogram (only time_left differs):
     rank correlation of score with -plan length; across groups: rank
     correlation of group-mean score with -group-mean plan length;
  b) the head's weights and the typical size of the histogram vs time
     contributions to score differences between passenger goals;
plus a reconstruction check of the stored scores from weights x inputs.

    PYTHONPATH=. python -m analysis.check_wl_timer --zip <cell3 zip>
"""
import argparse
import json
import os
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.join(HERE, "results", "scores")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    args = ap.parse_args()
    from analysis.models import load_policy
    from analysis.envs import obs_to_data, plan

    pol, _, _ = load_policy(args.zip)
    W = pol.path_value_net.path_value_net.weight.detach().numpy().ravel()
    b = float(pol.path_value_net.path_value_net.bias)
    V = pol.wl_vocab_size
    assert W.shape == (2 * (V + 1),)
    w_cur, w_h, w_t = W[:V + 1], W[V + 1:2 * V + 1], W[2 * V + 1]
    print(f"head: {pol.path_value_net}  (layer_norm={pol.path_value_net.layer_norm})")
    print(f"weights: projected time_left w_t = {w_t:+.4f}; current time_left {w_cur[V]:+.4f}; bias {b:+.4f}")
    print(f"projected-histogram weights: mean |w| {np.abs(w_h).mean():.4f}, max |w| {np.abs(w_h).max():.4f}, "
          f"range [{w_h.min():+.4f}, {w_h.max():+.4f}]")

    gt = json.load(open(os.path.join(SC, "gt.json")))
    stored = json.load(open(os.path.join(SC, "cell3.json")))
    obs = json.load(open(os.path.join(SC, "snapshot_head.json")))["obs"]

    recon_err, within, across = [], [], []
    d_hist, d_time, frac_time_wins = [], [], []
    hist_vs_len, time_vs_len = [], []
    group_sizes = []
    for sid, g in gt.items():
        P = [i for i, t in enumerate(g["types"]) if t == "pickup_deliver"]
        if len(P) < 2:
            continue
        o = obs[sid]
        cur = obs_to_data(o)
        cur_vec = np.concatenate([np.array(json.loads(o)["wl_histogram"], float), [float(cur.global_features[0, 0])]])
        H, T, L, S = [], [], [], []
        for i in P:
            proj, acts = plan(o, i)
            H.append(proj.wl_histogram.numpy().ravel().astype(float))
            T.append(float(proj.global_features[0, 0]))
            L.append(len(acts))
            S.append(stored[sid]["scores"][i])
        H, T, L, S = np.array(H), np.array(T), np.array(L), np.array(S)
        recon = cur_vec @ w_cur + H @ w_h + T * w_t + b
        recon_err.append(np.abs(recon - S).max())
        hc, tc = H @ w_h, T * w_t
        if np.ptp(L) > 0:
            if np.ptp(hc) > 0:
                hist_vs_len.append(spearmanr(hc, -L)[0])
            time_vs_len.append(spearmanr(tc, -L)[0])
        for a, c in combinations(range(len(P)), 2):
            dh, dt = abs(hc[a] - hc[c]), abs(tc[a] - tc[c])
            d_hist.append(dh)
            d_time.append(dt)
            frac_time_wins.append(dt > dh)
        groups = defaultdict(list)
        for k, i in enumerate(P):
            groups[g["hist_key"][i]].append(k)
        for ks in groups.values():
            group_sizes.append(len(ks))
            if len(ks) >= 2 and np.ptp(L[ks]) > 0:
                within.append(spearmanr(S[ks], -L[ks])[0])
        if len(groups) >= 3:
            gs = [np.mean(S[ks]) for ks in groups.values()]
            gl = [np.mean(L[ks]) for ks in groups.values()]
            if np.ptp(gl) > 0:
                across.append(spearmanr(gs, -np.array(gl))[0])
    within, across = np.array(within), np.array(across)
    d_hist, d_time = np.array(d_hist), np.array(d_time)
    print(f"reconstruction of stored scores from weights x inputs: max |error| {max(recon_err):.2e}")
    print(f"a) within histogram-sharing groups (only time_left differs): {len(within)} groups, "
          f"Spearman(score, -plan length) mean {within.mean():+.3f}; = +1 in {100 * np.mean(within > 0.999):.1f}%, "
          f"= -1 in {100 * np.mean(within < -0.999):.1f}%")
    print(f"   across groups (group-mean score vs group-mean plan length): {len(across)} states, "
          f"Spearman mean {across.mean():+.3f}")
    print(f"   group sizes among passenger goals: {dict(sorted(__import__('collections').Counter(group_sizes).items()))}")
    print(f"b) per pair of passenger goals in a state: |time contribution difference| median {np.median(d_time):.4f} "
          f"(p90 {np.percentile(d_time, 90):.4f}); |histogram contribution difference| median {np.median(d_hist):.4f} "
          f"(p90 {np.percentile(d_hist, 90):.4f}); time term larger in {100 * np.mean(frac_time_wins):.1f}% of pairs")
    print(f"   rank correlation with -plan length, across a state's passengers: time term {np.mean(time_vs_len):+.3f}, "
          f"histogram term {np.mean(hist_vs_len):+.3f}")
    print(f"   typical plan-length gap 20 frames -> time_left gap {20 / 2000:.3f} -> score gap {abs(w_t) * 20 / 2000:.4f}")


if __name__ == "__main__":
    main()
