# Cell 4 at L=1: vilg vocab, speed-up and code-path verification

Cell 4 (WL encoder + vILG graph encoding, `--graph-convention vilg`) was trained at WL
depth L=2 on branch `cell4-wl-vilg` (tip `ad3b977`). This branch, `cell4-vilg-L1`,
reruns it at **L=1** so that every WL cell uses the same depth.

**What differs from the Cell 4 L=2 runs: only L and the vocab.** The policy-input code
path is verified byte-identical to `cell4-wl-vilg@ad3b977` (section 2). Run with the same
flags as the Cell 4 L=2 runs, changing only:

```
--wl-vocab-path sage/domains/utils/wl_vocab_taxi_city_vilg_L1_full.json --wl-num-iterations 1
```

The branch was created from `cell6-wl-atom@46260c9`, which already contains Cell 4's code
plus the atom encoding, the corrected WL vocab tooling, the periodic checkpoint callback
and the WL diagnostics. Commits added here:

| commit | change |
|---|---|
| `20b3c99` | vilg: skip building observation JSON on intermediate plan steps |
| `96395b6` | `wl_colours`: collect colour ids in a list instead of per-node tensor writes |
| `f990e8a` | WL vocab guard: warn on metadata-less vilg vocabs; vilg tests for both loaders |
| `1884673` | `wl_collision_depth`: depth-bucketed frozen collision rate for a saved vocab |
| `01a9222` | vilg L=1 vocab `wl_vocab_taxi_city_vilg_L1_full.json` |

## 1. Speed-up

A Cell 4 seed took ~64 h on RCP, vs ~20-26 h for Cell 6 (atom).

### Profile (before)

`docs/tools/profile_vilg_pipeline.py`: a short CPU training run (city, vilg, Cell 4's L=2
vocab, 32 envs, 2,080 timesteps = 13 rollouts x 5 decisions x 32 envs, seed 0), with
timing wrappers on each pipeline stage. Mac CPU. py-spy needs root on macOS, so explicit
timers were used instead.

| stage | time | share |
|---|---:|---:|
| env-side WL (inside `env_to_vilg_graph`) | 408.5 s | 41.8% |
| vilg graph construction (excl. WL) | 216.1 s | 22.1% |
| JSON serialisation (`graph_to_json`) | 153.9 s | 15.7% |
| planner, total (9,568 `Planner.plan` calls) | 124.0 s | 12.7% |
| - of which planner-side WL | 84.8 s | 8.7% |
| JSON decoding (`json_to_graph`) | 35.0 s | 3.6% |
| vec-env buffer copies | ~24 s | 2.5% |
| policy network forward/backward | ~2 s | 0.2% |
| **total** | **977.9 s** | |

There were 45,258 simulator steps and 45,290 observation JSONs built, but only ~2,112
distinct observations were read by the policy: **95% of the observation JSON was built
and thrown away**. `collect_rollouts` executes each env's plan step by step but reads
only the observation after the env's last plan step. Every earlier step's JSON is
overwritten in the vec-env buffer by the same env's next step.

### Changes

- **Skip intermediate observations, vilg only (`20b3c99`).** `collect_rollouts` passes
  `obs_mask = plan_feedback_a2c.observation_needed(plan_step, plan_lengths)` to
  `AsyncVecEnv.step`. It is true exactly at each env's last plan step, i.e. the step whose
  observation that env's next decision reads (plans differ in length across envs; an env
  is never stepped after its plan ends).
  - Where `obs_mask` is false, the env is asked (`GraphTaxiEnv.skip_next_observation`,
    one-shot) not to build the JSON, and its buffer slot is left unchanged.
  - Only `graph_convention="vilg"` honours the request; oracle_sage and atom decline it
    and build every observation exactly as before.
  - If a skipped step ends the episode (timeout mid-plan), the observation is built anyway,
    so `terminal_observation` is unchanged; the reset observation is always built.
  - `obs_mask=None` (the default) builds every observation, so every other caller
    (`build_wl_vocab`, `wl_depth_sweep`, `wl_collision_check`, tests) behaves exactly as
    before. Building the JSON has no side effects on the simulator, so skipping it cannot
    change the trajectory.
- **Faster WL (`96395b6`).** `wl_colours`/`refine` wrote each node's colour into a tensor
  one element at a time; those writes dominated WL's run time. Ids are now resolved into a
  Python list in the same node order and turned into one tensor at the end (city vilg
  graph, L=2: 11.8 ms -> 2.4 ms). This is shared code (all conventions); outputs are
  identical (verified, section 2).

Not done, by decision: moving vilg WL to the decoder side (with intermediate observations
skipped, it would compute WL ~2.2x per observation the policy reads instead of once,
because `evaluate_actions` re-decodes), and a compact vilg JSON (serialisation is ~3% of
the remaining time).

### Timing

Same profiler, same command:

| | wall | per decision | observation JSONs built | env-side WL | planner (of which WL) |
|---|---:|---:|---:|---:|---:|
| before (`46260c9`) | 977.9 s | 470 ms | 45,290 | 408.5 s | 124.0 s (84.8 s) |
| after skip (`20b3c99`) | 217.8 s | 105 ms | 2,112 | 18.8 s | 125.1 s (85.8 s) |
| after skip + WL (`96395b6`) | 149.4 s | 72 ms | 2,112 | 6.3 s | 67.4 s (27.9 s) |

**6.5x faster per decision overall.** Simulator steps per run differ (45,258 / 49,432 /
46,306) because `gnn_global.py` passes `--seed` to the envs only, not to torch, so
action sampling differs run to run (pre-existing). The comparison is per decision; each
run made 2,080 decisions. After both changes the planner is the largest remaining cost
(45%; planner graph operations ~39 s, planner WL ~28 s), then JSON decoding (23%).

Command (on the Mac this needs the stack shims built into the script):
```
PYTHONPATH=. python docs/tools/profile_vilg_pipeline.py <outdir> -- \
  --env-name city-taxi-unmasked-v1 --planner --feedback --shared-gnn --graph-convention vilg \
  --wl-vocab-path sage/domains/utils/wl_vocab_taxi_city_vilg_L2.json --wl-num-iterations 2 \
  --num-processes 32 --num-env-steps 2000 --entropy-coef 5e-5 --learning-rate 3e-4 --seed 0 \
  --checkpoint-interval 0 --log-interval 5 --log-dir <outdir>/tb --save-dir <outdir>/save
```

## 2. Verification: byte-identical policy inputs

`docs/tools/policy_input_harness.py` drives an `AsyncVecEnv` of 3 city envs (seeds 0,
300, 600; plan lengths differ across envs within a decision, as in training) through the
`collect_rollouts` plan loop, passing `obs_mask` when the code version supports it. At
each of 250 decisions, for every env, it records a digest (dtype, shape, sha256 of the
bytes) of every tensor the policy reads:

- `x`, `edge_index`, `edge_attr`, `mask`, `global_features`, `wl_colours`, `wl_histogram`;
- for the decoded live observation and for K=3 planner projections (passenger goals,
  i.e. pickup + delivery, or location goals);
- plus the raw JSON's sha256 and the planner's action lists.

On every episode end it records `terminal_observation` (JSON and decoded tensors), the
following reset observation, and whether the episode ended mid-plan. Goal sequences are
recorded on the reference commit and replayed on the new code
(`docs/tools/policy_input_compare.py` compares the two).

Every run covered 750 env-decisions with 385 pickup plans, 375 deliveries and
**12 episode ends, all 12 in the middle of a plan** (city's 2,000-step timeout).

| convention | reference | new code | tensors compared | JSON built (ref -> new) | result |
|---|---|---|---:|---:|---|
| vilg, Cell 4 L=2 vocab | `cell4-wl-vilg@ad3b977` | `20b3c99` | 21,000 | 24,270 -> 765 | **byte-identical** |
| vilg, Cell 4 L=2 vocab | `cell4-wl-vilg@ad3b977` | `20b3c99`, no `obs_mask` | 21,000 | 24,270 -> 24,270 | byte-identical |
| vilg, Cell 4 L=2 vocab | `cell4-wl-vilg@ad3b977` | `96395b6` | 21,000 | 24,270 -> 765 | **byte-identical** |
| oracle_sage (default vocab) | `46260c9` | `20b3c99` and `96395b6` | 21,000 | 24,270 -> 24,270 | byte-identical |
| atom, `atom_L2_full` | `46260c9` | `20b3c99` and `96395b6` | 21,000 | 24,270 -> 24,270 | byte-identical |

oracle_sage and atom are given `obs_mask` too; their unchanged JSON counts confirm they
ignore it.

Commands (reference worktrees were detached checkouts; no branch was modified):
```
git worktree add --detach ../oracle-sage-verify-ad3b977 ad3b977
git worktree add --detach ../oracle-sage-verify-46260c9 46260c9
V=$PWD/sage/domains/utils
(cd ../oracle-sage-verify-ad3b977 && PYTHONPATH=$PWD python $OLDPWD/docs/tools/policy_input_harness.py \
   --convention vilg --vocab $V/wl_vocab_taxi_city_vilg_L2.json --L 2 --decisions 250 \
   --mode record --out vilg_ref.json)
PYTHONPATH=. python docs/tools/policy_input_harness.py --convention vilg \
   --vocab $V/wl_vocab_taxi_city_vilg_L2.json --L 2 --decisions 250 \
   --mode replay --goals vilg_ref.json --out vilg_new.json
python docs/tools/policy_input_compare.py vilg_ref.json vilg_new.json
# oracle_sage: --convention oracle_sage (no --vocab); atom: --convention atom
#   --vocab $V/wl_vocab_taxi_city_atom_L2_full.json --L 2; reference worktree 46260c9
```

The WL change was also checked directly:
- `tests/test_wl_colours_list_refine.py` compares against a verbatim copy of the previous
  implementation. It covers real city graphs (live states and projections) for all three
  conventions, L=0-3, frozen and growing mode (growing-mode vocab contents and insertion
  order compared), plus edge cases.
- Building a vocab with the old (`20b3c99`) and new code on the same corpus gives
  **byte-identical vocab files** for all three conventions. Settings: city, L=2, 2 full
  episodes, sample_every 50, seed 0, eps 0.2, goals_per_state 2. Vocab sizes: oracle_sage
  769, vilg 522, atom 4,774.

The skip logic has its own tests (`tests/test_vilg_obs_skip.py`): `observation_needed` per
env; the observation buffer is identical to a control run after every plan; a forced
mid-plan timeout gives identical `terminal_observation` and reset observation; the default
builds every observation; oracle_sage and atom decline the skip.

## 3. Vocab metadata guard

`validate_wl_vocab_metadata` is shared by both vocab loaders (`configure_wl_vocab_override`
and `WLPlanFeedbackPolicy._load_vocab`). A vocab that records metadata is fully enforced
for every convention: recorded convention and L must match the run. Metadata-less vilg
vocabs (such as Cell 4's `wl_vocab_taxi_city_vilg_L2.json`) are still accepted, but now
emit a `UserWarning` that the convention and L cannot be verified. oracle_sage
(metadata-less default vocab, no warning) and atom (metadata required) are unchanged. The
new L=1 vocab records metadata, so `--wl-num-iterations 2` with it is rejected at startup.

## 4. The vilg L=1 vocab

`sage/domains/utils/wl_vocab_taxi_city_vilg_L1_full.json`: **111 entries** (incl. OOV).
It is built with exactly the procedure and settings of `wl_vocab_taxi_city_atom_L1_full.json`
(see `docs/cell6_wl_diagnostics.md`, "Build procedure standard"); only the convention
differs:

```
python -m sage.domains.utils.build_wl_vocab --graph-convention vilg --scenario city -L 1 \
  --episodes 84 --sample-every 10 --seed 0 --eps 0.2 --goals-per-state 2 --log-every 2000 \
  --out sage/domains/utils/wl_vocab_taxi_city_vilg_L1_full.json
```

The recorded metadata is `graph_convention: vilg`, `num_iterations: 1`, and
`build_procedure: {episodes: 84, sample_every: 10, eps: 0.2, goals_per_state: 2, seed: 0,
full_episodes: true, git_commit: f990e8a}`. The corpus is 50,400 graphs; the build took
2,089 s on Mac CPU.

### Growth curve

| graphs | vocab size | delta |
|---:|---:|---:|
| 2,000 | 79 | +79 |
| 4,000 | 82 | +3 |
| 6,000 | 87 | +5 |
| 8,000 | 90 | +3 |
| 10,000 | 91 | +1 |
| 12,000 | 95 | +4 |
| 14,000 | 102 | +7 |
| 16,000 | 107 | +5 |
| 18,000 | 107 | +0 |
| 20,000 | 107 | +0 |
| 22,000 | 109 | +2 |
| 24,000 | 110 | +1 |
| 26,000 - 50,000 | 110 | +0 at every checkpoint |
| 50,400 (final, frozen incl. OOV) | 111 | - |

Flat from 24,000 graphs: converged at this corpus scale (cf. atom L1_full: 827 entries,
also flat).

### Depth-bucketed held-out OOV

Same standard and seeds as `docs/cell6_wl_diagnostics.md`: B-random seed 500001 (uniform
random actions, no projections), B-policy seed 900001 (greedy, eps=0, plus 3 planner
projections per state), 8 full episodes each, sample_every 20.

```
python -m sage.domains.utils.wl_depth_sweep --graph-convention vilg --scenario city \
  --vocab-path sage/domains/utils/wl_vocab_taxi_city_vilg_L1_full.json \
  --held-out-seed 500001 --held-out-episodes 8 --held-out-sample-every 20 \
  --policy-seed 900001 --policy-episodes 8 --policy-sample-every 20 --policy-goals-per-state 3
```

| bucket | L=1 OOV[B-random] | L=1 OOV[B-policy] | Cell 4 L=2 OOV[B-random] | Cell 4 L=2 OOV[B-policy] |
|---|---:|---:|---:|---:|
| 0-60 | 0.0000% | 0.0000% | 0.0000% | 0.1113% |
| 60-250 | 0.0000% | 0.0000% | 0.0035% | 2.1898% |
| 250-1000 | 0.0000% | 0.0000% | 0.0016% | 9.9726% |
| 1000+ | 0.0000% | 0.0002% (4 of 2,492,623 nodes) | 0.0358% | 21.7916% |

The Cell 4 L=2 columns were re-measured on this branch with the same command
(`--vocab-path .../wl_vocab_taxi_city_vilg_L2.json -L 2`) and match
`docs/cell6_wl_diagnostics.md` exactly. Node and graph counts are identical across the
two vocabs (same corpora).

### Depth-bucketed frozen move-move collision rate

Pairs of candidate goals whose ground-truth projected states differ but whose
frozen-vocab WL histograms are identical, i.e. pairs the policy cannot tell apart.
Measured with `sage/domains/utils/wl_collision_depth.py` (committed in `1884673`):
greedy policy, 8 full city episodes, a state every 20 steps, k=15 random candidates per
state, seed 700001. The candidate states and pairs do not depend on the vocab, so every
row below is measured on the same 800 states and 76,370 move-move pairs:

```
python -m sage.domains.utils.wl_collision_depth --graph-convention vilg \
  --vocab-path sage/domains/utils/wl_vocab_taxi_city_vilg_L1_full.json \
  --seed 700001 --episodes 8 --sample-every 20 --k 15
```

| bucket | states | move-move pairs | **vilg L=1 (new)** | vilg L=2 (Cell 4) | atom L=1 (Cell 6, `atom_L1_full`) | atom L=2 (`atom_L2_full`) | oracle_sage L=1 (Cell 3) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0-60 | 24 | 2,506 | **26.62%** (667) | 4.31% (108) | 4.31% (108) | 0.12% (3) | 26.62% (667) |
| 60-250 | 80 | 7,956 | **25.60%** (2,037) | 3.34% (266) | 3.57% (284) | 0.03% (2) | 26.01% (2,069) |
| 250-1000 | 296 | 28,089 | **21.54%** (6,051) | 2.03% (569) | 2.50% (702) | 0.03% (9) | 23.60% (6,629) |
| 1000+ | 400 | 37,819 | **18.99%** (7,183) | 2.20% (833) | 2.63% (993) | 0.04% (15) | 23.34% (8,828) |

The vilg L=1 vocab has ~0 OOV, so its collisions come from L=1's resolving power on
vilg, not from vocab coverage. vILG materialises every road as its own `adjacent`
proposition node, so one road hop is two WL hops (see the known-answer test in
`docs/cell6_wl_diagnostics.md`). At L=1 a location's colour cannot see the neighbouring
locations. This matches the growing-vocab calibration there (city vilg move-move: 29.0%
at L=1, 4.4% at L=2).

**At L=1, vilg resolves move-move candidates about as well as Cell 3's oracle_sage L=1 and
far worse than its own L=2 or atom L=1.** "Same L" therefore does not mean "same
resolving power" across conventions: vilg L=2 matches atom L=1 almost exactly here.
Other pair types for vilg L=1, all buckets: dropoff-move 79/299, pickup-pickup 10/173,
move-pickup 0/7,149, dropoff-pickup 0/9.

The atom L=2 row differs from the table in `docs/cell6_wl_diagnostics.md` (0.04-0.16%,
same settings). That table came from an uncommitted script whose candidate RNG wiring
is not recorded, so its exact counts cannot be reproduced; the magnitudes agree.

## 5. CPU smoke run

```
PYTHONPATH=. python sage/experiments/gnn_global.py --env-name city-taxi-unmasked-v1 \
  --planner --feedback --shared-gnn --graph-convention vilg \
  --wl-vocab-path sage/domains/utils/wl_vocab_taxi_city_vilg_L1_full.json --wl-num-iterations 1 \
  --checkpoint-interval 12000 --num-processes 32 --num-env-steps 1000 \
  --entropy-coef 5e-5 --learning-rate 3e-4 --seed 0 --log-interval 1
```

On the Mac this was run through a two-line wrapper that applies the gym 0.26 stack shims
(`Generator.randint`, `disable_env_checker`); RCP's gym 0.18 needs neither. Device: CPU.

The run completed: 7 iterations (1,120 timesteps, 20,333 simulator steps), final model
saved, no traceback, no NaN, no vocab-metadata warning. All logged losses are finite:

| update | policy_loss | value_loss | path_value_loss | entropy_loss | explained_variance |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.893 | 0.070 | 47.4 | -5.83 | 0.936 |
| 2 | 1.53 | 0.165 | 48.8 | -5.86 | 0.859 |
| 3 | 0.961 | 0.148 | 46.6 | -5.88 | 0.850 |
| 4 | 0.275 | 0.121 | 41.1 | -5.89 | 0.876 |
| 5 | -0.117 | 0.069 | 34.4 | -5.89 | 0.916 |
| 6 | -0.202 | 0.064 | 28.1 | -5.89 | 0.920 |

No checkpoint was written: `--checkpoint-interval 12000` is beyond the run's 1,120
timesteps.

## Test suite

`python -m pytest tests docs --ignore=docs/test_vilg.py`: 192 passed, 5 skipped (Mac,
gym 0.26 / torch 2.13 / PyG 2.8). `docs/test_vilg.py` is a module-level script that
unpacks a 5-tuple from `env_to_graph`; it has failed at collection since WL was added to
`env_to_graph` (it already fails at `46260c9`) and is unrelated to this work.
