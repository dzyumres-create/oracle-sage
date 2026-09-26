# Cell 6: WL depth-selection diagnostics

Three tools for choosing L (WL refinement depth) per `graph_convention`, by measurable,
reproducible criteria instead of the never-committed claims Cell 4's commit messages made.
None of them are wired into training — they are standalone scripts (or, for (c), an
always-on but off-by-default logging hook).

## Tools

- **(a) Held-out OOV + vocab growth** — `sage/domains/utils/wl_depth_sweep.py`. Builds a
  vocab from corpus A (random-action stepping, `build_wl_vocab.sample_action`, matching
  Cell 4's own method), freezes it, then measures OOV-node-fraction against two disjoint
  held-out corpora: B-random (same random-action method, disjoint seed) and B-policy
  (`build_wl_vocab.greedy_action` — nearest-passenger-then-destination — plus
  `Planner.plan()` projections of a few random candidate goals per visited state, so
  planner-touched states get real held-out coverage too, not just live-stepped ones).
  Reports vocab-size-vs-corpus-size growth checkpoints alongside both OOV numbers.
- **(b) Collision test** — `sage/domains/utils/wl_collision_check.py`. Drives
  `Planner.plan()` the way `project_actions` does: samples states, projects several
  candidate goals per state, and for every pair whose GROUND-TRUTH projected state
  differs (decoded via `planner.py`'s own `graph_to_networkx`/`_vilg`/`graph_to_state_atom`)
  but whose WL histogram is identical, counts a collision. Three histogram measurements
  per pair: growing (fresh shared vocab per L, no OOV possible — pure encoding
  expressiveness), frozen (the vocab (a) would build/freeze at that L — what a deployed
  policy actually sees, where OOV can itself manufacture spurious collisions), and floor
  (each candidate's own colour refinement run to a fixed point, capped at
  `--stable-cap` — pairs still collided there are WL-indistinguishable at ANY depth).
  Broken down by candidate-pair type (move-move, move-pickup, pickup-pickup,
  move-dropoff, pickup-dropoff, dropoff-dropoff), classified the same way
  `Planner.plan()` itself dispatches a goal.
- **(c) Runtime OOV logging** — `attach_wl(data, graph_convention="atom", site=None)`
  in `sage/domains/gym_taxi/utils/representations.py`. When `site` is given
  (`"decoder"` from `json_to_atom_graph`, `"planner"` from the atom planner's
  `_atoms_to_projection`), logs `wl_atom/oov_fraction_{site}` via
  `logger.record_mean` — the same SB3 logger `graph_plan_feedback_policy.py` already
  uses for `action_selection/*`, so it surfaces automatically at whatever
  `--log-interval` a real training run is using, with zero extra wiring. Purely a
  side-effect read of the already-computed colours — verified (test below) to never
  change `attach_wl`'s output.

Both (a) and (b) reuse `build_wl_vocab.py`'s env-construction/`sample_action` machinery
by import — `build_wl_vocab.py` itself gained `road_graph`/`next_hop`/`greedy_action`/
`to_planner_data` for this purpose, generic across `graph_convention`.

## Tests

- `tests/test_wl_diagnostics.py` — unit coverage for the building blocks in all three
  tools (greedy policy legality, `to_planner_data` shapes, `run_depth` checkpoints,
  `measure_held_out_oov`, `classify_goal`/`state_key`/`wl_to_stable_partition`/
  `select_candidates`), plus `TestKnownAnswerCollisionDepth` (see Calibration below).
- `tests/test_wl_atom_wiring.py::TestAttachWlSiteLoggingDoesNotChangeOutputs` — confirms
  `attach_wl`'s `wl_colours`/`wl_histogram` output is bit-identical regardless of `site`.

Run: `python -m pytest tests/test_wl_diagnostics.py tests/test_wl_atom_wiring.py -v`

## Calibration (against vilg, before trusting the tool on atom)

Cell 4's own claim ("L=1 collides completely on real move-vs-move candidates; L=2
resolves it", commit `b57707d`) was never backed by a committed script — treated as a
qualitative reference, not ground truth to literally reproduce. Calibration instead
checked the TOOL's own correctness:

1. **Subset invariant** (growing mode: colliding at L+1 must imply colliding at L, since
   a shared vocab only ever gains signatures) — **PASS**, 0 violations across every run
   in this document (predictable5: 67,998 pairs; city: 25,199 pairs; plus atom's own
   67,998/16,800). Frozen mode is NOT asserted (OOV can legitimately violate it) but was
   reported every time — see the atom/city table below for a case where it actually did.
2. **Isomorphism ground truth** (predictable5 only, `--check-isomorphism`): every pair
   still colliding at the floor was checked with `nx.is_isomorphic` (node/edge match on
   the exact one-hot features). Result: **964/964 floor-colliding vilg pairs were
   genuinely isomorphic** — zero false positives, i.e. the tool never reports a "floor
   collision" for two structurally-distinguishable graphs.
3. **Known-answer test** (`TestKnownAnswerCollisionDepth`, now a permanent regression
   test): a hand-built two-branch vilg graph (hub `H`, branches `H-A1-A2` and `H-B1-B2`,
   passenger at `A2` only). Derivation: vilg materialises each road as its own
   `adjacent` proposition node, so 1 road-hop = 2 WL-graph-hops; the passenger's
   distinguishing feature is its own `in(P,A2)` proposition, one more hop beyond `A2`
   itself — so from `A1`, reaching it costs 2 (to `A2`) + 1 (to the proposition) = 3
   WL-hops. Predicted: `A1`/`B1` collide through L=2, separate at L=3. **Verified
   empirically exactly as derived.**

All three passed → tool treated as calibrated.

## Commands and seeds (as actually run)

All commands assume the `sage` conda env and `PYTHONPATH=<repo root>`.

**Calibration — vilg, predictable5** (`--check-isomorphism --inspect-L 2`):
```
python -m sage.domains.utils.wl_collision_check --graph-convention vilg --scenario predictable5 \
  -L 1 2 3 4 --episodes 10 --steps-per-episode 60 --sample-every 3 \
  --vocab-episodes 15 --vocab-steps-per-episode 60 --stable-cap 15 \
  --check-isomorphism --inspect-L 2 --inspect-limit 3
```
State seed 0, vocab-build seed 700000 (disjoint) — 200 states, exhaustive candidates.

**Calibration — vilg, city:**
```
python -m sage.domains.utils.wl_collision_check --graph-convention vilg --scenario city \
  -L 1 2 3 4 --episodes 12 --steps-per-episode 60 --sample-every 3 --no-exhaustive --k 15 \
  --planner-seed 7 --vocab-episodes 8 --vocab-steps-per-episode 50 --stable-cap 15
```
State seed 0, vocab-build seed 700000, candidate-subset seed 7 — 240 states, k=15.

**(a) OOV sweep, predictable5** (run for `--graph-convention {atom,oracle_sage,vilg}`):
```
python -m sage.domains.utils.wl_depth_sweep --graph-convention <conv> --scenario predictable5 \
  -L 1 2 3 4 --seed 0 --episodes 20 --steps-per-episode 50 \
  --held-out-seed 500000 --held-out-episodes 10 --held-out-steps-per-episode 50 \
  --policy-seed 900000 --policy-episodes 10 --policy-steps-per-episode 50 --policy-goals-per-state 3
```

**(a) OOV sweep, city** (run for `--graph-convention {atom,oracle_sage,vilg}`):
```
python -m sage.domains.utils.wl_depth_sweep --graph-convention <conv> --scenario city \
  -L 1 2 3 4 --seed 0 --episodes 30 --steps-per-episode 60 \
  --held-out-seed 500000 --held-out-episodes 15 --held-out-steps-per-episode 60 \
  --policy-seed 900000 --policy-episodes 15 --policy-steps-per-episode 60 --policy-goals-per-state 3
```

**(b) Collision test, atom, predictable5:**
```
python -m sage.domains.utils.wl_collision_check --graph-convention atom --scenario predictable5 \
  -L 1 2 3 4 --episodes 10 --steps-per-episode 60 --sample-every 3 \
  --vocab-episodes 15 --vocab-steps-per-episode 60 --stable-cap 15
```

**(b) Collision test, atom, city:**
```
python -m sage.domains.utils.wl_collision_check --graph-convention atom --scenario city \
  -L 1 2 3 4 --episodes 8 --steps-per-episode 60 --sample-every 3 --no-exhaustive --k 15 \
  --planner-seed 7 --vocab-episodes 8 --vocab-steps-per-episode 50 --stable-cap 15
```

All vocab-build/held-out seeds (500000, 700000, 900000) are disjoint from the
state-sampling seed (0) and from each other, by construction (`assert`ed in both
scripts).

## Results: (a) held-out OOV

**predictable5** (corpus A: 20 episodes x 50 steps; B-random: 10x50; B-policy: 10x50, 3
goals/state):

| conv | L | vocab size | OOV[B-random] | OOV[B-policy] |
|---|---|---:|---:|---:|
| atom | 1 | 127 | 0.004% | 0.030% |
| atom | 2 | 1,539 | 4.07% | 2.12% |
| atom | 3 | 8,602 | 30.85% | 13.41% |
| atom | 4 | 25,412 | 59.88% | 31.09% |
| oracle_sage | 1 | 26 | 0.00% | 0.79% |
| oracle_sage | 2 | 271 | 2.50% | 4.28% |
| oracle_sage | 3 | 1,535 | 15.95% | 14.21% |
| oracle_sage | 4 | 4,747 | 42.55% | 32.28% |
| vilg | 1 | 30 | 0.04% | 4.44% |
| vilg | 2 | 155 | 0.13% | 13.48% |
| vilg | 3 | 505 | 0.91% | 17.94% |
| vilg | 4 | 2,095 | 5.28% | 36.39% |

**city** (corpus A: 30x60; B-random: 15x60; B-policy: 15x60, 3 goals/state):

| conv | L | vocab size | OOV[B-random] | OOV[B-policy] |
|---|---|---:|---:|---:|
| atom | 1 | 213 | 0.022% | 0.016% |
| atom | 2 | 6,439 | 3.07% | 2.27% |
| atom | 3 | 50,545 | 37.98% | 36.35% |
| atom | 4 | 139,022 | 81.23% | 80.93% |
| oracle_sage | 1 | 29 | 0.047% | 0.244% |
| oracle_sage | 2 | 640 | 0.80% | 0.97% |
| oracle_sage | 3 | 7,443 | 18.62% | 17.62% |
| oracle_sage | 4 | 25,467 | 64.21% | 64.55% |
| vilg | 1 | 36 | 0.001% | 0.083% |
| vilg | 2 | 240 | 0.062% | 0.237% |
| vilg | 3 | 996 | 0.333% | 0.533% |
| vilg | 4 | 6,773 | 3.36% | 3.28% |

All three conventions: vocab size grows combinatorially past L=2-3 at this corpus scale,
and OOV explodes correspondingly. B-policy OOV consistently exceeds B-random (planner
projections + real pickups/deliveries visit structurally different states than pure
random walk alone).

## Results: (b) collision rates (growing% / frozen% / floor%)

**vilg** (predictable5: 200 states, exhaustive; city: 240 states, k=15, seed 7):

| scenario | pair_type | L=1 | L=2 | L=3 | L=4 | floor |
|---|---|---|---|---|---|---:|
| predictable5 | move-move | 29.16/29.53 | 7.78/8.45 | 4.09/4.75 | 2.35/3.08 | 1.57% |
| city | move-move | 29.01/29.01 | 4.44/4.44 | 0.47/0.47 | 0.05/0.06 | 0.00% |

**atom** (predictable5: 200 states, exhaustive, seed 0; city: 160 states, k=15, seed 7):

| scenario | pair_type | L=1 | L=2 | L=3 | L=4 | floor |
|---|---|---|---|---|---|---:|
| predictable5 | move-move | 13.34/13.34 | 9.31/9.31 | 8.15/8.15 | 8.09/8.09 | 8.09% |
| city | move-move | 4.23/4.23 | 0.05/0.05 | 0.00/0.03 | 0.00/**6.04** | 0.00% |

All other pair types (dropoff-pickup, move-pickup, pickup-pickup, dropoff-dropoff) were
0% across the board for both conventions/scenarios in every run — pickup/dropoff targets
are locally distinguishable via the taxi's own carrying-status colour; only move-move
(and, more weakly, dropoff-move) actually collides.

**Notable finding — atom/city, L=4**: frozen move-move jumps back up to 6.04% despite
growing/floor both being 0%, and the frozen-mode subset invariant recorded 1,002
violations there (vs. 10 on vilg/predictable5, 0 elsewhere) — the L=4 vocab (139,022
entries, from only a 1,800-graph build corpus) is 81% OOV on held-out data, so deeper
refinement is actively hurting real deployed behavior via OOV-driven spurious
collisions, not helping. Atom already reaches its floor at L=3 on city (0.03% frozen,
matching 0.00% floor almost exactly) - L=4 buys nothing at this corpus scale and adds
vocab-coverage risk. This is exactly why (a)'s held-out OOV and (b)'s frozen-mode
numbers both matter alongside growing/floor: growing/floor alone would have recommended
"go deeper is free," which the frozen numbers show is false once a specific vocab has to
actually cover it.

## Depth-bucketed OOV: the corrected standard (post-tooling-fix)

A live training smoke run logged runtime OOV (`wl_atom/oov_fraction_{decoder,planner}`,
tool (c) above) around 3.3% — roughly 10x the ~0.3% held-out OOV (a) had reported for
the same vocab. Root cause, confirmed by re-measuring OOV bucketed by
simulator-step-in-episode depth: every corpus builder above (`sample_graphs`,
`collect_graph_corpus`, `collect_policy_corpus`) reset each episode after a fixed step
count (`--steps-per-episode`, typically 50-60) instead of running it to its natural
end, so held-out corpora only ever sampled the shallow part of the state space. A real
city episode runs to its ~2000-step timeout; OOV climbs steeply with depth for every
convention (see tables below), so short-episode held-out corpora systematically
under-reported the OOV a real training run actually sees.

**Tooling fix** (`build_wl_vocab.py`, `wl_depth_sweep.py`):
- every corpus builder now calls `env.seed(seed)` (previously only the global numpy
  RNG was seeded, so `--seed` never actually pinned maze/passenger-spawn generation —
  held-out corpora were not reliably reproducible or seed-disjoint from the build
  corpus)
- episodes run to their natural end (`run_full_episode_states`), sampling a state
  every `--sample-every` steps, instead of resetting early
- the vocab-building corpus now mixes half epsilon-greedy episodes (`eps=0.2` — mostly
  greedy nearest-passenger-then-destination, occasionally random, so sampled states
  aren't confined to the greedy policy's own narrow on-path slice) and half
  uniform-random episodes, plus `goals_per_state` planner-projected candidate goals
  per sampled state — not uniform-random actions only
- held-out OOV is now reported **bucketed by depth** (`BUCKETS = [(0,60), (60,250),
  (250,1000), (1000,inf)]`), on **two disjoint held-out policies**: B-random (uniform
  random actions throughout, no projections) and B-policy (pure greedy, `eps=0`, plus
  planner projections) — this is now the standard report for every WL cell, not just
  a flat/pooled OOV number
- `wl_depth_sweep.py --vocab-path <file>` measures an already-saved, already-frozen
  vocab against this standard without rebuilding it (used for Cell 3/Cell 4/the old
  atom vocab below, none of which needed rebuilding)

**The <1% OOV target was tried and dropped.** The original acceptance bar for the new
atom/L=2 build was "held-out OOV under 1% in every depth bucket, both policies." Even
after the tooling fix, rebuilding from a ~50K-graph corpus with the corrected
procedure (below) left OOV at 1.2-3.0% in the three deeper buckets — 2-3x over target
— and the vocab-growth curve was still adding new signatures at the 50K mark (not
flat), so the deficit is a real, structural long tail of the atom encoding at L=2 on
city, not a bug. Scaling further would likely narrow it somewhat but was not expected
to clear 1% at every bucket (diminishing returns were already visible in the growth
curve), and OOV alone was never actually the thing that mattered — it is only a proxy
for whether the frozen vocab confuses two structurally different candidate states.
That was checked directly: the **depth-bucketed frozen move-move collision rate**
(the real safety criterion) is 0.04-0.16% at every depth for the new atom vocab (see
below) — far below any level of concern, despite the elevated OOV. Going forward, OOV
is reported for every WL cell as a **descriptive property of the encoding at a given
corpus scale** (how much of the state space a vocab of this size actually covers), not
a pass/fail gate; the frozen move-move collision rate, measured depth-bucketed, is the
actual acceptance criterion.

**Build procedure standard, applied uniformly across WL cells going forward**: full
episodes, the same `--sample-every`, the same half-epsilon-greedy/half-random policy
mix (`eps=0.2`, `goals_per_state=2`), and a corpus size of ~50K sampled graphs
(recorded in the saved vocab's `build_procedure` metadata, including the git commit it
was built at).

### Growth curve — new atom vocab (city, L=2, seed 0, 84 full episodes, sample_every=10)

| graphs | vocab size | Δ |
|---:|---:|---:|
| 2,000 | 17,975 | +17,975 |
| 10,000 | 48,050 | +5,808 |
| 20,000 | 72,797 | +4,910 |
| 30,000 | 87,439 | +1,570 |
| 40,000 | 96,060 | +1,302 |
| 50,000 | 105,148 | +1,774 |
| 50,400 (final, frozen incl. OOV) | 105,340 | — |

Deceleration is clear (from +17,975 to ~+1,300-2,300 per 2,000-graph checkpoint) but
not flat — the tail is still adding signatures, consistent with a genuine long tail
rather than a stalled/broken build.

### Depth-bucketed held-out OOV, all four vocabs (city scenario)

Held-out seeds: B-random 500001, B-policy 900001 (both disjoint from every build
seed), 8 episodes each, `sample_every=20`, `goals_per_state=3` for B-policy.

**Cell 3** (`wl_vocab_taxi_city_L1_edgefixed.json`, oracle_sage, L=1 — pre-existing,
no rebuild, measured only):

| bucket | OOV[B-random] | OOV[B-policy] |
|---|---:|---:|
| 0-60 | 0.0000% | 0.0619% |
| 60-250 | 0.0000% | 0.1466% |
| 250-1000 | 0.0008% | 0.1428% |
| 1000+ | 0.0000% | 0.1385% |

Excellent at every depth for both policies — well under 1%, no rebuild needed.

**Cell 4** (`wl_vocab_taxi_city_vilg_L2.json`, vilg, L=2 — pre-existing, no rebuild,
measured only):

| bucket | OOV[B-random] | OOV[B-policy] |
|---|---:|---:|
| 0-60 | 0.0000% | 0.1113% |
| 60-250 | 0.0035% | 2.1898% |
| 250-1000 | 0.0016% | 9.9726% |
| 1000+ | 0.0358% | 21.7916% |

Confirms the same episode-depth effect on vilg: negligible under B-random, but
B-policy OOV climbs to ~22% at depth — vilg's convention keeps delivered passengers'
predicate nodes alive indefinitely (see `env_to_vilg_graph`), so the graph a
planner-projected state sees keeps growing in structurally novel ways deep into a long
episode.

**Atom, OLD vocab** (`wl_vocab_taxi_city_atom_L2.json`, size 33,336, built from 50,000
graphs with the *pre-fix* short-episode tooling — **superseded**, kept only for
reference/comparison; do not use for new work):

| bucket | OOV[B-random] | OOV[B-policy] |
|---|---:|---:|
| 0-60 | 0.2669% | 0.2063% |
| 60-250 | 3.1316% | 2.0299% |
| 250-1000 | 4.6171% | 4.3136% |
| 1000+ | 4.6280% | 4.8657% |

**Atom, NEW vocab** (`wl_vocab_taxi_city_atom_L2_full.json`, size 105,340, built from
50,400 graphs with the corrected full-episode tooling — **current production vocab**):

| bucket | OOV[B-random] | OOV[B-policy] |
|---|---:|---:|
| 0-60 | 0.2609% | 0.1492% |
| 60-250 | 1.9543% | 1.1890% |
| 250-1000 | 2.9088% | 2.5701% |
| 1000+ | 2.8927% | 3.0422% |

The corrected tooling roughly halves OOV at depth versus the old vocab (e.g. 1000+
B-policy: 4.87% -> 3.04%) but does not clear 1% — see "the <1% target was tried and
dropped" above for why this is accepted.

### Depth-bucketed frozen move-move collision — atom, NEW vocab

Greedy policy (`eps=0`) + planner projections, 8 full episodes, `sample_every=20`,
seed 700001 (disjoint from both the build and the OOV held-out seeds), `k=15`
candidates/state:

| bucket | states | pairs | collisions | collision% |
|---|---:|---:|---:|---:|
| 0-60 | 24 | 2,478 | 4 | 0.1614% |
| 60-250 | 80 | 7,784 | 6 | 0.0771% |
| 250-1000 | 296 | 27,854 | 16 | 0.0574% |
| 1000+ | 400 | 37,954 | 16 | 0.0422% |

Collision rate is low and, if anything, *decreasing* with depth (more candidates per
deep state means more distinguishing context, not less) — despite OOV increasing with
depth over the same range. This is the direct evidence that elevated OOV at depth is
not translating into the policy actually confusing distinct candidate states, which is
why OOV was demoted from a pass/fail gate to a descriptive metric (see above).

## Verification

Full test suite green throughout (`python -m pytest tests/ -q`): 158 passed, 5 skipped
(CUDA-only), 216 subtests passed, 0 failed — confirmed after every code change in this
document's scope, including the corpus-builder tooling fix (env seeding, full-episode
sampling, policy mix, depth-bucketed OOV/collision reporting) described above.
