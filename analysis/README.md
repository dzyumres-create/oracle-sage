# Cell 5: expressiveness / indistinguishability analysis

Offline analysis only. No training, no GPU, no changes to the simulator, the env, the
planner, or any convention's behaviour — `indistinguishability.py` only *imports*
(read-only) the real graph converters in `sage/domains/gym_taxi/utils/representations.py`
and the real WL machinery in `sage/domains/utils/wl_colours.py`.

Run with (from the repo root):

```
PYTHONPATH=. python analysis/indistinguishability.py            # full run (~40 min)
PYTHONPATH=. python analysis/indistinguishability.py --quick    # ~20s smoke test
```

(`PYTHONPATH=.` is needed because `sage` must be importable from the repo root; running
the script directly puts `analysis/` itself on `sys.path[0]` instead.)

## Part 1 — current Taxi domain (hypothesis: near-zero collisions everywhere)

**Method.** 5 seeds (0, 300, 600, 900, 1200) × 1000 states each, sampled every 5
simulator steps from a live `scenario="city"` `GraphTaxiEnv`, taking uniformly-random
*valid* actions. Each snapshot's graph is built under all three conventions using the
real converters (`env_to_graph`, `env_to_vilg_graph`, `atoms_to_graph`), then run through
this script's own **exact** WL colour refinement (a fresh, growing vocabulary per run —
*not* the trained/frozen vocab the real pipeline uses) for L = 1..5. "Optimal action" is
**approximated** as the best subgoal by shortest-path distance to the nearest
deliverable passenger (carrying a passenger → head for/attempt dropoff at their
destination; otherwise → head for/attempt pickup at the nearest waiting passenger by
road-graph hop count) — not a true optimal-action oracle, and reported as a
`(kind, hop_distance)` pair rather than a raw node id, since node ids aren't comparable
across different randomly-generated mazes.

**Result** (5000 raw snapshots → 1738 distinct ground-atom-set states):

| convention  | L | distinct states | distinct hashes | collisions | bad collisions |
|---|---|---|---|---|---|
| oracle_sage | 1 | 1738 | 538  | 14243 | 13288 |
| oracle_sage | 2 | 1738 | 1631 | 130   | 97    |
| oracle_sage | 3 | 1738 | 1717 | 21    | 1     |
| oracle_sage | 4 | 1738 | 1718 | 20    | 0     |
| oracle_sage | 5 | 1738 | 1718 | 20    | 0     |
| vilg        | 1 | 1738 | 538  | 14243 | 13288 |
| vilg        | 2 | 1738 | 1110 | 1736  | 1570  |
| vilg        | 3 | 1738 | 1631 | 130   | 97    |
| vilg        | 4 | 1738 | 1711 | 27    | 7     |
| vilg        | 5 | 1738 | 1717 | 21    | 1     |
| atom        | 1 | 1738 | 1110 | 1736  | 1570  |
| atom        | 2 | 1738 | 1711 | 27    | 7     |
| atom        | 3 | 1738 | 1718 | 20    | 0     |
| atom        | 4 | 1738 | 1718 | 20    | 0     |
| atom        | 5 | 1738 | 1718 | 20    | 0     |

**Confirms the hypothesis**: bad collisions → 0 for every convention by L=3–4. A clean
side-finding fell out of the numbers: `atom`'s row at L is *exactly* `oracle_sage`'s row
at L (same distinct-hash/collision/bad-collision counts), while `vilg` reaches that same
row one L later (`vilg` L+1 ≡ `oracle_sage`/`atom` L). This matches the extra hop vilg's
proposition-node indirection costs on every relationship. All three converge to the same
fixed point (1718 distinct hashes, 20 residual raw collisions, 0 bad) — the 20 remaining
collisions are genuine WL-indistinguishable state pairs (at this domain's arity, some
states really are graph-symmetric) that never matter for the approximated optimal action.

## Part 2 — prototype of the extended domain (ternary `request`)

**A correctness problem found and fixed before any real numbers were produced** (see the
long comment at the top of the `PART 2` section in `indistinguishability.py`): the first
version of the "crossed-pair" construction — two passengers sharing origins with swapped
destinations, comparing a state against one with their tables exchanged — turned out to
be **always isomorphic under every encoding, not just object encoding**, because
exchanging two objects' entire data is exactly equivalent to renaming them, and renaming
is invisible to any relational encoding (object, atom, object-atom alike). This was
verified two ways before being trusted: an exhaustive search over the entire 2×2×2
configuration space found zero pairs with the intended split (atom/object-atom actually
showed *more* raw collisions there than object encoding — 48 and 31 vs 26 — all from the
same relabeling symmetry), and a direct node-colour check confirmed the two passengers'
and two destinations' WL colours were identical under every encoding at that scale.

Two independent fixes make the comparison meaningful, both verified empirically:

1. **Only one member of a crossed pair boards** (`picked_up_at`) in the compared
   snapshot — the buddy is present (its table still creates the ambiguity) but hasn't
   been picked up yet. This alone breaks the passenger-relabelling escape.
2. **Locations sit in a real `adjacent(l1, l2)` road network** (a maze, exactly like the
   current domain's binary `adjacent` predicate), so a hypothetical destination
   relabelling isn't a symmetry either — two locations with different neighbours can't be
   swapped without changing the maze's own edge set.

### Hand-checkable verification (both cases assert PASSED)

**Case 1** — 2 passengers, 2 origins, 2 destinations, `k=2`, fully crossed, on a 4-node
path maze `0-1-2-3` (so destinations 2 and 3 have different degree, hence aren't
interchangeable). Only passenger p boards, at origin 0.

**Case 2** — isolates the higher-arity effect from the maze's contribution: 3
passengers, **no maze/adjacency atoms at all**. p, q are the crossed pair; a third
passenger r's own request anchors one destination directly.

Both cases give the identical qualitative split:

| convention   | L=1 | L=2..5 |
|---|---|---|
| object       | collides (literally identical graph, not just isomorphic) | collides |
| atom         | does **not** collide | does not collide |
| object_atom  | **collides** (2 hops between the fact and its matching request atom, not 1) | does **not** collide |

Hop distance (destination location node → the `picked_up_at` fact, hand-check case 1):
**object = 1 hop, atom = 2 hops, object_atom = 3 hops** — object-atom's extra
proposition-node indirection needs strictly more message-passing depth to even have the
relevant information in the same neighbourhood, which is exactly why it needs L≥2 to
resolve the join while atom resolves it at L=1.

### Crossing-rate sweep — the key empirical question

*"How high must crossing_rate be before object encoding shows many bad collisions while
atom/object-atom show none?"*

**Answer: any crossing at all.** Swept crossing_rate 0.0→1.0 (n_passengers=10, n_origins=
n_destinations=6, k=2, 30 independent random worlds per point, each on its own random
maze). The moment `round(crossing_rate·n_passengers/2) ≥ 1` (crossing_rate ≥ 0.2 at
n_passengers=10 — 0.1 rounds to 0 pairs and shows no effect at all), **every** world's
State A collides with its own flipped State B under object encoding, at every L, 100% of
the time — and **never** collides under atom encoding, at any L. `object_atom` collides
100% at L=1 (matching the 2-hop distance above) and 0% from L=2 on. This pattern is
completely flat across crossing_rate 0.2–1.0 (verified once a crossed pair exists at
all, more crossing_rate doesn't make object encoding "worse" — one bad pair is already
100% bad) and is stable across `requests_per_passenger` k ∈ {2,3,4} and n_passengers ∈
{4,10,20} (same split every time; see the script's own printed sweep tables for the full
per-L, per-parameter breakdown). One rare exception: at n_passengers=4, one of 30 random
mazes let `atom` still collide at L=1 (dropping to 0 at L=2) and `object_atom` needed L=3
in a couple of draws — small mazes occasionally place destinations close enough together
that the anchoring signal takes one extra round to arrive; this is exactly the kind of
"needs more reach" result the maze construction is meant to surface, not a flaw in the
encodings.

**This decides the real-domain generator design**: even a *single* crossed passenger
pair, embedded in the real road network, should already produce a clean, 100%-repeatable
demonstration that object encoding collides while atom/object-atom don't — there's no
need to engineer a high crossing rate across many passengers to see the effect.

## What a proper fix would involve / caveats

- The 20 residual (non-bad) collisions in Part 1's converged state, and the rare
  slower-converging draws in Part 2's sweep, both reflect genuine WL-indistinguishable
  structure at this domain's scale — not measurement noise, and not something a fixed L
  can be guaranteed to eliminate in general (WL has real, provable limits).
- Part 2's `object`/`atom`/`object_atom` encoders are reimplemented in
  `indistinguishability.py` itself (not imported), generalising each real convention's
  own rule to the ternary `request` predicate exactly as specified; they are not wired
  into the simulator/env/planner and cannot be, without separate implementation work
  outside this analysis task.

## Part 2: Ternary Generator Collision Counter Results (PROTOTYPE encoders)

> Prototype: `analysis/ternary_collision_counter.py`'s adapter drops the taxi and `in` facts and uses
> `indistinguishability.py`'s re-implemented encoders, graph level only, growing vocab only. Superseded by
> *Buddy-pair collisions with the production converters* below, which also explains every difference.

### 1. Counterfactual Collision Table (Real Crossed States vs. Flipped Counterfactuals)

Evaluated across $n = 200$ real crossed states and their corresponding flipped counterfactuals ($A$ vs. $B$ via `flip_crossed_pairs()`).

| Convention | $L=1$ | $L=2$ | $L=3$ | $L=4$ | $L=5$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **object** | 100% | 100% | 100% | 100% | 100% |
| **atom** | 13% | 0% | 0% | 0% | 0% |
| **object_atom ($\nu$ILG)** | 100% | 13% | 2% | 0% | 0% |

- **Independently-Sampled States (Null Baseline):** Evaluating 2,000 independent snapshots (623 distinct states) showed 0% collisions across all layers and conventions. Because the adapter drops the taxi object to match the prototype representation, states differ across a sparse combinatorial selection of 400 location nodes, avoiding random collisions.
- **Counterfactual Real States:** Demonstrates qualitative alignment with the synthetic prototype. The standard **object** graph remains completely blind (100% collision across all depths $L$). While the synthetic maze showed a sharp $100\% \to 0\%$ drop at $L=2$, full-city topologies exhibit slight noise: **atom** displays a 13% residual collision at $L=1$, and **object_atom ($\nu$ILG)** requires $L=4$ to fully eliminate ambiguity.

---

### 2. Coverage & Structural Ambiguity

Analysis of $n = 1,395$ carried-passenger decision states:

- **Crossed States (Buddy Present):** 58.2%
- **Object-Encoding Ambiguous:** 58.2%

**Equivalence Proof:**
The exact parity between crossed states and object-encoding ambiguity is deterministic rather than coincidental at $k=2$. Under generator Rule 2, buddy pairs share the identical destination multiset via cyclical rotation without redraws. Whenever an unboarded buddy remains present at the shared origin, its destination is guaranteed to belong to the carried passenger's destination multiset. Therefore, $\text{crossed} \implies \text{ambiguous}$ holds deterministically under this generator, with no extra non-buddy-driven ambiguity observed in this sample.
---

## Frozen WL vocabularies on the ternary domain (production converters)

**Finding: frozen WL at L=2 is impractical on this domain.** A frozen vocab is built by
`sage/domains/utils/build_wl_vocab.py --domain ternary` (growing-mode
`wl_colours(domain="ternary")` over a corpus of full `GraphTaxiEnv(ternary=True)` episodes:
first half `scripted_policy` with 20% random actions, second half uniform random legal
actions; a live state every 10 steps plus 2 planner projections; seed 0). Held-out OOV
is measured by `analysis/ternary_vocab_oov.py` on seeds 1001–1005 (two full episodes each,
same mix; 6,000 graphs, including late-episode states and projections). "Request" nodes are
request propositions (vilg), request atoms (atom), and for object — which has none — the
passenger and location nodes.

L=2, 84-episode corpus (50,400 graphs):

| convention | vocab size | file | OOV overall | passenger | request |
|---|---|---|---|---|---|
| object (oracle_sage) | 57,743 | 7.8 MB | 5.46% | **25.13%** | 5.47% |
| vILG | 10,230 | 1.1 MB | 0.13% | 0.00% | **1.86%** |
| atom | 526,374 | **115 MB** | **15.34%** | **27.43%** | **93.16%** |

Held-out OOV against corpus size (same seed and mix; 21 / 42 / 84 / 336 / 650 episodes),
at the node type that fails. The L=2 rows were measured on the first held-out set (seeds
1001–1005, 10 episodes); the atom L=1 rows on the enlarged set (seeds 1001–1010, 20
episodes), with 95% CIs from a bootstrap over held-out episodes:

| vocab | nodes | 21 | 42 | 84 | 336 | 650 |
|---|---|---|---|---|---|---|
| object L=2 | passenger | 43.56% | 34.15% | 25.13% | — | — |
| vILG L=2 | request | 7.54% | 2.95% | 1.86% | — | — |
| atom L=1 | request | 7.87% [5.50, 10.24] | 4.15% [2.49, 5.98] | 2.39% [1.33, 3.59] | 1.01% [0.47, 1.72] | **0.69% [0.29, 1.23]** |
| atom L=1 | all nodes | 0.48% | 0.25% | 0.16% | 0.07% | 0.04% |
| atom L=1 | vocab size | 5,184 | 7,474 | 10,504 | 19,100 | 25,605 |

(On the 10-episode set the atom L=1 request-atom OOV was 8.61 / 4.44 / 2.53 / 1.02% at
21 / 42 / 84 / 336 episodes -- the same curve.) Atom L=1's request-atom OOV falls 1.9× and
1.7× for the first two doublings of the corpus, then ~1.5× per doubling (84→336, 336→650).

At L=2 a node's signature combines the L=1 classes of several neighbours (for an object
passenger: four or five locations, each already encoding its degree and the facts at it),
so nearly every combination in a new episode is unseen: object L=2 OOV falls only ~1.3×
per doubling of the corpus, and atom L=2 is both 93% OOV at request atoms and over
GitHub's 100 MB file limit. Frozen L=2 vocabs are therefore not built or committed; L=2
(and L=3) appear below only as growing-vocab analysis.

**Committed L=1 vocabs** (`sage/domains/utils/wl_vocab_taxi_city_ternary_{conv}_L1.json`):
650-episode corpus (390,000 graphs); held-out seeds 1001–1010 (20 episodes, 12,000 graphs,
including late-episode states and projections); 95% CI over held-out episodes. Gate: < 1%
overall and at passenger nodes, atom request atoms <= 1.5%.

| convention | vocab size | file | OOV overall | passenger | request |
|---|---|---|---|---|---|
| object (oracle_sage) | 885 | 149 KB | 0.003% [0.000, 0.008] | 0.000% | 0.003% [0.000, 0.008] |
| vILG | 498 | 80 KB | 0.001% [0.000, 0.002] | 0.000% | 0.000% |
| atom | 25,605 | 5.9 MB | 0.039% [0.014, 0.073] | 0.000% | 0.688% [0.288, 1.231] |

---

## Buddy-pair collisions with the production converters

`analysis/ternary_collision_production.py` (tests: `tests/test_ternary_collision_production.py`).
States come from the vocab corpus design (`build_wl_vocab.ternary_corpus`) on seeds 2001–2007
(whole seeds, added until the crossed states held ≥ 200 distinct buddy pairs): two full
episodes per seed (scripted with 20% random actions, then uniform random), a live state every
10 steps plus 2 planner projections per state — 2,800 live states and 5,600 projections.
Every state is encoded by the policy-side converters (facts → JSON → `json_to_ternary_graph_*`)
and projected by each convention's own `Planner(ternary=True)`; colours come from
`wl_colours(domain="ternary")`. Buddy identity is read from the simulator (analysis only).

- **Counterfactual** of a crossed state (carried p, buddy q waiting): the simulator's own
  `facts()` with p's and q's request tables swapped. It equals the prototype's
  `build_flipped_facts` as a fact set in 689/689 crossed live states (that function's list is
  never in canonical order); all 689 are proven **not isomorphic** to the original by a
  relational invariant independent of WL (road distances from p's true destination), and p's
  true destination differs in all 689 — so M4 (the decision-relevant subset) is every pair.
- **M1** graph level (critic / path_value_net): identical final-colour histograms, state vs
  counterfactual, over live states and their projections that keep the pair (2,062
  observations). **M2** node level (actor): the carried passenger's own colour identical in
  both. **M3** two waiting buddies with the same colour within a state (62,506 observations).
- One shared growing vocab per (convention, L), so equal colours mean WL-equivalent nodes.
  n is given as observations / distinct buddy pairs; 95% CIs are a percentile bootstrap over
  distinct pairs (2,000 resamples), since consecutive states of a pair are autocorrelated.
- Checks: object encodings of state and counterfactual are identical tensors in 2,062/2,062
  observations (so object M1 = M2 = 100% is structural); 290 random collisions and
  non-collisions re-derived with the independent reference WL of `tests/test_wl_ternary.py`
  all agree; the hand-built cases of `tests/test_wl_ternary.py` give the hand-derived answers
  through this script.

**Collision rates, growing vocab** (M1, M2: n = 2,062 observations / 204 distinct crossed
pairs; M3: n = 62,506 observations / 310 distinct waiting pairs):

| encoding | metric | L=1 | L=2 | L=3 |
|---|---|---|---|---|
| object | M1 graph | 100% | 100% | 100% |
| object | M2 carried p | 100% | 100% | 100% |
| object | M3 waiting buddies | 100% | 66.7% [58.9, 74.4] | 61.5% [53.0, 69.1] |
| vILG | M1 graph | 100% | 20.6% [13.2, 29.8] | 2.9% [0.7, 6.0] |
| vILG | M2 carried p | 100% | 100% | 20.6% [13.4, 30.1] |
| vILG | M3 waiting buddies | 100% | 100% | 21.4% [15.1, 28.8] |
| atom | M1 graph | 20.6% [13.3, 29.7] | 0% | 0% |
| atom | M2 carried p | 100% | 20.6% [12.9, 30.1] | 2.9% [0.7, 5.9] |
| atom | M3 waiting buddies | 100% | 20.5% [14.2, 27.2] | 1.8% [0.4, 4.2] |

Live states only (689 / 204 pairs) differ from these by at most 0.7 points. The pattern is
the one the hand-built cases predict: atom at L is vILG at L+1 (the proposition-node hop), and
the node-level view lags the graph-level view by one L (the carried passenger only sees its
own destination's distinguishing structure one hop later than the request node does).

**Frozen vocab, L=1** (the committed 650-episode vocabs; same n):

| encoding | M1 | M2 | M3 | M5 (frozen-only collisions, M1/M2/M3) | observations with an OOV node |
|---|---|---|---|---|---|
| object | 100% | 100% | 100% | 0 / 0 / 0 | 2.1% [0.5, 4.4] |
| vILG | 100% | 100% | 100% | 0 / 0 / 0 | 1.2% [0.0, 3.0] |
| atom | 20.6% [13.2, 29.8] | 100% | 100% | 0 / 0 / 0 | 28.4% [18.8, 39.3] |

The frozen vocab adds no collision at L=1: identical to the growing vocab in every cell.

**Differences from the prototype table, and their causes.** M1 on the same pairs three ways:

| sample | encoders | vILG L=1/2/3 | atom L=1/2/3 |
|---|---|---|---|
| published sample (exact replay: 200 states, 40 distinct pairs) | prototype | 100 / 13.0 / 2.0% | 13.0 / 0 / 0% |
| published sample | production, taxi and `in` dropped | 100 / 13.0 / 2.0% | 13.0 / 0 / 0% |
| published sample | production | 100 / 12.5 / 2.0% | 12.5 / 0 / 0% |
| this sample (689 live states, 204 pairs) | prototype | 100 / 21.0 / 2.9% | 21.0 / 0 / 0% |
| this sample | production, taxi and `in` dropped | 100 / 21.0 / 2.9% | 21.0 / 0 / 0% |
| this sample | production | 100 / 19.9 / 2.8% | 19.9 / 0 / 0% |

(object: 100% at every L in every row.) So: (1) the prototype encoders and the production
converters agree exactly once the taxi and `in` facts are dropped — no other encoder detail
matters; (2) the taxi and `in` facts break about 1 in 100–200 collisions (13.0 → 12.5%,
21.0 → 19.9%); (3) the remaining gap (13% vs ~20% for atom L=1 / vILG L=2) is the state
sample: the published 200 states are 40 distinct pairs from scripted play only, and the
bootstrap CI over this sample's 204 pairs is [13, 30]%, which the published 13% just touches.
The prototype's qualitative conclusions stand: object is blind to the exchange at every L;
atom separates every crossed pair at L=2 at graph level; vILG needs one L more.
