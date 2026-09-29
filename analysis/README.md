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
