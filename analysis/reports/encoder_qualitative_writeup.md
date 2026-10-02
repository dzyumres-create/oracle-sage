# What policy does each encoder learn? GNN vs WL on Oracle-SAGE city taxi (Cell 1 vs Cell 3)

*Draft, 2 October 2026. Same representation (object encoding), different encoder: Cell 1 = GNN (original Oracle-SAGE), Cell 3 = WL colour refinement, L = 1. Seed 0 of each.*

## Summary

1. **Both actors propose passengers almost at random; the discriminators make the difference.** Cell 1's actor gives every passenger the same probability because its logits saturate at a ±30 clamp; Cell 3's actor cannot tell passengers apart at all, because at WL depth 1 every waiting passenger has the same colour. Neither actor prefers short trips.
2. **Cell 1's discriminator is a near-perfect plan-length ranker, over all goals.** Its scores rank goals by plan length with Spearman 0.999, *including goals that deliver nothing*: every move or no-op with a shorter plan than the shortest passenger outscores every passenger. It behaves well only because its actor almost never proposes non-passenger goals. This holds although Cell 1's path-value head was never trained (a pre-existing optimizer-timing bug left it at its random initialisation); the shared GNN encoder learned to project a timer onto that fixed direction.
3. **Cell 3's discriminator recognises deliveries but ranks passengers weakly.** It never prefers a non-delivering goal over all passengers, but its ranking of passengers follows plan length only with Spearman 0.33 (passengers only; 0.36 over all goals). It uses the time-left input with the correct sign, but on the time/2000 scale that input moves scores about a quarter as much as the histogram terms, which hardly track plan length.
4. **Net effect: Cell 3 chooses longer trips.** Selected trips are on average 11 frames longer than the shortest available for Cell 1 and 20 frames for Cell 3 (greedy: 6). This matches the end-to-end gap: about 95% of both models' decisions are pickup-and-deliver trips, and Cell 3 spends 38.7 frames per decision against Cell 1's 32.9, so its deficit (48 vs 57 deliveries per episode) comes from longer plans, not different goal types.
5. **Both learned policies are well below a greedy heuristic** (nearest passenger, then its destination): 71.8 deliveries per episode, against 56.7 (Cell 1) and 48.2 (Cell 3) on the same seeds.

## Setup

**How a decision works.** The actor's masked softmax over all nodes is sampled three times without replacement (`th.multinomial`), giving 3 candidate goals. The planner projects each candidate, the discriminator (`path_value_net`) scores each projection, and the highest score is chosen. The chosen plan is executed to completion. A passenger goal plans the whole trip (pickup and delivery); a location goal moves the taxi; the taxi node or the taxi's own location is a 1-frame no-op.

**Discriminator inputs.** Cell 1: the GNN encoding of the current and projected graphs (shared with the actor). Cell 3: the WL colour histogram of the current and projected graphs, each with time left, through one linear layer.

**State set.** 495 states at the models' own decision points (the taxi is always empty there): 165 each from greedy, Cell 1 and Cell 3 trajectories, 55 per source in each depth bucket (early: frame < 250; middle: 250–1000; late: ≥ 1000). Each model sees each state on its own simulator (Cell 1 with the stale tether edges it was trained with, Cell 3 on the fixed one); the dynamics are identical, only Cell 1's graph carries extra stale edges.

**Ground truth.** Plan length in frames from the training planner, for every goal; for passengers also split into pickup distance and trip length. "Shortest" means the shortest passenger plan, since only passenger goals deliver. Rollout values are not used yet.

**Read-outs per model and state.** The actor's full distribution, the discriminator's score for every goal (~416 per state), and 20 seeded draws of 3 candidates with the selection made as in training (ties broken at random; ties occurred in 0.02% of Cell 1's and 0.25% of Cell 3's draws).

**Verification before analysis.**
- Loading reproduces the trained weights exactly and gives deterministic outputs.
- Cell 1's decisions are bit-identical under its original code and the current branch.
- The decision read-out matches the training-time decision code (60/60 sampled decisions).
- Cell 1's and Cell 3's simulators give identical dynamics, rewards, spawns and plans for 3 × 2,000 frames.
- Forked copies of an environment are independent and stay identical under the same actions, spawns included.
- Each loaded model reproduces its training plateau (below).

## Results

### Performance reference (10 episodes, same env seeds)

| Policy | Deliveries per episode: mean (sd), range |
|---|---|
| Greedy (nearest passenger, then destination) | 71.8 (5.1), 65–80 |
| Cell 1 on its own (stale-edge) simulator | 56.7 (3.2), 53–63 (training plateau ~55) |
| Cell 1 on the fixed simulator | 56.4 (2.7), 53–61 |
| Cell 3 on the fixed simulator | 48.2 (3.7), 40–54 (training plateau ~47) |

Greedy beats Cell 1 on every seed (by 11–20) and Cell 3 on every seed (by 17–29). Cell 1 gives the same result on clean graphs in 9 of 10 episodes, so it did not come to depend on the stale edges it trained with.

### Goal selection: actor vs discriminator

Overall, over 495 states × 20 draws:

| | Cell 1 (GNN) | Cell 3 (WL) |
|---|---|---|
| Selected trip, frames over the shortest (mean / median) | +11.2 / +9 | +20.4 / +19 |
| Selected goal is a shortest-plan passenger | 23.9% | 13.9% |
| Selected goal is a move or no-op | 7.8% | 7.3% |
| **Actor:** shortest passenger among the 3 candidates | 29.8% | 17.7% |
| **Discriminator:** shortest picked when proposed (random: 33%) | **92.0%** | **69.0%** |
| Discriminator's top passenger over *all* passengers, frames over shortest | **+0.0** | **+14.7** |
| Discriminator rank agreement with plan length, passengers only (Spearman) | **0.998** | **0.334** |
| Discriminator rank agreement with plan length, all goals (Spearman) | **0.999** | **0.364** |
| Actor rank agreement with plan length, passengers only | 0.05 | undefined (constant) |
| Actor probability on passengers | 100% | 67% |

For reference: greedy's choice is +6.3 frames over the shortest, a random passenger +22.3, and an actor uniform over all ~416 goals would propose the shortest 0.8% of the time.

By depth bucket:

| | Cell 1: frames over shortest | proposed | picked if proposed | Cell 3: frames over shortest | proposed | picked if proposed |
|---|---|---|---|---|---|---|
| early | +7.8 | 56% | 77% | +15.8 | 27% | 79% |
| middle | +11.5 | 18% | 100% | +21.5 | 13% | 60% |
| late | +13.4 | 16% | 100% | +23.0 | 13% | 68% |

The source of a state (greedy, Cell 1 or Cell 3 trajectory) changes these numbers little (Cell 1: +10.1 to +12.8; Cell 3: +18.3 to +23.4).

**Attribution.**
- **Cell 1:** the actor is the bottleneck. It proposes passengers uniformly, so the shortest one is among the candidates only 30% of the time (less as passengers accumulate), and the discriminator then picks it almost always. Cell 1's behaviour is close to "the shortest of three random passengers".
- **Cell 3:** both stages contribute. The actor is uniform over passengers and also spends a third of its probability on locations. The discriminator picks the shortest proposed passenger only 69% of the time, and its own favourite passenger is on average 15 frames longer than the shortest.

The two models make the same most-frequent choice in only 9.7% of states (5.5% late in the episode).

### Main finding: Cell 1's discriminator ranks every goal by plan length, delivering or not

Over all ~416 goals per state, Cell 1's discriminator score has Spearman 0.999 with shorter plans. Every move or no-op goal shorter than the shortest passenger outscores every passenger, in 100% of the cases (all 495 states have such goals, since the no-op takes 1 frame). The discriminator is effectively a timer, not a delivery-value estimate.

Two consequences:
- It works because the actor puts essentially all its probability on passengers, so non-passenger goals reach the discriminator only when fewer than three passengers are waiting.
- That is exactly where Cell 1 goes wrong: early in an episode with one or two passengers, the third candidate is a location or no-op, and the discriminator prefers it to a long delivery. Hence Cell 1's 23.5% move/no-op choices in early states, against 0% in middle and late states. In state 171 (frame 0, one passenger, proposed in every draw), Cell 1 chooses a short move or a no-op in 19 of 20 draws.

Cell 1's path-value head (a 64 → 1 linear layer) was never in the optimizer: a construction-order bug left it at its random initialisation, confirmed from the saved optimizer state and from its weights sitting in PyTorch's default initialisation range. Its path-value loss nevertheless fell to 2–4, tracking the value loss. The shared GNN encoder adapted so that the fixed random readout reproduces the return. Our hypothesis is that time left is the cheapest feature for that; we have not tested it directly. Cell 3 (WL, trained head) shows the opposite behaviour (0% of non-delivering goals preferred). Whether Cell 1's timer behaviour comes from the frozen head or from the GNN itself is what the Cell 1-fixed run will tell (see the placeholder below).

### WL: structural indistinguishability and the time-left scale

**The actor cannot tell passengers apart.** At L = 1 all waiting passengers share one WL colour in all 495 states, so Cell 3's actor gives them exactly equal probability. Its proposals are uniform over passengers by construction, not by learning.

**Many projections collide.** Collisions in the discriminator's inputs, over passenger goals:

| | Histogram only | Full input (histogram + time left) |
|---|---|---|
| Passenger goals sharing their key with another passenger | 56.9% | 2.2% |
| Shortest passenger distinguishable from all longer ones | 51.1% of states (early 68%, middle 38%, late 48%) | 100% |
| Distinct keys among all ~416 goals | 22.0 | 160.3 |

By histogram alone, about half of the passenger projections are indistinguishable from another passenger's, and in half of the states the shortest passenger looks identical to a longer one. Time left separates almost all of them, so the information needed to rank passengers is always present in the full input. Identical full inputs always gave identical scores (no exceptions).

**The timer is used, but on too small a scale.** The head is one linear layer over [current histogram, current time left, projected histogram, projected time left]; reconstructing its scores from weights × inputs matches to 6 × 10⁻⁶.
- Within groups of passengers that share a projected histogram (so only time left differs), the score ranks shorter plans higher in 100% of 1,630 groups. Across groups, the rank agreement is 0.26 (passengers only).
- The projected time-left weight is +0.337, against histogram weights of mean |w| 0.042. But time left is encoded as (time remaining)/2000, so a 20-frame difference in plan length is a 0.010 difference in input and a 0.0034 difference in score.
- Between two passengers in a state, the time term changes the score by a median of 0.0025; the histogram term by 0.0095 (larger in 80% of pairs). The histogram term hardly tracks plan length (Spearman +0.08, passengers only).
- The two time weights (+0.395 current, +0.337 projected) are by far the largest in the head, the only weights well past the initialisation range (±0.099). Adam moves a weight by at most about the learning rate per update, so 1,250 updates at 3 × 10⁻⁴ allow a change of about 0.375. Both values include their random initial value (up to ±0.099), so +0.395 is within init + 0.375 = 0.474. This fits a scale or update-budget limit on the time weights, though it does not prove it.
- The discriminator's training target is the 5-decision return (γ = 0.99 per decision, bootstrapped by the value net), with rewards inside a plan discounted per frame. Plan length therefore enters the target through the discount on the plan's own delivery (γ^(L−1): 0.83 for a 20-frame plan, 0.68 for a 40-frame plan) and through the time left at the next decision. There is no per-frame discount between decisions.

**Revised reading:** Cell 3's discriminator uses the time-left input with the correct sign. Among passengers whose projections share a histogram it always prefers the shorter plan. But on the time/2000 scale, time-left differences contribute about a quarter of the score differences that the histogram terms do, so plan length decides only part of its ranking (Spearman 0.33 over passengers only).

### The actor clamp (all 5 Cell 1 seeds)

`masked_segmented_softmax` clamps actor logits to [−30, 30] before the softmax:

```python
def masked_segmented_softmax(energies, mask, batch_ind):
    energies = energies.clamp(-30,30)
    energies[~mask.bool()] = -np.inf
    probs = softmax(energies, batch_ind)
    return probs.flatten()
```

A logit beyond ±30 receives zero gradient. Checked with autograd on logits [35, 30, 29, 0, −31]: the gradients are [0, 0.113, 0.197, 0, 0]. The first and last are zero because of the clamp. The logit at 0 also gets a zero gradient, but for a different reason: next to logits of 30 its probability is about e⁻²⁹, so its gradient is negligible. Once a passenger's logit passes +30 it is frozen there, and the actor can no longer learn to prefer one passenger over another.

| Cell 1 seed | Passenger logits ≥ +30 | States where all passengers get equal probability | Probability on the shortest passenger (uniform share) |
|---|---|---|---|
| 0 | 73.9% | 366 / 495 | 14.6% (14.6%) |
| 300 | 100.0% | 495 / 495 | 14.6% (14.6%) |
| 600 | 92.1% | 456 / 495 | 14.6% (14.6%) |
| 900 | 87.5% | 433 / 495 | 14.6% (14.6%) |
| 1200 | 70.9% | 349 / 495 | 14.6% (14.6%) |

In every seed, passengers take essentially all the actor's probability, while non-passenger logits sit far below (median maximum −10 to −27). The clamp is part of the training code, so this is the trained behaviour, not an artefact of the analysis.

### Example states

| Figure | What it shows |
|---|---|
| `figures/state_283.png` | Late state: the models disagree. Cell 1's most frequent choice is the shortest passenger; Cell 3's is one 27 frames longer. |
| `figures/state_090.png` | Cell 3's shortest passenger shares its WL histogram with a longer passenger; only time left separates them. |
| `figures/state_171.png` | Frame 0, one passenger, always proposed: Cell 1 chooses a short move or a no-op in 19 of 20 draws. |
| `figures/state_015.png` | Same situation: Cell 1 picks non-delivering goals in all 20 draws. |
| `figures/state_040.png` | Cell 3 proposes the shortest passenger in 40% of draws but picks it in only 12% of those. |
| `figures/state_037.png` | Cell 3's largest detour among early states (+32 frames on average). |

Each figure shows the city map (walls, taxi, waiting passengers, the shortest passenger S and greedy's choice G, each model's most frequent choice with its route) and, for every goal either model chose, the share of the 20 draws choosing it, with its plan length. Rows are sorted by plan length; goals that deliver nothing are greyed.

## Implications for the research question

The question is which encoder learns which policy, and whether differences come from what the encoder can represent.

- **The one expressiveness limit found is the WL actor at L = 1.** Every waiting passenger has the same WL colour, so the WL actor cannot prefer one passenger over another, whatever it learns. Deeper WL (L ≥ 2) might separate passengers by their neighbourhoods (for example, through their destinations). That has not been tested here; the L = 2 runs had other problems (vocabulary size, coverage).
- **The other limitations come from learning or implementation, not representation:**
  - *WL discriminator:* its input distinguishes the shortest passenger in every state, and it uses time left with the correct sign. The weak ranking comes from the time-left scale (time/2000) and, plausibly, the number of updates available to grow that weight.
  - *GNN actor:* the ±30 logit clamp saturates passenger logits and removes their gradient, so the actor cannot express a preference among passengers even though the GNN could.
  - *GNN discriminator:* its timer-like readout ignores whether a goal delivers. This may come from the frozen random head, which leaves the encoder only a fixed direction to fit returns with; the Cell 1-fixed run will show whether a trained head behaves differently.
- So on this task, the representation difference between GNN and WL explains the actors' inability to rank passengers only for WL, and only at L = 1. The rest of the GNN-WL gap comes from training details that differ between the two cells or affect them differently.

## Implications beyond Cell 1 vs Cell 3

- **Cells 2 and 5 are also pre-fix.** Cell 2's saved optimizer state shows the path-value head missing from the optimizer, with its weights at initialisation (seeds 0, 300, 600). Cell 5 was trained from a branch descended from Cell 2's, without the fix (from the branch history; no Cell 5 model has been inspected).
- **The clamp is in shared training code** (`masked_segmented_softmax`), so it likely affects every GNN cell's actor. It has only been measured for Cell 1.
- **The GNN and WL rows of the main experiment differ in the optimizer fix.** The GNN cells (1, 2, 5) were trained with a frozen path-value head; the WL cells (3, 4, 6) with a trained one. Any GNN-vs-WL comparison in the learning curves includes this difference as well as the encoder. Cell 1-fixed is the first GNN cell with the fix.

## Caveats

- **One seed per model.** The actor-clamp result holds for all five Cell 1 seeds; the other results are for seed 0 of each cell.
- **Ground truth is plan length only.** A shorter trip is not always better (destinations and other passengers matter later). Rollout values, which will continue each goal with the greedy policy, are planned; they will measure a goal's value under greedy continuation, not under each model's own policy.
- **Different simulators.** Cell 1 was trained on a simulator with accumulating stale tether edges; Cell 3 on the fixed one. Each is analysed on its own. Cell 1's performance does not change on clean graphs.
- **Cell 1's discriminator head was never trained.** Its behaviour reflects the shared encoder adapting to a frozen random readout.
- **The state set covers decision points only.** At the models' own decision points the taxi is always empty, so "deliver the carried passenger" and "phantom delivery" goals never arise there and are not analysed.
- **Software stack.** The analysis ran on a Mac with torch 2.13, gym 0.26 and numpy 2.2, not the training stack (torch 1.7, gym 0.18). The training code ran unmodified, with compatibility shims in the analysis code only (legacy gym seeding, removed numpy aliases, a deepcopy-safe environment fork). Both models' training plateaus were reproduced on this stack.
- **WL depth.** WL was analysed at L = 1 only, the depth chosen for all WL cells (Cell 4's existing L = 2 run is being redone at L = 1). Whether deeper WL separates passengers is untested.
- **Code lineage.** Cell 1 ran from RCP's main checkout with uncommitted patches; its simulator is rebuilt here from committed code. All patch files are accounted for and none touches the simulator or planner, but the full diff has not yet been inspected.

## Cell 1-fixed (placeholder; training on RCP)

GNN, object encoding, with both the stale-edge fix and the optimizer-timing fix (path-value head trained); otherwise Cell 1's settings. When its model is available:

- Score it on the same 495 states (on the fixed simulator) and repeat all metrics above.
- Main question: **does its discriminator still prefer non-delivering short goals?** Report rank agreement with plan length over all goals, and the share of shorter move/no-op goals that outscore every passenger, against Cell 1's 0.999 / 100% and Cell 3's 0.36 / 0% (Spearman over all goals).
- Also: actor clamp saturation, attribution (proposed / picked if proposed), and deliveries per episode against Cell 1 and greedy.

*Results: pending.*
