# What policy does each encoder learn? GNN vs WL on Oracle-SAGE city taxi (Cell 1 vs Cell 3)

*Draft, 5 October 2026. Same representation (object encoding), different encoder: Cell 1 = GNN (original Oracle-SAGE), Cell 3 = WL colour refinement, L = 1. Seed 0 of each, with Cell 1 seed 300 as a replication, plus Cell 1-fixed: the GNN retrained with two bug fixes.*

## Summary

1. **Both actors propose passengers almost at random.** Cell 1's actor gives every passenger the same probability because its logits saturate at a ±30 clamp. Cell 3's actor cannot tell passengers apart at all: at WL depth 1 every waiting passenger has the same colour. Neither actor prefers short trips.
2. **The GNN discriminator ranks every goal by plan length, delivering or not.** Its scores follow plan length with Spearman 0.999 over all goals, and every move or no-op shorter than the shortest passenger outscores every passenger. This holds for old Cell 1, whose discriminator head was never trained (a pre-existing optimizer bug), and equally for Cell 1-fixed, whose head was trained. So the frozen head does not cause it.
3. **The WL discriminator recognises deliveries but ranks passengers weakly.** It almost never prefers a non-delivering goal over all passengers (0.01%), but its scores follow plan length only with Spearman 0.33 among passengers (0.36 over all goals). It uses the time-left input with the correct sign, but on the time/2000 scale that input moves scores about a quarter as much as the histogram terms, which hardly track plan length.
4. **Once the shortest passenger is proposed, both discriminators pick it equally often: 73.7% (GNN) vs 72.7% (WL), pooled over draws.** The confidence interval of the difference includes zero. They fail in opposite places:
   - the GNN early in the episode, when a short non-delivering goal is among the candidates;
   - the WL discriminator in middle and late states, against longer passengers.
   The GNN picks the shortest passenger more often overall (22.0% vs 12.8%) because its actor proposes it more often (29.8% vs 17.7%): the WL actor spends a third of its probability on locations. The trip-length gap (+11 vs +20 frames over the shortest) also reflects ranking: the GNN discriminator's favourite passenger is always the shortest; the WL discriminator's is 15 frames longer on average.
5. **In performance, the GNN behaves like "the shortest of 3 random passengers" and WL like "a random passenger".** The GNN models make 55.5–56.7 deliveries per episode against 57.1 for that reference policy. Cell 3 makes 48.2 against 46.6, with overlapping confidence intervals. All are far below greedy (71.8) and a policy that always takes the shortest-plan passenger (70.7).
6. **The two training bugs were silent and harmless here; the actor clamp is silent but binding.** Cell 1-fixed (stale-edge fix and optimizer fix) matches old Cell 1 on every behavioural measure and delivers 55.5 per episode, within old Cell 1's range.

## Setup

**How a decision works.** The actor's masked softmax over all nodes is sampled three times without replacement (`th.multinomial`), giving 3 candidate goals. The planner projects each candidate, the discriminator (`path_value_net`) scores each projection, and the highest score is chosen. The chosen plan is executed to completion. A passenger goal plans the whole trip (pickup and delivery); a location goal moves the taxi; the taxi node or the taxi's own location is a 1-frame no-op.

**Discriminator inputs.**
- *GNN (Cells 1 and 1-fixed):* the GNN encoding of the current and projected graphs, shared with the actor, through a linear layer.
- *WL (Cell 3):* the WL colour histogram of the current and projected graphs, each with time left, through one linear layer.

**Models.**
- **Cell 1:** seeds 0 and 300; trained with the stale-edge simulator and a frozen discriminator head.
- **Cell 3:** seed 0; trained on the fixed simulator with a trained head.
- **Cell 1-fixed:** seed 0; Cell 1's settings with both fixes.

**State set.** 495 states at the models' own decision points (the taxi is always empty there):
- 165 states each from greedy, Cell 1 and Cell 3 trajectories;
- 55 per source in each depth bucket: early (frame < 250), middle (250–1000), late (≥ 1000).

Each model sees each state on the simulator it was trained on:
- old Cell 1 with the stale tether edges it was trained with;
- Cell 3 and Cell 1-fixed on the fixed simulator.

The dynamics are identical across simulators; only old Cell 1's graph carries extra stale edges.

**Ground truth.** Plan length in frames from the training planner, for every goal. For passengers it is also split into pickup distance and trip length. "Shortest" means the shortest passenger plan, since only passenger goals deliver; passengers tied at that length all count as shortest (40 of 495 states have a tie). Rollout values are not used yet.

**Read-outs per model and state.**
- the actor's full distribution;
- the discriminator's score for every goal (~416 per state);
- 20 seeded draws of 3 candidates, with the selection made as in training. Ties are broken at random; they occurred in 0.01–0.02% of the GNN models' draws and 0.25% of Cell 3's.

**Rate definitions.** All rates are pooled over draws (9,900 per model), so *proposed × picked-if-proposed = picked* exactly. A draw "proposes the shortest" if any shortest passenger is among its 3 candidates, and "picks" it if its choice is one of them. An earlier draft averaged picked-if-proposed per state, which overweighted states with few proposals and gave 92.0% for Cell 1; the pooled value is 73.7%.

**Uncertainty.** 95% confidence intervals come from 1,000 bootstrap resamples of the 495 states, with pooled rates recomputed in each resample. Differences between models are paired (both models on the same resampled states). Deliveries per episode are bootstrapped over the 10 episodes.

**Verification before analysis.**
- Loading reproduces the trained weights exactly and gives deterministic outputs (all three models).
- Old Cell 1's decisions are bit-identical under its original code and the current branch.
- The decision read-out matches the training-time decision code (60/60 sampled decisions per model).
- Cell 1's and Cell 3's simulators give identical dynamics, rewards, spawns and plans for 3 × 2,000 frames.
- Forked copies of an environment are independent and stay identical under the same actions, spawns included.
- Each loaded model reproduces its training plateau (below).

## Results

### Performance (10 episodes, env seeds 100–109; plans run open-loop as in training)

| Policy | Deliveries per episode [95% CI] | sd, range |
|---|---|---|
| Greedy (nearest passenger, then its destination; re-decides every frame) | 71.8 [68.7, 74.9] | 5.1, 65–80 |
| Shortest-plan passenger (reference) | 70.7 [68.4, 73.5] | 4.3, 65–79 |
| Best of 3 random passengers (reference) | 57.1 [55.1, 59.2] | 3.6, 52–62 |
| Cell 1 on its own (stale-edge) simulator | 56.7 [55.0, 58.7] | 3.2, 53–63 (training plateau ~55) |
| Cell 1 on the fixed simulator | 56.4 [55.1, 58.1] | 2.7, 53–61 |
| Cell 1-fixed on the fixed simulator | 55.5 [54.1, 57.0] | 2.4, 52–59 (training plateau 55.7) |
| Cell 3 on the fixed simulator | 48.2 [45.8, 50.3] | 3.7, 40–54 (training plateau ~47) |
| Random passenger (reference) | 46.6 [43.6, 49.3] | 4.7, 37–53 |

The reference policies choose among waiting passengers only (a 1-frame no-op if none is waiting):
- *shortest-plan:* the passenger with the shortest total plan;
- *best of 3 random:* 3 distinct passengers sampled uniformly, then the shortest of them;
- *random passenger:* one passenger sampled uniformly.

**Plan length is a valid ground truth.** Always taking the shortest-plan passenger performs as well as greedy: paired difference −1.1, better on 4 seeds, worse on 5, tied on 1. That's 14–24 deliveries above the learned models, so choosing by plan length is a strong policy in this domain.

**Learned policies against the references.**
- Both GNN models sit at the "best of 3 random passengers" level. That is what an actor that proposes passengers uniformly plus a discriminator that picks the shortest proposal would give.
- Cell 3 sits at the "random passenger" level (48.2 vs 46.6, intervals overlapping).
- Old Cell 1 gives the same result on clean graphs in 9 of 10 episodes, so it did not come to depend on the stale edges it trained with.

### Goal selection: actor vs discriminator (495 states × 20 draws, pooled; 95% CIs)

| | Cell 1 (GNN) | Cell 1-fixed (GNN) | Cell 3 (WL) | Cell 1 − Cell 3 (paired) |
|---|---|---|---|---|
| Selected trip, frames over shortest | 11.2 [10.6, 11.7] | 11.1 [10.6, 11.6] | 20.4 [19.7, 21.0] | −9.2 [−9.7, −8.7] |
| Shortest picked, % of draws | 22.0 [20.4, 23.7] | 22.4 [20.8, 24.1] | 12.9 [11.8, 13.9] | 9.1 [7.3, 10.9] |
| **Actor:** shortest proposed, % of draws | 29.8 [27.3, 32.3] | 30.4 [27.9, 32.8] | 17.7 [16.5, 18.7] | 12.1 [10.3, 14.0] |
| **Discriminator:** picked if proposed, % (random: 33%) | 73.7 [67.6, 80.2] | 73.9 [67.8, 80.0] | 72.7 [69.3, 75.7] | 1.1 [−7.1, 9.3] |
| Move/no-op picks, % of draws | 7.8 [5.6, 10.3] | 7.9 [5.6, 10.3] | 7.3 [6.0, 8.7] | 0.5 [−0.7, 1.8] |
| Discriminator's top passenger, frames over shortest | 0.00 [0.00, 0.00] | 0.04 [0.02, 0.05] | 14.7 [13.3, 16.1] | −14.7 [−16.1, −13.3] |
| Discriminator Spearman with plan length, passengers only | 0.998 | 0.996 | 0.334 [0.31, 0.36] | 0.66 [0.64, 0.69] |
| Discriminator Spearman with plan length, all goals | 0.999 | 0.999 | 0.364 [0.35, 0.37] | 0.64 [0.63, 0.64] |
| Actor probability on the shortest passenger, % | 14.6 [12.6, 16.7] | 14.6 [12.6, 16.7] | 6.2 [5.9, 6.6] | 8.4 [6.7, 10.2] |
| Actor probability on passengers, % | 100 | 100 | 67 | |

For reference:
- greedy's choice is +6.3 frames over the shortest;
- a random passenger is +22.3;
- an actor uniform over all ~416 goals would propose the shortest 0.8% of the time.

Among passenger picks only, the share that is a shortest passenger is 23.9% (Cell 1), 24.3% (Cell 1-fixed) and 13.9% (Cell 3).

By depth bucket (pooled):

| | Cell 1: frames over shortest | proposed | picked if proposed | move/no-op picks | Cell 3: frames over shortest | proposed | picked if proposed | move/no-op picks |
|---|---|---|---|---|---|---|---|---|
| early | +7.8 | 56.3% | **58.3%** | 23.5% | +15.8 | 26.8% | 81.5% | 18.6% |
| middle | +11.5 | 17.7% | 100% | 0% | +21.5 | 13.2% | **59.7%** | 1.6% |
| late | +13.4 | 15.5% | 100% | 0% | +23.0 | 12.9% | **67.7%** | 1.7% |

Cell 1-fixed's buckets match Cell 1's: picked if proposed 57.9 / 99.6 / 99.7%; move/no-op picks 23.5 / 0 / 0%. The source of a state (greedy, Cell 1 or Cell 3 trajectory) changes these numbers little.

**Attribution.**
- **Overall, the discriminators are equally good at keeping a proposed shortest passenger** (73.7% vs 72.7%; difference 1.1 points, CI −7.1 to 9.3), but they lose it in opposite places:
  - the GNN loses it in early states, to a short move or no-op among the candidates (below);
  - in middle and late states, where its candidates are almost always passengers, it keeps it every time;
  - the WL discriminator loses it in middle and late states (60–68%), to longer passengers.
- **The gap in picking the shortest passenger comes from the actors:** the GNN actor proposes it in 29.8% of draws, the WL actor in 17.7%. Both actors are uniform over passengers, but the WL actor also spends a third of its probability on locations.
- **Ranking still drives trip length.** When the shortest passenger is not proposed, the GNN discriminator picks the shortest passenger among its candidates, because its ranking follows plan length (Spearman 0.998). The WL discriminator's does not (0.33). Hence +11 vs +20 frames over the shortest.

The two models make the same most-frequent choice in only 9.7% of states (5.5% late in the episode).

### Main finding: the GNN discriminator ranks every goal by plan length, delivering or not

Over all ~416 goals per state, the GNN discriminator's score has Spearman 0.999 with shorter plans, in both old Cell 1 and Cell 1-fixed. Every move or no-op goal shorter than the shortest passenger outscores every passenger, in 100% of cases for both models. All 495 states have such goals, since the no-op takes 1 frame. The discriminator behaves like a timer, not like an estimate of delivery value. Cell 3's discriminator does the opposite: a shorter non-delivering goal beat every passenger in 0.01% of cases.

**Consequences.**
- Non-passenger goals reach the GNN discriminator only when fewer than three passengers are waiting, because its actor puts essentially all its probability on passengers.
- In those early states the third candidate is a location or no-op, and the discriminator prefers it to a long delivery. Hence the GNN's 23.5% move/no-op choices in early states, against 0% in middle and late states. In state 171 (frame 0, one passenger, proposed in every draw), Cell 1 chooses a short move or a no-op in 19 of 20 draws.
- **It is untested whether this costs deliveries.** Waiting a frame or moving early may cost little or nothing, as more passengers spawn. "Best of 3 random passengers", which always delivers when a passenger is waiting, scores the same as Cell 1 (57.1 vs 56.7). An early-state rollout comparison (no-op first vs deliver first, from the same forked states) would settle it; it is listed as an optional follow-up.

**Not caused by the frozen head.** Old Cell 1's discriminator head (a 64 → 1 linear layer) was never in the optimizer: a construction-order bug left it at its random initialisation, confirmed from the saved optimizer state and from its weights sitting in PyTorch's default initialisation range. Its path-value loss still fell to 2–4, tracking the value loss, as the shared GNN encoder adapted to the fixed random readout.

In Cell 1-fixed the head is in the optimizer and did train: 7 of 64 weights moved beyond the initialisation range, and the final path-value loss was about 0.4–0.6. Yet the discriminator ranks goals the same way: Spearman 0.999 over all goals, 100% of shorter non-delivering goals preferred, and early picked-if-proposed 57.9% vs 58.3%.

**Hypothesis for why the timer persists (not tested):**
- The discriminator is trained only on the goals actually executed. These are mostly deliveries in crowded states, where among comparable passenger trips "more time left afterwards = better" holds.
- Non-delivering goals are rarely executed, so the learned preference for time left is extrapolated to them without correction.
- Per-decision discounting makes a wasted 1-frame decision cheap in the training target: the return is discounted per decision, not per frame, so a no-op costs one discount step regardless of how little it achieves.

### WL: structural indistinguishability and the time-left scale

**The actor cannot tell passengers apart.** At L = 1 all waiting passengers share one WL colour in all 495 states, so Cell 3's actor gives them exactly equal probability. Its proposals are uniform over passengers by construction, not by learning.

**Many projections collide.** Collisions in the discriminator's inputs, over passenger goals:

| | Histogram only | Full input (histogram + time left) |
|---|---|---|
| Passenger goals sharing their key with another passenger | 56.9% | 2.2% |
| Shortest passenger distinguishable from all longer ones | 51.1% of states (early 68%, middle 38%, late 48%) | 100% |
| Distinct keys among all ~416 goals | 22.0 | 160.3 |

By histogram alone, about half of the passenger projections are indistinguishable from another passenger's. In half of the states the shortest passenger looks identical to a longer one. Time left separates almost all of them, so the information needed to rank passengers is always present in the full input. Identical full inputs always gave identical scores (no exceptions).

**The timer is used, but on too small a scale.** The head is one linear layer over [current histogram, current time left, projected histogram, projected time left]. Reconstructing its scores from weights × inputs matches to 6 × 10⁻⁶.
- **Within histogram groups:** among passengers that share a projected histogram (so only time left differs), the score ranks shorter plans higher in 100% of 1,630 groups. Across groups, the rank agreement is 0.26 (passengers only).
- **Weights vs scale:** the projected time-left weight is +0.337, against histogram weights of mean |w| 0.042. But time left is encoded as (time remaining)/2000, so a 20-frame difference in plan length is a 0.010 difference in input and a 0.0034 difference in score.
- **Typical contributions:** between two passengers in a state, the time term changes the score by a median of 0.0025 and the histogram term by 0.0095 (larger in 80% of pairs). The histogram term hardly tracks plan length (Spearman +0.08, passengers only).
- **Possible update-budget limit:** the two time weights (+0.395 current, +0.337 projected) are by far the largest in the head, and the only weights well past the initialisation range (±0.099). Adam moves a weight by at most about the learning rate per update, so 1,250 updates at 3 × 10⁻⁴ allow a change of about 0.375. Both values include their random initial value (up to ±0.099), so +0.395 is within init + 0.375 = 0.474. This fits a scale or update-budget limit on the time weights, though it does not prove it.
- **Training target:** the 5-decision return (γ = 0.99 per decision, bootstrapped by the value net), with rewards inside a plan discounted per frame. Plan length therefore enters the target in two ways:
  - through the discount on the plan's own delivery (γ^(L−1): 0.83 for a 20-frame plan, 0.68 for a 40-frame plan);
  - through the time left at the next decision.

  There is no per-frame discount between decisions.

**Reading:** Cell 3's discriminator uses the time-left input with the correct sign: among passengers whose projections share a histogram it always prefers the shorter plan. But on the time/2000 scale, time-left differences contribute about a quarter of the score differences that the histogram terms do. So plan length decides only part of its ranking (Spearman 0.33, passengers only).

### The actor clamp (all 5 Cell 1 seeds and Cell 1-fixed)

`masked_segmented_softmax` clamps actor logits to [−30, 30] before the softmax:

```python
def masked_segmented_softmax(energies, mask, batch_ind):
    energies = energies.clamp(-30,30)
    energies[~mask.bool()] = -np.inf
    probs = softmax(energies, batch_ind)
    return probs.flatten()
```

A logit beyond ±30 receives zero gradient. Checked with autograd on logits [35, 30, 29, 0, −31], the gradients are [0, 0.113, 0.197, 0, 0]:
- the first and last are zero because of the clamp;
- the logit at 0 also gets a zero gradient, for a different reason: next to logits of 30 its probability is about e⁻²⁹, so its gradient is negligible.

Once a passenger's logit passes +30 it is frozen there, and the actor can no longer learn to prefer one passenger over another.

| Model | Passenger logits ≥ +30 | States where all passengers get equal probability | Probability on the shortest passenger (uniform share) |
|---|---|---|---|
| Cell 1 seed 0 | 73.9% | 366 / 495 | 14.6% (14.6%) |
| Cell 1 seed 300 | 100.0% | 495 / 495 | 14.6% (14.6%) |
| Cell 1 seed 600 | 92.1% | 456 / 495 | 14.6% (14.6%) |
| Cell 1 seed 900 | 87.5% | 433 / 495 | 14.6% (14.6%) |
| Cell 1 seed 1200 | 70.9% | 349 / 495 | 14.6% (14.6%) |
| Cell 1-fixed seed 0 | 100.0% | 495 / 495 | 14.6% (14.6%) |

In every model, passengers take essentially all the actor's probability, while non-passenger logits sit far below (median maximum −10 to −27). The clamp is part of the training code, so this is the trained behaviour, not an artefact of the analysis.

### Seed replication

Headline metrics on the same 495 states (pooled):

| | Cell 1 s0 | Cell 1 s300 | Cell 3 s0 | Cell 3 s300 |
|---|---|---|---|---|
| Selected trip, frames over shortest | 11.15 | 11.03 | 20.36 | *pending* |
| Shortest picked, % | 21.99 | 21.92 | 12.85 | *pending* |
| Shortest proposed, % | 29.82 | 29.78 | 17.68 | *pending* |
| Picked if proposed, % | 73.75 | 73.61 | 72.69 | *pending* |
| Move/no-op picks, % | 7.83 | 7.83 | 7.31 | *pending* |
| Discriminator Spearman, passengers only / all goals | 0.998 / 0.999 | 0.997 / 0.999 | 0.334 / 0.364 | *pending* |
| Discriminator's top passenger, frames over shortest | 0.00 | 0.01 | 14.67 | *pending* |
| Shorter move/no-op beats every passenger, % | 100 | 100 | 0.01 | *pending* |

Cell 1 seed 300 reproduces seed 0 almost exactly. Cell 3 seed 300 will be scored the same way once its model is copied to the Mac.

### Example states

| Figure | What it shows |
|---|---|
| `figures/state_283.png` | Late state: the models disagree. Cell 1's most frequent choice is the shortest passenger; Cell 3's is one 27 frames longer. |
| `figures/state_090.png` | Cell 3's shortest passenger shares its WL histogram with a longer passenger; only time left separates them. |
| `figures/state_171.png` | Frame 0, one passenger, always proposed: Cell 1 chooses a short move or a no-op in 19 of 20 draws. |
| `figures/state_015.png` | Same situation: Cell 1 picks non-delivering goals in all 20 draws. |
| `figures/state_040.png` | Cell 3 proposes the shortest passenger in 40% of draws but picks it in only 12% of those. |
| `figures/state_037.png` | Cell 3's largest detour among early states (+32 frames on average). |

Each figure has two panels:
- *Map:* the city (walls, taxi, waiting passengers), the shortest passenger S and greedy's choice G, and each model's most frequent choice with its route.
- *Choices:* for every goal either model chose, the share of the 20 draws choosing it and its plan length. Rows are sorted by plan length; goals that deliver nothing are greyed.

## Implications for the research question

The question is which encoder learns which policy, and whether differences come from what the encoder can represent.

- **The one expressiveness limit found is the WL actor at L = 1.** Every waiting passenger has the same WL colour, so the WL actor cannot prefer one passenger over another, whatever it learns. Deeper WL (L ≥ 2) might separate passengers by their neighbourhoods (for example, through their destinations). That has not been tested here; the L = 2 runs had other problems (vocabulary size, coverage).
- **The other limitations come from learning or implementation, not representation:**
  - *WL discriminator:* its input distinguishes the shortest passenger in every state, and it uses time left with the correct sign. The weak ranking comes from the time-left scale (time/2000) and, plausibly, the number of updates available to grow that weight.
  - *GNN actor:* the ±30 logit clamp saturates passenger logits and removes their gradient, so the actor cannot express a preference among passengers even though the GNN could.
  - *GNN discriminator:* its timer-like readout ignores whether a goal delivers. It is **not** due to the frozen head: Cell 1-fixed, with a trained head, behaves identically. A hypothesis (training only on executed goals, per-decision discounting) is given above.
- **The two training bugs were silent and, as measured here, harmless.** The stale tether edges and the frozen discriminator head changed neither behaviour nor performance (Cell 1-fixed vs Cell 1).
- **The actor clamp is silent but binding.** It makes the GNN actor uniform over passengers, so the GNN policy amounts to "the shortest of 3 random passengers" (55.5–56.7 vs 57.1 deliveries).
- **Cell 3 is close to a random-passenger policy** (48.2 vs 46.6, intervals overlapping): a uniform actor over passengers, plus a discriminator whose weak ranking adds little.
- **Overall:** on this task, the representation difference between GNN and WL explains the actors' inability to rank passengers only for WL, and only at L = 1. The rest of the GNN-WL gap comes from training details that affect the two encoders differently.

## Implications beyond Cell 1 vs Cell 3

- **Cells 2 and 5 are also pre-fix.** Cell 2's saved optimizer state shows the discriminator head missing from the optimizer, with its weights at initialisation (seeds 0, 300, 600). Cell 5 was trained from a branch descended from Cell 2's, without the fix (from the branch history; no Cell 5 model has been inspected). Cell 1-fixed suggests this bug is harmless for GNN cells, but that is shown only for the object encoding.
- **The clamp is in shared training code** (`masked_segmented_softmax`), so it likely affects every GNN cell's actor. It has been measured only for Cell 1 and Cell 1-fixed.
- **The GNN and WL rows of the main experiment differ in the optimizer fix.** The GNN cells (1, 2, 5) were trained with a frozen discriminator head; the WL cells (3, 4, 6) with a trained one. For the object encoding, Cell 1-fixed shows this difference doesn't matter.

## Caveats

- **Seeds.** Cell 1: seeds 0 and 300 scored (the actor-clamp result holds for all five seeds). Cell 3: seed 0 only so far. Cell 1-fixed: seed 0.
- **Ground truth is plan length only.** A shorter trip is not always better (destinations and other passengers matter later), though the shortest-plan policy matching greedy suggests it is a good proxy. Rollout values are planned. They will continue each goal with the greedy policy, so they will measure a goal's value under greedy continuation, not under each model's own policy.
- **Different simulators.** Old Cell 1 was trained on a simulator with accumulating stale tether edges; Cell 3 and Cell 1-fixed on the fixed one. Each is analysed on its own. Old Cell 1's performance does not change on clean graphs.
- **The state set covers decision points only.** At the models' own decision points the taxi is always empty, so "deliver the carried passenger" and "phantom delivery" goals never arise there and are not analysed.
- **Software stack.** The analysis ran on a Mac with torch 2.13, gym 0.26 and numpy 2.2, not the training stack (torch 1.7, gym 0.18). The training code ran unmodified, with compatibility shims in the analysis code only (legacy gym seeding, removed numpy aliases, a deepcopy-safe environment fork). All three models' training plateaus were reproduced on this stack.
- **WL depth.** WL was analysed at L = 1 only, the depth chosen for all WL cells (Cell 4's existing L = 2 run is being redone at L = 1). Whether deeper WL separates passengers is untested.
- **Code lineage.** Old Cell 1 ran from RCP's main checkout with uncommitted patches; its simulator is rebuilt here from committed code. All patch files are accounted for and none touches the simulator or planner, but the full diff has not yet been inspected.

## Cell 1-fixed

GNN, object encoding, seed 0, trained on RCP from a `cell3-wl-oraclesage` checkout without `--wl-vocab-path` (the log directory matches that launch command; the log itself records no branch or commit). It has both fixes (stale tether edges removed; discriminator head built before the optimizer) and otherwise Cell 1's settings. The zip confirms the optimizer fix directly. The stale-edge fix follows from the branch and can't be verified from the zip.

**Checks.**
- The discriminator head is in the optimizer (58 of 58 tensors, all with Adam state; old Cell 1: 56 of 58). It trained: 7 of 64 weights are beyond the initialisation range, against 0 in old Cell 1.
- Final path-value loss 0.61 (mean of the last 10 logs 0.37, against 2.4–4.5 for old Cell 1). Training plateau 55.0–55.7.
- Loading checks pass (exact weights, bit-identical outputs, 60/60 training-time decisions).

**Performance.** 55.5 deliveries per episode [54.1, 57.0] over seeds 100–109. That is within old Cell 1's 5-seed training range (54.6–56.1) and its Mac evaluation (56.4–56.7). The bug fixes did not change performance.

**Behaviour.** Indistinguishable from old Cell 1. The paired bootstrap difference includes zero for every behavioural metric; the largest is the shortest picked overall, +0.4 points [−0.7, 1.5].

| | Old Cell 1 | Cell 1-fixed | Cell 3 |
|---|---|---|---|
| Discriminator Spearman with plan length, all goals | 0.999 | 0.999 | 0.364 |
| Discriminator Spearman, passengers only | 0.998 | 0.996 | 0.334 |
| Shorter move/no-op beats every passenger | 100% | 100% | 0.01% |
| Picked if proposed: early / middle / late | 58.3 / 100 / 100% | 57.9 / 99.6 / 99.7% | 81.5 / 59.7 / 67.7% |
| Move/no-op picks: early / middle / late | 23.5 / 0 / 0% | 23.5 / 0 / 0% | 18.6 / 1.6 / 1.7% |
| Shortest proposed / picked if proposed / picked | 29.8 / 73.7 / 22.0% | 30.4 / 73.9 / 22.4% | 17.7 / 72.7 / 12.8% |
| Selected trip, frames over shortest | 11.15 | 11.13 | 20.36 |
| Passenger logits ≥ +30 | 73.9% | 100% | — |
| Probability on the shortest passenger (uniform share 14.6%) | 14.6% | 14.6% | 6.2% |

**Answers to the planned questions.**
- *Does the discriminator still prefer non-delivering short goals?* Yes, exactly as before. A trained head changes nothing.
- *Early-state behaviour?* Unchanged.
- *Does the actor saturate at the clamp?* Yes, fully: all passenger logits are at or above +30.

## Optional follow-ups

- **Early-state rollout comparison:** from forked early states, compare "no-op or short move first, then deliver" against "deliver first" over a fixed horizon, to test whether the GNN's early non-delivering choices cost deliveries.
- **Rollout values** (deferred): goal values under greedy continuation for a subset of states.
- **Cell 3 seed 300:** score when its model is on the Mac.
- **Deeper WL:** check whether L = 2 colours separate waiting passengers on this state set.
