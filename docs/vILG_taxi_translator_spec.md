# νILG → Taxi Translator: Design Spec (Cell 2)

## 1. What this replaces

Oracle-SAGE's existing translator does this, per object/predicate:

| PDDL element | Oracle-SAGE graph element |
|---|---|
| object | node |
| unary predicate `p(o)` | binary attribute on node `o` |
| unary function `f(o)` | real-valued attribute on node `o` |
| binary predicate/function `p(o1,o2)` | edge `o1 → o2`, attribute = which predicate |

Your Cell 2 translator needs to instead produce a **νILG** (Chen & Thiébaux, Def. 3.1), where
*every grounded predicate instance gets its own node*, not just an attribute or edge:

| PDDL element | νILG graph element |
|---|---|
| object | node (`O`) |
| grounded proposition true in state, or appearing in goal | node (`Xp(s0) ∪ Gp`) |
| argument position of that proposition | labeled edge from the proposition node to the object node |

Taxi has no numeric fluents in its base form, so you can ignore νILG's numeric-node machinery
(`Xn(s0)`, `Gn`) entirely for this domain — one simplification in your favour.

## 2. Node set

```
V = O  ∪  (Xp(s0) ∪ Gp)
```

- `O`: one node per object (taxi, passengers, locations) — same as Oracle-SAGE, unchanged.
- `Xp(s0) ∪ Gp`: one node per **distinct grounded atom** that is either true in the current
  state, or part of the goal. Note this is a *union* — an atom that's true now and also a goal
  (e.g. `delivered(p1)` once satisfied) gets exactly one node, not two.

Important: predicates that are **false and not a goal** never get a node. Only true-now or
goal-relevant atoms materialize. This means the node set is state-dependent and changes size
every timestep as passengers get picked up/dropped off — worth confirming your GNN pipeline
already tolerates variable graph size per step (Oracle-SAGE's GNN is size-agnostic by design,
so this should be fine, but flag it explicitly when you implement it).

## 3. Categorical node features `Fcat`

```
Fcat(u) =
    OBJ(u)                                        if u ∈ O
    (PRED(u), achieved_propositional_goal)        if u ∈ Xp(s0) ∩ Gp
    (PRED(u), unachieved_propositional_goal)      if u ∈ Gp \ Xp(s0)
    (PRED(u), achieved_propositional_nongoal)     if u ∈ Xp(s0) \ Gp
```

- `OBJ(u)` = object type: `taxi` / `passenger` / `location` (same one-hot as Oracle-SAGE uses today).
- `PRED(u)` = which predicate this atom was grounded from: `empty`, `delivered`, `in`,
  `destination`, `adjacent`.

Note `unachieved_propositional_goal` atoms (e.g. `delivered(p3)` before it happens) still get a
node even though they're not true yet — this is how the GNN "sees" the goal at all under νILG.
That's a structural difference from Oracle-SAGE's convention, where goal info is injected
separately (not via the same graph). Confirm with Andrew whether Oracle-SAGE's existing
goal-injection mechanism should be **removed** for Cell 2 (since νILG now carries goal info
natively) — running both would be redundant/leaky.

## 4. Edges

```
E = { ⟨p, o_i⟩ : p = σ(o_1,...,o_n) ∈ Xp(s0) ∪ Gp,  i ∈ [1..n] }
L(⟨p, o_i⟩) = i     (edge label = argument position, 1 or 2 for taxi's max arity)
```

- `empty(t)`, `delivered(pi)` → 1 edge each (arity 1) → node `p` connects to node `t`/`pi` with label 1.
- `in(x,l)`, `destination(pi,l)`, `adjacent(l1,l2)` → 2 edges each (arity 2) → node `p` connects
  to arg-1 object with label 1, and to arg-2 object with label 2.

Edge label = argument position, **not** predicate identity (predicate identity already lives on
the proposition node's `Fcat`). This is the opposite of Oracle-SAGE's convention, where edge
attributes *were* the predicate identity.

## 5. Continuous features / global node

Taxi is purely categorical, so `Fcon(u) = 0` for all nodes — no numeric machinery needed.

**Open design question:** Oracle-SAGE's thesis representation stores elapsed timestep as an
attribute on a special global node `u`. νILG as defined (Chen & Thiébaux) has no equivalent
global node concept. You'll need to decide and confirm with Andrew:
- (a) keep Oracle-SAGE's global node alongside the νILG object+proposition graph, feeding
  timestep in as before, or
- (b) drop it / fold timestep in some other way.
Since the Training Plan says architecture stays fixed and only graph construction changes,
(a) is probably the intended answer — but the global node isn't part of νILG's formal
definition, so it's worth stating explicitly in your report as an implementation choice, not
something inherited from the GOOSE paper.

## 6. Feature vectorization (for GNN input)

Following GOOSE Sec. 4's transform: fix a domain-wide vocabulary `Σ_V` once
(not per-instance), then one-hot encode:

```
Σ_V = {taxi, passenger, location}                                   # object types
    ∪ {(pred, status) : pred ∈ {empty, delivered, in, destination, adjacent},
                          status ∈ {achieved_goal, unachieved_goal, achieved_nongoal}}
```

Not all (pred, status) combinations are reachable in practice — e.g. `adjacent` is never a
goal predicate, so it only ever appears with `achieved_nongoal`. That's fine; the unreachable
one-hot slots are just always zero. Keep the fixed-size vocabulary so the GNN's input layer
dimension doesn't change across episodes/instances.

```
X(u) = one_hot(Fcat(u), Σ_V)      # Fcon is all zero for taxi, so no concatenation needed
```

## 7. Pseudocode

```python
def build_vilg(state, goal, object_types, sigma_v):
    """
    state: current grounded predicates true now, e.g. {('in', 't', 'l0'), ('empty', 't'), ...}
    goal:  fixed set of grounded goal predicates, e.g. {('delivered', 'p1'), ('delivered', 'p2')}
    object_types: dict {object_name: type}, unchanged from Oracle-SAGE's object list
    sigma_v: fixed domain-wide categorical vocabulary (built once, not per instance)
    """
    nodes = {}
    edges = []

    # 1. object nodes — identical to Oracle-SAGE
    for obj, otype in object_types.items():
        nodes[obj] = {"kind": "object", "feat": one_hot(otype, sigma_v)}

    # 2. proposition nodes — union of true-now and goal atoms, deduplicated by the atom itself
    atoms = state | goal
    for atom in atoms:
        pred, *args = atom
        if atom in state and atom in goal:
            status = "achieved_propositional_goal"
        elif atom in goal:
            status = "unachieved_propositional_goal"
        else:
            status = "achieved_propositional_nongoal"

        prop_node_id = atom  # atoms are unique by construction, safe to use as node id
        nodes[prop_node_id] = {"kind": "proposition", "feat": one_hot((pred, status), sigma_v)}

        # 3. edges: proposition node -> each argument, labeled by position
        for position, arg_obj in enumerate(args, start=1):
            edges.append((prop_node_id, arg_obj, position))

    return nodes, edges


def to_gnn_input(nodes, edges):
    """Pack into whatever tensor format Oracle-SAGE's existing GN framework expects —
    node feature matrix, edge index list, edge label (position) list."""
    node_ids = list(nodes.keys())
    idx = {nid: i for i, nid in enumerate(node_ids)}
    node_feats = [nodes[nid]["feat"] for nid in node_ids]
    edge_index = [(idx[src], idx[dst]) for src, dst, _ in edges]
    edge_labels = [pos for _, _, pos in edges]
    return node_feats, edge_index, edge_labels
```

## 8. Action space (already decided, restated for completeness)

Per the Training Plan: the actor's selectable action space stays **object nodes only** — same
as Oracle-SAGE. Proposition nodes never appear as selectable ancillary-goal targets; they exist
purely so the GNN's message passing can route predicate/goal information into the object node
embeddings. When you implement action selection, filter the candidate node list to
`nodes[n]["kind"] == "object"` before the autoregressive goal-selection step.

## 9. Side effects worth flagging in your report

### 9.1 Effective message-passing depth shrinks

Under Oracle-SAGE's convention, two objects related by a binary predicate (e.g. taxi and its
current location via `in`) are **1 hop apart** — connected directly by an edge. Under νILG,
they're **2 hops apart** — taxi → `in(t,l0)` node → location. Every binary relation now costs
an extra message-passing step to traverse.

This isn't a bug, but it's directly relevant to your research question: since your GNN encoder
has a *fixed* trained depth, moving to νILG effectively shrinks its reachable radius in terms of
"objects apart," even before you vary task horizon. You may want to either (a) explicitly note
this as a confound when comparing Cell 1 vs Cell 2 results, or (b) compensate by adding one
extra message-passing layer for the νILG condition and justify that choice. Worth raising with
Andrew before you lock in the experiment design — it affects whether Cell 1 vs Cell 2 is a clean
comparison of "graph convention" alone, or convention entangled with effective depth.

### 9.2 Persistent goal nodes grow the graph in aggregate, not monotonically per step

Once Step 4 is implemented (destination(pid, loc) persists and flips to
`achieved_propositional_goal` instead of the passenger node being deleted on delivery), the
original expectation was that Cell 2's node count would **strictly grow** across an episode,
never shrinking the way Cell 1's does. Having actually implemented and hand-verified this
(`docs/test_vilg_delivery.py`), the corrected picture is more precise than that:

- The goal node itself — `destination(pid, loc)` — never disappears once created. It persists
  at a stable identity and correctly flips `unachieved_propositional_goal` →
  `achieved_propositional_goal` on delivery. That part of the "doesn't shrink" intuition holds.
- But total node count **is not monotonic at the per-step level**. At the exact delivery
  transition, `attempt_dropoff` removes the now-stale `in(pid, taxi)` "carrying" edges (the
  passenger genuinely is no longer in the taxi, and that atom isn't a goal, so νILG correctly
  gives it no node). In the hand-traced example, this drops total node count by 1 at that one
  step (38 → 37), even though the goal node it might look like "shrank" is untouched.
- So the real invariant is: **goal-relevant nodes never shrink; non-goal, non-adjacent
  propositions (like "currently carrying") can still appear and disappear as normal, and their
  churn can transiently outweigh goal-node accumulation at any single step.** Over a full
  episode with several deliveries, Cell 2's node count still trends upward in aggregate relative
  to Cell 1 (which deletes the entire passenger subgraph, goal atom included, on every
  delivery) — but "strictly growing, step over step" is not a guarantee you can rely on or
  assert in analysis code.

If your experiment design or analysis leaned on a per-step monotonic growth assumption (e.g. to
sanity-check episode traces, or to reason about padding/batching by node count), swap it for the
aggregate/goal-node-specific version above.

## 10. Open questions to confirm before/while coding

1. Global node / timestep handling (Section 5) — keep Oracle-SAGE's global node or not?
2. Should Oracle-SAGE's existing separate goal-injection mechanism be removed for Cell 2, since
   νILG now encodes goal status on proposition nodes natively?
3. Do you compensate for the "effective depth shrinks under νILG" effect (Section 9), or treat
   it as an intentional part of what you're measuring?
4. Vocabulary `Σ_V` — confirm the exact predicate list is fixed for all Taxi variants (small/
   medium/large) so the one-hot dimension doesn't change between grid sizes, which would break
   using the same trained GNN across variants.
