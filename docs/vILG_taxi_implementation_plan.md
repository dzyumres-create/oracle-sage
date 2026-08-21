# Implementing the νILG Translator in oracle-sage — Step by Step

Repo: https://github.com/AndrewPaulChester/oracle-sage
Target: Cell 2 (GNN encoder + GOOSE's νILG graph construction, Taxi domain)

## Step 1 — new converter function
**File:** `sage/domains/gym_taxi/utils/representations.py`

Add `env_to_vilg_graph(env)` next to the existing `env_to_graph(env)` (don't delete the
original — you need both for the Cell 1 vs Cell 2 comparison).

```python
def env_to_vilg_graph(env):
    node_feats = []      # object node features first, then proposition node features
    node_kind  = []       # parallel list: True = object node, False = proposition node
                           # (used to build the action mask later)
    edges = []             # list of (src_idx, dst_idx, position_label)

    # --- object nodes: identical source data to env_to_graph, same order/indexing ---
    sorted_object_nodes = sorted(env.graph.nodes.items())
    obj_index = {}
    for i, (nid, data) in enumerate(sorted_object_nodes):
        obj_index[nid] = i
        node_feats.append(data['attr'])      # reuse existing 3-way type one-hot unchanged
        node_kind.append(True)

    # --- proposition nodes: one per forward-direction edge in env.graph ---
    next_idx = len(sorted_object_nodes)
    for (u, v, attr) in env.graph.edges(data=True):
        if attr['attr'][-1] != 1:
            continue   # skip reverse-direction duplicate; νILG edges are undirected

        pred_onehot = attr['attr'][:-1]       # reuse existing 3-way predicate one-hot
        # status: for taxi, nothing is currently "goal" in this minimal version —
        # see Step 4 before finalising this. Placeholder below treats all as achieved_nongoal.
        status_onehot = [0, 0, 1]             # [achieved_goal, unachieved_goal, achieved_nongoal]
        node_feats.append(pred_onehot + status_onehot)
        node_kind.append(False)

        edges.append((next_idx, obj_index[u], 1))   # position 1
        edges.append((next_idx, obj_index[v], 2))   # position 2
        next_idx += 1

    node_feats = np.array(node_feats, dtype=np.float64)
    edge_index = np.array([[s, d] for (s, d, _) in edges]).T
    # edge label (argument position) replaces predicate-identity edge attr from Oracle-SAGE's convention
    edge_feats = np.array([[1, 0] if pos == 1 else [0, 1] for (_, _, pos) in edges])

    # mask: only object nodes are selectable actions — proposition nodes excluded
    mask = np.array(node_kind, dtype=bool)
    # then apply the SAME existing planning/SR-DRL masking logic on top, restricted to object nodes
    if env.planning == False:
        restricted = np.zeros(len(node_feats), dtype=bool)
        restricted[obj_index[env.taxi.location]] = True
        for x in env.graph[env.taxi.location]:
            restricted[obj_index[x]] = True
        mask = mask & restricted
    # else: mask stays "all object nodes True, all proposition nodes False"

    global_feats = np.zeros(EMB_SIZE, dtype=np.float64)
    global_feats[0] = (env.timeout - env.time) / env.timeout

    return node_feats, edge_feats, edge_index, mask, global_feats
```

This is a first-pass sketch, not final code — treat the status/goal placeholder as step 4's job,
and treat variable names as illustrative (match your actual coding style / existing helper
patterns in the file).

## Step 2 — wire it into the JSON interchange
**File:** same file, `env_to_json`. Add an equivalent `env_to_vilg_json`:

```python
def env_to_vilg_json(env):
    return graph_to_json(*env_to_vilg_graph(env))
```

`graph_to_json`/`json_to_graph` (in `sage/domains/utils/representations.py`) don't need any
changes — they're already convention-agnostic; they just serialize whatever arrays you hand them.

## Step 3 — declare the new dimensions
**File:** `sage/domains/gym_taxi/envs/taxi_env.py`, around line 539-540.

Currently:
```python
self.observation_space = JsonGraph(
    converter=json_to_graph, node_dimension=3, edge_dimension=4, planner=Planner())
```

For Cell 2 you'll need a way to select the νILG converter/dimensions — cleanest is a
constructor flag (e.g. `graph_convention="oracle_sage"` vs `"vilg"`) threaded through from
wherever `TaxiEnv` is instantiated (check `train` script / experiment configs for how domain
variants are currently selected — likely the same place size/passenger-count variants are
chosen), so Cell 1 and Cell 2 can both be launched from the same codebase without duplicating
the env file. New dimensions:
- `node_dimension`: object one-hot (3) — but proposition nodes need `3 (predicate) + 3 (status)
  = 6` dims, and both node kinds must share one feature vector length. Pad the shorter one with
  zeros (e.g. object nodes get `[type_onehot(3), 0,0,0]`, proposition nodes get
  `[0,0,0, pred_onehot(3), status_onehot(3)]` — 9 total) rather than silently reusing 3 for both,
  since object type and predicate identity are different vocabularies and shouldn't collide in
  the same one-hot slots.
- `edge_dimension`: 2 (argument position 1 or 2) instead of 4.

## Step 4 — decide how "goal" is represented (the real open question)

This is the part the sketch above ducks, and it's genuinely different in Taxi vs. GOOSE's usual
IPC benchmarks: **when a passenger is delivered, `attempt_dropoff` currently calls
`self.graph.remove_node(pid)`** — the passenger node vanishes entirely from `env.graph`.

νILG's goal-status categories (`achieved_propositional_goal` /
`unachieved_propositional_goal` / `achieved_propositional_nongoal`) assume the goal atom's
*node* persists across the transition from unachieved → achieved, so the GNN can observe the
state flip. If the passenger node is deleted on delivery, there's no node left to carry an
`achieved_propositional_goal` label — the goal simply disappears from the graph instead of
being marked satisfied.

Two ways to resolve this, worth raising with Andrew before you commit to one:
1. **Stop removing delivered-passenger nodes**, and instead add an explicit `delivered(pid)`
   proposition node that flips from `unachieved_propositional_goal` to
   `achieved_propositional_goal` on dropoff. This is the more faithful νILG implementation, but
   it means Cell 2's graph never shrinks the way Cell 1's does — episodes end up with strictly
   growing node counts, another structural difference from Cell 1 beyond just "convention."
2. **Keep node removal as-is** and treat "no `delivered` node present" as implicitly meaning
   "not currently a live obligation" — i.e., don't try to represent completed deliveries at all,
   only currently-pending ones. Simpler, closer to current behaviour, but arguably not really
   "using νILG's goal representation" in the sense the GOOSE paper intends, since achieved-vs-
   unachieved status is exactly the distinction νILG's node typing is designed to expose to the GNN.

I'd lean toward flagging this explicitly in your report either way — it's a genuine domain-fit
issue between νILG (designed for static single-shot IPC goals) and Taxi's episodic,
goal-recurring structure, not just an implementation detail.

## Step 5 — update the consumers of `node_dimension`/`edge_dimension`
**File:** `sage/agent/graph_policy.py` — `NodeExtractor.__init__` (line ~95) and
`GNNExtractor.__init__` (line ~111-119) already read these values off
`observation_space.node_dimension` / a passed-in `edge_dim`, so **no code change needed here**
as long as Step 3's `JsonGraph(...)` construction is updated correctly — this was designed to
be convention-agnostic already, which works in your favour.

## Step 6 — sanity checks before running real training
1. Instantiate a single `TaxiEnv`, call `env_to_vilg_graph` directly, and print node/edge counts
   by hand against a small known state (reuse the thesis's 4-location, 1-passenger example from
   Chester's thesis Example 6.1.1 as a hand-checkable ground truth).
2. Confirm `mask` never allows a proposition node to be selected — assert
   `mask[~node_kind_array].sum() == 0` in a unit test.
3. Confirm graph size actually changes across a few steps (passenger spawns → more proposition
   nodes; if you go with Step 4 option 1, delivered passengers should NOT shrink the graph).
4. Run Cell 1 and Cell 2 on the *same* fixed random seed for one short rollout and diff the
   action masks at each step — they should select the same underlying object even though the
   graphs are shaped differently, which is a good smoke test that the action space truly stayed
   identical between conditions.
