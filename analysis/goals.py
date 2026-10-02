"""Goal types for city taxi, as the training planner (Planner.plan) treats them.

Ground-truth labels (see Step 0 f):
  noop             - taxi node (always an empty plan: graph_to_networkx never sets
                     taxi.passenger, so 'deliver via taxi node' does not exist), or
                     the taxi's current location. 1 frame, nothing happens.
  move             - any other location node.
  deliver          - the passenger currently in the taxi: drive to its destination, drop off.
  pickup_deliver   - a waiting passenger while the taxi is empty: full trip.
  phantom          - a waiting passenger while the taxi is carrying another one:
                     the pickup and (usually) the drop-off fail in the simulator,
                     but the projection shows the waiting passenger delivered.
"""
import numpy as np

GOAL_TYPES = ("noop", "move", "deliver", "pickup_deliver", "phantom")


def parse_state(data):
    """(taxi_node, taxi_location, carried passenger node or None, waiting passenger nodes)
    from a torch_geometric Data in the oracle_sage encoding."""
    from sage.domains.gym_taxi.simulator.planner import graph_to_networkx
    st = graph_to_networkx(data)
    carried = [p.node for p in st.passengers if p.location == st.taxi.node]
    waiting = [p.node for p in st.passengers if p.location != st.taxi.node]
    return st.taxi.node, st.taxi.location, (carried[0] if carried else None), waiting


def classify_goals(data, goals=None):
    """Goal type for each node id in `goals` (default: all nodes)."""
    taxi, loc, carried, waiting = parse_state(data)
    x = data.x.cpu().numpy()
    n = len(x)
    goals = range(n) if goals is None else goals
    waiting = set(waiting)
    out = []
    for g in goals:
        g = int(g)
        if g == taxi or g == loc:
            out.append("noop")
        elif x[g, 0] == 1:
            out.append("move")
        elif g == carried:
            out.append("deliver")
        elif g in waiting:
            out.append("phantom" if carried is not None else "pickup_deliver")
        else:
            raise ValueError(f"unclassifiable goal {g}")
    return out
