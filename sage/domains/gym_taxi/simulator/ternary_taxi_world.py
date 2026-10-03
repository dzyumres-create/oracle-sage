"""
.. module:: ternary_taxi_world
   :synopsis: A ternary-predicate variant of the taxi world, for testing whether graph
   encodings that can only see pairwise object co-occurrence (the object encoding)
   remain expressive once a fact's meaning depends on three arguments at once.

See extended_domain_handoff_ternary_taxi.md for the design rationale. Summary: a
passenger's destination is never a stored value and never a standalone fact -- it can
only be recovered by joining picked_up_at(p, l) with request(p, o, d) on o == l,
including inside this simulator's own dropoff check. This is deliberate: it is the one
thing that makes the encodings diverge, so nothing in this module may compute or cache
a passenger's destination ahead of that join, and no field, fact or comment here may
be named "destination".

This is a NEW, independent simulator -- it does not subclass, import state from, or
share any NamedTuple with TaxiWorldSimulator (sage.domains.gym_taxi.simulator.taxi_world),
so that module, and the old (non-ternary) domain it drives, stay byte-identical.

This class holds a fact-level state, not a graph. sage.domains.gym_taxi.utils.representations
already builds all three graph conventions (object, vILG, atom) by walking
TaxiWorldSimulator's networkx graph directly -- the ternary-domain converters (a later
commit) build instead from facts(), the single query interface this class exposes.

Unlike TaxiWorldSimulator's self.graph (which folds locations, the taxi and passengers
into one networkx graph, and mutates it in place as pickups/dropoffs/moves happen),
this class keeps the road network (self.roads) as a static adjacency structure over
locations only, and tracks the taxi and passengers separately as plain data (TernaryTaxi,
TernaryPassenger). facts() is rebuilt fresh from that data on every call, so there is no
incrementally-mutated graph to keep in sync.
"""
import networkx as nx

from typing import Dict, List, NamedTuple, Optional, Tuple

from sage.domains.gym_taxi.simulator.ternary_planner import object_hypothesis_count
from sage.domains.gym_taxi.utils.config import MAX_EPISODE_LENGTH
from sage.domains.gym_taxi.utils.ternary_representations import facts_to_object_graph
from sage.domains.gym_taxi.utils.utils import generate_city_maze


DEFAULT_REWARDS = {"base": 0, "failed-action": 0, "drop-off": 1}

# Ambiguous-delivery diagnostic: per-episode counters, returned in act()'s info dict on
# every call (so Monitor's info_keywords pick up the totals on whichever step ends the
# episode). A dropoff with nobody aboard counts ONLY towards dropoff_empty; every other
# counter refers to dropoffs attempted while carrying a passenger. "ambiguous" means the
# OBJECT encoding of the true state has more than one valid hypothesis for the carried
# passenger's cluster (ternary_planner.object_hypothesis_count > 1) -- computed whatever
# graph convention the env actually uses, so it measures exposure to the ambiguity.
DROPOFF_DIAGNOSTIC_KEYS = (
    "dropoff_attempts",
    "dropoff_attempts_ambiguous",
    "dropoff_failures",
    "dropoff_failures_ambiguous",
    "dropoff_failures_own_other_dest",
    "dropoff_empty",
)

# facts() emits type atoms (one per object) first, in ascending object-id order --
# taxi (0), then locations (1..size*size), then passengers -- so that a future
# "row k is object k" graph convention (as the atom encoding already relies on for
# TaxiWorldSimulator -- see env_to_atoms) gets the taxi at row 0, matching
# graph_policy.py's mask[:,0]=True assumption that node 0 is the taxi. Relational
# facts follow, grouped by predicate in this order, then sorted lexicographically by
# (int) argument tuple within each predicate -- see facts()'s docstring for why.
RELATION_PREDICATES = ["adjacent", "in", "request", "picked_up_at"]


class TernaryTaxi(NamedTuple):
    node: int
    location: int
    passenger: Optional[int]


class TernaryPassenger(NamedTuple):
    # None while carried (see TernaryTaxi.passenger for containment instead).
    location: Optional[int]
    # Static per passenger, set at spawn and never modified: (origin, destination)
    # pairs. Deliberately NOT a mapping keyed by origin -- keeping it a plain tuple of
    # pairs means nothing here silently indexes straight to "the" destination; every
    # consumer, including this simulator's own dropoff check, must join explicitly.
    requests: Tuple[Tuple[int, int], ...]
    # None until picked up; then the location the passenger boarded at, so it can be
    # joined against requests to find where they are going.
    picked_up_at: Optional[int]
    # The buddy's passenger id, or None if this passenger was never paired or their
    # buddy has already been delivered.
    buddy: Optional[int]


class DropoffRecord(NamedTuple):
    """One dropoff attempted while carrying a passenger (see DROPOFF_DIAGNOSTIC_KEYS)."""
    pid: int
    location: int
    success: bool
    object_ambiguous: bool
    # Only ever True on a failure: the taxi stood on one of pid's OWN request
    # destinations, just not the one its picked_up_at selects -- the signature of a
    # wrong guess between pid's own pairings.
    own_other_dest: bool


class TernaryTaxiWorldSimulator(object):
    def __init__(
        self,
        random,
        size,
        delivery_limit=1,
        concurrent_passengers=2,
        timeout=MAX_EPISODE_LENGTH,
        passenger_creation_probability=1,
        pair_creation_probability=None,
        requests_per_passenger=2,
        random_walls=True,
        rewards=None,
        planning=False,
        observation_fn=None,
        initial_pair=None,
    ):
        """
        Houses the game state and transition dynamics for the ternary-predicate taxi
        world.

        :param random: a seeded numpy RandomState (or equivalent) -- see the RNG-order
            note on generate_road_network/add_taxi for why calls must happen in this
            exact order.
        :param size: side length of the (square) road grid.
        :param concurrent_passengers: must be >= 2 -- buddies always spawn as a pair,
            so there must always be room for two.
        :param pair_creation_probability: probability of attempting to spawn a new
            pair on each step, checked only when there is room for one (see
            try_spawn_pair). Defaults to passenger_creation_probability / 2, so a pair
            of passengers arrives at the same expected rate individual passengers do
            in the old domain (each pair contributes 2 passengers at half the
            per-step probability).
        :param requests_per_passenger: k -- see spawn_pair/facts() for the request
            structure this produces.
        :param observation_fn: called as observation_fn(self) at the end of act() to
            build the value act() returns alongside the reward. Left unset in this
            commit (see act()) -- the env-wiring commit supplies it.
        :param initial_pair: optional (requests_p, location_p, requests_q, location_q)
            tuple (see spawn_pair) to deterministically seed the one pair every
            simulator starts with, bypassing the RNG. Used by tests to build
            hand-checkable cases; production callers leave this None and get a
            randomly drawn pair exactly as TaxiWorldSimulator.add_passenger draws one
            passenger.
        """
        assert concurrent_passengers >= 2, (
            "TernaryTaxiWorldSimulator always spawns buddies as a pair, so "
            f"concurrent_passengers must be >= 2 (got {concurrent_passengers})"
        )
        self.random = random
        self.seed_id = hash(self.random)
        self.time = 0
        self.size = size
        self.delivery_limit = delivery_limit
        self.concurrent_passengers = concurrent_passengers
        self.timeout = timeout
        self.passenger_creation_probability = passenger_creation_probability
        self.pair_creation_probability = (
            pair_creation_probability
            if pair_creation_probability is not None
            else passenger_creation_probability / 2
        )
        self.requests_per_passenger = requests_per_passenger
        self.rewards = rewards if rewards is not None else DEFAULT_REWARDS
        self.done = False
        self.planning = planning
        self.observation_fn = observation_fn
        # A fresh simulator per episode (BaseTaxiEnv.reset rebuilds it), so these are
        # per-episode by construction.
        self.dropoff_counts = {key: 0 for key in DROPOFF_DIAGNOSTIC_KEYS}
        self.dropoff_records: List[DropoffRecord] = []

        # RNG call order matters: generate_road_network then add_taxi, in that order
        # and with no RNG draws in between, exactly mirrors TaxiWorldSimulator's own
        # __init__ order -- so the same seed produces the same maze and the same taxi
        # start location as the old domain (see tests/test_ternary_world.py's
        # TestMatchesOldDomainRNG). Passenger spawning necessarily diverges from here
        # (a pair, not a single passenger, with a different sampling scheme), so
        # nothing past this point is expected to match the old domain's RNG stream.
        self.roads = self.generate_road_network(random_walls)
        assert len(self.roads.nodes) == size * size, (
            f"generate_road_network produced {len(self.roads.nodes)} locations, not "
            f"size*size={size * size} -- generate_city_maze ignores `size` and always "
            f"builds a fixed 20x20 maze (a pre-existing TaxiWorldSimulator quirk), so "
            f"random_walls=True silently breaks for any size != 20. "
            f"_renumber_passengers/spawn_pair's next-id arithmetic (size*size + 1) "
            f"depends on this holding."
        )
        self.taxi = self.add_taxi()

        self.passengers: Dict[int, TernaryPassenger] = {}
        if initial_pair is not None:
            self.spawn_pair(*initial_pair)
        else:
            self._spawn_random_pair()

    def generate_road_network(self, random_walls):
        """
        Same generator, same relabelling as TaxiWorldSimulator.generate_road_network,
        so location ids line up 1..size*size the same way for the same seed. Unlike
        TaxiWorldSimulator, no 'attr' one-hot is stored on nodes/edges here -- this
        class exposes state only through facts(), and the (eventual) converters build
        whichever per-convention node/edge features they need from that.
        """
        if random_walls:
            network = generate_city_maze(self.random)
        else:
            network = nx.grid_2d_graph(self.size, self.size, create_using=nx.DiGraph)
        mapping = {k: v + 1 for v, k in enumerate(network.nodes)}
        nx.relabel_nodes(network, mapping, copy=False)
        return network

    def add_taxi(self):
        """Same draw, same formula as TaxiWorldSimulator.add_taxi -- see the RNG-order
        note in __init__."""
        location = int(self.random.choice(range(1, self.size * self.size + 1)))
        return TernaryTaxi(0, location, None)

    def facts(self):
        """
        The canonical, sorted fact list for the current state -- the single interface
        the (eventual) graph converters use; nothing else in this class or its callers
        should reach past this into self.roads/self.taxi/self.passengers directly.

        Ordering guarantee, in two parts:

        1. Type atoms (location/taxi/passenger -- one per object) come FIRST, sorted
           by ascending object id: taxi (id 0), then every location (1..size*size,
           ascending), then every current passenger (ascending). Id ranges are
           disjoint and each block is already internally contiguous/ascending, so
           concatenating [taxi, sorted(locations), sorted(passengers)] already is the
           ascending-id order -- there is no need to sort the three blocks together.
           This block is "row k is object k" for any future graph convention that
           uses one row per type atom (exactly what the (non-ternary) atom encoding's
           env_to_atoms already relies on for TaxiWorldSimulator), and specifically
           puts the taxi at row/id 0, matching graph_policy.py's mask[:,0]=True
           assumption that node 0 is always the taxi.
        2. Relational facts (adjacent/in/request/picked_up_at) come after, grouped by
           predicate in RELATION_PREDICATES order, then sorted lexicographically by
           argument tuple within each predicate group. Every argument is cast to a
           plain Python int before sorting or returning, so ordering is always
           numeric (1, 2, ..., 10, ...), never lexicographic-string (1, 10, 2, ...) --
           regardless of whether an id originated as a numpy scalar internally.

        Together, this means two calls on states that are logically identical but
        reached via different action sequences (e.g. two different orders of picking
        up two passengers) always produce byte-identical fact lists -- there is no
        dependence on dict insertion order or on the order actions happened to occur
        in.

        Predicates (7, matching the atom encoding's node-type / edge-type split):
          taxi(t)              exactly one fact; t == 0
          location(l)          one per road-network node, l in [1, size*size]
          passenger(p)          one per currently-live passenger id
          adjacent(a, b)       one per directed road edge in self.roads
          in(x, y)              taxi(0)'s location: in(0, l); a waiting passenger's
                                location: in(p, l); or a carried passenger's
                                containment in the taxi: in(p, 0)
          request(p, o, d)     one per (origin, destination) pair in p's static
                                requests -- always exactly requests_per_passenger
                                facts per passenger, spawn to delivery
          picked_up_at(p, l)   only once p has been picked up; l is where they
                                boarded -- this is the ONLY fact a dropoff can join
                                against request(p, o, d) to find where p is going;
                                there is no destination(p, l) fact, ever

        :return: list of (predicate, args_tuple), ordered as described above
        """
        type_atoms = [("taxi", (int(self.taxi.node),))]
        type_atoms.extend(("location", (int(location),)) for location in sorted(self.roads.nodes))
        type_atoms.extend(("passenger", (int(pid),)) for pid in sorted(self.passengers))

        relational = []
        for a, b in self.roads.edges():
            relational.append(("adjacent", (int(a), int(b))))
        relational.append(("in", (int(self.taxi.node), int(self.taxi.location))))
        for pid, passenger in self.passengers.items():
            if passenger.location is not None:
                relational.append(("in", (int(pid), int(passenger.location))))
            else:
                relational.append(("in", (int(pid), int(self.taxi.node))))
        for pid, passenger in self.passengers.items():
            for origin, destination in passenger.requests:
                relational.append(("request", (int(pid), int(origin), int(destination))))
        for pid, passenger in self.passengers.items():
            if passenger.picked_up_at is not None:
                relational.append(("picked_up_at", (int(pid), int(passenger.picked_up_at))))

        predicate_order = {predicate: i for i, predicate in enumerate(RELATION_PREDICATES)}
        relational.sort(key=lambda fact: (predicate_order[fact[0]], fact[1]))

        return type_atoms + relational

    def _get_state_json(self):
        """
        BaseTaxiEnv.reset() (sage/domains/gym_taxi/envs/taxi_env.py) unconditionally
        calls self.sim._get_state_json() to build the initial observation -- unlike
        act(), which only builds one when explicitly stepped. TaxiWorldSimulator has
        its own _get_state_json that dispatches by graph_convention; this simulator
        has no such dispatch (observation_fn already IS the single, convention-
        agnostic hook -- see _ternary_observation_fn in ternary_representations.py),
        so this is a thin shim rather than a real second code path: it exists purely
        so BaseTaxiEnv.reset() keeps working unmodified for a ternary sim exactly as
        it does for the old one.
        """
        return self.observation_fn(self)

    def act(self, action):
        """
        Advances the game state by one step.

        :param action: a node id -- self.taxi.node (0) to attempt a dropoff, a live
            passenger id to attempt a pickup, or a location id to attempt a move.
        :returns: (observation_fn(self), reward, done, info) -- info is a copy of the
            episode's running dropoff_counts (DROPOFF_DIAGNOSTIC_KEYS)
        :raises KeyError: if action is a location id with no road from the taxi's
            current location (the same condition under which TaxiWorldSimulator's
            attempt_move raises KeyError), or an id that is none of the above.
        :raises NotImplementedError: if observation_fn was never set -- a ternary
            GraphTaxiEnv always sets it (_ternary_observation_fn); this only fires
            for a simulator built directly, without going through the env.
        """
        reward = self._apply(action)

        self.try_spawn_pair()
        if self.delivery_limit == 0:
            self.done = True
        self.time += 1

        if self.observation_fn is None:
            raise NotImplementedError(
                "TernaryTaxiWorldSimulator.observation_fn is not set -- this commit "
                "only builds the simulator and its facts(); the env-wiring commit "
                "supplies observation_fn. Call facts() directly until then."
            )
        return self.observation_fn(self), reward, self.done, dict(self.dropoff_counts)

    def _apply(self, action):
        """The dynamics for a single action, split out from act() so a caller that
        only wants the fact-level transition (e.g. a planner or a test) doesn't have
        to also have an observation_fn configured."""
        if action == self.taxi.node:
            return self._attempt_dropoff()
        elif action in self.passengers:
            return self._attempt_pickup(action)
        elif action in self.roads.nodes:
            return self._attempt_move(action)
        raise KeyError(f"unrecognised action id {action}")

    def _attempt_pickup(self, pid):
        """
        Succeeds iff the taxi is empty and at pid's current location. On success, pid
        is no longer anywhere in particular (location becomes None -- see in(p, taxi)
        in facts()) and picked_up_at records where they boarded, which is the only
        thing that will let a later dropoff find their destination.
        """
        if self.taxi.passenger is not None:
            return self.rewards["failed-action"]
        passenger = self.passengers[pid]
        if passenger.location != self.taxi.location:
            return self.rewards["failed-action"]
        self.taxi = self.taxi._replace(passenger=pid)
        self.passengers[pid] = passenger._replace(location=None, picked_up_at=passenger.location)
        return self.rewards["base"]

    def _attempt_dropoff(self):
        """
        Resolves the carried passenger's destination the only way it can ever be
        resolved: the unique request whose origin equals where they were picked up.
        Rule 1 (every passenger's own requests have distinct origins -- enforced by
        spawn_pair/_draw_random_pair) is exactly what makes that join unambiguous; it
        is asserted here rather than silently taking the first match, since a
        violation would mean this simulator's own invariant broke, not a normal
        runtime condition.
        """
        pid = self.taxi.passenger
        if pid is None:
            self.dropoff_counts["dropoff_empty"] += 1
            return self.rewards["failed-action"]
        passenger = self.passengers[pid]
        matching_destinations = [
            destination for origin, destination in passenger.requests if origin == passenger.picked_up_at
        ]
        assert len(matching_destinations) == 1, (
            f"passenger {pid}: expected exactly one request with origin == "
            f"picked_up_at ({passenger.picked_up_at}), found {matching_destinations}"
        )
        destination = matching_destinations[0]
        self._record_dropoff_attempt(pid, passenger, destination)
        if self.taxi.location != destination:
            return self.rewards["failed-action"]

        del self.passengers[pid]
        if passenger.buddy is not None and passenger.buddy in self.passengers:
            buddy = self.passengers[passenger.buddy]
            self.passengers[passenger.buddy] = buddy._replace(buddy=None)
        self.taxi = self.taxi._replace(passenger=None)
        self.delivery_limit -= 1
        self._renumber_passengers()
        return self.rewards["drop-off"]

    def _object_ambiguous(self, pid):
        """Whether the OBJECT encoding of the current (true) state admits more than one
        hypothesis for pid's cluster -- the planner's own decoder logic, via
        object_hypothesis_count, never a reimplementation. Diagnostic only: it reads
        facts() and changes no state."""
        meta = {"time": self.time, "timeout": self.timeout, "planning": True}
        node_feats, edge_feats, edge_index, _mask, _global = facts_to_object_graph(self.facts(), meta)
        return object_hypothesis_count(node_feats, edge_index, edge_feats, pid) > 1

    def _record_dropoff_attempt(self, pid, passenger, destination):
        """Updates the ambiguous-delivery diagnostic for a dropoff attempted while
        carrying pid, BEFORE the dropoff changes any state (a success deletes pid)."""
        success = self.taxi.location == destination
        object_ambiguous = self._object_ambiguous(pid)
        own_other_dest = (not success) and self.taxi.location in {d for _o, d in passenger.requests}
        self.dropoff_records.append(
            DropoffRecord(int(pid), int(self.taxi.location), success, object_ambiguous, own_other_dest)
        )
        counts = self.dropoff_counts
        counts["dropoff_attempts"] += 1
        if object_ambiguous:
            counts["dropoff_attempts_ambiguous"] += 1
        if not success:
            counts["dropoff_failures"] += 1
            if object_ambiguous:
                counts["dropoff_failures_ambiguous"] += 1
            if own_other_dest:
                counts["dropoff_failures_own_other_dest"] += 1

    def _attempt_move(self, action):
        start = self.taxi.location
        if start == action:
            return self.rewards["base"]  # no-op
        if not self.roads.has_edge(start, action):
            raise KeyError(f"no road from {start} to {action}")
        self.taxi = self.taxi._replace(location=action)
        return self.rewards["base"]

    def try_spawn_pair(self):
        """Mirrors TaxiWorldSimulator.try_spawn_passenger's short-circuit shape: the
        capacity check happens first and consumes no RNG, so a full simulator draws
        nothing from self.random on a step where there isn't room for a pair."""
        if (
            len(self.passengers) + 2 <= self.concurrent_passengers
            and self.random.uniform() < self.pair_creation_probability
        ):
            self._spawn_random_pair()

    def _draw_random_pair(self):
        """
        Draws a fresh buddy pair per the extended-domain handoff's generator rules:
        rule 1 (each passenger's own requests have distinct origins) and rule 2
        (buddies share origins, with destinations rotated by one so the pair is
        genuinely crossed rather than a relabelling of each other).

        2 * requests_per_passenger distinct locations are drawn; the first half are
        the shared origins, the second half the destinations. p's i-th request is
        (origins[i], destinations[i]); q's i-th request is (origins[i],
        destinations[(i+1) % k]) -- same origin, rotated destination, so p and q's
        request lists agree on every origin but disagree on at least one destination
        for k >= 2.

        :return: (requests_p, location_p, requests_q, location_q), as spawn_pair
            expects
        """
        k = self.requests_per_passenger
        locations = self.random.choice(range(1, self.size * self.size + 1), 2 * k, replace=False)
        origins = [int(x) for x in locations[:k]]
        destinations = [int(x) for x in locations[k:]]
        requests_p = tuple((origins[i], destinations[i]) for i in range(k))
        requests_q = tuple((origins[i], destinations[(i + 1) % k]) for i in range(k))
        # Each buddy spawns at an origin of its own, drawn independently -- not
        # necessarily the same origin as the other, and not necessarily origins[0].
        location_p = int(self.random.choice(origins))
        location_q = int(self.random.choice(origins))
        return requests_p, location_p, requests_q, location_q

    def _spawn_random_pair(self):
        self.spawn_pair(*self._draw_random_pair())

    def spawn_pair(self, requests_p, location_p, requests_q, location_q):
        """
        Adds one buddy pair to self.passengers, bypassing the RNG -- the hook
        tests use (via __init__'s initial_pair, or by calling this directly) to build
        hand-checkable cases with a specific, known pair instead of whatever
        _draw_random_pair would have produced.

        :param requests_p: tuple of (origin, destination) pairs for the first buddy
        :param location_p: the first buddy's starting location (must be one of
            requests_p's origins, by rule 1/2, though this is not itself checked here
            -- callers building hand-checked cases are expected to satisfy it, exactly
            as TaxiWorldSimulator.add_passenger does not itself re-validate its draw)
        :param requests_q: as requests_p, for the second buddy
        :param location_q: as location_p, for the second buddy
        :return: (pid_p, pid_q), the newly assigned ids
        """
        base = self.size * self.size + 1 + len(self.passengers)
        pid_p, pid_q = base, base + 1
        self.passengers[pid_p] = TernaryPassenger(location_p, tuple(requests_p), None, pid_q)
        self.passengers[pid_q] = TernaryPassenger(location_q, tuple(requests_q), None, pid_p)
        return pid_p, pid_q

    def _renumber_passengers(self):
        """
        Repacks passenger ids back into a contiguous block starting at
        size*size + 1, in spawn order (i.e. sorted by their current id, which is
        monotonic in spawn order since ids are only ever assigned by spawn_pair
        appending to the top of the current range) -- the same compacting spawn_pair
        relies on for its own next-id arithmetic, and the same idea as
        TaxiWorldSimulator.resort_passengers, scoped to the passenger id block only
        (locations 1..size*size and taxi 0 never move).
        """
        old_ids = sorted(self.passengers.keys())
        base = self.size * self.size + 1
        mapping = {old_id: base + i for i, old_id in enumerate(old_ids)}
        renumbered: Dict[int, TernaryPassenger] = {}
        for old_id, passenger in self.passengers.items():
            new_buddy = mapping[passenger.buddy] if passenger.buddy is not None else None
            renumbered[mapping[old_id]] = passenger._replace(buddy=new_buddy)
        self.passengers = renumbered
        if self.taxi.passenger is not None:
            self.taxi = self.taxi._replace(passenger=mapping[self.taxi.passenger])
