"""
analysis/ternary_collision_production.py, against ground truth that does not reuse it:

  - the hand-built cases of tests/test_wl_ternary.py, run through the script's own metric
    functions (pair_metrics / buddies_metric): M1 must reproduce
    EXPECTED_SAME_HISTOGRAM_AS_FLIP, M3 must reproduce EXPECTED_BUDDY_SPLIT, and M2 (the
    carried passenger's colour, C vs C_FLIP) the hand derivation below;
  - the counterfactual: on a real crossed state it differs from the state exactly in p's
    and q's request facts, exchanged, and equals build_flipped_facts as a set;
  - nonisomorphism_proof: proves C vs C_FLIP, and recognises W vs W_FLIP (a pure
    relabelling of p and q -- the prototype's earlier bug) and C vs C as isomorphic;
  - passenger_after (id tracking through a projected delivery) against plan_on_facts'
    projected facts on a real state;
  - a short end-to-end run: object M1 = M2 = 100% at every L with identical tensors, every
    counterfactual proven non-isomorphic, and no spot-check disagreement with the
    reference WL.

Run from the repo root with:
    PYTHONPATH=. python -m pytest tests/test_ternary_collision_production.py -v
"""
import random as pyrandom
import unittest

from analysis import ternary_collision_production as tcp
from analysis.ternary_collision_counter import build_flipped_facts
from sage.domains.gym_taxi.simulator.ternary_planner import plan_on_facts
from tests.test_ternary_planner import make_ternary_sim
from sage.domains.gym_taxi.simulator.ternary_scripted_policy import scripted_policy
from tests.test_ternary_world import crossed_state
from tests.test_wl_ternary import (
    CASE_C,
    CASE_C_FLIP,
    CASE_W,
    CASE_W_FLIP,
    EXPECTED_BUDDY_SPLIT,
    EXPECTED_SAME_HISTOGRAM_AS_FLIP,
)

CONVENTIONS = ("oracle_sage", "vilg", "atom")
META = {"time": 0, "timeout": 100, "planning": True}
CASES = {"W": (CASE_W, CASE_W_FLIP), "C": (CASE_C, CASE_C_FLIP)}

# M2 on the hand case: is the carried passenger p (row 5) coloured the same in C and in
# C_FLIP (one shared growing vocab)? Derived by hand:
#   object: the two graphs are identical tensors -> same at every L.
#   vilg:   L=1 p sees (in,1), (req,1)x2, (pua,1) -- types only -> same. L=2 p sees its
#           neighbours' L=1 colours: in(5,0), pua(5,1) unchanged, and both requests are
#           still one class at L=1 -> same. L=3: its requests' L=2 colours encode (origin,
#           destination) location classes -- (1,3), (2,4) in C vs (1,4), (2,3) in C_FLIP,
#           and locations 1..4 are pairwise distinct at L=1 in both -> different.
#   atom:   L=1 p's out-edges are all label (1,1), to in/req/req/pua -> same. L=2 p sees its
#           requests' L=1 colours: in C, R1 (5,1,3) sees pua(5,1) via {11,22} AND in(0,3)
#           via {32}, R2 (5,2,4) sees pua via {11} only; in C_FLIP R1' (5,1,4) sees pua via
#           {11,22} but no in(0,*), and R2' (5,2,3) sees pua via {11} and in(0,3) via
#           {32} -> the multisets differ -> different from L=2.
EXPECTED_CARRIED_SAME_AS_FLIP = {
    "oracle_sage": {1: True, 2: True, 3: True},
    "vilg": {1: True, 2: True, 3: False},
    "atom": {1: True, 2: False, 3: False},
}


class TestHandBuiltCasesThroughTheScript(unittest.TestCase):
    def test_m1_histogram_against_flip(self):
        for (case, convention), per_level in sorted(EXPECTED_SAME_HISTOGRAM_AS_FLIP.items()):
            state, flip = CASES[case]
            for level, expected in per_level.items():
                with self.subTest(case=case, convention=convention, L=level):
                    m1, _m2, _oov = tcp.pair_metrics(
                        tcp.encode(state, META, convention), tcp.encode(flip, META, convention),
                        5, 5, convention, level, {})
                    self.assertEqual(m1, expected)

    def test_m2_carried_passenger_against_flip(self):
        for convention, per_level in EXPECTED_CARRIED_SAME_AS_FLIP.items():
            for level, expected in per_level.items():
                with self.subTest(convention=convention, L=level):
                    _m1, m2, _oov = tcp.pair_metrics(
                        tcp.encode(CASE_C, META, convention), tcp.encode(CASE_C_FLIP, META, convention),
                        5, 5, convention, level, {})
                    self.assertEqual(m2, expected)

    def test_m3_waiting_buddies(self):
        for convention in CONVENTIONS:
            split_at = EXPECTED_BUDDY_SPLIT[("W", convention)]
            for level in (1, 2, 3):
                with self.subTest(convention=convention, L=level):
                    (same, oov), = tcp.buddies_metric(tcp.encode(CASE_W, META, convention), [(5, 6)], convention, level, {})
                    self.assertEqual(same, split_at is None or level < split_at)
                    self.assertFalse(oov)


def first_crossed_sim(seed):
    sim = make_ternary_sim(seed)
    rng = pyrandom.Random(seed)
    for _ in range(3000):
        if crossed_state(sim):
            return sim
        sim.act(scripted_policy(sim, rng))
    raise AssertionError(f"no crossed state within 3000 steps for seed {seed}")


class TestCounterfactual(unittest.TestCase):
    def test_exchanges_exactly_the_two_request_tables(self):
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                sim = first_crossed_sim(seed)
                p = sim.taxi.passenger
                q = sim.passengers[p].buddy
                facts = sim.facts()
                cf = tcp.counterfactual_sim(sim, p, q).facts()

                def requests(fs, pid):
                    return sorted(args[1:] for pred, args in fs if pred == "request" and args[0] == pid)

                self.assertEqual(requests(cf, p), requests(facts, q))
                self.assertEqual(requests(cf, q), requests(facts, p))
                self.assertNotEqual(requests(cf, p), requests(facts, p))
                rest = lambda fs: [f for f in fs if not (f[0] == "request" and f[1][0] in (p, q))]
                self.assertEqual(rest(cf), rest(facts))
                # same fact set as the prototype's build_flipped_facts (whose list order is
                # not canonical: it appends the swapped requests at the end)
                _p, _q, flipped = build_flipped_facts(sim)
                self.assertEqual((_p, _q), (p, q))
                self.assertEqual(sorted(flipped), sorted(cf))
                # the original simulator is untouched
                self.assertEqual(sim.facts(), facts)

    def test_nonisomorphism_proof(self):
        self.assertEqual(tcp.nonisomorphism_proof(CASE_C, CASE_C_FLIP), "invariant")
        self.assertEqual(tcp.nonisomorphism_proof(CASE_W, CASE_W_FLIP), "isomorphic")
        self.assertEqual(tcp.nonisomorphism_proof(CASE_C, CASE_C), "isomorphic")


class TestPassengerAfter(unittest.TestCase):
    def test_against_plan_on_facts_delivery(self):
        sim = make_ternary_sim(3)
        rng = pyrandom.Random(3)
        while len(sim.passengers) < 6 or sim.taxi.passenger is not None:
            sim.act(rng.choice([sim.taxi.location] + list(sim.roads.successors(sim.taxi.location))))
        facts = sim.facts()
        passengers = sorted(sim.passengers)
        goal = passengers[2]
        _actions, projected = plan_on_facts(facts, goal)

        def requests(fs, pid):
            return sorted(args[1:] for pred, args in fs if pred == "request" and args[0] == pid)

        for pid in passengers:
            new = tcp.passenger_after(pid, goal, set(passengers))
            if pid == goal:
                self.assertIsNone(new)
            else:
                self.assertEqual(requests(projected, new), requests(facts, pid))
        # a location goal renumbers nothing
        self.assertEqual(tcp.passenger_after(passengers[-1], 5, set(passengers)), passengers[-1])


class TestShortRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = tcp.run(seeds=(7,), episodes=2, sample_every=40, max_steps=600, use_frozen=False,
                             spot_checks=2, log=False, compare_prototype=False)

    def test_crossed_pairs_exist_and_are_proven_nonisomorphic(self):
        crossed = self.report["crossed_live_states"]
        self.assertGreater(crossed, 0)
        cf = self.report["counterfactual"]
        self.assertEqual(cf["flipped_matches"], crossed)
        self.assertEqual(cf["dest_differs"], crossed)
        self.assertNotIn("isomorphic", cf["nonisomorphic"])

    def test_object_m1_m2_are_total(self):
        r = self.report["conventions"]["oracle_sage"]
        hits, n = r["object_identical_tensors"]
        self.assertGreater(n, 0)
        self.assertEqual(hits, n)
        for level in tcp.GROWING_LEVELS:
            row = r["summary"][f"growing_L{level}"]
            self.assertEqual(row["M1"][0], row["M1"][1])
            self.assertEqual(row["M2"][0], row["M2"][1])

    def test_spot_checks_agree_with_reference_wl(self):
        for convention in CONVENTIONS:
            r = self.report["conventions"][convention]
            with self.subTest(convention=convention):
                self.assertEqual(r["spot_check_disagreements"], [])
                self.assertGreater(sum(v[0] for v in r["spot_checks"].values()), 0)


class TestPrototypeComparison(unittest.TestCase):
    def test_replay_of_the_published_sample_and_encoder_equivalence(self):
        """published_sample_pairs asserts its pairs ARE the prototype sampler's (via
        facts_to_ext_atoms); on them the production converters with the taxi and `in`
        facts dropped must agree with the prototype's own encoders for atom and vILG
        (object differs by design: the prototype keeps one edge per fact and pair, and no
        argument positions)."""
        pairs = tcp.published_sample_pairs(seeds=(0,), per_seed_target=6)
        self.assertEqual(len(pairs), 6)
        result = tcp.prototype_comparison(pairs, max_l=2)
        for convention in ("vilg", "atom"):
            with self.subTest(convention=convention):
                self.assertEqual(result["prototype"][convention], result["production_without_taxi_in"][convention])
        for level in (1, 2):
            self.assertEqual(result["production"]["oracle_sage"][level], [6, 6])

    def test_drop_taxi_and_in(self):
        dropped = tcp.drop_taxi_and_in(CASE_C)
        self.assertEqual([pred for pred, _ in dropped[:6]], ["location"] * 4 + ["passenger"] * 2)
        self.assertNotIn("in", [pred for pred, _ in dropped])
        self.assertIn(("request", (4, 0, 2)), dropped)  # request(5, 1, 3) shifted down by one
        self.assertIn(("picked_up_at", (4, 0)), dropped)


if __name__ == "__main__":
    unittest.main()
