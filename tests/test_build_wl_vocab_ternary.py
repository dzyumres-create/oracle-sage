"""
build_wl_vocab on the ternary domain (ternary_corpus / build_ternary_vocab).

  - Same seed, same arguments -> byte-identical vocab file; a different seed -> a
    different one.
  - The file's colours are exactly what direct wl_colours(domain="ternary") calls give on
    the same sampled graphs: the growing vocab built here, call by call, equals the file's
    entries in order, and every sampled graph coloured with the file's frozen vocab gets
    the same colours as the direct growing call (so no corpus graph is OOV).
  - The corpus takes the production path: every live sample's tensors equal the ternary
    converter applied directly to the simulator's facts() at that step, and every
    projection equals plan_ternary on the live sample.
  - Metadata: domain="ternary", convention and L recorded; a standard loader refuses it.

Kept small (2 episodes capped at 300 steps) so it runs in seconds; the full corpus is the
same code with the defaults.

Run from the repo root with:
    PYTHONPATH=. python -m pytest tests/test_build_wl_vocab_ternary.py -v
"""
import functools
import json
import os
import tempfile
import unittest

import torch as th

from sage.agent.wl_plan_feedback_policy import _load_vocab
from sage.domains.gym_taxi.simulator.ternary_planner import plan_ternary
from sage.domains.gym_taxi.utils.ternary_representations import (
    facts_to_atom_graph,
    facts_to_object_graph,
    facts_to_vilg_graph,
)
from sage.domains.utils import build_wl_vocab
from sage.domains.utils.wl_colours import OOV_SIGNATURE, freeze_vocab, wl_colours

CONVENTIONS = ("oracle_sage", "vilg", "atom")
SMALL = {"episodes": 2, "sample_every": 25, "max_steps": 300}
FACTS_TO_GRAPH = {"oracle_sage": facts_to_object_graph, "vilg": facts_to_vilg_graph, "atom": facts_to_atom_graph}


@functools.lru_cache(maxsize=None)
def collect(convention, seed=0):
    """The corpus as plain records, reading what each sample needs before advancing
    (cached: several tests read the same corpus; nothing mutates the records)."""
    records = []
    for sample in build_wl_vocab.ternary_corpus(seed=seed, graph_convention=convention, **SMALL):
        record = {"episode": sample.episode, "step": sample.step, "goal": sample.goal, "data": sample.data}
        if sample.goal is None:
            sim = sample.env.sim
            record["facts"] = sim.facts()
            record["meta"] = {"time": sim.time, "timeout": sim.timeout, "planning": sim.planning}
            record["carrying"] = sim.taxi.passenger is not None
        records.append(record)
    return records


class TernaryVocabTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def path(self, name):
        return os.path.join(self._tmpdir.name, name)

    def build(self, name, convention, level, seed=0):
        out, _total, _size = build_wl_vocab.build_ternary_vocab(
            convention, level, out_path=self.path(name), seed=seed, log_every=0, **SMALL,
        )
        with open(out, "rb") as f:
            return f.read()


class TestDeterminism(TernaryVocabTest):
    def test_same_seed_byte_identical_different_seed_differs(self):
        for convention in CONVENTIONS:
            with self.subTest(convention=convention):
                first = self.build("a.json", convention, 2)
                second = self.build("b.json", convention, 2)
                other = self.build("c.json", convention, 2, seed=1)
                self.assertEqual(first, second)
                self.assertNotEqual(first, other)


class TestFileMatchesDirectColouring(TernaryVocabTest):
    def test_colours_in_file_equal_direct_wl_colours(self):
        for convention in CONVENTIONS:
            records = collect(convention)
            for level in (1, 2):
                with self.subTest(convention=convention, L=level):
                    path = self.path(f"{convention}_L{level}.json")
                    build_wl_vocab.build_ternary_vocab(convention, level, out_path=path, log_every=0, **SMALL)

                    direct_vocab = {}
                    direct_colours = []
                    for record in records:
                        d = record["data"]
                        colours, _ = wl_colours(
                            d.x, d.edge_index, d.edge_attr, num_iterations=level, vocab=direct_vocab,
                            frozen=False, graph_convention=convention, domain="ternary",
                        )
                        direct_colours.append(colours)
                    freeze_vocab(direct_vocab)

                    loaded = _load_vocab(path, convention, level, "ternary")
                    self.assertEqual(list(loaded.items()), list(direct_vocab.items()))

                    oov = loaded[OOV_SIGNATURE]
                    for record, expected in zip(records, direct_colours):
                        d = record["data"]
                        colours, _ = wl_colours(
                            d.x, d.edge_index, d.edge_attr, num_iterations=level, vocab=loaded,
                            frozen=True, graph_convention=convention, domain="ternary",
                        )
                        self.assertTrue(th.equal(colours, expected))
                        self.assertFalse(bool((colours == oov).any()))

    def test_metadata(self):
        path = self.path("v.json")
        build_wl_vocab.build_ternary_vocab("atom", 2, out_path=path, log_every=0, **SMALL)
        with open(path) as f:
            payload = json.load(f)
        self.assertEqual(payload["domain"], "ternary")
        self.assertEqual(payload["graph_convention"], "atom")
        self.assertEqual(payload["num_iterations"], 2)
        self.assertEqual(payload["build_procedure"]["seed"], 0)
        self.assertEqual(payload["build_procedure"]["episodes"], 2)
        with self.assertRaises(ValueError):
            build_wl_vocab.load_vocab(path)
        with self.assertRaises(ValueError):
            _load_vocab(path, "atom", 2)

    def test_default_name(self):
        self.assertEqual(
            build_wl_vocab.ternary_default_out_path("oracle_sage", 1).name,
            "wl_vocab_taxi_city_ternary_oracle_sage_L1.json",
        )
        self.assertEqual(build_wl_vocab.ternary_default_out_path("atom", 2).name, "wl_vocab_taxi_city_ternary_atom_L2.json")


class TestCorpusTakesTheProductionPath(unittest.TestCase):
    def test_live_samples_equal_converter_on_facts_and_projections_equal_plan_ternary(self):
        for convention in CONVENTIONS:
            records = collect(convention)
            live = None
            n_live = n_projection = 0
            with self.subTest(convention=convention):
                for record in records:
                    d = record["data"]
                    if record["goal"] is None:
                        live = record
                        n_live += 1
                        node_feats, edge_feats, edge_index, mask, global_feats = FACTS_TO_GRAPH[convention](
                            record["facts"], record["meta"])
                        self.assertTrue(th.equal(d.x, th.as_tensor(node_feats, dtype=th.float32)))
                        self.assertTrue(th.equal(d.edge_attr, th.as_tensor(edge_feats, dtype=th.float32)))
                        self.assertTrue(th.equal(d.edge_index, th.as_tensor(edge_index, dtype=th.long)))
                        self.assertTrue(th.equal(d.mask, th.as_tensor(mask)))
                    else:
                        n_projection += 1
                        expected, _actions = plan_ternary(live["data"], record["goal"], convention)
                        for name in ("x", "edge_index", "edge_attr", "global_features"):
                            self.assertTrue(th.equal(getattr(d, name), getattr(expected, name)), name)
                self.assertGreater(n_live, 0)
                self.assertEqual(n_projection, 2 * n_live)

    def test_trajectory_does_not_depend_on_convention(self):
        facts = {c: [r["facts"] for r in collect(c) if r["goal"] is None] for c in CONVENTIONS}
        goals = {c: [r["goal"] for r in collect(c) if r["goal"] is not None] for c in CONVENTIONS}
        self.assertEqual(facts["oracle_sage"], facts["vilg"])
        self.assertEqual(facts["oracle_sage"], facts["atom"])
        self.assertEqual(goals["oracle_sage"], goals["vilg"])
        self.assertEqual(goals["oracle_sage"], goals["atom"])

    def test_policy_mix(self):
        """First half scripted (it picks a passenger up within 300 steps), second half
        random (episode 1); every episode starts at step 0."""
        records = collect("vilg")
        episodes = sorted({r["episode"] for r in records})
        self.assertEqual(episodes, [0, 1])
        self.assertTrue(any(r["carrying"] for r in records if r["goal"] is None and r["episode"] == 0))
        for episode in episodes:
            self.assertEqual(min(r["step"] for r in records if r["episode"] == episode), 0)


if __name__ == "__main__":
    unittest.main()
