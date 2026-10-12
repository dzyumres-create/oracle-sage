"""
The vocab "domain" field ("standard" | "ternary"): every loader checks it, and existing
vocab files load exactly as before.

Loaders covered (every place a vocab file is read for use):
  - wl_vocab_cache.configure_wl_vocab_override (env_to_graph / Planner.plan's path) and
    wl_vocab_cache.get_wl_vocab (which reads whatever was configured);
  - wl_plan_feedback_policy._load_vocab (the policy's own path);
  - build_wl_vocab.load_vocab and load_vocab_with_metadata (analysis tooling).

Ground truth for "loads exactly as before" is a JSON decode written in this file
(decode_entries below), compared entry by entry and in order, plus the committed files'
SHA-256, recorded at 3b06e2b, so the files themselves cannot have changed either.

Run from the repo root with:
    PYTHONPATH=. python -m pytest tests/test_wl_vocab_domain.py -v
"""
import hashlib
import json
import os
import tempfile
import unittest
import warnings

from sage.agent.wl_plan_feedback_policy import _load_vocab
from sage.domains.gym_taxi.utils import wl_vocab_cache
from sage.domains.gym_taxi.utils.wl_vocab_cache import (
    WL_VOCAB_PATH,
    configure_wl_vocab_override,
    get_wl_vocab,
    reset_wl_vocab_override,
    validate_wl_vocab_metadata,
)
from sage.domains.utils import build_wl_vocab
from sage.domains.utils.wl_colours import OOV_SIGNATURE

VOCAB_DIR = WL_VOCAB_PATH.parent

# SHA-256 of every vocab file committed before the domain field existed (3b06e2b).
EXISTING_VOCAB_SHA256 = {
    "wl_vocab_taxi_city_L1.json": "4493b8f869117e6c63b1c4fa40dff45fa8b5f919a6b14a0f8868e2600ad9ed53",
    "wl_vocab_taxi_city_L1_edgefixed.json": "883613f7af910465dc2787cac102516b9bc20cf3eb5a4d3f5f0d49000075cc99",
    "wl_vocab_taxi_city_atom_L1_full.json": "c12203833b5d8cf280dbe4ca637dd2522e7a6b0c65b519f3822e3b936ae6b024",
    "wl_vocab_taxi_city_atom_L2.json": "108ea7118ed475d14586a634b2d02f5b5ca7fd84c4b48832aa304173a5a85cdd",
    "wl_vocab_taxi_city_atom_L2_full.json": "4959b81d298ecd747c46f3375ac9a651b1b9cc16915c5aa71ed50121105d77f4",
    "wl_vocab_taxi_city_vilg_L1_full.json": "49c0043a748e6083147cf5e41e0907a79d109bc5ffcb744af12c86dccbfbcba3",
    "wl_vocab_taxi_city_vilg_L2.json": "bf9336421034851dc37aa99e026068186071544b14db1e3660db87ec712b89ee",
    "wl_vocab_taxi_predictable5_L1.json": "87692c76c7bae44387e05621dac7df29d0e4892f6ed9dbaac4c335cc66add3cb",
    "wl_vocab_taxi_predictable5_vilg_L2.json": "0e0916c11591be09915083b3659c901bbd4cf5d51b2c15ffbec1432338a01ffe",
}


def decode_entries(payload):
    """Independent decode of a vocab file's entries: signature -> id, in file order."""
    vocab = {}
    for entry in payload["entries"]:
        sig = entry["signature"]
        if sig["kind"] == "oov":
            key = OOV_SIGNATURE
        elif sig["kind"] == "init":
            key = ("init", sig["type_id"])
        else:
            key = (sig["own_colour"], tuple((c, l) for c, l in sig["neighbours"]))
        vocab[key] = entry["id"]
    return vocab


def write_vocab(path, **metadata):
    """A minimal valid vocab file (one OOV entry) with exactly the given top-level keys."""
    payload = {"vocab_size": 1}
    payload.update(metadata)
    payload["entries"] = [{"signature": {"kind": "oov"}, "id": 0}]
    with open(path, "w") as f:
        json.dump(payload, f)


class TempVocabDir(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.files = {
            # every shape a standard vocab can take
            "standard_bare": {},
            "standard_meta": {"graph_convention": "vilg", "num_iterations": 1},
            "standard_explicit": {"graph_convention": "vilg", "num_iterations": 1, "domain": "standard"},
            "ternary": {"graph_convention": "vilg", "num_iterations": 1, "domain": "ternary"},
        }
        for name, metadata in self.files.items():
            write_vocab(self.path(name), **metadata)

    def tearDown(self):
        self._tmpdir.cleanup()
        reset_wl_vocab_override()

    def path(self, name):
        return os.path.join(self._tmpdir.name, name + ".json")


def _loaders():
    """name -> callable(path, domain), each going through one real loader."""
    def configure(path, domain):
        configure_wl_vocab_override(path, 1, graph_convention="vilg", domain=domain)
        return get_wl_vocab(domain)

    return {
        "configure_wl_vocab_override": configure,
        "_load_vocab": lambda path, domain: _load_vocab(path, "vilg", 1, domain),
        "load_vocab": lambda path, domain: build_wl_vocab.load_vocab(path, domain=domain),
        "load_vocab_with_metadata": lambda path, domain: build_wl_vocab.load_vocab_with_metadata(path, domain=domain)[0],
    }


class TestEveryLoaderChecksDomain(TempVocabDir):
    def setUp(self):
        super().setUp()
        warnings.simplefilter("ignore", UserWarning)  # metadata-less vilg vocabs warn

    def tearDown(self):
        warnings.resetwarnings()
        super().tearDown()

    def test_ternary_run_rejects_every_standard_vocab(self):
        for loader_name, load in _loaders().items():
            for name in ("standard_bare", "standard_meta", "standard_explicit"):
                with self.subTest(loader=loader_name, vocab=name):
                    with self.assertRaises(ValueError):
                        load(self.path(name), "ternary")

    def test_standard_run_rejects_ternary_vocab(self):
        for loader_name, load in _loaders().items():
            with self.subTest(loader=loader_name):
                with self.assertRaises(ValueError):
                    load(self.path("ternary"), "standard")

    def test_matching_domains_load(self):
        for loader_name, load in _loaders().items():
            for name, domain in (("ternary", "ternary"), ("standard_bare", "standard"),
                                 ("standard_meta", "standard"), ("standard_explicit", "standard")):
                with self.subTest(loader=loader_name, vocab=name):
                    self.assertEqual(load(self.path(name), domain), {OOV_SIGNATURE: 0})
                    reset_wl_vocab_override()

    def test_ternary_vocab_without_full_metadata_raises_everywhere(self):
        write_vocab(self.path("ternary_no_L"), graph_convention="vilg", domain="ternary")
        write_vocab(self.path("ternary_no_conv"), num_iterations=1, domain="ternary")
        write_vocab(self.path("bad_domain"), graph_convention="vilg", num_iterations=1, domain="tenary")
        for loader_name, load in _loaders().items():
            for name in ("ternary_no_L", "ternary_no_conv", "bad_domain"):
                for domain in ("standard", "ternary"):
                    with self.subTest(loader=loader_name, vocab=name, domain=domain):
                        with self.assertRaises(ValueError):
                            load(self.path(name), domain)

    def test_unknown_expected_domain_raises(self):
        for loader_name, load in _loaders().items():
            with self.subTest(loader=loader_name):
                with self.assertRaises(ValueError):
                    load(self.path("standard_bare"), "Ternary")

    def test_ternary_run_must_name_its_convention_and_L_is_checked(self):
        with self.assertRaises(ValueError):
            configure_wl_vocab_override(self.path("ternary"), 1, domain="ternary")
        with self.assertRaises(ValueError):
            _load_vocab(self.path("ternary"), None, 1, "ternary")
        with self.assertRaises(ValueError):
            _load_vocab(self.path("ternary"), "atom", 1, "ternary")
        with self.assertRaises(ValueError):
            _load_vocab(self.path("ternary"), "vilg", 2, "ternary")

    def test_rejected_override_never_takes_effect(self):
        with self.assertRaises(ValueError):
            configure_wl_vocab_override(self.path("ternary"), 1, graph_convention="vilg")
        self.assertFalse(wl_vocab_cache.is_wl_override_configured())

    def test_get_wl_vocab_checks_the_active_domain(self):
        # No override: the default is a standard vocab; there is no ternary default.
        get_wl_vocab()
        with self.assertRaises(ValueError):
            get_wl_vocab("ternary")
        configure_wl_vocab_override(self.path("ternary"), 1, graph_convention="vilg", domain="ternary")
        self.assertEqual(get_wl_vocab("ternary"), {OOV_SIGNATURE: 0})
        with self.assertRaises(ValueError):
            get_wl_vocab()
        reset_wl_vocab_override()
        get_wl_vocab()

    def test_validator_direct(self):
        meta = {"graph_convention": "atom", "num_iterations": 2, "domain": "ternary"}
        validate_wl_vocab_metadata(meta, "atom", 2, "test", "ternary")
        with self.assertRaises(ValueError):
            validate_wl_vocab_metadata(meta, "atom", 2, "test")  # default expected domain: standard
        with self.assertRaises(ValueError):
            validate_wl_vocab_metadata(None, "atom", 2, "test", "ternary")
        with self.assertRaises(ValueError):
            validate_wl_vocab_metadata({"graph_convention": "atom", "num_iterations": 2}, "atom", 2, "test", "ternary")


class TestSaveVocab(TempVocabDir):
    VOCAB = {OOV_SIGNATURE: 0, ("init", 0): 1, (1, ((1, 3),)): 2}

    def test_standard_layout_unchanged(self):
        """A standard save writes exactly the keys, in exactly the order, it wrote before
        the domain field existed -- no "domain" key."""
        p = self.path("saved")
        build_wl_vocab.save_vocab(self.VOCAB, p, "vilg", 1, build_procedure={"seed": 0})
        with open(p) as f:
            raw = f.read()
        expected = json.dumps({
            "vocab_size": 3, "graph_convention": "vilg", "num_iterations": 1,
            "entries": [
                {"signature": {"kind": "oov"}, "id": 0},
                {"signature": {"kind": "init", "type_id": 0}, "id": 1},
                {"signature": {"kind": "refine", "own_colour": 1, "neighbours": [[1, 3]]}, "id": 2},
            ],
            "build_procedure": {"seed": 0},
        })
        self.assertEqual(raw, expected)

    def test_ternary_save_records_domain_and_round_trips(self):
        p = self.path("saved")
        build_wl_vocab.save_vocab(self.VOCAB, p, "atom", 2, domain="ternary")
        with open(p) as f:
            payload = json.load(f)
        self.assertEqual(list(payload), ["vocab_size", "graph_convention", "num_iterations", "domain", "entries"])
        self.assertEqual(payload["domain"], "ternary")
        self.assertEqual(build_wl_vocab.load_vocab(p, domain="ternary"), self.VOCAB)
        self.assertEqual(_load_vocab(p, "atom", 2, "ternary"), self.VOCAB)
        with self.assertRaises(ValueError):
            build_wl_vocab.load_vocab(p)

    def test_unknown_domain_refused(self):
        with self.assertRaises(ValueError):
            build_wl_vocab.save_vocab(self.VOCAB, self.path("saved"), "atom", 2, domain="ternery")


class TestExistingVocabsUnchanged(unittest.TestCase):
    def tearDown(self):
        reset_wl_vocab_override()

    def test_files_byte_identical(self):
        on_disk = sorted(f for f in os.listdir(VOCAB_DIR) if f.startswith("wl_vocab_") and f.endswith(".json")
                         and "ternary" not in f)
        self.assertEqual(on_disk, sorted(EXISTING_VOCAB_SHA256))
        for name, sha in EXISTING_VOCAB_SHA256.items():
            with self.subTest(vocab=name):
                with open(VOCAB_DIR / name, "rb") as f:
                    self.assertEqual(hashlib.sha256(f.read()).hexdigest(), sha)

    def test_every_loader_returns_the_same_entries_in_the_same_order(self):
        for name in sorted(EXISTING_VOCAB_SHA256):
            path = str(VOCAB_DIR / name)
            with open(path) as f:
                payload = json.load(f)
            expected = decode_entries(payload)
            convention = payload.get("graph_convention")
            level = payload.get("num_iterations", 1)
            with self.subTest(vocab=name):
                self.assertNotIn("domain", payload)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    loaded = {
                        "_load_vocab_checked": _load_vocab(path, convention, level),
                        "load_vocab": build_wl_vocab.load_vocab(path),
                    }
                    if convention is None:
                        # the bare call shape (no L) only ever worked for metadata-less files:
                        # L is always checked once metadata is recorded (unchanged behaviour)
                        loaded["_load_vocab"] = _load_vocab(path)
                    vocab, metadata = build_wl_vocab.load_vocab_with_metadata(path)
                    loaded["load_vocab_with_metadata"] = vocab
                    configure_wl_vocab_override(path, level, graph_convention=convention)
                    loaded["get_wl_vocab"] = get_wl_vocab()
                self.assertEqual(metadata, {k: v for k, v in payload.items() if k != "entries"})
                # the metadata the runtime reads keeps its old shape: no "domain" key
                self.assertEqual(
                    wl_vocab_cache.get_wl_vocab_metadata(),
                    None if convention is None else {"graph_convention": convention, "num_iterations": level},
                )
                for loader, got in loaded.items():
                    self.assertEqual(list(got.items()), list(expected.items()), loader)
                reset_wl_vocab_override()

    def test_default_vocab_unchanged(self):
        with open(WL_VOCAB_PATH) as f:
            expected = decode_entries(json.load(f))
        self.assertEqual(list(get_wl_vocab().items()), list(expected.items()))
        self.assertEqual(list(get_wl_vocab("standard").items()), list(expected.items()))


if __name__ == "__main__":
    unittest.main()
