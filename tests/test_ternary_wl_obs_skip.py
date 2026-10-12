"""
The vilg observation skip (20b3c99) must decline for every ternary convention.

TernaryTaxiWorldSimulator.act has no build_obs argument, so honouring the skip there raised
TypeError on the first intermediate plan step of any ternary vilg run. With the skip
declined, a ternary observation stream driven through AsyncVecEnv with collect_rollouts'
obs_mask must be byte-identical to the same stream on ternary-gnn @ e84fb5d, where
AsyncVecEnv had no obs_mask and every observation was built.

Ground truth: TERNARY_OBS_STREAM_SHA256, recorded by running
analysis/ternary_obs_stream_hash.py against a checkout of e84fb5d (a different commit, with
no observation skip at all), keyed by stack. The standard-domain vilg skip is still
covered by tests/test_vilg_obs_skip.py.
"""
import unittest

import gym as gym_module

from analysis.ternary_obs_stream_hash import (
    CONVENTIONS,
    install_randint_shim,
    stack_key,
    ternary_obs_stream_hash,
)

# sha256 of analysis/ternary_obs_stream_hash.py's stream, computed at ternary-gnn @ e84fb5d.
# The three conventions share one value: the ternary wire format (facts JSON) is the same
# for every convention, and the goals are object rows, which every convention numbers the
# same way. Still checked per convention, since the skip decision is per convention.
#   gym0.26: Mac (gym 0.26.2, numpy 2.2, networkx 3.4).
TERNARY_OBS_STREAM_SHA256 = {
    "gym0.26": {
        "oracle_sage": "48eea4957f30e57f3b90cec1f509f2c519634fed361f53cd36f615532eb25d69",
        "vilg": "48eea4957f30e57f3b90cec1f509f2c519634fed361f53cd36f615532eb25d69",
        "atom": "48eea4957f30e57f3b90cec1f509f2c519634fed361f53cd36f615532eb25d69",
    },
    "gym0.18": {
        "oracle_sage": "48eea4957f30e57f3b90cec1f509f2c519634fed361f53cd36f615532eb25d69",
        "vilg": "48eea4957f30e57f3b90cec1f509f2c519634fed361f53cd36f615532eb25d69",
        "atom": "48eea4957f30e57f3b90cec1f509f2c519634fed361f53cd36f615532eb25d69",
    },
}

REFERENCE_COMMAND = (
    "on this stack, in a checkout of ternary-gnn @ e84fb5d, run "
    "`git show ternary-wl:analysis/ternary_obs_stream_hash.py > /tmp/ternary_obs_stream_hash.py && "
    "for c in oracle_sage vilg atom; do PYTHONPATH=. python /tmp/ternary_obs_stream_hash.py $c; done` "
    "and add the printed hashes to TERNARY_OBS_STREAM_SHA256[{stack!r}] in tests/test_ternary_wl_obs_skip.py"
)


def _make_env(convention, ternary):
    from sage.domains.gym_taxi import REWARDS
    from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv

    install_randint_shim()
    scenario = "city_ternary" if ternary else "city"
    env = GraphTaxiEnv(representation="graph", scenario=scenario, mask=False, rewards=REWARDS["v1"],
                       graph_convention=convention, ternary=ternary)
    env.seed(0)
    env.reset()
    return env


class TestTernaryDeclinesSkip(unittest.TestCase):
    def test_ternary_declines_for_every_convention(self):
        for convention in CONVENTIONS:
            with self.subTest(convention=convention):
                env = _make_env(convention, ternary=True)
                self.assertFalse(env.skip_next_observation())
                obs, _reward, _done, _info = env.step(env.sim.taxi.location)  # no-op move
                self.assertIsInstance(obs, str)
                self.assertTrue(obs.startswith("{"))

    def test_standard_vilg_still_skips(self):
        env = _make_env("vilg", ternary=False)
        self.assertTrue(env.skip_next_observation())
        obs, _reward, done, _info = env.step(env.sim.taxi.location)  # no-op move
        self.assertFalse(done)
        self.assertIsNone(obs)


class TestTernaryObservationStreamUnchanged(unittest.TestCase):
    def test_stream_matches_e84fb5d_reference(self):
        stack = stack_key()
        references = TERNARY_OBS_STREAM_SHA256.get(stack, {})
        missing = [c for c in CONVENTIONS if c not in references]
        if missing:
            self.fail(
                f"no ternary observation-stream reference for stack {stack!r} (gym {gym_module.__version__}), "
                f"conventions {missing}: " + REFERENCE_COMMAND.format(stack=stack)
            )
        for convention in CONVENTIONS:
            with self.subTest(convention=convention):
                digest, stats = ternary_obs_stream_hash(convention, require_obs_mask=True)
                # the run really did ask some envs to skip, and really ended episodes
                self.assertTrue(stats["supports_obs_mask"])
                self.assertGreater(stats["obs_mask_false"], 0)
                self.assertGreater(stats["terminal_observations"], 0)
                self.assertEqual(digest, references[convention])


if __name__ == "__main__":
    unittest.main()
