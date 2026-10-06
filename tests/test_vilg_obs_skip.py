"""
Tests for skipping the observation JSON on intermediate plan steps (vilg only):
plan_feedback_a2c.observation_needed, AsyncVecEnv.step's obs_mask, and
GraphTaxiEnv.skip_next_observation.

Every test runs two identically-seeded AsyncVecEnvs through the same plans, exactly the
way PlanFeedback_A2C.collect_rollouts steps them: a control (obs_mask=None, every
observation built - the behaviour before this change) and one given
observation_needed(...) as obs_mask. Whatever the policy can read afterwards (the
observation buffer after each plan, terminal_observation, the reset observation) must be
identical strings.

Run from the repo root with:
    python -m pytest tests/test_vilg_obs_skip.py -v
"""
import unittest
from copy import deepcopy

import numpy as np

# --- numpy/gym compat shim for constructing a "city" env on the Mac's newer gym/numpy;
# same pattern as tests/test_atom_wiring.py. Production code does not depend on it.
import gym.utils.seeding as _seeding

if not hasattr(np.random.Generator, "randint"):
    class _RandintCompatGenerator:
        def __init__(self, generator):
            self._generator = generator

        def randint(self, low, high=None):
            if hasattr(self._generator, "randint"):
                return self._generator.randint(low, high) if high is not None else self._generator.randint(low)
            return self._generator.integers(low, high)

        def __getattr__(self, name):
            return getattr(self._generator, name)

    _original_np_random = _seeding.np_random

    def _np_random_with_randint(seed=None):
        generator, seed = _original_np_random(seed)
        return _RandintCompatGenerator(generator), seed

    _seeding.np_random = _np_random_with_randint

from sage.agent.async_vec_env import AsyncVecEnv
from sage.agent.plan_feedback_a2c import observation_needed
from sage.domains.gym_taxi import REWARDS
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.simulator.taxi_world import TaxiWorldSimulator
from sage.domains.gym_taxi.utils.wl_vocab_cache import configure_wl_vocab_override, reset_wl_vocab_override, WL_VOCAB_PATH

VILG_VOCAB = WL_VOCAB_PATH.parent / "wl_vocab_taxi_city_vilg_L2.json"
ATOM_VOCAB = WL_VOCAB_PATH.parent / "wl_vocab_taxi_city_atom_L1_full.json"
SEEDS = [0, 300, 600]


def make_vec_env(convention, seeds=SEEDS):
    def _make(seed):
        def _f():
            env = GraphTaxiEnv(representation="graph", scenario="city", mask=False,
                               rewards=REWARDS["v1"], graph_convention=convention)
            env.seed(seed)
            return env
        return _f
    vec_env = AsyncVecEnv([_make(s) for s in seeds])
    return vec_env, vec_env.reset()


def plans_for(vec_env, obs, rng, passenger_prob=0.5):
    """One plan per env from the env's own planner, to a passenger (pickup + delivery)
    or a location goal - real multi-step plans of different lengths across envs."""
    space = vec_env.envs[0].observation_space
    plans = []
    for d in space.converter(obs).to_data_list():
        selectable = d.mask.bool()
        passengers = np.nonzero((selectable & (d.x[:, 2] == 1)).numpy())[0].tolist()
        locations = np.nonzero((selectable & (d.x[:, 0] == 1)).numpy())[0].tolist()
        pool = passengers if (passengers and rng.rand() < passenger_prob) else locations
        _, actions = space.planner.plan(deepcopy(d), int(pool[rng.randint(len(pool))]))
        plans.append([int(a) for a in actions])
    return plans


def run_plans(vec_env, plans, use_obs_mask):
    """The plan loop of PlanFeedback_A2C.collect_rollouts, verbatim apart from recording."""
    plan_lengths = np.array([len(p) for p in plans])
    max_plan_length = max(plan_lengths)
    env_finished = plan_lengths == 0
    grid = -np.ones((len(plans), max_plan_length), dtype=int)
    for i, p in enumerate(plans):
        grid[i, :plan_lengths[i]] = p
    plan_step = 0
    terminals = []
    new_obs = None
    while not env_finished.all() and plan_step < max_plan_length:
        actions = grid[:, plan_step]
        env_finished = np.logical_or(env_finished, actions == -1)
        obs_mask = observation_needed(plan_step, plan_lengths) if use_obs_mask else None
        new_obs, rewards, dones, infos = vec_env.step(actions, np.logical_not(env_finished), obs_mask)
        for i in np.nonzero(np.logical_and(dones, np.logical_not(env_finished)))[0]:
            terminals.append((int(i), plan_step, infos[i]["terminal_observation"], str(new_obs[i][0])))
        env_finished = np.logical_or(env_finished, dones)
        plan_step += 1
    return new_obs, terminals


class CountJsonBuilds:
    def __enter__(self):
        self.count = 0
        self._orig = TaxiWorldSimulator._get_state_json

        def counted(sim):
            self.count += 1
            return self._orig(sim)
        TaxiWorldSimulator._get_state_json = counted
        return self

    def __exit__(self, *exc):
        TaxiWorldSimulator._get_state_json = self._orig


class TestObservationNeeded(unittest.TestCase):
    def test_true_exactly_at_each_envs_last_plan_step(self):
        plan_lengths = np.array([3, 1, 0, 5])
        expected = {
            0: [False, True, False, False],
            1: [False, False, False, False],
            2: [True, False, False, False],
            4: [False, False, False, True],
        }
        for step, row in expected.items():
            np.testing.assert_array_equal(observation_needed(step, plan_lengths), row)

    def test_accepts_plain_list(self):
        np.testing.assert_array_equal(observation_needed(0, [1, 2]), [True, False])


class _ConventionCase(unittest.TestCase):
    convention = None
    vocab = None
    num_iterations = None

    @classmethod
    def setUpClass(cls):
        if cls.vocab is not None:
            configure_wl_vocab_override(cls.vocab, cls.num_iterations)
        else:
            reset_wl_vocab_override()

    @classmethod
    def tearDownClass(cls):
        reset_wl_vocab_override()

    def run_both(self, decisions, shorten_timeout_at=None):
        """Runs control and obs_mask vec envs side by side; returns
        (control buffers, masked buffers, control terminals, masked terminals, builds)."""
        control, control_obs = make_vec_env(self.convention)
        masked, masked_obs = make_vec_env(self.convention)
        self.assertEqual([str(o[0]) for o in control_obs], [str(o[0]) for o in masked_obs])
        rng_c, rng_m = np.random.RandomState(7), np.random.RandomState(7)
        out = {"control": [], "masked": [], "t_control": [], "t_masked": [], "builds_control": 0, "builds_masked": 0}
        for dec in range(decisions):
            plans_c = plans_for(control, control_obs, rng_c)
            plans_m = plans_for(masked, masked_obs, rng_m)
            self.assertEqual(plans_c, plans_m)
            if shorten_timeout_at is not None and dec == shorten_timeout_at:
                # make env 0's episode time out in the middle of its plan
                for venv in (control, masked):
                    env = venv.envs[0].unwrapped
                    env.sim.timeout = env.steps + 2
                self.assertGreater(len(plans_c[0]), 2)
            with CountJsonBuilds() as c:
                control_obs, t_c = run_plans(control, plans_c, use_obs_mask=False)
            out["builds_control"] += c.count
            with CountJsonBuilds() as c:
                masked_obs, t_m = run_plans(masked, plans_m, use_obs_mask=True)
            out["builds_masked"] += c.count
            out["control"].append([str(o[0]) for o in control_obs])
            out["masked"].append([str(o[0]) for o in masked_obs])
            out["t_control"] += t_c
            out["t_masked"] += t_m
        return out


class TestVilgSkipsIntermediateObservations(_ConventionCase):
    convention = "vilg"
    vocab = VILG_VOCAB
    num_iterations = 2

    def test_buffers_identical_and_fewer_builds(self):
        out = self.run_both(decisions=4)
        self.assertEqual(out["control"], out["masked"])
        # one observation per env per decision instead of one per simulator step
        self.assertEqual(out["builds_masked"], 4 * len(SEEDS))
        self.assertGreater(out["builds_control"], 3 * out["builds_masked"])

    def test_episode_ending_mid_plan(self):
        out = self.run_both(decisions=2, shorten_timeout_at=1)
        self.assertEqual(len(out["t_masked"]), 1)
        (env_c, step_c, term_c, reset_c), = out["t_control"]
        (env_m, step_m, term_m, reset_m), = out["t_masked"]
        self.assertEqual((env_c, step_c), (env_m, step_m))
        self.assertEqual(env_m, 0)
        self.assertEqual(step_m, 1)  # timeout on the 2nd step of a longer plan
        self.assertIsInstance(term_m, str)
        self.assertEqual(term_c, term_m)
        self.assertEqual(reset_c, reset_m)
        self.assertEqual(out["control"], out["masked"])

    def test_skip_request_is_one_shot(self):
        vec_env, _ = make_vec_env("vilg", seeds=[0])
        env = vec_env.envs[0]
        self.assertTrue(env.skip_next_observation())
        obs, _, done, info = env.step(env.sim.taxi.location)
        self.assertIsNone(obs)
        self.assertIsNone(info["s_true"])
        self.assertFalse(done)
        obs, _, _, info = env.step(env.sim.taxi.location)
        self.assertIsInstance(obs, str)
        self.assertEqual(info["s_true"], obs)

    def test_reset_clears_pending_skip(self):
        vec_env, _ = make_vec_env("vilg", seeds=[0])
        env = vec_env.envs[0]
        env.skip_next_observation()
        env.reset()
        obs, _, _, _ = env.step(env.sim.taxi.location)
        self.assertIsInstance(obs, str)


class TestDefaultBuildsEveryObservation(_ConventionCase):
    """obs_mask=None (every caller except collect_rollouts) keeps building every step."""
    convention = "vilg"
    vocab = VILG_VOCAB
    num_iterations = 2

    def test_vec_env_default(self):
        vec_env, obs = make_vec_env("vilg")
        plans = plans_for(vec_env, obs, np.random.RandomState(0))
        with CountJsonBuilds() as c:
            run_plans(vec_env, plans, use_obs_mask=False)
        self.assertEqual(c.count, sum(len(p) for p in plans))

    def test_env_and_sim_default(self):
        vec_env, _ = make_vec_env("vilg", seeds=[0])
        env = vec_env.envs[0]
        obs, _, _, info = env.step(env.sim.taxi.location)
        self.assertIsInstance(obs, str)
        self.assertEqual(info["s_true"], obs)
        obs, _, _, _ = env.sim.act(env.sim.taxi.location)
        self.assertIsInstance(obs, str)


class TestOracleSageIgnoresObsMask(_ConventionCase):
    convention = "oracle_sage"

    def test_identical_and_every_observation_built(self):
        out = self.run_both(decisions=3)
        self.assertEqual(out["control"], out["masked"])
        self.assertEqual(out["builds_control"], out["builds_masked"])

    def test_declines_skip(self):
        vec_env, _ = make_vec_env("oracle_sage", seeds=[0])
        self.assertFalse(vec_env.envs[0].skip_next_observation())


class TestAtomIgnoresObsMask(_ConventionCase):
    convention = "atom"
    vocab = ATOM_VOCAB
    num_iterations = 1

    def test_identical_and_every_observation_built(self):
        out = self.run_both(decisions=3)
        self.assertEqual(out["control"], out["masked"])
        self.assertEqual(out["builds_control"], out["builds_masked"])

    def test_declines_skip(self):
        vec_env, _ = make_vec_env("atom", seeds=[0])
        self.assertFalse(vec_env.envs[0].skip_next_observation())


if __name__ == "__main__":
    unittest.main()
