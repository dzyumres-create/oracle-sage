"""
Tests for Task C: periodic mid-training checkpointing
(sage/agent/checkpoint_callback.py's PeriodicCheckpointCallback, wired into
sage/experiments/gnn_global.py via --checkpoint-interval).

Constraint: must not change training behaviour (no RNG draws, no change to call order,
no env access). Full-training-run log comparison (--checkpoint-interval 0 vs 160,
same seed) was attempted as originally specified, but found to be blocked by a
PRE-EXISTING, unrelated gap: gnn_global.py's --seed is threaded to make_vec_env's env
RNG only, never to the algorithm constructor, so PlanFeedback_A2C.set_random_seed(None)
is a no-op and the policy's random weight init (hence the entire trajectory) differs
run-to-run regardless of --checkpoint-interval or any other flag -- confirmed directly:
two runs with IDENTICAL settings (both --checkpoint-interval 0) diverged just as much as
the 0-vs-160 comparison did. Per the user's decision, this gap is left alone (out of
Task C's scope); TestCallbackConsumesNoRandomness below is the agreed rigorous
substitute -- it isolates the callback itself (not the whole pipeline) and proves it
consumes zero randomness directly, independent of the pipeline's own pre-existing
non-reproducibility elsewhere.

Loading a checkpoint back (PlanFeedback_A2C.load(path)) is currently broken for every
graph_convention -- see the comment above this file's `if __name__ ==` block for the
full diagnosis (JsonGraph doesn't survive pickling, and separately this SB3 fork's
load() doesn't forward custom_objects to where it could work around that). Both are
pre-existing gaps outside Task C's scope, so there is no load-and-forward test here.

Run from the repo root with:
    python -m pytest tests/test_checkpoint_callback.py -v
"""
import os
import tempfile
import unittest

import numpy as np
import torch as th

from sage.agent.checkpoint_callback import PeriodicCheckpointCallback

# --- numpy/gym compat shim for constructing a "city" env in this sandbox's drifted
# gym/numpy -- see tests/test_atom_wiring.py for the identical pattern/rationale.
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


class _FakeModel:
    """Stands in for BaseAlgorithm: records save() calls, writes a real (empty) .zip
    file so glob-based pruning logic can be tested without a real (slow) model."""

    def __init__(self):
        self.saved_paths = []
        self.num_timesteps = 0

    def save(self, path):
        self.saved_paths.append(path)
        open(path + ".zip", "w").close()


class TestPeriodicCheckpointCallback(unittest.TestCase):
    def _make(self, save_dir, interval=160, keep=2):
        cb = PeriodicCheckpointCallback(save_dir, interval=interval, keep=keep)
        cb.model = _FakeModel()
        return cb

    def test_saves_at_expected_crossings_non_divisor_step_size(self):
        # num_timesteps advances by 13 (does not evenly divide 160) -- exercises the
        # "crosses a multiple", not "equals a multiple", requirement directly.
        with tempfile.TemporaryDirectory() as d:
            cb = self._make(d, interval=160, keep=2)
            for i in range(1, 30):
                cb.model.num_timesteps = i * 13
                cb.num_timesteps = cb.model.num_timesteps
                cb._on_step()
            # first crossing of 160 is at 13*13=169; next crossing (320) at 13*25=325
            self.assertEqual(cb.model.saved_paths, [os.path.join(d, "checkpoint_169"), os.path.join(d, "checkpoint_325")])

    def test_keeps_only_the_n_most_recent_and_final_model_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "final_model.zip"), "w").close()
            cb = self._make(d, interval=160, keep=2)
            for i in range(1, 100):
                cb.model.num_timesteps = i * 13
                cb.num_timesteps = cb.model.num_timesteps
                cb._on_step()

            files = sorted(os.listdir(d))
            checkpoints = [f for f in files if f.startswith("checkpoint_")]
            self.assertEqual(len(checkpoints), 2)
            self.assertIn("final_model.zip", files)

    def test_creates_save_dir_if_missing(self):
        with tempfile.TemporaryDirectory() as parent:
            missing_dir = os.path.join(parent, "does", "not", "exist", "yet")
            self.assertFalse(os.path.exists(missing_dir))
            cb = self._make(missing_dir, interval=160)
            cb.model.num_timesteps = 200
            cb.num_timesteps = 200
            cb._on_step()
            self.assertTrue(os.path.isdir(missing_dir))
            self.assertTrue(os.path.exists(os.path.join(missing_dir, "checkpoint_200.zip")))

    def test_no_save_before_first_crossing(self):
        with tempfile.TemporaryDirectory() as d:
            cb = self._make(d, interval=160)
            for step in [10, 50, 100, 159]:
                cb.model.num_timesteps = step
                cb.num_timesteps = step
                cb._on_step()
            self.assertEqual(cb.model.saved_paths, [])

    def test_interval_must_be_positive(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(AssertionError):
                PeriodicCheckpointCallback(d, interval=0)


class TestCallbackConsumesNoRandomness(unittest.TestCase):
    """The agreed substitute for the (blocked, see module docstring) full-training-run
    log comparison: proves the callback itself never draws from the torch or numpy RNG,
    directly and in isolation -- including through a REAL model.save() I/O call, not a
    fake one, so this covers the actual save path Task C's callback exercises."""

    def test_rng_state_unchanged_across_a_real_save(self):
        from sage.agent.graph_plan_feedback_policy import GNNPlanFeedbackPolicy
        from sage.agent.plan_feedback_a2c import PlanFeedback_A2C
        from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
        from sage.domains.gym_taxi import REWARDS
        from sage.agent.async_vec_env import AsyncVecEnv
        from sage.forks.stable_baselines3.stable_baselines3.common.env_util import make_vec_env
        from sage.domains.utils import spaces as sage_spaces
        import gym as gym_module

        def make_env():
            return GraphTaxiEnv(representation="graph", scenario="predictable5", mask=False, rewards=REWARDS["v1"], graph_convention="atom")

        env = make_vec_env(make_env, n_envs=1, seed=0, monitor_kwargs={"info_keywords": ("len100", "len200")}, vec_env_cls=AsyncVecEnv)
        policy_kwargs = {
            "optimizer_class": th.optim.AdamW, "optimizer_kwargs": {"weight_decay": 0.0001},
            "ortho_init": False, "exploration_initial_eps": 0.0, "exploration_final_eps": 0.0,
            "exploration_fraction": 0.0, "shared_gnn": True, "layer_norm": False,
            "num_planning_choices": 1, "features_extractor_kwargs": {"gnn_steps": 2},
        }
        model = PlanFeedback_A2C(
            GNNPlanFeedbackPolicy, env, verbose=0,
            supported_action_spaces=(sage_spaces.BinaryAction, gym_module.spaces.Discrete, sage_spaces.Autoregressive),
            n_steps=1, policy_kwargs=policy_kwargs,
        )

        with tempfile.TemporaryDirectory() as d:
            cb = PeriodicCheckpointCallback(d, interval=1)
            cb.model = model
            cb.num_timesteps = 1

            th_state_before = th.get_rng_state()
            np_state_before = np.random.get_state()

            cb._on_step()  # triggers a REAL model.save()

            th_state_after = th.get_rng_state()
            np_state_after = np.random.get_state()

            self.assertTrue(th.equal(th_state_before, th_state_after))
            for a, b in zip(np_state_before[1:], np_state_after[1:]):
                if isinstance(a, np.ndarray):
                    self.assertTrue(np.array_equal(a, b))
                else:
                    self.assertEqual(a, b)

            self.assertTrue(os.path.exists(os.path.join(d, "checkpoint_1.zip")))

        env.close()


# NOT FIXED HERE -- checkpoint *loading* is currently broken for every graph_convention
# (oracle_sage, vilg, atom alike), by two separate, pre-existing gaps outside Task C's
# scope. There is deliberately no test in this file that loads a checkpoint back;
# see checkpoint_callback.py's module docstring for the full writeup, and the diff
# report ("report but do not fix") for what a proper fix of each would involve.
#
#   1. JsonGraph (sage/domains/utils/spaces.py) does not survive a pickle round trip: it
#      subclasses gym.spaces.Box but never calls Box.__init__(), so it never sets
#      .low/.high. Newer gym's Box.__setstate__ unconditionally does
#      self.low_repr = _short_repr(self.low), which raises
#      AttributeError: 'JsonGraph' object has no attribute 'low'. observation_space is a
#      JsonGraph under all three conventions, so this hits all of them identically --
#      confirmed directly for both graph_convention="oracle_sage" and "atom" (identical
#      exception, same attribute, same call site). action_space (a plain
#      gym.spaces.Discrete for all three conventions) is NOT itself affected in isolation.
#   2. Separately, this vendored SB3 fork's BaseAlgorithm.load() (base_class.py:584) does
#      not forward a custom_objects argument to load_from_zip_file() -- and
#      load_from_zip_file() (save_util.py:359) does not forward one to json_to_data()
#      either. json_to_data() itself DOES already support custom_objects (it substitutes
#      a caller-supplied value for a named top-level key before ever attempting to
#      cloudpickle-deserialise it), but that support is unreachable through .load() as
#      currently written: PlanFeedback_A2C.load(f, custom_objects={...}) still raises the
#      exact same JsonGraph AttributeError, because custom_objects silently becomes a
#      stray kwarg merged into model.__dict__ post-hoc instead of ever reaching
#      json_to_data. Confirmed directly by calling json_to_data() by hand with the same
#      custom_objects dict -- it skips observation_space/action_space correctly and
#      succeeds -- versus through PlanFeedback_A2C.load(), which does not.
#
# Bug (2) is likely also why the RCP failure names "action_space" rather than
# "observation_space": save_util.py's json_to_data() has a real, separate loop-variable-
# scoping issue in its `except RuntimeError:` branch -- `deserialized_object` is not
# reset before the unconditional `return_data[data_key] = deserialized_object` line right
# after the try/except, so on Python (no block scoping) a value left over from a
# previously-processed dict key can leak into the currently-failing key's slot. This
# wasn't independently re-confirmed against RCP's exact pinned versions, but it is the
# most plausible explanation on hand for why the RCP warning names the key that ISN'T
# the actual JsonGraph offender.
#
# Fixing either would mean editing sage/domains/utils/spaces.py or the SB3 fork under
# sage/forks/stable_baselines3/ -- both explicitly out of scope for this task.


if __name__ == "__main__":
    unittest.main()
