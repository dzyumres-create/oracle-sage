"""
Tests for the chunked training update (Feedback_A2C.update_chunks / --update-chunks).

Every update below starts from the same snapshot: one real rollout on the ternary atom
env (policy weights, fresh optimizer state, rollout buffer, torch/numpy RNG states taken
at the moment train() would run). For each variant we record the gradients at the
optimizer step (after clipping), the parameters after it, every logged value, and the
RNG states afterwards.

Run from the repo root with:
    PYTHONPATH=. python -m pytest tests/test_chunked_update.py -v -s
Do not run while a training run is using the GPU.
"""
import contextlib
import copy
import os
import subprocess
import unittest
from unittest import mock

import gym as gym_module
import numpy as np
import torch as th
from torch.nn import functional as F

# test_ternary_planner installs the gym-0.26 randint shim at import time.
from tests.test_ternary_planner import POLICY_KWARGS

import sage.experiments.gnn_global as gnn_global
from sage.agent.async_vec_env import AsyncVecEnv
from sage.agent.feedback_a2c import Feedback_A2C
from sage.agent.graph_plan_feedback_policy import GNNPlanFeedbackPolicy
from sage.agent.plan_feedback_a2c import PlanFeedback_A2C
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi.utils.compat import spec_kwargs
from sage.domains.utils import spaces as sage_spaces
from sage.domains.utils.spaces import Autoregressive
from sage.forks.stable_baselines3.stable_baselines3.common import logger
from sage.forks.stable_baselines3.stable_baselines3.common.env_util import make_vec_env
from sage.forks.stable_baselines3.stable_baselines3.common.utils import explained_variance
from gym import spaces

TERNARY_ENV_ID = "city-taxi-ternary-unmasked-v1"
CUDA = th.cuda.is_available()


# ==========================================================================================
# Feedback_A2C.train() exactly as it was before update_chunks existed (commit 219720c),
# copied verbatim, as the reference the update_chunks=1 path must reproduce bit for bit.
# ==========================================================================================

def reference_train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):

            actions = rollout_data.actions
            if (isinstance(self.action_space, spaces.Discrete) or
               isinstance(self.action_space, Autoregressive) ):
                # Convert discrete action from float to long
                actions = actions.long().flatten()


            values, log_prob, entropy, path_values = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            policy_loss = -(advantages * log_prob)

            if self.tis_heuristic is not None:
                probs = th.exp(log_prob.detach())
                policy_loss *= (probs*self.tis_heuristic).clamp(0,1) #clamp is the truncated in truncated importance sampling

            policy_loss = policy_loss.mean()


            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns*(1-rollout_data.explored), values*(1-rollout_data.explored))

            # Entropy loss favor exploration
            if entropy is None or self.sample_entropy:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            if self.pvf_coef == 0:
                loss = self.policy_coef*policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
            else:
                path_values = path_values.flatten()
                path_value_loss = F.mse_loss(rollout_data.returns, path_values)
                loss = self.policy_coef*policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.pvf_coef * path_value_loss

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/explained_variance", explained_var)
        logger.record("train/entropy_loss", entropy_loss.item())
        logger.record("train/entropy", entropy.item())
        logger.record("train/policy_loss", policy_loss.item())
        logger.record("train/value_loss", value_loss.item())
        if self.pvf_coef != 0:
            logger.record("train/path_value_loss", path_value_loss.item())
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", th.exp(self.policy.log_std).mean().item())


# ==========================================================================================
# Shared fixture: one real rollout, snapshotted at the moment train() would run
# ==========================================================================================

def copy_buffer(buffer):
    """Deep copy of the rollout buffer's data, sharing its observation/action spaces:
    JsonGraph can't be deep-copied (it subclasses gym's Box without Box.__init__, the
    same defect that breaks model reloading), and the spaces are never mutated."""
    shared = {id(buffer.observation_space): buffer.observation_space, id(buffer.action_space): buffer.action_space}
    return copy.deepcopy(buffer, shared)


def build_model_with_rollout(n_envs, n_steps, device="cpu", convention="atom"):
    env = make_vec_env(
        lambda: GraphTaxiEnv(**spec_kwargs(gym_module.spec(TERNARY_ENV_ID)), graph_convention=convention),
        n_envs=n_envs, seed=0, monitor_kwargs={"info_keywords": gnn_global.info_keywords_for(TERNARY_ENV_ID)},
        vec_env_cls=AsyncVecEnv,
    )
    model = PlanFeedback_A2C(
        GNNPlanFeedbackPolicy, env, verbose=0, device=device,
        supported_action_spaces=(sage_spaces.BinaryAction, gym_module.spaces.Discrete, sage_spaces.Autoregressive),
        n_steps=n_steps, policy_kwargs=dict(POLICY_KWARGS, exploration_fraction=0.1),
    )
    snapshot = {}

    def capture():
        snapshot["buffer"] = copy_buffer(model.rollout_buffer)
        snapshot["policy"] = copy.deepcopy(model.policy.state_dict())
        # .grad too: path_value_net is not in the optimizer, so zero_grad() never clears its
        # gradient, which accumulates across updates and enters clip_grad_norm_'s global
        # norm. Every update must start from the same leftover gradients to be comparable.
        snapshot["grads"] = {name: None if p.grad is None else p.grad.detach().clone()
                             for name, p in model.policy.named_parameters()}
        snapshot["optimizer"] = copy.deepcopy(model.policy.optimizer.state_dict())
        snapshot["progress"] = model._current_progress_remaining
        snapshot["n_updates"] = model._n_updates
        snapshot["torch_rng"] = th.get_rng_state()
        snapshot["cuda_rng"] = th.cuda.get_rng_state_all() if CUDA else None
        snapshot["np_rng"] = np.random.get_state()

    with mock.patch.object(model, "train", side_effect=capture):
        model.learn(total_timesteps=n_envs * n_steps)  # exactly one rollout, then "train"
    assert snapshot, "train() was never reached"
    return env, model, snapshot


def run_update(model, snapshot, update_chunks, reference=False):
    """One training update from the snapshot. Returns the clipped gradients at the
    optimizer step, the parameters after it, the logged values, and the RNG states."""
    model.policy.load_state_dict(snapshot["policy"])
    for name, p in model.policy.named_parameters():
        saved = snapshot["grads"][name]
        p.grad = None if saved is None else saved.clone()
    model.policy.optimizer.load_state_dict(snapshot["optimizer"])
    model.rollout_buffer = copy_buffer(snapshot["buffer"])
    model._current_progress_remaining = snapshot["progress"]
    model._n_updates = snapshot["n_updates"]
    th.set_rng_state(snapshot["torch_rng"])
    if snapshot["cuda_rng"] is not None:
        th.cuda.set_rng_state_all(snapshot["cuda_rng"])
    np.random.set_state(snapshot["np_rng"])
    model.update_chunks = update_chunks

    grads = {}
    real_step = model.policy.optimizer.step

    def step_spy(*args, **kwargs):
        for name, p in model.policy.named_parameters():
            if p.grad is not None:
                grads[name] = p.grad.detach().clone()
        return real_step(*args, **kwargs)

    logged = {}
    real_record = logger.record

    def record_spy(key, value, *args, **kwargs):
        logged[key] = value
        return real_record(key, value, *args, **kwargs)

    with mock.patch.object(model.policy.optimizer, "step", side_effect=step_spy), \
            mock.patch.object(logger, "record", side_effect=record_spy):
        if reference:
            reference_train(model)
        else:
            model.train()
    return {
        "grads": grads,
        "params": {name: p.detach().clone() for name, p in model.policy.named_parameters()},
        "logged": logged,
        "torch_rng": th.get_rng_state(),
        "np_rng": np.random.get_state(),
    }


def max_abs(a, b):
    return max((a[k] - b[k]).abs().max().item() for k in a)


def whole_grad_rel(a, b):
    """||g_a - g_b|| / ||g_a|| over the whole gradient vector. Per-tensor relative error
    is meaningless here: several tensors have an analytically ZERO gradient (the
    GlobalAttention gate biases and action_net.bias -- a constant added before a softmax
    cancels), so their computed gradient is pure rounding noise (~1e-8)."""
    total = sum(g.double().norm().item() ** 2 for g in a.values()) ** 0.5
    diff = sum((a[k] - b[k]).double().norm().item() ** 2 for k in a) ** 0.5
    return diff / total


def max_rel_logged(a, b):
    return max(abs(a[k] - b[k]) / max(abs(a[k]), 1e-12) for k in a)


def worst_logged(a, b, rtol, atol):
    """The logged key closest to failing |a - b| <= atol + rtol * |b|, as
    (key, |a - b|, atol + rtol * |b|)."""
    rows = [(k, abs(a[k] - b[k]), atol + rtol * abs(b[k])) for k in a]
    return max(rows, key=lambda row: row[1] / row[2])


def to_float64(model, snapshot):
    """Switches the update to float64 end to end: parameters (and the snapshot's copy of
    them), a fresh optimizer over the double parameters, the features extractor's inputs
    (both observations and projections pass through it), and the buffer's targets."""
    model.policy.double()
    snapshot["policy"] = {k: (v.double() if v.is_floating_point() else v) for k, v in snapshot["policy"].items()}
    snapshot["grads"] = {k: (None if g is None else g.double()) for k, g in snapshot["grads"].items()}
    extractor = model.policy.features_extractor
    real_forward = extractor.forward

    def forward64(batch):
        batch.x = batch.x.double()
        batch.edge_attr = batch.edge_attr.double()
        batch.global_features = batch.global_features.double()
        return real_forward(batch)

    extractor.forward = forward64
    buffer = snapshot["buffer"]
    for name in ("advantages", "returns", "values", "explored", "log_probs"):
        setattr(buffer, name, getattr(buffer, name).astype(np.float64))
    model.policy.optimizer = model.policy.optimizer_class(
        model.policy.parameters(), lr=model.lr_schedule(1), **model.policy.optimizer_kwargs
    )
    snapshot["optimizer"] = model.policy.optimizer.state_dict()


def assert_same_rng(test, a, b):
    test.assertTrue(th.equal(a["torch_rng"], b["torch_rng"]), "torch RNG state differs after the update")
    for x, y in zip(a["np_rng"], b["np_rng"]):
        if isinstance(x, np.ndarray):
            test.assertTrue(np.array_equal(x, y), "numpy RNG state differs after the update")
        else:
            test.assertEqual(x, y, "numpy RNG state differs after the update")


def assert_bitwise_equal(test, a, b, what):
    """Two run_update results: same gradients, parameters, logged values and RNG states,
    bit for bit."""
    test.assertEqual(set(a["grads"]), set(b["grads"]), what)
    for name in a["grads"]:
        test.assertTrue(th.equal(a["grads"][name], b["grads"][name]), f"{what}: grad {name}")
    for name in a["params"]:
        test.assertTrue(th.equal(a["params"][name], b["params"][name]), f"{what}: param {name}")
    test.assertEqual(a["logged"], b["logged"], what)
    assert_same_rng(test, a, b)


@contextlib.contextmanager
def single_cpu_thread():
    """With several CPU threads, some reduction kernels sum in a run-dependent order for
    some tensor sizes, so the same update isn't bitwise reproducible run to run (seen
    on the Mac for chunked updates, and on RCP -- torch 1.7.1, 8 cores -- for the
    unchunked one). One thread makes the order fixed."""
    threads = th.get_num_threads()
    th.set_num_threads(1)
    try:
        yield
    finally:
        th.set_num_threads(threads)


# ==========================================================================================
# Default unchanged: update_chunks=1 is the old train(), bit for bit (any stack, no skip)
# ==========================================================================================

class TestDefaultUnchanged(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env, cls.model, cls.snapshot = build_model_with_rollout(n_envs=4, n_steps=5)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_update_chunks_1_bitwise_identical_to_original_train(self):
        """Single-threaded (see single_cpu_thread; multi-threaded, this failed on RCP).
        The control runs the reference train() twice first: if that differs, the run
        itself isn't reproducible here, and the main comparison would mean nothing."""
        with single_cpu_thread():
            reference = run_update(self.model, self.snapshot, 1, reference=True)
            control = run_update(self.model, self.snapshot, 1, reference=True)
            new = run_update(self.model, self.snapshot, 1)
        assert_bitwise_equal(self, reference, control,
                             "control: reference train() differs from itself (nondeterminism, not a code difference)")
        assert_bitwise_equal(self, reference, new, "update_chunks=1 differs from the reference train()")

    def test_update_chunks_1_never_enters_chunked_path(self):
        with mock.patch.object(self.model, "_train_chunked") as chunked:
            run_update(self.model, self.snapshot, 1)
        chunked.assert_not_called()

    def test_invalid_update_chunks_raise(self):
        for bad in (0, -1, 2.0):
            with self.assertRaises(ValueError):
                PlanFeedback_A2C(
                    GNNPlanFeedbackPolicy, self.env, verbose=0, device="cpu", update_chunks=bad,
                    supported_action_spaces=(sage_spaces.BinaryAction, gym_module.spaces.Discrete, sage_spaces.Autoregressive),
                    n_steps=5, policy_kwargs=dict(POLICY_KWARGS, exploration_fraction=0.1),
                )


# ==========================================================================================
# Equivalence: N in {2, 3, 7, B} vs the unchunked update
# ==========================================================================================

class TestChunkedEquivalence(unittest.TestCase):
    """
    Same update, N in {2, 3, 7, B} vs N = 1, in the training dtype (float32) and in
    float64. In exact arithmetic the chunked gradient equals the unchunked one; what
    remains is floating-point summation order. float64 shows that: the whole-gradient
    difference falls from ~1e-5 (float32) to ~1e-14 (float64), which a logic error (e.g.
    a wrong chunk weight) could not do. So the strict checks are in float64, and the
    float32 checks use tolerances sized to float32 rounding.
    """

    N_ENVS, N_STEPS = 4, 5  # B = 20; N = 7 gives uneven chunks (3,3,3,3,3,3,2), N = B one graph each

    # float32 logged values. A relative check alone fails on keys that are small
    # differences of larger float32 quantities: value_loss is a mean of squared residuals
    # (returns - values), ~1e-4 here, while the rounding of the value predictions scales
    # with |values|, so its relative error is amplified (measured ~3e-6 on the Mac; on
    # RCP a logged value reached 2.05e-5 at N = 7 while the whole-gradient difference was
    # ~9e-8). Every other key stays <= 2e-7 relative on the Mac. RTOL matches the
    # whole-gradient tolerance; ATOL is a floor for values near zero (value_loss's
    # measured absolute difference is ~2e-10). Neither is what would catch a logic error
    # such as a wrong chunk weight: the float64 test is, with relative 1e-12 on the same
    # logged values.
    LOGGED_RTOL, LOGGED_ATOL = 1e-4, 1e-6

    @classmethod
    def setUpClass(cls):
        cls.env, cls.model, cls.snapshot = build_model_with_rollout(n_envs=cls.N_ENVS, n_steps=cls.N_STEPS)
        cls.batch = cls.N_ENVS * cls.N_STEPS
        cls.chunk_counts = (2, 3, 7, cls.batch)
        cls.lr = cls.model.lr_schedule(1)
        cls.float32 = {n: run_update(cls.model, cls.snapshot, n) for n in (1,) + cls.chunk_counts}
        to_float64(cls.model, cls.snapshot)
        cls.float64 = {n: run_update(cls.model, cls.snapshot, n) for n in (1,) + cls.chunk_counts}

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def _table(self, results, dtype):
        base = results[1]
        print(f"\n[chunked equivalence, {dtype}] ternary atom, B={self.batch}, CPU; differences vs update_chunks=1")
        print(f"   {'N':>3}  {'whole-grad rel':>14}  {'max |grad diff|':>15}  {'max |param diff|':>16}  {'max rel logged':>14}")
        for n in self.chunk_counts:
            r = results[n]
            print(f"   {n:>3}  {whole_grad_rel(base['grads'], r['grads']):>14.2e}  {max_abs(base['grads'], r['grads']):>15.2e}"
                  f"  {max_abs(base['params'], r['params']):>16.2e}  {max_rel_logged(base['logged'], r['logged']):>14.2e}")
        print(f"   (largest |grad| {max(g.abs().max().item() for g in base['grads'].values()):.2e}; lr {self.lr:g})")

    def test_float32(self):
        """Training dtype. Gradient: whole-vector relative difference <= 1e-4 (measured
        ~1e-5: float32 sums over millions of node/edge terms, regrouped by chunk).
        Parameters: within one learning-rate step -- AdamW's first step moves each
        weight by lr * g / (|g| + eps), about +-lr however small g is, so rounding noise
        on a near-zero gradient element can move it by up to lr; the gradient checks,
        not this one, are what would catch a wrong chunk weight. Logged values:
        |a - b| <= LOGGED_ATOL + LOGGED_RTOL * |b| (see LOGGED_RTOL). RNG states:
        identical (the evaluate path's discarded multinomial draws happen once per graph,
        in the same order)."""
        self._table(self.float32, "float32")
        base = self.float32[1]
        for n in self.chunk_counts:
            with self.subTest(update_chunks=n):
                r = self.float32[n]
                self.assertEqual(set(base["grads"]), set(r["grads"]))
                self.assertLessEqual(whole_grad_rel(base["grads"], r["grads"]), 1e-4)
                self.assertLessEqual(max_abs(base["params"], r["params"]), self.lr)
                self.assertEqual(set(base["logged"]), set(r["logged"]))
                key, diff, bound = worst_logged(r["logged"], base["logged"], self.LOGGED_RTOL, self.LOGGED_ATOL)
                self.assertLessEqual(diff, bound, f"logged {key}: |diff| {diff:.2e} > {bound:.2e}")
                assert_same_rng(self, base, r)

    def test_float64_matches_to_rounding(self):
        """The strict check: in float64 the chunked update matches element by element."""
        self._table(self.float64, "float64")
        base = self.float64[1]
        self.assertEqual(next(iter(base["grads"].values())).dtype, th.float64)
        for n in self.chunk_counts:
            with self.subTest(update_chunks=n):
                r = self.float64[n]
                self.assertLessEqual(whole_grad_rel(base["grads"], r["grads"]), 1e-12)
                for name, g in base["grads"].items():
                    self.assertTrue(th.allclose(r["grads"][name], g, rtol=1e-9, atol=1e-14), f"grad {name}")
                for name, p in base["params"].items():
                    self.assertTrue(th.allclose(r["params"][name], p, rtol=1e-9, atol=1e-11), f"param {name}")
                self.assertLessEqual(max_rel_logged(base["logged"], r["logged"]), 1e-12)
                assert_same_rng(self, base, r)


# ==========================================================================================
# CUDA: lower peak memory in a small configuration where N=1 also fits
# ==========================================================================================

def free_gpu_memory_bytes():
    """Free memory on the current CUDA device as the driver reports it, i.e. net of other
    processes, or None if it can't be determined. This process's cached blocks are
    released first so they count as free. th.cuda.mem_get_info doesn't exist in torch
    1.7, so there it asks nvidia-smi, for the GPU named by CUDA_VISIBLE_DEVICES (an
    index or a UUID) or, if unset, the GPU with the device's index -- which assumes
    nvidia-smi and CUDA order GPUs alike, true unless a machine mixes GPU models -- or,
    if nvidia-smi sees exactly one GPU (a container), that one."""
    th.cuda.empty_cache()
    if hasattr(th.cuda, "mem_get_info"):
        return th.cuda.mem_get_info()[0]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = [[field.strip() for field in line.split(",")] for line in out.strip().splitlines()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    device = th.cuda.current_device()
    wanted = visible.split(",")[device].strip() if visible else str(device)
    for index, uuid, free_mib in gpus:
        if wanted in (index, uuid):
            return int(free_mib) * 2**20
    return int(gpus[0][2]) * 2**20 if len(gpus) == 1 else None


class TestChunkedMemoryCuda(unittest.TestCase):
    # The N = 1 update below peaks at 2.15 GiB (max_memory_allocated on RCP; ~1.6 GiB
    # estimated on the Mac from the CPU peak RSS), plus the CUDA context and cuBLAS
    # workspaces, which max_memory_allocated doesn't count.
    REQUIRED_FREE_GIB = 3.0

    @unittest.skipIf(not CUDA, "requires CUDA")
    def test_peak_memory_lower_and_gradients_close(self):
        """Fails, rather than skips, when other processes leave too little GPU memory,
        with that as the message instead of a cryptic cuBLAS error.

        Memory: the peak at N = 4 must be under half the peak at N = 1 (measured on
        RCP: 0.555 vs 2.152 GiB).

        Gradients: N = 4 vs N = 1 whole-gradient relative difference <=
        max(1e-3, 10 * control), where the control is N = 1 run twice from the same
        snapshot, i.e. the GPU's own run-to-run noise. GPU float32 rounding is larger
        than on CPU: CUDA scatter-add sums are non-deterministic, and torch 1.7+ may run
        matmuls and convolutions in TF32 on Ampere or newer GPUs (10-bit mantissa; the
        flags are printed), so RCP measured 1.46e-4 where the CPU gives ~1e-5. This
        check only guards against gross errors; exactness of the chunked gradient is
        established by the CPU float64 test (TestChunkedEquivalence, relative 1e-12)."""
        free = free_gpu_memory_bytes()
        if free is None:
            self.fail("cannot determine free GPU memory (no th.cuda.mem_get_info, nvidia-smi unavailable "
                      "or doesn't identify this GPU)")
        if free < self.REQUIRED_FREE_GIB * 2**30:
            self.fail(f"insufficient free GPU memory: other processes are using it "
                      f"({free / 2**30:.2f} GiB free, this test needs {self.REQUIRED_FREE_GIB:g} GiB)")
        try:
            env, model, snapshot = build_model_with_rollout(n_envs=8, n_steps=5, device="cuda")
            peaks = {}
            results = {}
            for run, n in (("N=1", 1), ("N=1 control", 1), ("N=4", 4)):
                th.cuda.synchronize()
                th.cuda.reset_peak_memory_stats()
                results[run] = run_update(model, snapshot, n)
                th.cuda.synchronize()
                peaks[run] = th.cuda.max_memory_allocated()
            env.close()
        except RuntimeError as error:
            if "out of memory" not in str(error) and "ALLOC_FAILED" not in str(error):
                raise
            self.fail(f"ran out of GPU memory although {free / 2**30:.2f} GiB were free at the start: other "
                      f"processes probably took memory meanwhile ({error})")
        control = whole_grad_rel(results["N=1"]["grads"], results["N=1 control"]["grads"])
        rel = whole_grad_rel(results["N=1"]["grads"], results["N=4"]["grads"])
        tolerance = max(1e-3, 10 * control)
        print(f"\n[chunked memory, CUDA, B=40] peak during train(): N=1 {peaks['N=1'] / 2**30:.3f} GiB, "
              f"N=4 {peaks['N=4'] / 2**30:.3f} GiB")
        print(f"   whole-gradient relative diff: N=4 vs N=1 {rel:.2e}; control (N=1 twice) {control:.2e}; "
              f"tolerance {tolerance:.2e}")
        print(f"   TF32: torch.backends.cuda.matmul.allow_tf32 = "
              f"{getattr(th.backends.cuda.matmul, 'allow_tf32', 'n/a')}, "
              f"torch.backends.cudnn.allow_tf32 = {getattr(th.backends.cudnn, 'allow_tf32', 'n/a')}")
        self.assertLess(peaks["N=4"], 0.5 * peaks["N=1"])
        self.assertLessEqual(rel, tolerance)
        self.assertIn("train/max_mem_gb", results["N=4"]["logged"])
        self.assertNotIn("train/max_mem_gb", results["N=1"]["logged"])


# ==========================================================================================
# gnn_global wiring
# ==========================================================================================

class TestGnnGlobalWiring(unittest.TestCase):
    BASE = ["--env-name", TERNARY_ENV_ID, "--graph-convention", "atom", "--planner", "--feedback",
            "--shared-gnn", "--num-processes", "2", "--checkpoint-interval", "0"]

    def _variant(self, argv):
        with mock.patch.object(gnn_global, "run") as run:
            gnn_global.main(argv)
        return run.call_args[0][0]

    def test_flag_reaches_algorithm_kwargs_only_when_set(self):
        self.assertNotIn("update_chunks", self._variant(self.BASE)["algorithm_kwargs"])
        self.assertEqual(self._variant(self.BASE + ["--update-chunks", "4"])["algorithm_kwargs"]["update_chunks"], 4)
        self.assertNotIn("update_chunks", self._variant(self.BASE + ["--update-chunks", "1"])["algorithm_kwargs"])

    def test_raises_without_feedback_or_below_one(self):
        no_feedback = [a for a in self.BASE if a != "--feedback"]
        with mock.patch.object(gnn_global, "run") as run:
            with self.assertRaises(ValueError):
                gnn_global.main(no_feedback + ["--update-chunks", "2"])
            with self.assertRaises(ValueError):
                gnn_global.main(self.BASE + ["--update-chunks", "0"])
            run.assert_not_called()
            gnn_global.main(no_feedback)  # the default stays valid without --feedback

    def test_model_built_by_run_gets_update_chunks(self):
        """run() as gnn_global calls it, stopped before training. gym.make on gym 0.26
        rejects JsonGraph, so make_vec_env builds the same registered env from its spec."""
        real_make_vec_env = gnn_global.make_vec_env

        def make_vec_env_from_spec(env_id, **kwargs):
            env_kwargs = kwargs.pop("env_kwargs", {}) or {}
            return real_make_vec_env(lambda: GraphTaxiEnv(**spec_kwargs(gym_module.spec(env_id)), **env_kwargs), **kwargs)

        class Stop(Exception):
            pass

        for argv, expected in ((self.BASE, 1), (self.BASE + ["--update-chunks", "4"], 4)):
            captured = {}

            def stop_learn(model_self, *args, **kwargs):
                captured["model"] = model_self
                raise Stop()

            with mock.patch.object(gnn_global, "make_vec_env", side_effect=make_vec_env_from_spec), \
                    mock.patch.object(PlanFeedback_A2C, "learn", stop_learn):
                with self.assertRaises(Stop):
                    gnn_global.main(argv)
            self.assertIsInstance(captured["model"], Feedback_A2C)
            self.assertEqual(captured["model"].update_chunks, expected)
            captured["model"].env.close()


if __name__ == "__main__":
    unittest.main()
