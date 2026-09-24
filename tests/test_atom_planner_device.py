"""
Tests for the CUDA/device bug found on RCP: graph_to_atoms (called from
graph_to_state_atom <- plan_atom <- Planner.plan <- project_actions,
graph_plan_feedback_policy.py:66) did `np.asarray(x)` directly, which raises
`TypeError: can't convert cuda:0 device type tensor to numpy` when the input Data lives
on CUDA -- which it does during real GPU training, since symbolic_batch (extract_features,
policies.py) is already moved to self.device before project_actions ever runs.

Fix: graph_to_atoms (sage/domains/gym_taxi/utils/representations.py) now accepts a
torch.Tensor on ANY device via a small _to_numpy() helper (falls back to np.asarray for
non-tensor input, so numpy/list input is unaffected). plan_atom's own output device
handling (_atoms_to_projection's `device = reference_graph.x.device`) was already correct
-- confirmed here, not just asserted.

The oracle_sage/vilg planner paths were checked (not changed, per instructions):
graph_to_networkx/graph_to_networkx_vilg already call `.cpu().numpy()` explicitly, so
their READ side is device-safe. Their WRITE side (move_taxi, remove_node_from_graph,
move_taxi_vilg) mutates graph.x/.edge_index/.edge_attr via plain torch indexing, which
preserves whatever device the tensors already live on -- also safe. One inconsistency
worth flagging (not fixed, out of scope): remove_node_from_graph_vilg's
`keep_nodes = th.ones(graph.x.shape[0], dtype=th.bool)` does NOT pass device=, unlike the
`status = th.zeros(..., device=graph.x.device)` two lines above it in the same function --
on a CUDA graph this would construct a CPU tensor and then boolean-index a CUDA tensor
with it, which raises in most PyTorch versions ("indices should be either on cpu or on
the same device as the indexed tensor"). This is a latent bug in the (untouched,
Cell-2-era) vilg planning path, not something this task's changes introduced.

Run from the repo root with:
    python -m pytest tests/test_atom_planner_device.py -v
"""
import unittest

import numpy as np
import torch as th
from torch_geometric.data import Data

from sage.domains.gym_taxi.simulator.taxi_world import TaxiWorldSimulator
from sage.domains.gym_taxi.utils.config import CITY
from sage.domains.gym_taxi.utils.representations import env_to_atom_graph
from sage.domains.gym_taxi.simulator.planner import Planner
from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
from sage.domains.gym_taxi import REWARDS
from sage.agent.async_vec_env import AsyncVecEnv
from sage.forks.stable_baselines3.stable_baselines3.common.env_util import make_vec_env

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


def make_cpu_atom_graph(sim):
    nf, ef, ei, mask, gf = env_to_atom_graph(sim)
    g = Data(x=th.as_tensor(nf, dtype=th.float32), edge_index=th.as_tensor(ei, dtype=th.long), edge_attr=th.as_tensor(ef, dtype=th.float32))
    g.mask = th.as_tensor(mask, dtype=th.bool)
    g.global_features = th.as_tensor(gf, dtype=th.float32).unsqueeze(0)
    return g


def to_device(graph, device):
    g = Data(x=graph.x.to(device), edge_index=graph.edge_index.to(device), edge_attr=graph.edge_attr.to(device))
    g.mask = graph.mask.to(device)
    g.global_features = graph.global_features.to(device)
    return g


class TestPlanAtomOnCuda(unittest.TestCase):
    """A real atom graph moved to cuda, planned, and compared against the CPU-path
    result -- the exact scenario that crashed on RCP."""

    @unittest.skipIf(not th.cuda.is_available(), "requires CUDA")
    def test_cuda_projection_matches_cpu_and_stays_on_device(self):
        sim = TaxiWorldSimulator(
            np.random.RandomState(0), size=5, random_walls=False, planning=True,
            graph_convention="atom", passenger_creation_probability=0,
        )
        pid = next(iter(sim.passengers))

        cpu_graph = make_cpu_atom_graph(sim)
        cuda_graph = to_device(cpu_graph, th.device("cuda"))

        planner = Planner(graph_convention="atom")
        cpu_projection, cpu_actions = planner.plan(cpu_graph, pid)
        cuda_projection, cuda_actions = planner.plan(cuda_graph, pid)

        for name, tensor in [
            ("x", cuda_projection.x), ("edge_index", cuda_projection.edge_index),
            ("edge_attr", cuda_projection.edge_attr), ("mask", cuda_projection.mask),
            ("global_features", cuda_projection.global_features),
        ]:
            self.assertEqual(tensor.device.type, "cuda", f"projection.{name} is not on cuda")

        self.assertEqual(list(cpu_actions), list(cuda_actions))
        self.assertTrue(th.equal(cuda_projection.x.cpu(), cpu_projection.x))
        self.assertTrue(th.equal(cuda_projection.edge_index.cpu(), cpu_projection.edge_index))
        self.assertTrue(th.equal(cuda_projection.edge_attr.cpu(), cpu_projection.edge_attr))
        self.assertTrue(th.equal(cuda_projection.mask.cpu(), cpu_projection.mask))
        self.assertTrue(th.allclose(cuda_projection.global_features.cpu(), cpu_projection.global_features))

        # dtypes must match Step 3a's spec regardless of device
        self.assertEqual(cuda_projection.x.dtype, th.float32)
        self.assertEqual(cuda_projection.edge_index.dtype, th.int64)
        self.assertEqual(cuda_projection.edge_attr.dtype, th.float32)
        self.assertEqual(cuda_projection.mask.dtype, th.bool)
        self.assertEqual(cuda_projection.global_features.dtype, th.float32)


class TestPolicyForwardPassOnAvailableDevice(unittest.TestCase):
    """The actual crash path: PlanFeedback_A2C's policy.forward() on real vec-env
    observations, on whichever device SB3's device="auto" resolves to -- CPU on the Mac,
    CUDA on RCP. This is the smoke test the bug report describes; it must not raise."""

    def test_forward_runs_end_to_end_without_raising(self):
        from sage.agent.plan_feedback_a2c import PlanFeedback_A2C
        from sage.agent.graph_plan_feedback_policy import GNNPlanFeedbackPolicy
        from sage.domains.utils import spaces as sage_spaces
        import gym as gym_module

        def make_env():
            return GraphTaxiEnv(representation="graph", scenario="city", mask=False, rewards=REWARDS["v1"], graph_convention="atom")

        env = make_vec_env(make_env, n_envs=1, seed=0, monitor_kwargs={"info_keywords": ("len100", "len200")}, vec_env_cls=AsyncVecEnv)

        policy_kwargs = {
            "optimizer_class": th.optim.AdamW, "optimizer_kwargs": {"weight_decay": 0.0001},
            "ortho_init": False,
            "exploration_initial_eps": 0.0, "exploration_final_eps": 0.0, "exploration_fraction": 0.0,
            "shared_gnn": True,
            "layer_norm": False,
            "num_planning_choices": 5,  # >1, so this takes the project_actions/gnn_extractor2 path
            "features_extractor_kwargs": {"gnn_steps": 5},
        }
        model = PlanFeedback_A2C(
            GNNPlanFeedbackPolicy, env, verbose=0, device="auto",
            supported_action_spaces=(sage_spaces.BinaryAction, gym_module.spaces.Discrete, sage_spaces.Autoregressive),
            n_steps=5, policy_kwargs=policy_kwargs,
        )
        device = model.device
        self.assertEqual(model.policy.action_net.weight.device.type, device.type)

        obs = env.reset()  # raw vec-env observation array, exactly what policy.forward() receives during real rollouts
        actions, values, log_prob, explored, plans = model.policy.forward(obs)

        self.assertEqual(actions.device.type, device.type)
        self.assertEqual(values.device.type, device.type)
        self.assertEqual(log_prob.device.type, device.type)
        self.assertEqual(values.shape, (1, 1))

        env.close()


if __name__ == "__main__":
    unittest.main()
