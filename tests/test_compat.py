"""
Tests for sage/domains/gym_taxi/utils/compat.py. Each helper is checked against the
library actually installed AND against a stand-in shaped like the other stack's API, so
both branches run on any machine (the Mac has PyG 2.x / gym 0.26; RCP has PyG 1.7.2 /
gym 0.18).

Run from the repo root with:
    PYTHONPATH=. python -m pytest tests/test_compat.py -v
"""
import unittest

import gym
import torch as th
from torch_geometric.data import Batch, Data

import sage.domains.gym_taxi  # registers the envs
from sage.domains.gym_taxi.utils.compat import data_keys, spec_kwargs


class _KeysAsListProperty(object):
    """PyG 1.7.2 shape: Data.keys is a property returning a list."""

    @property
    def keys(self):
        return ["x", "edge_index"]


class _KeysAsMethod(object):
    """PyG 2.x shape: Data.keys() is a method."""

    def keys(self):
        return ["x", "edge_index"]


class _Gym018Spec(object):
    """gym 0.18 shape: kwargs live on spec._kwargs, with no spec.kwargs."""

    def __init__(self, kwargs):
        self._kwargs = kwargs


class TestDataKeys(unittest.TestCase):
    def test_list_property_shape(self):
        self.assertEqual(data_keys(_KeysAsListProperty()), ["x", "edge_index"])

    def test_method_shape(self):
        self.assertEqual(data_keys(_KeysAsMethod()), ["x", "edge_index"])

    def test_installed_pyg_data_and_batch(self):
        d = Data(x=th.zeros(2, 3), edge_index=th.zeros(2, 1, dtype=th.long))
        d.mask = th.ones(2, dtype=th.bool)
        self.assertEqual(set(data_keys(d)), {"x", "edge_index", "mask"})
        batch = Batch.from_data_list([d, d])
        self.assertTrue({"x", "edge_index", "mask"} <= set(data_keys(batch)))
        self.assertEqual(set(data_keys(batch.to_data_list()[0])), {"x", "edge_index", "mask"})


class TestSpecKwargs(unittest.TestCase):
    def test_gym018_shape(self):
        kwargs = {"scenario": "city_ternary", "ternary": True}
        self.assertEqual(spec_kwargs(_Gym018Spec(kwargs)), kwargs)

    def test_installed_gym_spec(self):
        kwargs = spec_kwargs(gym.spec("city-taxi-ternary-unmasked-v1"))
        self.assertEqual(kwargs["scenario"], "city_ternary")
        self.assertIs(kwargs["ternary"], True)

    def test_returns_a_copy(self):
        spec = gym.spec("city-taxi-ternary-unmasked-v1")
        kwargs = spec_kwargs(spec)
        kwargs["scenario"] = "mutated"
        self.assertEqual(spec_kwargs(spec)["scenario"], "city_ternary")


if __name__ == "__main__":
    unittest.main()
