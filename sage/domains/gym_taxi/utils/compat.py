"""
.. module:: compat
   :synopsis: Small shims for library APIs that differ between the RCP stack (torch 1.7.1,
   torch_geometric 1.7.2, gym 0.18.0, networkx 2.6.3) and newer local stacks
   (e.g. torch_geometric 2.x, gym 0.26). Use these instead of the raw attribute anywhere
   the code must run on both.
"""


def data_keys(data):
    """
    Attribute names stored on a torch_geometric Data/Batch.

    PyG 1.7.2 exposes ``Data.keys`` as a property returning a list; PyG 2.x makes it a
    method. Calling ``data.keys()`` therefore fails on 1.7.2, and ``list(data.keys)``
    fails on 2.x.

    :param data: a torch_geometric Data or Batch
    :return: list of attribute names
    """
    keys = data.keys
    return list(keys() if callable(keys) else keys)


def spec_kwargs(spec):
    """
    The constructor kwargs a registered gym EnvSpec was registered with.

    gym 0.26 exposes them as ``spec.kwargs``; gym 0.18 stores them as ``spec._kwargs``.

    :param spec: a gym EnvSpec, e.g. ``gym.spec("city-taxi-ternary-unmasked-v1")``
    :return: a shallow copy of the kwargs dict, so callers can't mutate the registry
    """
    if hasattr(spec, "kwargs"):
        return dict(spec.kwargs)
    return dict(spec._kwargs)
