"""
Load a saved model's policy (PlanFeedback_A2C.save / PeriodicCheckpointCallback zips).

The fork's own BaseAlgorithm.load does not work for these models: the zip's "data" pickles
the JsonGraph observation space, which subclasses gym's Box without calling Box.__init__
and so cannot be unpickled, and the fork never forwards custom_objects to replace it.

This loader never unpickles the spaces. It builds the policy with the given spaces (from a
live env of the same kind) and the zip's own policy_kwargs, then loads policy.pth and
policy.optimizer.pth with load_state_dict. The few pickled entries it does read go through
a restricted unpickler that only allows the policy classes and optimizers listed below.

Moved here from analysis/inspect_logits.py (branch ternary-cell1-relclamp, fd52445), where
it was written for Cell 1 only; inspect_logits now uses this module.
"""
import base64
import io
import json
import pickle
import zipfile

import torch as th

POLICY_CLASSES = {
    ("sage.agent.graph_policy", "GNNPolicy"),
    ("sage.agent.graph_feedback_policy", "GNNFeedbackPolicy"),
    ("sage.agent.graph_plan_feedback_policy", "GNNPlanFeedbackPolicy"),
    ("sage.agent.wl_plan_feedback_policy", "WLPlanFeedbackPolicy"),
}
OPTIMIZER_CLASSES = {
    ("torch.optim.adamw", "AdamW"),
    ("torch.optim.adam", "Adam"),
    ("torch.optim.rmsprop", "RMSprop"),
}


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickles only the globals in `allowed` ({(module, name)}); anything else raises."""

    def __init__(self, data, allowed):
        super(RestrictedUnpickler, self).__init__(io.BytesIO(data))
        self.allowed = allowed

    def find_class(self, module, name):
        if (module, name) in self.allowed:
            return super(RestrictedUnpickler, self).find_class(module, name)
        raise pickle.UnpicklingError(f"restricted unpickler: {module}.{name} is not allowed")


def decode_serialized(entry, allowed):
    """One SB3 "data" JSON entry ({":type:", ":serialized:" base64 pickle}) through the restricted
    unpickler."""
    return RestrictedUnpickler(base64.b64decode(entry[":serialized:"]), allowed).load()


def read_zip(zip_path):
    """(data JSON, policy state_dict, optimizer state_dict or None) -- tensors mapped to CPU."""
    with zipfile.ZipFile(zip_path) as z:
        data = json.loads(z.read("data").decode())
        policy_state = th.load(io.BytesIO(z.read("policy.pth")), map_location="cpu")
        optimizer_state = None
        if "policy.optimizer.pth" in z.namelist():
            optimizer_state = th.load(io.BytesIO(z.read("policy.optimizer.pth")), map_location="cpu")
    return data, policy_state, optimizer_state


def saved_policy_kwargs(data):
    """The policy_kwargs the model was built with (optimizer classes allowed)."""
    return decode_serialized(data["policy_kwargs"], OPTIMIZER_CLASSES)


def saved_policy_class(data):
    """The policy class the model was built with (only the classes in POLICY_CLASSES)."""
    return decode_serialized(data["policy_class"], POLICY_CLASSES)


def load_weights(policy, policy_state, optimizer_state=None, source="model"):
    """Loads a policy state_dict (strict) and, if given, the optimizer state into `policy`.
    load_state_dict checks the optimizer's group sizes but not shapes, so shapes are checked."""
    policy.load_state_dict(policy_state)
    if optimizer_state is None:
        return
    policy.optimizer.load_state_dict(optimizer_state)
    for group in policy.optimizer.param_groups:
        for p in group["params"]:
            state = policy.optimizer.state.get(p, {})
            if "exp_avg" in state and state["exp_avg"].shape != p.shape:
                raise ValueError(f"{source}: optimizer state shape {tuple(state['exp_avg'].shape)} != {tuple(p.shape)}")


def load_policy(zip_path, observation_space, action_space, policy_class=None, policy_kwargs=None,
                override_kwargs=None, device="cpu", load_optimizer=True):
    """
    (policy, data): the saved policy, rebuilt and loaded, in eval mode.

    :param observation_space, action_space: from a live env of the same domain and graph
        convention as the model's (e.g. GraphTaxiEnv(...).observation_space)
    :param policy_class: None reads it from the zip (restricted to POLICY_CLASSES)
    :param policy_kwargs: None uses the zip's own policy_kwargs
    :param override_kwargs: entries replacing policy_kwargs ones, e.g. {"wl_vocab_path": ...}
        when the vocab lives elsewhere on this machine, or {"clamp_mode": ...}
    :param load_optimizer: also restore the optimizer (AdamW moments, learning rate)
    """
    data, policy_state, optimizer_state = read_zip(zip_path)
    kwargs = saved_policy_kwargs(data) if policy_kwargs is None else dict(policy_kwargs)
    kwargs.update(override_kwargs or {})
    policy_class = policy_class or saved_policy_class(data)
    learning_rate = data.get("learning_rate")
    # a constant learning rate is stored as a number; a schedule is pickled (not read here).
    # Either way, loading the optimizer state restores the rate the run last used.
    learning_rate = learning_rate if isinstance(learning_rate, (int, float)) else 0.0
    policy = policy_class(observation_space, action_space, lambda _progress: learning_rate, **kwargs)
    # move first: Optimizer.load_state_dict puts its state on each parameter's device
    policy = policy.to(device)
    load_weights(policy, policy_state, optimizer_state if load_optimizer else None, source=str(zip_path))
    policy.eval()
    return policy, data
