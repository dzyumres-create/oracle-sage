"""
.. module:: checkpoint_callback
   :synopsis: Periodic mid-training checkpointing (Cell 5, Task C).
"""
import glob
import os
import re

from sage.forks.stable_baselines3.stable_baselines3.common.callbacks import BaseCallback


class PeriodicCheckpointCallback(BaseCallback):
    """
    Saves model.save(f"{save_dir}/checkpoint_{num_timesteps}") each time
    self.num_timesteps crosses a multiple of `interval`, keeping only the `keep` most
    recent checkpoint_*.zip files in save_dir. final_model.zip is never touched -- it
    doesn't match the checkpoint_*.zip glob this callback prunes.

    "Crosses a multiple", not "equals a multiple": PlanFeedback_A2C's
    self.num_timesteps advances by env.num_envs once per rollout step (not once per
    literal env.step() -- a rollout step can execute several real env steps while
    running a multi-step plan), so it will not land exactly on every multiple of
    `interval` unless interval happens to be a multiple of num_envs. Comparing
    `num_timesteps // interval` against the same ratio at the last checkpoint detects
    every interval boundary crossed since, exactly once, regardless of step size.

    What this callback touches, precisely (per the task's "no behaviour change" /
    "report what the callback touches" requirement):
      - reads self.num_timesteps (set by BaseCallback.on_step(), from
        self.model.num_timesteps, BEFORE _on_step() runs -- this callback never reads
        or writes self.model.num_timesteps directly) and self.model itself (only to
        call .save(), never to mutate any of its attributes);
      - on a triggered save: os.makedirs(save_dir) (idempotent, exist_ok=True),
        self.model.save(path) (BaseAlgorithm.save() -- confirmed by reading its
        source: copies self.__dict__ and torch state_dicts to a zip; no RNG draws, no
        env access, no mutation of the model), then os.listdir-equivalent
        (glob.glob) + os.remove on this save_dir's own checkpoint_*.zip files only;
      - does NOT touch self.training_env, self.locals, self.globals, action/observation
        sampling, or any other callback's state -- CallbackList invokes each callback's
        on_step() independently and this one returns True unconditionally, so it can
        never itself stop training or alter what TensorboardCallback (or any other
        callback in the list) sees.

    :param save_dir: directory .zip checkpoints are written to (created if missing)
    :param interval: total-timesteps interval between checkpoints; the caller is
        responsible for not constructing this callback at all when checkpointing
        should be disabled (interval <= 0) -- see gnn_global.py
    :param keep: number of most recent checkpoints to retain (ranked by the numeric
        timestep in their filename, not file mtime)

    Loading these checkpoints back is currently BROKEN for every graph_convention
    (oracle_sage, vilg, atom) -- do not assume PlanFeedback_A2C.load(path) works when
    evaluating a trained model. Two separate, pre-existing gaps, neither introduced by
    this callback and neither fixed here:

      1. observation_space is a JsonGraph (sage/domains/utils/spaces.py), which
         subclasses gym.spaces.Box but never calls Box.__init__(), so it never sets
         .low/.high. Newer gym's Box.__setstate__ unconditionally reads self.low during
         unpickling, so cloudpickle.loads() raises
         AttributeError: 'JsonGraph' object has no attribute 'low' the moment a saved
         checkpoint's observation_space is deserialized.
      2. The obvious workaround -- load(path, custom_objects={"observation_space": ...,
         "action_space": ...}), which substitutes a live space instead of deserializing
         the saved one -- does NOT work through this vendored SB3 fork as currently
         written: BaseAlgorithm.load() (base_class.py) never forwards custom_objects to
         load_from_zip_file(), which never forwards it to json_to_data() -- the one
         function that actually implements custom_objects substitution. So
         PlanFeedback_A2C.load(path, custom_objects={...}) still hits the same
         AttributeError as a plain .load(path) call.

    A proper fix needs both: (a) giving JsonGraph real .low/.high (e.g. dummy full
    arrays) or rebasing it on plain gym.spaces.Space instead of Box, so it survives
    pickling at all, and (b) threading custom_objects through
    BaseAlgorithm.load()/load_from_zip_file() to json_to_data() as a belt-and-braces
    fallback. Both changes are convention-agnostic -- JsonGraph and load() are shared by
    all three conventions, so fixing this is not atom-specific and is out of scope for
    Cell 5.
    """

    def __init__(self, save_dir: str, interval: int, keep: int = 2, verbose: int = 0):
        super().__init__(verbose)
        assert interval > 0, "PeriodicCheckpointCallback requires interval > 0; do not construct it when checkpointing is disabled"
        self.save_dir = save_dir
        self.interval = interval
        self.keep = keep
        self._last_checkpointed_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps // self.interval > self._last_checkpointed_step // self.interval:
            os.makedirs(self.save_dir, exist_ok=True)
            path = os.path.join(self.save_dir, f"checkpoint_{self.num_timesteps}")
            self.model.save(path)
            self._last_checkpointed_step = self.num_timesteps
            if self.verbose > 0:
                print(f"Saved checkpoint to {path}.zip")
            self._prune_old_checkpoints()
        return True

    def _prune_old_checkpoints(self) -> None:
        checkpoints = []
        for filepath in glob.glob(os.path.join(self.save_dir, "checkpoint_*.zip")):
            match = re.fullmatch(r"checkpoint_(\d+)\.zip", os.path.basename(filepath))
            if match:
                checkpoints.append((int(match.group(1)), filepath))
        checkpoints.sort(key=lambda entry: entry[0])
        for _step, filepath in checkpoints[: -self.keep] if self.keep > 0 else checkpoints:
            os.remove(filepath)
