from typing import Any, Dict, List, Optional, Type, Union, Tuple
from numpy.core.numeric import roll
import math

import gym
import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from sage.domains.utils.spaces import Autoregressive

from sage.forks.stable_baselines3.stable_baselines3.common import logger
from sage.forks.stable_baselines3.stable_baselines3.a2c.a2c import A2C
from sage.forks.stable_baselines3.stable_baselines3.common.policies import ActorCriticPolicy
from sage.forks.stable_baselines3.stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from sage.forks.stable_baselines3.stable_baselines3.common.utils import explained_variance


from sage.forks.stable_baselines3.stable_baselines3.common import logger
from sage.forks.stable_baselines3.stable_baselines3.common.base_class import BaseAlgorithm
from sage.forks.stable_baselines3.stable_baselines3.common.buffers import RolloutBuffer
from sage.forks.stable_baselines3.stable_baselines3.common.callbacks import BaseCallback
from sage.forks.stable_baselines3.stable_baselines3.common.policies import ActorCriticPolicy
from sage.forks.stable_baselines3.stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from sage.forks.stable_baselines3.stable_baselines3.common.utils import safe_mean
from sage.forks.stable_baselines3.stable_baselines3.common.vec_env import VecEnv

from sage.agent.epsilon_buffer import EpsilonRolloutBuffer
from sage.agent.feedback_a2c import Feedback_A2C
from sage.agent.diag_occupancy import OccupancyDiagnostics


class PlanFeedback_A2C(Feedback_A2C):
    """
    Feedback Advantage Actor Critic (A2C)
    Modified A2C algorithm which has an additional path value loss designed to help train the path value function in feedback-sage.


    Code: This implementation is built off the stable baselines3 implementation of A2C

    :param diag_occupancy: None (default), or OccupancyDiagnostics' kwargs ({"out_dir", "run_name"})
        to turn on the logging-only --diag-occupancy diagnostics (sage/agent/diag_occupancy.py).
        The env must then be built with GraphTaxiEnv(diag_occupancy=True).
    """

    def __init__(self, *args, diag_occupancy: Optional[Dict[str, Any]] = None, **kwargs):
        super(PlanFeedback_A2C, self).__init__(*args, **kwargs)
        # Set only when enabled, so a default model's __dict__ (and saved data) is unchanged;
        # Feedback_A2C.diag = None is the class-level default.
        if diag_occupancy is not None:
            self.diag = OccupancyDiagnostics(**diag_occupancy)

    def _excluded_save_params(self) -> List[str]:
        # the diagnostics hold file paths and per-run state, not model state
        return super(PlanFeedback_A2C, self)._excluded_save_params() + ["diag"]

    def learn(self, *args, **kwargs):
        try:
            return super(PlanFeedback_A2C, self).learn(*args, **kwargs)
        finally:
            if self.diag is not None:
                self.diag.close()

    def collect_rollouts(
        self, env: VecEnv, callback: BaseCallback, rollout_buffer: EpsilonRolloutBuffer, n_rollout_steps: int
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        self.policy._on_step(self._current_progress_remaining)
        diag = self.diag  # --diag-occupancy: logging only, None unless enabled
        if diag is not None:
            diag.begin_rollout(self)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            if diag is not None:
                self.policy.diag_stash = {}
            with th.no_grad():
                # Convert to pytorch tensor
                actions, values, log_probs, explored, plans = self.policy.forward(self._last_obs)
            actions = actions.cpu().numpy()
            if diag is not None:
                diag.on_forward(self, env, n_steps, actions, log_probs, plans)

            plan_lengths = np.array([len(p) for p in plans])
            max_plan_length = max(plan_lengths)
            env_finished = plan_lengths == 0
            plans_grid = -np.ones((len(plan_lengths),max_plan_length),dtype=int) #create a grid of plans
            for i,p in enumerate(plans):
                plans_grid[i,:plan_lengths[i]]=p
            plan_step=0
            plan_rewards = np.zeros(len(plan_lengths))

            while not env_finished.all() and plan_step<max_plan_length:
                env_actions = plans_grid[:,plan_step]
                env_finished = np.logical_or(env_finished,(env_actions == -1)) #if any plans have invalid actions, they are finished
                self.env_steps+=np.logical_not(env_finished).sum()
                new_obs, rewards, dones, infos = env.step(env_actions,np.logical_not(env_finished))
                if diag is not None:
                    diag.on_env_step(np.logical_not(env_finished), infos, dones)
                plan_rewards += ((math.pow(self.gamma,plan_step))*rewards)*(1-env_finished) #aggregate reward for all continuing environments
                env_finished = np.logical_or(env_finished,dones)
                plan_step +=1

            self.num_timesteps += env.num_envs
            if diag is not None:
                diag.end_decision(plan_rewards, self.num_timesteps)

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if (isinstance(self.action_space, gym.spaces.Discrete) or
               isinstance(self.action_space, Autoregressive) ):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, plan_rewards, self._last_dones, values, log_probs, explored)
            self._last_obs = new_obs
            self._last_dones = dones

        with th.no_grad():
            # #Fixing episode timeout - only correct for num-steps = 1
            # true_obs = [[x['s_true']] for x in infos]
            # true_done = np.array([x['d_true'] for x in infos],dtype=bool)
            # Compute value for the last timestep
            #obs_tensor = th.as_tensor(new_obs).to(self.device)
            if diag is not None:
                self.policy.diag_stash = {}
            bootstrap_actions, values, _, _, _ = self.policy.forward(new_obs)
            if diag is not None:
                diag.on_bootstrap(self, env, bootstrap_actions.cpu().numpy())

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        if diag is not None:
            diag.end_rollout(rollout_buffer)
        logger.record("time/env_steps", self.env_steps)

        callback.on_rollout_end()

        return True
