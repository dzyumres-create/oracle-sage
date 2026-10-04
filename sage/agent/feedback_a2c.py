from typing import Any, Dict, Optional, Type, Union, Tuple
from numpy.core.numeric import roll

import gym
import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

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
from sage.domains.utils.spaces import Autoregressive

class Feedback_A2C(A2C):
    """
    Feedback Advantage Actor Critic (A2C)
    Modified A2C algorithm which has an additional path value loss designed to help train the path value function in feedback-sage.


    Code: This implementation is built off the stable baselines3 implementation of A2C

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param rms_prop_eps: RMSProp epsilon. It stabilizes square root computation in denominator
        of RMSProp update
    :param use_rms_prop: Whether to use RMSprop (default) or Adam as optimizer
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param normalize_advantage: Whether to normalize or not the advantage
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        policy_coef: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        pvf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        sample_entropy: bool = False,
        tis_heuristic: float = None,
        update_chunks: int = 1,
        log_grad_norms: bool = False,
        supported_action_spaces: Optional[Tuple[spaces.Space, ...]] = (
            spaces.Box,
            spaces.Discrete,
            spaces.MultiDiscrete,
            spaces.MultiBinary,
        )
    ):

        super(Feedback_A2C, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            policy_coef=policy_coef,
            max_grad_norm=max_grad_norm,
            rms_prop_eps=rms_prop_eps,
            use_rms_prop=use_rms_prop,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            normalize_advantage = normalize_advantage,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=_init_setup_model,
            sample_entropy=sample_entropy,
            supported_action_spaces=supported_action_spaces,
        )

        self.pvf_coef=pvf_coef
        self.tis_heuristic=tis_heuristic
        self.env_steps = 0
        # Number of chunks the training update's forward/backward is split into (see
        # _train_chunked). 1 keeps the original single-pass train() unchanged.
        if not isinstance(update_chunks, int) or update_chunks < 1:
            raise ValueError(f"update_chunks must be an int >= 1, got {update_chunks!r}")
        self.update_chunks = update_chunks
        # Logging only (see _log_grad_norms); never changes the update.
        self.log_grad_norms = log_grad_norms

        if _init_setup_model: #overwrite base rollout buffer with explored version.
            self.rollout_buffer = EpsilonRolloutBuffer(
                self.n_steps,
                self.observation_space,
                self.action_space,
                self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.n_envs,
            )

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        if self.update_chunks > 1:
            return self._train_chunked()
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
            if self.log_grad_norms:
                self._log_grad_norms()

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

    def _log_grad_norms(self) -> None:
        """
        Logging only (--log-grad-norms): the gradient norms going into this update's
        clip_grad_norm_, read after backward and before clipping. Reads .grad, changes
        nothing.

        path_value_net is not in the optimizer (it is created after it), so
        optimizer.zero_grad() never clears its .grad: it accumulates across updates, yet
        clip_grad_norm_(self.policy.parameters(), ...) still counts it in the global norm.
          train/grad_norm_head   norm of path_value_net's (accumulated) gradient
          train/grad_norm_train  norm over the parameters the optimizer updates
          train/grad_norm_total  the global norm clip_grad_norm_ computes
          train/clip_coef        the factor it scales gradients by: min(1, max_norm / (total + 1e-6)),
                                 torch's own formula
        """
        def norm(params):
            norms = [p.grad.detach().norm(2) for p in params if p.grad is not None]
            return th.norm(th.stack(norms), 2).item() if norms else 0.0

        head = norm(self.policy.path_value_net.parameters()) if hasattr(self.policy, "path_value_net") else 0.0
        trained = norm(p for group in self.policy.optimizer.param_groups for p in group["params"])
        total = norm(self.policy.parameters())
        logger.record("train/grad_norm_head", head)
        logger.record("train/grad_norm_train", trained)
        logger.record("train/grad_norm_total", total)
        logger.record("train/clip_coef", min(1.0, self.max_grad_norm / (total + 1e-6)))

    def _train_chunked(self) -> None:
        """
        train() with the forward/backward split into self.update_chunks chunks, to lower
        the update's peak memory without changing the learning.

        Same single batch (one rollout_buffer.get, so the same permutation draw), split
        into contiguous slices of that order. Every loss term in train() is a mean over
        the batch (policy, value and path-value losses over samples; the policy's entropy
        over graphs), and nothing in the network couples graphs, so weighting chunk i's
        loss by n_i / B makes the summed gradients equal train()'s gradient. Gradients
        accumulate across chunks; clipping and the optimizer step happen once, on the
        full gradient. Logged values are the same weighted sums, so they match train()'s.
        Equal up to floating-point summation order, not bitwise.
        """
        use_cuda = self.device.type == "cuda"
        if use_cuda:
            th.cuda.reset_peak_memory_stats(self.device)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # Same single full batch train() uses (consumed the same way)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            pass

        actions = rollout_data.actions
        if (isinstance(self.action_space, spaces.Discrete) or
           isinstance(self.action_space, Autoregressive) ):
            # Convert discrete action from float to long
            actions = actions.long().flatten()

        # Normalize advantage over the FULL batch, before chunking (not present in the original implementation)
        advantages = rollout_data.advantages
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = actions.shape[0]
        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0, "entropy": 0.0, "path_value_loss": 0.0}

        self.policy.optimizer.zero_grad()
        for chunk in np.array_split(np.arange(batch_size), self.update_chunks):
            if len(chunk) == 0:
                continue
            weight = len(chunk) / batch_size
            index = th.as_tensor(chunk, dtype=th.long, device=actions.device)

            values, log_prob, entropy, path_values = self.policy.evaluate_actions(
                rollout_data.observations[chunk], actions[index]
            )
            values = values.flatten()
            chunk_advantages = advantages[index]
            returns = rollout_data.returns[index]
            explored = rollout_data.explored[index]

            # Policy gradient loss
            policy_loss = -(chunk_advantages * log_prob)
            if self.tis_heuristic is not None:
                probs = th.exp(log_prob.detach())
                policy_loss *= (probs*self.tis_heuristic).clamp(0,1) #clamp is the truncated in truncated importance sampling
            policy_loss = policy_loss.mean()

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(returns*(1-explored), values*(1-explored))

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
                path_value_loss = F.mse_loss(returns, path_values)
                loss = self.policy_coef*policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.pvf_coef * path_value_loss
                totals["path_value_loss"] += weight * path_value_loss.item()

            # Accumulate this chunk's share of the full-batch gradient
            (weight * loss).backward()

            totals["policy_loss"] += weight * policy_loss.item()
            totals["value_loss"] += weight * value_loss.item()
            totals["entropy_loss"] += weight * entropy_loss.item()
            totals["entropy"] += weight * entropy.item()

        if self.log_grad_norms:
            self._log_grad_norms()

        # Clip grad norm on the full accumulated gradient, then a single step
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/explained_variance", explained_var)
        logger.record("train/entropy_loss", totals["entropy_loss"])
        logger.record("train/entropy", totals["entropy"])
        logger.record("train/policy_loss", totals["policy_loss"])
        logger.record("train/value_loss", totals["value_loss"])
        if self.pvf_coef != 0:
            logger.record("train/path_value_loss", totals["path_value_loss"])
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        if use_cuda:
            logger.record("train/max_mem_gb", th.cuda.max_memory_allocated(self.device) / 2**30)


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

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor
                actions, values, log_probs, explored = self.policy.forward(self._last_obs)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_dones, values, log_probs, explored)
            self._last_obs = new_obs
            self._last_dones = dones

        with th.no_grad():
            #Fixing episode timeout - only correct for num-steps = 1
            true_obs = [[x['s_true']] for x in infos]
            true_done = np.array([x['d_true'] for x in infos],dtype=bool)
            # Compute value for the last timestep
            #obs_tensor = th.as_tensor(new_obs).to(self.device)
            _, values, _, _ = self.policy.forward(true_obs)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=true_done)
        logger.record("time/env_steps", self.num_timesteps)

        callback.on_rollout_end()

        return True
