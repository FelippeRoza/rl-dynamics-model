# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import gymnasium as gym
import numpy as np

import mbrl.models
import mbrl.planning
import mbrl.types
from mbrl.util.replay_buffer import ReplayBuffer


def cost_rollout_agent_trajectories(
    env: gym.Env,
    steps_or_trials_to_collect: int,
    agent: mbrl.planning.Agent,
    agent_kwargs: Dict,
    trial_length: Optional[int] = None,
    replay_buffer: Optional[ReplayBuffer] = None,
    collect_full_trajectories: bool = False,
    agent_uses_low_dim_obs: bool = False,
    seed: Optional[int] = None,
) -> List[float]:
    """Rollout agent trajectories in the given environment.

    Rollouts trajectories in the environment using actions produced by the given agent.
    Optionally, it stores the saved data into a replay buffer.

    Args:
        env (gym.Env): the environment to step.
        steps_or_trials_to_collect (int): how many steps of the environment to collect. If
            ``collect_trajectories=True``, it indicates the number of trials instead.
        agent (:class:`mbrl.planning.Agent`): the agent used to generate an action.
        agent_kwargs (dict): any keyword arguments to pass to `agent.act()` method.
        trial_length (int, optional): a maximum length for trials (env will be reset regularly
            after this many number of steps). Defaults to ``None``, in which case trials
            will end when the environment returns ``terminated=True`` or ``truncated=True``.
        callback (callable, optional): a function that will be called using the generated
            transition data `(obs, action. next_obs, reward, terminated, truncated)`.
        replay_buffer (:class:`mbrl.util.ReplayBuffer`, optional):
            a replay buffer to store data to use for training.
        collect_full_trajectories (bool): if ``True``, indicates that replay buffers should
            collect full trajectories. This only affects the split between training and
            validation buffers. If ``collect_trajectories=True``, the split is done over
            trials (full trials in each dataset); otherwise, it's done across steps.
        agent_uses_low_dim_obs (bool): only valid if env is of type
            :class:`mbrl.env.MujocoGymPixelWrapper` and replay_buffer is not ``None``.
            If ``True``, instead of passing the obs
            produced by env.reset/step to the agent, it will pass
            obs = env.get_last_low_dim_obs(). This is useful for rolling out an agent
            trained with low dimensional obs, but collect pixel obs in the replay buffer.

    Returns:
        (list(float)): Total rewards obtained at each complete trial.
    """
    if (
        replay_buffer is not None
        and replay_buffer.stores_trajectories
        and not collect_full_trajectories
    ):
        # Might be better as a warning but it's possible that users will miss it.
        raise RuntimeError(
            "Replay buffer is tracking trajectory information but "
            "collect_trajectories is set to False, which will result in "
            "corrupted trajectory data."
        )

    step = 0
    trial = 0
    total_rewards: List[float] = []
    while True:

        obs, info = env.reset(seed=seed)
        cost = info['cost']

        agent.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        while not terminated and not truncated:
            if replay_buffer is not None:
                next_obs, reward, terminated, truncated, info = step_env_and_add_to_buffer(
                    env,
                    obs,
                    cost,
                    agent,
                    agent_kwargs,
                    replay_buffer,
                    agent_uses_low_dim_obs=agent_uses_low_dim_obs,
                )
            else:
                if agent_uses_low_dim_obs:
                    raise RuntimeError(
                        "Option agent_uses_low_dim_obs is only valid if a "
                        "replay buffer is given."
                    )
                action = agent.act(obs, **agent_kwargs)
                next_obs, reward, terminated, truncated, info = env.step(action)
            obs = next_obs
            cost = info['cost']
            total_reward += reward
            step += 1
            if not collect_full_trajectories and step == steps_or_trials_to_collect:
                total_rewards.append(total_reward)
                return total_rewards
            if trial_length and step % trial_length == 0:
                if (
                    collect_full_trajectories
                    and not terminated
                    and replay_buffer is not None
                ):
                    replay_buffer.close_trajectory()
                break
        trial += 1
        total_rewards.append(total_reward)
        if collect_full_trajectories and trial == steps_or_trials_to_collect:
            break
    return total_rewards


def step_env_and_add_to_buffer(
    env: gym.Env,
    obs: np.ndarray,
    cost: np.ndarray,
    agent: mbrl.planning.Agent,
    agent_kwargs: Dict,
    replay_buffer: ReplayBuffer,
    agent_uses_low_dim_obs: bool = False,
) -> Tuple[np.ndarray, float, bool, bool, Dict]:
    """Steps the environment with an agent's action and populates the replay buffer.

    Args:
        env (gym.Env): the environment to step.
        obs (np.ndarray): the latest observation returned by the environment (used to obtain
            an action from the agent).
        agent (:class:`mbrl.planning.Agent`): the agent used to generate an action.
        agent_kwargs (dict): any keyword arguments to pass to `agent.act()` method.
        replay_buffer (:class:`mbrl.util.ReplayBuffer`): the replay buffer
            containing stored data.
        callback (callable, optional): a function that will be called using the generated
            transition data `(obs, action. next_obs, reward, terminated, truncated)`.
        agent_uses_low_dim_obs (bool): only valid if env is of type
            :class:`mbrl.env.MujocoGymPixelWrapper`. If ``True``, instead of passing the obs
            produced by env.reset/step to the agent, it will pass
            obs = env.get_last_low_dim_obs(). This is useful for rolling out an agent
            trained with low dimensional obs, but collect pixel obs in the replay buffer.

    Returns:
        (tuple): next observation, reward, terminated, truncated and meta-info, respectively,
        as generated by `env.step(agent.act(obs))`.
    """

    if agent_uses_low_dim_obs and not hasattr(env, "get_last_low_dim_obs"):
        raise RuntimeError(
            "Option agent_uses_low_dim_obs is only compatible with "
            "env of type mbrl.env.MujocoGymPixelWrapper."
        )
    if agent_uses_low_dim_obs:
        agent_obs = getattr(env, "get_last_low_dim_obs")()
    else:
        agent_obs = obs
    action = agent.act(agent_obs, **agent_kwargs)
    try:
        next_obs, reward, terminated, truncated, info = env.step(action)
    except:
        next_obs, reward, terminated, info = env.step(action)

    next_cost = info['cost']
    max_next_cost = max(next_cost)

    truncated = False

    replay_buffer.add(cost, np.concatenate([obs, action]), next_cost, reward, terminated, truncated)

    return next_obs, reward, terminated, truncated, info
