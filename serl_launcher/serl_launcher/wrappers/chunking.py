from collections import deque
from typing import Optional, Dict, Any, List, Tuple

import gymnasium as gym
import gymnasium.spaces
import torch
import numpy as np


def stack_obs(obs: List[Dict]) -> Dict:
    """Stack observations along first dimension."""
    dict_list = {k: [dic[k] for dic in obs] for k in obs[0]}
    
    def _stack_arrays(x: List[Any]) -> Any:
        if isinstance(x[0], np.ndarray):
            return np.stack(x)
        elif isinstance(x[0], torch.Tensor):
            return torch.stack(x)
        elif isinstance(x[0], (list, tuple)):
            return type(x[0])([_stack_arrays([y[i] for y in x]) for i in range(len(x[0]))])
        return x

    return {k: _stack_arrays(v) for k, v in dict_list.items()}


def space_stack(space: gym.Space, repeat: int) -> gym.Space:
    """Create a stacked version of a gym space."""
    if isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=space.dtype,
        )
    elif isinstance(space, gym.spaces.Discrete):
        return gym.spaces.MultiDiscrete([space.n] * repeat)
    elif isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict(
            {k: space_stack(v, repeat) for k, v in space.spaces.items()}
        )
    else:
        raise TypeError(f"Unsupported space type: {type(space)}")


class ChunkingWrapper(gym.Wrapper):
    """
    Enables observation histories and receding horizon control.

    Accumulates observations into obs_horizon size chunks. Starts by repeating the first obs.
    Executes act_exec_horizon actions in the environment.
    
    Args:
        env: The environment to wrap
        obs_horizon: Number of observations to stack
        act_exec_horizon: Number of actions to execute (None for single action)
    """

    def __init__(
        self, 
        env: gym.Env, 
        obs_horizon: int, 
        act_exec_horizon: Optional[int]
    ):
        super().__init__(env)
        self.env = env
        self.obs_horizon = obs_horizon
        self.act_exec_horizon = act_exec_horizon

        self.current_obs = deque(maxlen=self.obs_horizon)

        # Create stacked observation and action spaces
        self.observation_space = space_stack(
            self.env.observation_space, 
            self.obs_horizon
        )
        if self.act_exec_horizon is None:
            self.action_space = self.env.action_space
        else:
            self.action_space = space_stack(
                self.env.action_space, 
                self.act_exec_horizon
            )

    def step(self, action: Any, *args) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute action(s) in environment."""
        act_exec_horizon = self.act_exec_horizon
        if act_exec_horizon is None:
            action = [action]
            act_exec_horizon = 1

        assert len(action) >= act_exec_horizon, \
            f"Action length {len(action)} must be >= act_exec_horizon {act_exec_horizon}"

        # Execute actions sequentially
        for i in range(act_exec_horizon):
            obs, reward, done, trunc, info = self.env.step(action[i], *args)
            self.current_obs.append(obs)
            
        return stack_obs(self.current_obs), reward, done, trunc, info

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        """Reset environment and initialize observation stack."""
        obs, info = self.env.reset(**kwargs)
        self.current_obs.clear()
        self.current_obs.extend([obs] * self.obs_horizon)
        return stack_obs(self.current_obs), info


def post_stack_obs(obs: Dict, obs_horizon: int = 1) -> Dict:
    """Stack observations after collection."""
    if obs_horizon != 1:
        # TODO: Support proper stacking
        raise NotImplementedError("Only obs_horizon=1 is supported for now")
    
    return {k: v[None] for k, v in obs.items()}