import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import gymnasium as gym
import torch
import numpy as np


def supply_rng(f, generator=None):
    """Supply a random number generator to a function."""
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(0)
        
    def wrapped(*args, **kwargs):
        nonlocal generator
        kwargs['generator'] = generator
        return f(*args, **kwargs)

    return wrapped


def flatten(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
    """Flatten a nested dictionary into a flat dictionary with dot-separated keys."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def filter_info(info: Dict) -> Dict:
    """Filter out specific keys from info dictionary."""
    filter_keys = [
        "object_names",
        "target_object",
        "initial_positions",
        "target_position",
        "goal",
    ]
    return {k: v for k, v in info.items() if k not in filter_keys}


def add_to(dict_of_lists: Dict[str, List], single_dict: Dict):
    """Add values from single_dict to corresponding lists in dict_of_lists."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(policy_fn, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    """
    Evaluate a policy for a specified number of episodes.
    
    Args:
        policy_fn: Function that takes observations and returns actions
        env: Gymnasium environment
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary of evaluation statistics
    """
    stats = defaultdict(list)
    for _ in range(num_episodes):
        observation, info = env.reset()
        add_to(stats, flatten(info))
        done = False
        while not done:
            action = policy_fn(observation)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            add_to(stats, flatten(info))
        add_to(stats, flatten(info, parent_key="final"))

    return {k: np.mean(v) for k, v in stats.items()}


def evaluate_with_trajectories(
    policy_fn, env: gym.Env, num_episodes: int
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Evaluate a policy and collect full trajectories.
    
    Args:
        policy_fn: Function that takes observations and returns actions
        env: Gymnasium environment
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Tuple of (statistics dictionary, list of trajectory dictionaries)
    """
    trajectories = []
    stats = defaultdict(list)

    for _ in range(num_episodes):
        trajectory = defaultdict(list)
        observation, info = env.reset()
        add_to(stats, flatten(info))
        done = False
        while not done:
            action = policy_fn(observation)
            next_observation, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=r,
                done=done,
                info=info,
            )
            add_to(trajectory, transition)
            add_to(stats, flatten(info))
            observation = next_observation
        add_to(stats, flatten(info, parent_key="final"))
        trajectories.append(trajectory)

    return {k: np.mean(v) for k, v in stats.items()}, trajectories


def bootstrap_std(arr: np.ndarray, f=np.mean, n: int = 30) -> float:
    """
    Compute bootstrap standard deviation of a statistic.
    
    Args:
        arr: Input array
        f: Function to compute statistic
        n: Number of bootstrap samples
        
    Returns:
        Bootstrap standard deviation
    """
    arr = np.array(arr)
    bootstrap_stats = [
        f(arr[np.random.choice(len(arr), len(arr))]) 
        for _ in range(n)
    ]
    return np.std(bootstrap_stats)


def parallel_evaluate(policy_fn, eval_envs, num_eval: int, verbose: bool = True) -> Tuple[List, List]:
    """
    Evaluate policy in parallel environments.
    
    Args:
        policy_fn: Function that takes observations and returns actions
        eval_envs: VectorEnv containing multiple environments
        num_eval: Total number of evaluations to perform
        verbose: Whether to print evaluation results
        
    Returns:
        Tuple of (episode rewards, episode time rewards)
    """
    n_envs = len(eval_envs.reset())
    eval_episode_rewards = []
    eval_episode_time_rewards = []
    counter = np.zeros(n_envs)

    obs = eval_envs.reset()
    if verbose:
        print("Evaluating Envs")
    n_per = int(math.ceil(num_eval / n_envs))
    n_to_eval = n_per * n_envs
    while len(eval_episode_rewards) < n_to_eval:
        action = policy_fn(obs)

        # Observe reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        for n, info in enumerate(infos):
            if "episode" in info.keys() and counter[n] < n_per:
                eval_episode_rewards.append(info["episode"]["r"])
                eval_episode_time_rewards.append(info["episode"]["time_r"])
                counter[n] += 1
                
    if verbose:
        print(
            f"Evaluation using {len(eval_episode_rewards)} episodes: "
            f"mean reward {np.mean(eval_episode_rewards):.5f} "
            f"+- {bootstrap_std(eval_episode_rewards):.5f} \n"
        )
    return eval_episode_rewards, eval_episode_time_rewards
