from threading import Lock
from typing import Union, Iterable
import pickle as pkl
import numpy as np
from copy import deepcopy

import gymnasium as gym
import torch
from serl_launcher.data.replay_buffer import ReplayBuffer
from serl_launcher.data.memory_efficient_replay_buffer import (
    MemoryEfficientReplayBuffer,
)

from agentlace.data.data_store import DataStoreBase


class ReplayBufferDataStore(ReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
    ):
        ReplayBuffer.__init__(self, observation_space, action_space, capacity)
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    # ensure thread safety
    def insert(self, *args, **kwargs):
        with self._lock:
            super(ReplayBufferDataStore, self).insert(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(ReplayBufferDataStore, self).sample(*args, **kwargs)

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO


class MemoryEfficientReplayBufferDataStore(MemoryEfficientReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        image_keys: Iterable[str] = ("image",),
        **kwargs,
    ):
        MemoryEfficientReplayBuffer.__init__(
            self, observation_space, action_space, capacity, pixel_keys=image_keys, **kwargs
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    # ensure thread safety
    def insert(self, *args, **kwargs):
        with self._lock:
            super(MemoryEfficientReplayBufferDataStore, self).insert(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(MemoryEfficientReplayBufferDataStore, self).sample(
                *args, **kwargs
            )

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO


def populate_data_store(
    data_store: DataStoreBase,
    demos_path: str,
) -> DataStoreBase:
    """
    Utility function to populate demonstrations data into data_store.
    
    Args:
        data_store: The data store to populate
        demos_path: Path to demonstration files
        
    Returns:
        Populated data store
    """
    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                # Convert numpy arrays to torch tensors if needed
                if isinstance(transition, dict):
                    transition = {
                        k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                        for k, v in transition.items()
                    }
                data_store.insert(transition)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store


def populate_data_store_with_z_axis_only(
    data_store: DataStoreBase,
    demos_path: str,
) -> DataStoreBase:
    """
    Utility function to populate demonstrations data into data_store.
    This will remove the x and y cartesian coordinates from the state.
    
    Args:
        data_store: The data store to populate
        demos_path: Path to demonstration files
        
    Returns:
        Populated data store
    """
    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                tmp = deepcopy(transition)
                
                # Process state data
                state = tmp["observations"]["state"]
                if isinstance(state, np.ndarray):
                    state = torch.from_numpy(state)
                
                # Extract relevant coordinates
                tmp["observations"]["state"] = torch.cat([
                    state[:, :4],
                    state[:, 6].unsqueeze(1),
                    state[:, 10:],
                ], dim=-1)
                
                # Process next state data
                next_state = tmp["next_observations"]["state"]
                if isinstance(next_state, np.ndarray):
                    next_state = torch.from_numpy(next_state)
                    
                tmp["next_observations"]["state"] = torch.cat([
                    next_state[:, :4],
                    next_state[:, 6].unsqueeze(1),
                    next_state[:, 10:],
                ], dim=-1)
                
                data_store.insert(tmp)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store
