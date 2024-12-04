from threading import Lock
from typing import Union, Iterable

import gymnasium as gym
import torch
from replay_buffer_torch import ReplayBuffer
from memory_efficient_replay_buffer_torch import (
    MemoryEfficientReplayBuffer,
)

from agentlace.data.data_store import DataStoreBase


class ReplayBufferDataStore(ReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        device: str = "cpu"
    ):
        ReplayBuffer.__init__(self, observation_space, action_space, capacity, device=device)
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    def insert(self, *args, **kwargs):
        """Thread-safe insertion into replay buffer"""
        with self._lock:
            super(ReplayBufferDataStore, self).insert(*args, **kwargs)

    def sample(self, *args, **kwargs):
        """Thread-safe sampling from replay buffer"""
        with self._lock:
            return super(ReplayBufferDataStore, self).sample(*args, **kwargs)

    def latest_data_id(self) -> int:
        """Get the index of the latest inserted data"""
        return self._insert_index

    def get_latest_data(self, from_id: int):
        """Get data since the given ID (Not implemented)"""
        raise NotImplementedError("TODO")


class MemoryEfficientReplayBufferDataStore(MemoryEfficientReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        image_keys: Iterable[str] = ("image",),
        device: str = "cpu",
        **kwargs,
    ):
        MemoryEfficientReplayBuffer.__init__(
            self, 
            observation_space, 
            action_space, 
            capacity, 
            pixel_keys=image_keys, 
            device=device,
            **kwargs
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    def insert(self, *args, **kwargs):
        """Thread-safe insertion into replay buffer"""
        with self._lock:
            super(MemoryEfficientReplayBufferDataStore, self).insert(*args, **kwargs)

    def sample(self, *args, **kwargs):
        """Thread-safe sampling from replay buffer"""
        with self._lock:
            return super(MemoryEfficientReplayBufferDataStore, self).sample(
                *args, **kwargs
            )

    def latest_data_id(self) -> int:
        """Get the index of the latest inserted data"""
        return self._insert_index

    def get_latest_data(self, from_id: int):
        """Get data since the given ID (Not implemented)"""
        raise NotImplementedError("TODO")


def populate_data_store(
    data_store: DataStoreBase,
    demos_path: str,
    device: str = "cpu"
) -> DataStoreBase:
    """
    Utility function to populate demonstrations data into data_store.
    
    Args:
        data_store: The data store to populate
        demos_path: Path to demonstration files
        device: Device to store tensors on
        
    Returns:
        Populated data store
    """
    import pickle as pkl
    import numpy as np
    from copy import deepcopy

    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                # Convert numpy arrays to torch tensors if needed
                if isinstance(transition, dict):
                    transition = {
                        k: (torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v)
                        for k, v in transition.items()
                    }
                data_store.insert(transition)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store


def populate_data_store_with_z_axis_only(
    data_store: DataStoreBase,
    demos_path: str,
    device: str = "cpu"
) -> DataStoreBase:
    """
    Utility function to populate demonstrations data into data_store.
    This will remove the x and y cartesian coordinates from the state.
    
    Args:
        data_store: The data store to populate
        demos_path: Path to demonstration files
        device: Device to store tensors on
        
    Returns:
        Populated data store
    """
    import pickle as pkl
    import numpy as np
    from copy import deepcopy

    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                tmp = deepcopy(transition)
                # Convert state arrays to torch tensors and concatenate
                state = torch.from_numpy(tmp["observations"]["state"]).to(device)
                next_state = torch.from_numpy(tmp["next_observations"]["state"]).to(device)
                
                tmp["observations"]["state"] = torch.cat([
                    state[:, :4],
                    state[:, 6:7],
                    state[:, 10:],
                ], dim=-1)
                
                tmp["next_observations"]["state"] = torch.cat([
                    next_state[:, :4],
                    next_state[:, 6:7],
                    next_state[:, 10:],
                ], dim=-1)
                
                data_store.insert(tmp)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store 