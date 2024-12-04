import collections
from typing import Any, Iterator, Optional, Sequence, Tuple, Union
import torch
import gymnasium as gym
import numpy as np
from serl_launcher.data.dataset import Dataset, DatasetDict


def _init_replay_dict(
    obs_space: gym.Space, 
    capacity: int,
    device: torch.device
) -> Union[torch.Tensor, DatasetDict]:
    """Initialize replay buffer storage based on observation space"""
    if isinstance(obs_space, gym.spaces.Box):
        return torch.empty(
            (capacity, *obs_space.shape), 
            dtype=torch.from_numpy(np.empty(1, dtype=obs_space.dtype)).dtype,
            device=device
        )
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity, device)
        return data_dict
    else:
        raise TypeError(f"Unsupported space type: {type(obs_space)}")


def _insert_recursively(
    dataset_dict: DatasetDict, 
    data_dict: DatasetDict, 
    insert_index: int
):
    """Recursively insert data into dataset dictionary"""
    if isinstance(dataset_dict, torch.Tensor):
        if isinstance(data_dict, np.ndarray):
            dataset_dict[insert_index] = torch.from_numpy(data_dict).to(dataset_dict.device)
        else:
            dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys(), (
            dataset_dict.keys(),
            data_dict.keys(),
        )
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError(f"Unsupported type: {type(dataset_dict)}")


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity, self.device)
        next_observation_data = _init_replay_dict(next_observation_space, capacity, self.device)
        
        # Initialize dataset dictionary with torch tensors
        dataset_dict = {
            "observations": observation_data,
            "next_observations": next_observation_data,
            "actions": torch.empty(
                (capacity, *action_space.shape), 
                dtype=torch.float32, 
                device=self.device
            ),
            "rewards": torch.empty(
                (capacity,), 
                dtype=torch.float32, 
                device=self.device
            ),
            "masks": torch.empty(
                (capacity,), 
                dtype=torch.float32, 
                device=self.device
            ),
            "dones": torch.empty(
                (capacity,), 
                dtype=torch.bool, 
                device=self.device
            ),
        }

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        """Insert data into replay buffer"""
        # Convert numpy arrays to torch tensors if needed
        if isinstance(data_dict, dict):
            data_dict = {
                k: (torch.from_numpy(v).to(self.device) if isinstance(v, np.ndarray) else v)
                for k, v in data_dict.items()
            }
            
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}, device=None):
        """Get iterator over batches with optional device transfer"""
        if device is None:
            device = self.device
            
        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                # Move batch to specified device
                if isinstance(data, dict):
                    data = {
                        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                        for k, v in data.items()
                    }
                queue.append(data)

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def download(self, from_idx: int, to_idx: int):
        """Download a range of data from the buffer"""
        indices = torch.arange(from_idx, to_idx, device=self.device)
        data_dict = self.sample(batch_size=len(indices), indx=indices)
        return to_idx, data_dict

    def get_download_iterator(self):
        """Iterator for downloading all data from buffer"""
        last_idx = 0
        while True:
            if last_idx >= self._size:
                raise RuntimeError(f"last_idx {last_idx} >= self._size {self._size}")
            last_idx, batch = self.download(last_idx, self._size)
            yield batch 