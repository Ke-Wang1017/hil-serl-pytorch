from typing import Dict, Iterable, Optional, Tuple, Union
import torch
import numpy as np
from gymnasium.utils import seeding
from dataclasses import dataclass

DataType = Union[np.ndarray, Dict[str, "DataType"]]
DatasetDict = Dict[str, DataType]


def _check_lengths(dataset_dict: DatasetDict, dataset_len: Optional[int] = None) -> int:
    """Check that all arrays in the dataset have consistent lengths."""
    for v in dataset_dict.values():
        if isinstance(v, dict):
            dataset_len = dataset_len or _check_lengths(v, dataset_len)
        elif isinstance(v, (np.ndarray, torch.Tensor)):
            item_len = len(v)
            dataset_len = dataset_len or item_len
            assert dataset_len == item_len, "Inconsistent item lengths in the dataset."
        else:
            raise TypeError(f"Unsupported type: {type(v)}")
    return dataset_len


def _subselect(dataset_dict: DatasetDict, index: np.ndarray) -> DatasetDict:
    """Select a subset of the dataset using the given indices."""
    new_dataset_dict = {}
    for k, v in dataset_dict.items():
        if isinstance(v, dict):
            new_v = _subselect(v, index)
        elif isinstance(v, (np.ndarray, torch.Tensor)):
            new_v = v[index]
        else:
            raise TypeError(f"Unsupported type: {type(v)}")
        new_dataset_dict[k] = new_v
    return new_dataset_dict


def _sample(dataset_dict: Union[np.ndarray, DatasetDict], indx: np.ndarray) -> DatasetDict:
    """Sample from the dataset using the given indices."""
    if isinstance(dataset_dict, (np.ndarray, torch.Tensor)):
        return dataset_dict[indx]
    elif isinstance(dataset_dict, dict):
        batch = {}
        for k, v in dataset_dict.items():
            batch[k] = _sample(v, indx)
    else:
        raise TypeError(f"Unsupported type: {type(dataset_dict)}")
    return batch


class Dataset:
    def __init__(self, dataset_dict: DatasetDict, seed: Optional[int] = None):
        """
        Initialize dataset with dictionary of arrays and optional random seed.
        
        Args:
            dataset_dict: Dictionary of arrays or nested dictionaries of arrays
            seed: Random seed for sampling
        """
        self.dataset_dict = dataset_dict
        self.dataset_len = _check_lengths(dataset_dict)
        
        # Initialize random state
        self._np_random = None
        self._seed = None
        if seed is not None:
            self.seed(seed)

    @property
    def np_random(self) -> np.random.RandomState:
        if self._np_random is None:
            self.seed()
        return self._np_random

    def seed(self, seed: Optional[int] = None) -> list:
        """Set random seed for sampling."""
        self._np_random, self._seed = seeding.np_random(seed)
        return [self._seed]

    def __len__(self) -> int:
        return self.dataset_len

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
    ) -> Dict:
        """
        Sample a batch of data.
        
        Args:
            batch_size: Number of samples to draw
            keys: Optional keys to sample from
            indx: Optional specific indices to sample
            device: Optional device to place tensors on
            
        Returns:
            Dictionary of sampled data
        """
        if indx is None:
            if hasattr(self.np_random, "integers"):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)

        batch = {}
        if keys is None:
            keys = self.dataset_dict.keys()

        for k in keys:
            if isinstance(self.dataset_dict[k], dict):
                batch[k] = _sample(self.dataset_dict[k], indx)
            else:
                data = self.dataset_dict[k][indx]
                if device is not None and isinstance(data, torch.Tensor):
                    data = data.to(device)
                batch[k] = data

        return batch

    def split(self, ratio: float) -> Tuple["Dataset", "Dataset"]:
        """
        Split dataset into train and test sets.
        
        Args:
            ratio: Fraction of data to use for training
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        assert 0 < ratio < 1
        index = np.arange(len(self), dtype=np.int32)
        self.np_random.shuffle(index)
        
        split_idx = int(self.dataset_len * ratio)
        train_index = index[:split_idx]
        test_index = index[split_idx:]

        train_dataset_dict = _subselect(self.dataset_dict, train_index)
        test_dataset_dict = _subselect(self.dataset_dict, test_index)
        return Dataset(train_dataset_dict), Dataset(test_dataset_dict)

    def _trajectory_boundaries_and_returns(self) -> Tuple[list, list, list]:
        """Get episode boundaries and returns."""
        episode_starts = [0]
        episode_ends = []
        episode_returns = []
        episode_return = 0

        for i in range(len(self)):
            episode_return += self.dataset_dict["rewards"][i]

            if self.dataset_dict["dones"][i]:
                episode_returns.append(episode_return)
                episode_ends.append(i + 1)
                if i + 1 < len(self):
                    episode_starts.append(i + 1)
                episode_return = 0.0

        return episode_starts, episode_ends, episode_returns

    def filter(self, take_top: Optional[float] = None, threshold: Optional[float] = None):
        """
        Filter trajectories based on returns.
        
        Args:
            take_top: Optional percentage of top trajectories to keep
            threshold: Optional minimum return threshold
        """
        assert (take_top is None) != (threshold is None), "Specify exactly one of take_top or threshold"

        starts, ends, returns = self._trajectory_boundaries_and_returns()

        if take_top is not None:
            threshold = np.percentile(returns, 100 - take_top)

        bool_indx = np.full((len(self),), False, dtype=bool)
        for i, (start, end, ret) in enumerate(zip(starts, ends, returns)):
            if ret >= threshold:
                bool_indx[start:end] = True

        self.dataset_dict = _subselect(self.dataset_dict, bool_indx)
        self.dataset_len = _check_lengths(self.dataset_dict)

    def normalize_returns(self, scaling: float = 1000):
        """Normalize rewards by return range and scale."""
        _, _, returns = self._trajectory_boundaries_and_returns()
        self.dataset_dict["rewards"] /= np.max(returns) - np.min(returns)
        self.dataset_dict["rewards"] *= scaling
