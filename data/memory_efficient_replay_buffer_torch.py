import copy
from typing import Iterable, Optional, Tuple
import torch
import gymnasium as gym
import numpy as np
from data.dataset_torch import DatasetDict, _sample
from data.replay_buffer_torch import ReplayBuffer
from gymnasium.spaces import Box

class MemoryEfficientReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        pixel_keys: Tuple[str, ...] = ("pixels",),
        include_next_actions: Optional[bool] = False,
        include_grasp_penalty: Optional[bool] = False,
        device: str = "cpu"  # Store data on CPU by default
    ):
        self.pixel_keys = pixel_keys
        self.device = torch.device(device)

        observation_space = copy.deepcopy(observation_space)
        self._num_stack = 0
        for pixel_key in self.pixel_keys:
            pixel_obs_space = observation_space.spaces[pixel_key]
            if self._num_stack==0:
                self._num_stack = pixel_obs_space.shape[0]
            else:
                assert self._num_stack == pixel_obs_space.shape[0]
            self._unstacked_dim_size = pixel_obs_space.shape[-1]
            low = pixel_obs_space.low[0]
            high = pixel_obs_space.high[0]
            unstacked_pixel_obs_space = Box(
                low=low, high=high, dtype=pixel_obs_space.dtype
            )
            observation_space.spaces[pixel_key] = unstacked_pixel_obs_space

        next_observation_space_dict = copy.deepcopy(observation_space.spaces)
        for pixel_key in self.pixel_keys:
            next_observation_space_dict.pop(pixel_key)
        next_observation_space = gym.spaces.Dict(next_observation_space_dict)

        self._first = True
        self._is_correct_index = torch.full((capacity,), False, dtype=torch.bool, device=self.device)

        super().__init__(
            observation_space,
            action_space,
            capacity,
            next_observation_space=next_observation_space,
            include_next_actions=include_next_actions,
            include_grasp_penalty=include_grasp_penalty,
            device=device
        )

    def insert(self, data_dict: DatasetDict):
        if self._insert_index == 0 and self._capacity == len(self) and not self._first:
            indxs = torch.arange(len(self) - self._num_stack, len(self), device=self.device)
            for indx in indxs:
                element = super().sample(1, indx=indx)
                self._is_correct_index[self._insert_index] = False
                super().insert(element)

        data_dict = copy.deepcopy(data_dict)
        
        # Convert numpy arrays to torch tensors if needed
        if isinstance(data_dict["observations"], dict):
            for k, v in data_dict["observations"].items():
                if isinstance(v, np.ndarray):
                    data_dict["observations"][k] = torch.from_numpy(v).to(self.device)
        if isinstance(data_dict["next_observations"], dict):
            for k, v in data_dict["next_observations"].items():
                if isinstance(v, np.ndarray):
                    data_dict["next_observations"][k] = torch.from_numpy(v).to(self.device)

        obs_pixels = {}
        next_obs_pixels = {}
        for pixel_key in self.pixel_keys:
            obs_pixels[pixel_key] = data_dict["observations"].pop(pixel_key)
            next_obs_pixels[pixel_key] = data_dict["next_observations"].pop(pixel_key)

        if self._first:
            for i in range(self._num_stack):
                for pixel_key in self.pixel_keys:
                    data_dict["observations"][pixel_key] = obs_pixels[pixel_key][i]

                self._is_correct_index[self._insert_index] = False
                super().insert(data_dict)

        for pixel_key in self.pixel_keys:
            data_dict["observations"][pixel_key] = next_obs_pixels[pixel_key][-1]

        self._first = data_dict["dones"]

        self._is_correct_index[self._insert_index] = True
        super().insert(data_dict)

        for i in range(self._num_stack):
            indx = (self._insert_index + i) % len(self)
            self._is_correct_index[indx] = False

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[torch.Tensor] = None,
        pack_obs_and_next_obs: bool = False,
    ) -> dict:
        """Samples from the replay buffer.

        Args:
            batch_size: Minibatch size.
            keys: Keys to sample.
            indx: Take indices instead of sampling.
            pack_obs_and_next_obs: whether to pack img and next_img into one image.
                It's useful when they have overlapping frames.

        Returns:
            A dictionary of batched data.
        """
        if indx is None:
            indx = torch.randint(len(self), (batch_size,), device=self.device)
            
            for i in range(batch_size):
                while not self._is_correct_index[indx[i]]:
                    indx[i] = torch.randint(len(self), (1,), device=self.device)
        else:
            raise NotImplementedError()

        if keys is None:
            keys = self.dataset_dict.keys()
        else:
            assert "observations" in keys

        keys = list(keys)
        keys.remove("observations")
        batch = super().sample(batch_size, keys, indx)

        obs_keys = self.dataset_dict["observations"].keys()
        obs_keys = list(obs_keys)
        for pixel_key in self.pixel_keys:
            obs_keys.remove(pixel_key)

        batch["observations"] = {}
        for k in obs_keys:
            batch["observations"][k] = _sample(
                self.dataset_dict["observations"][k], indx
            )

        for pixel_key in self.pixel_keys:
            obs_pixels = self.dataset_dict["observations"][pixel_key]
            # Convert to torch tensor if needed
            if isinstance(obs_pixels, np.ndarray):
                obs_pixels = torch.from_numpy(obs_pixels).to(self.device)
                
            # Create sliding window view
            obs_pixels = obs_pixels.unfold(0, self._num_stack + 1, 1)
            obs_pixels = obs_pixels[indx - self._num_stack]
            
            # Transpose from (B, H, W, C, T) to (B, T, H, W, C) to follow convention
            obs_pixels = obs_pixels.permute(0, 4, 1, 2, 3)

            if pack_obs_and_next_obs:
                batch["observations"][pixel_key] = obs_pixels
            else:
                batch["observations"][pixel_key] = obs_pixels[:, :-1, ...]
                if "next_observations" in keys:
                    batch["next_observations"][pixel_key] = obs_pixels[:, 1:, ...]

        return batch 