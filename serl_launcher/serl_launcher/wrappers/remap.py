from typing import Any, Dict, Tuple, Union

import gymnasium as gym
import gymnasium.spaces
import torch


class RemapWrapper(gym.ObservationWrapper):
    """
    Remap a dictionary observation space to some other flat structure specified by keys.
    
    Args:
        env: Environment to wrap.
        new_structure: A tuple/dictionary/singleton where leaves are keys in the original observation space.
    """
    def __init__(self, env: gym.Env, new_structure: Union[Tuple, Dict, str]):
        super().__init__(env)
        self.new_structure = new_structure

        # Create new observation space based on structure type
        if isinstance(new_structure, tuple):
            self.observation_space = gym.spaces.Tuple(
                [env.observation_space[v] for v in new_structure]
            )
        elif isinstance(new_structure, dict):
            self.observation_space = gym.spaces.Dict(
                {k: env.observation_space[v] for k, v in new_structure.items()}
            )
        elif isinstance(new_structure, str):
            self.observation_space = env.observation_space[new_structure]
        else:
            raise TypeError(f"Unsupported structure type: {type(new_structure)}")

    def observation(self, observation: Dict) -> Any:
        """
        Remap observation according to new structure.
        
        Args:
            observation: Original observation dictionary
            
        Returns:
            Remapped observation following new_structure
        """
        def _remap(structure: Any) -> Any:
            if isinstance(structure, (tuple, list)):
                return type(structure)([observation[v] for v in structure])
            elif isinstance(structure, dict):
                return {k: observation[v] for k, v in structure.items()}
            elif isinstance(structure, str):
                return observation[structure]
            else:
                raise TypeError(f"Unsupported structure type: {type(structure)}")

        return _remap(self.new_structure)
