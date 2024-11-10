import os
import pickle as pkl
import requests
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

import imageio
import torch
import numpy as np
import wandb
from typing import Dict, Any


def ask_for_frame(images_dict: Dict[int, np.ndarray]) -> int:
    """Display images and ask user to select first success frame."""
    # Create a new figure
    fig, axes = plt.subplots(5, 5, figsize=(15, 20))
    
    # Flatten the axes array for easier indexing
    axes = axes.flatten()
    for i, (idx, img) in enumerate(images_dict.items()):
        # Display the image
        axes[i].imshow(img)
        
        # Remove axis ticks
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        
        # Overlay the index number
        axes[i].text(10, 30, str(idx), color='white', fontsize=12, 
                     bbox=dict(facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.show(block=False)

    while True:
        try:
            first_success = int(input("First success frame number: "))
            assert first_success in images_dict.keys()
            break
        except:
            continue

    plt.close(fig)
    return first_success


def concat_batches(offline_batch: Dict, online_batch: Dict, axis: int = 1) -> Dict:
    """Concatenate offline and online batches."""
    batch = defaultdict(list)

    for k, v in offline_batch.items():
        if isinstance(v, dict):
            batch[k] = concat_batches(offline_batch[k], online_batch[k], axis=axis)
        else:
            if isinstance(v, torch.Tensor):
                batch[k] = torch.cat((v, online_batch[k]), dim=axis)
            else:
                batch[k] = np.concatenate((v, online_batch[k]), axis=axis)

    return batch


def load_recorded_video(video_path: str) -> wandb.Video:
    """Load and prepare video for wandb logging."""
    video = imageio.mimread(video_path, "MP4")
    video = np.array(video).transpose((0, 3, 1, 2))
    assert video.shape[1] == 3, "Numpy array should be (T, C, H, W)"
    return wandb.Video(video, fps=20)


def _unpack(batch: Dict) -> Dict:
    """
    Helps to minimize CPU to GPU transfer.
    Assuming that if next_observation is missing, it's combined with observation.
    
    Args:
        batch: A batch of data from the replay buffer
        
    Returns:
        Unpacked batch with separated observations
    """
    batch = batch.copy()
    
    for pixel_key in batch["observations"].keys():
        if pixel_key not in batch["next_observations"]:
            obs_pixels = batch["observations"][pixel_key][:, :-1, ...]
            next_obs_pixels = batch["observations"][pixel_key][:, 1:, ...]

            obs = batch["observations"].copy()
            obs[pixel_key] = obs_pixels
            next_obs = batch["next_observations"].copy()
            next_obs[pixel_key] = next_obs_pixels
            
            batch["observations"] = obs
            batch["next_observations"] = next_obs

    return batch


def load_resnet10_params(agent: Any, image_keys: tuple = ("image",), public: bool = True) -> Any:
    """
    Load pretrained resnet10 params from github release to an agent.
    
    Args:
        agent: The agent to load parameters into
        image_keys: Keys for image observations
        public: Whether to load from public release
        
    Returns:
        Agent with pretrained resnet10 params
    """
    file_name = "resnet10_params.pkl"
    if not public:  # if github repo is not public, load from local file
        with open(file_name, "rb") as f:
            encoder_params = pkl.load(f)
    else:  # when repo is released, download from url
        # Construct the full path to the file
        file_path = os.path.expanduser("~/.serl/")
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path = os.path.join(file_path, file_name)
        
        # Check if the file exists
        if os.path.exists(file_path):
            print(f"The ResNet-10 weights already exist at '{file_path}'.")
        else:
            url = f"https://github.com/rail-berkeley/serl/releases/download/resnet10/{file_name}"
            print(f"Downloading file from {url}")

            # Streaming download with progress bar
            try:
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get("content-length", 0))
                block_size = 1024  # 1 Kibibyte
                t = tqdm(total=total_size, unit="iB", unit_scale=True)
                with open(file_path, "wb") as f:
                    for data in response.iter_content(block_size):
                        t.update(len(data))
                        f.write(data)
                t.close()
                if total_size != 0 and t.n != total_size:
                    raise Exception("Error, something went wrong with the download")
            except Exception as e:
                raise RuntimeError(e)
            print("Download complete!")

        with open(file_path, "rb") as f:
            encoder_params = pkl.load(f)

    # Convert numpy arrays to torch tensors
    encoder_params = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v 
                     for k, v in encoder_params.items()}

    param_count = sum(p.numel() for p in encoder_params.values() if isinstance(p, torch.Tensor))
    print(f"Loaded {param_count/1e6}M parameters from ResNet-10 pretrained on ImageNet-1K")

    # Update agent parameters
    state_dict = agent.state.model.state_dict()
    
    for image_key in image_keys:
        encoder_prefix = f"modules_actor.encoder.encoder_{image_key}.pretrained_encoder."
        for k, v in encoder_params.items():
            if encoder_prefix + k in state_dict:
                state_dict[encoder_prefix + k] = v
                print(f"replaced {k} in pretrained_encoder")
    
    agent.state.model.load_state_dict(state_dict)
    return agent
