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
from copy import deepcopy

def ask_for_frame(images_dict):    
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

def concat_batches(offline_batch, online_batch, axis=1):
    """Concatenate two batches along specified axis"""
    batch = defaultdict(list)

    if isinstance(offline_batch, dict) and isinstance(online_batch, dict):
        for k, v in offline_batch.items():
            if isinstance(v, dict):
                batch[k] = concat_batches(offline_batch[k], online_batch[k], axis=axis)
            else:
                if isinstance(v, torch.Tensor) and isinstance(online_batch[k], torch.Tensor):
                    batch[k] = torch.cat((v, online_batch[k]), dim=axis)
                elif isinstance(v, np.ndarray) and isinstance(online_batch[k], np.ndarray):
                    batch[k] = np.concatenate((v, online_batch[k]), axis=axis)
                else:
                    raise TypeError(f"Unsupported type for concatenation: {type(v)} and {type(online_batch[k])}")
    return batch

def load_recorded_video(video_path: str):
    """Load and convert video for wandb logging"""
    video = np.array(imageio.mimread(video_path, "MP4")).transpose((0, 3, 1, 2))
    assert video.shape[1] == 3, "Numpy array should be (T, C, H, W)"
    return wandb.Video(video, fps=20)

def _unpack(batch):
    """
    Helps to minimize CPU to GPU transfer.
    Assuming that if next_observation is missing, it's combined with observation
    """
    for pixel_key in batch["observations"].keys():
        if pixel_key not in batch["next_observations"]:
            if isinstance(batch["observations"][pixel_key], torch.Tensor):
                obs_pixels = batch["observations"][pixel_key][:, :-1, ...]
                next_obs_pixels = batch["observations"][pixel_key][:, 1:, ...]
                
                obs = dict(batch["observations"])
                obs[pixel_key] = obs_pixels
                
                next_obs = dict(batch["next_observations"])
                next_obs[pixel_key] = next_obs_pixels
                
                batch = dict(batch)
                batch["observations"] = obs
                batch["next_observations"] = next_obs

    return batch

def load_resnet10_params(agent, image_keys=("image",), public=True):
    """
    Load pretrained resnet10 params from github release to an agent.
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

    # Convert numpy arrays to torch tensors if needed
    encoder_params = {
        k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v 
        for k, v in encoder_params.items()
    }

    param_count = sum(p.numel() for p in encoder_params.values() if isinstance(p, torch.Tensor))
    print(f"Loaded {param_count/1e6}M parameters from ResNet-10 pretrained on ImageNet-1K")

    # Update agent's encoder parameters
    for image_key in image_keys:
        encoder_name = f"encoder_{image_key}"
        if encoder_name in agent.actor.encoder.state_dict():
            encoder_state = agent.actor.encoder.state_dict()
            for k, v in encoder_params.items():
                if k in encoder_state:
                    encoder_state[k].copy_(v)
                    print(f"replaced {k} in pretrained_encoder")
            
            agent.actor.encoder.load_state_dict(encoder_state)
            # Also update critic's encoder if it shares the same architecture
            if hasattr(agent, 'critic') and hasattr(agent.critic, 'encoder'):
                agent.critic.encoder.load_state_dict(encoder_state)

    return agent 