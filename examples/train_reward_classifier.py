import glob
import os
import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from absl import app, flags

from serl_launcher.data.data_store import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier
from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 150, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")

def main(_):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)

    # Create buffer for positive transitions
    pos_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=20000,
        include_label=True,
    )

    # Load positive examples
    success_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*success*.pkl"))
    for path in success_paths:
        with open(path, "rb") as f:
            success_data = pkl.load(f)
            for trans in success_data:
                if "images" in trans['observations'].keys():
                    continue
                trans["labels"] = 1
                trans['actions'] = env.action_space.sample()
                pos_buffer.insert(trans)
            
    pos_iterator = pos_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=device,
    )
    
    # Create buffer for negative transitions
    neg_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=50000,
        include_label=True,
    )

    # Load negative examples
    failure_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*failure*.pkl"))
    for path in failure_paths:
        with open(path, "rb") as f:
            failure_data = pkl.load(f)
            for trans in failure_data:
                if "images" in trans['observations'].keys():
                    continue
                trans["labels"] = 0
                trans['actions'] = env.action_space.sample()
                neg_buffer.insert(trans)
            
    neg_iterator = neg_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=device,
    )

    print(f"failed buffer size: {len(neg_buffer)}")
    print(f"success buffer size: {len(pos_buffer)}")

    # Create initial samples
    pos_sample = next(pos_iterator)
    neg_sample = next(neg_iterator)
    sample = concat_batches(pos_sample, neg_sample, axis=0)

    # Create classifier and optimizer
    torch.manual_seed(0)
    classifier = create_classifier(
        sample["observations"], 
        config.classifier_keys,
        device=device
    )
    optimizer = optim.Adam(classifier.parameters())
    criterion = nn.BCEWithLogitsLoss()

    def data_augmentation_fn(observations):
        """Apply data augmentation to observations."""
        for pixel_key in config.classifier_keys:
            observations = observations.copy()
            observations[pixel_key] = batched_random_crop(
                observations[pixel_key], 
                padding=4, 
                num_batch_dims=2
            )
        return observations

    def train_step(model, batch, optimizer):
        """Single training step."""
        model.train()
        optimizer.zero_grad()
        
        logits = model(batch["observations"])
        loss = criterion(logits, batch["labels"].float())
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            predictions = (torch.sigmoid(logits) >= 0.5)
            accuracy = (predictions == batch["labels"]).float().mean()
            
        return loss.item(), accuracy.item()

    # Training loop
    for epoch in tqdm(range(FLAGS.num_epochs)):
        # Sample equal number of positive and negative examples
        pos_sample = next(pos_iterator)
        neg_sample = next(neg_iterator)
        
        # Merge and create labels
        batch = concat_batches(pos_sample, neg_sample, axis=0)
        
        # Apply data augmentation
        obs = data_augmentation_fn(batch["observations"])
        batch["observations"] = obs
        batch["labels"] = batch["labels"].unsqueeze(-1)
        
        # Training step
        train_loss, train_accuracy = train_step(classifier, batch, optimizer)

        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
        )

    # Save checkpoint
    checkpoint_path = os.path.join(os.getcwd(), "classifier_ckpt/")
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save({
        'epoch': FLAGS.num_epochs,
        'model_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(checkpoint_path, f"checkpoint_{FLAGS.num_epochs}.pt"))

if __name__ == "__main__":
    app.run(main)