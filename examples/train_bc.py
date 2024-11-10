#!/usr/bin/env python3

import glob
import time
import torch
import numpy as np
import tqdm
from absl import app, flags
import os
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.utils.launcher import (
    make_bc_agent,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from experiments.mappings import CONFIG_MAPPING
from experiments.config import DefaultTrainingConfig

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_string("bc_checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_integer("train_steps", 20_000, "Number of pretraining steps.")
flags.DEFINE_bool("save_video", False, "Save video of the evaluation.")
flags.DEFINE_boolean("debug", False, "Debug mode.")  # debug mode will disable wandb logging

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

def print_green(x: str):
    return print("\033[92m {}\033[00m".format(x))

def print_yellow(x: str):
    return print("\033[93m {}\033[00m".format(x))

##############################################################################

def eval(
    env,
    bc_agent: BCAgent,
    device: torch.device,
):
    """Evaluation loop for the agent."""
    success_counter = 0
    time_list = []
    
    for episode in range(FLAGS.eval_n_trajs):
        obs, _ = env.reset()
        done = False
        start_time = time.time()
        
        while not done:
            with torch.no_grad():
                actions = bc_agent.sample_actions(
                    observations=obs,
                    device=device
                )
            actions = actions.cpu().numpy()
            next_obs, reward, done, truncated, info = env.step(actions)
            obs = next_obs
            
            if done:
                if reward:
                    dt = time.time() - start_time
                    time_list.append(dt)
                    print(dt)
                success_counter += reward
                print(reward)
                print(f"{success_counter}/{episode + 1}")

    print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
    print(f"average time: {np.mean(time_list)}")

##############################################################################

def train(
    bc_agent: BCAgent,
    bc_replay_buffer,
    config: DefaultTrainingConfig,
    device: torch.device,
    wandb_logger=None,
):
    """Training loop for BC agent."""
    bc_replay_iterator = bc_replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size,
            "pack_obs_and_next_obs": False,
        },
        device=device,
    )
    
    # Pretrain BC policy
    for step in tqdm.tqdm(
        range(FLAGS.train_steps),
        dynamic_ncols=True,
        desc="bc_pretraining",
    ):
        batch = next(bc_replay_iterator)
        update_info = bc_agent.update(batch)
        
        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log({"bc": update_info}, step=step)
            
        if step > FLAGS.train_steps - 100 and step % 10 == 0:
            checkpoint = {
                'model_state_dict': bc_agent.state.model.state_dict(),
                'optimizer_state_dict': bc_agent.state.optimizer.state_dict(),
                'step': step,
            }
            torch.save(
                checkpoint,
                os.path.join(FLAGS.bc_checkpoint_path, f"checkpoint_{step}.pt")
            )
            
    print_green("bc pretraining done and saved checkpoint")

##############################################################################

def main(_):
    config: DefaultTrainingConfig = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    
    eval_mode = FLAGS.eval_n_trajs > 0
    env = config.get_environment(
        fake_env=not eval_mode,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

    # Set random seeds
    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FLAGS.seed)

    bc_agent: BCAgent = make_bc_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        device=device,
    )

    if not eval_mode:
        checkpoint_path = os.path.join(FLAGS.bc_checkpoint_path, f"checkpoint_{FLAGS.train_steps}.pt")
        assert not os.path.exists(checkpoint_path)

        bc_replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
        )

        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )

        demo_path = glob.glob(os.path.join(os.getcwd(), "demo_data", "*.pkl"))
        assert demo_path, "No demonstration data found"

        for path in demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if np.linalg.norm(transition['actions']) > 0.0:
                        bc_replay_buffer.insert(transition)
        print(f"bc replay buffer size: {len(bc_replay_buffer)}")

        print_green("starting learner loop")
        train(
            bc_agent=bc_agent,
            bc_replay_buffer=bc_replay_buffer,
            wandb_logger=wandb_logger,
            config=config,
            device=device,
        )

    else:
        checkpoint = torch.load(
            os.path.join(FLAGS.bc_checkpoint_path, "checkpoint_latest.pt"),
            map_location=device
        )
        bc_agent.state.model.load_state_dict(checkpoint['model_state_dict'])
        bc_agent.state.model.eval()

        print_green("starting actor loop")
        eval(
            env=env,
            bc_agent=bc_agent,
            device=device,
        )

if __name__ == "__main__":
    app.run(main)
