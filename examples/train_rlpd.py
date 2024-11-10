#!/usr/bin/env python3

import glob
import time
import torch
import numpy as np
import tqdm
from absl import app, flags
import os
import copy
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

# Define flags
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_boolean("debug", False, "Debug mode.")

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

def print_green(x: str):
    return print("\033[92m {}\033[00m".format(x))

##############################################################################

def actor(agent, data_store, intvn_data_store, env):
    """Actor loop implementation."""
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        # Load checkpoint
        checkpoint = torch.load(
            os.path.join(FLAGS.checkpoint_path, f"checkpoint_{FLAGS.eval_checkpoint_step}.pt"),
            map_location=device
        )
        agent.state.model.load_state_dict(checkpoint['model_state_dict'])
        agent.state.model.eval()

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            
            while not done:
                with torch.no_grad():
                    actions = agent.sample_actions(
                        observations=obs,
                        argmax=False,
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
        return

    # Continue with regular actor loop...
    # (Rest of actor implementation)

##############################################################################

def learner(agent, replay_buffer, demo_buffer, wandb_logger=None):
    """Learner loop implementation."""
    # Find latest checkpoint
    start_step = 0
    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        checkpoints = glob.glob(os.path.join(FLAGS.checkpoint_path, "checkpoint_*.pt"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            start_step = checkpoint['step'] + 1
            agent.state.model.load_state_dict(checkpoint['model_state_dict'])
            agent.state.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}

    # Create server and start training loop...
    # (Rest of learner implementation)

##############################################################################

def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    
    # Set random seeds
    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Create environment
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

    # Create agent based on setup mode
    if config.setup_mode == 'single-arm-fixed-gripper' or config.setup_mode == 'dual-arm-fixed-gripper':   
        agent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            device=device,
        )
        include_grasp_penalty = False
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            device=device,
        )
        include_grasp_penalty = True
    elif config.setup_mode == 'dual-arm-learned-gripper':
        agent = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            device=device,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # Rest of main implementation...
    # (Continue with buffer creation, training setup, etc.)

if __name__ == "__main__":
    app.run(main)
