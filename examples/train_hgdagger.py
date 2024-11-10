#!/usr/bin/env python3

import time
import torch
from natsort import natsorted
import numpy as np
import tqdm
from absl import app, flags
import os
import copy
import glob
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from pynput import keyboard

from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.utils.timer_utils import Timer
from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore
from serl_launcher.utils.launcher import (
    make_bc_agent,
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
flags.DEFINE_string("demo_buffer_path", None, "Path to folder of demo buffers.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_integer("pretrain_steps", 20_000, "Number of pretraining steps.")
flags.DEFINE_boolean("debug", False, "Debug mode.")

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

def print_green(x: str):
    return print("\033[92m {}\033[00m".format(x))

def print_yellow(x: str):
    return print("\033[93m {}\033[00m".format(x))

should_reset = False

def on_press(key):
    global should_reset
    if key == keyboard.Key.esc:
        should_reset = True
        print("ESC pressed. Resetting...")

##############################################################################

def actor(agent: BCAgent, data_store, env):
    """Actor loop implementation."""
    if FLAGS.eval_checkpoint_step:
        global should_reset
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
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

                if done or should_reset:
                    should_reset = False
                    if reward:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print(dt)
                    success_counter += reward
                    print(reward)
                    print(f"{success_counter}/{episode + 1}")
                    break

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")
        return

    # Regular actor loop
    start_step = 0
    if FLAGS.checkpoint_path and os.path.exists(os.path.join(FLAGS.checkpoint_path, "demo_buffer")):
        buffer_files = glob.glob(os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl"))
        if buffer_files:
            latest_file = natsorted(buffer_files)[-1]
            start_step = int(os.path.basename(latest_file)[12:-4]) + 1

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_store,
        wait_for_server=True,
        timeout_ms=3000,
    )

    def update_params(params):
        """Update agent parameters from server."""
        agent.state.model.load_state_dict(params)

    client.recv_network_callback(update_params)

    # Training loop implementation
    demo_transitions = []
    obs, _ = env.reset()
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")

        # Sample actions
        with timer.context("sample_actions"):
            with torch.no_grad():
                actions = agent.sample_actions(
                    observations=obs,
                    device=device,
                    argmax=False,
                )
            actions = actions.cpu().numpy()

        # Step environment
        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)
            
            # Clean info dict
            info.pop('left', None)
            info.pop('right', None)

            # Handle interventions
            if "intervene_action" in info:
                actions = info.pop("intervene_action")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False

            # Create transition
            transition = {
                'observations': obs,
                'actions': actions,
                'next_observations': next_obs,
                'rewards': np.float32(reward),
                'masks': 1.0 - done,
                'dones': done,
            }

            if already_intervened:
                data_store.insert(transition)
                demo_transitions.append(copy.deepcopy(transition))

            obs = next_obs
            running_return += reward

            # Handle episode end
            if done or truncated:
                info['episode']['intervention_count'] = intervention_count
                info['episode']['intervention_steps'] = intervention_steps
                client.request("send-stats", {"environment": info})
                pbar.set_description(f"last return: {running_return}")
                
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                
                client.update()
                obs, _ = env.reset()

        # Save buffer periodically
        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            os.makedirs(demo_buffer_path, exist_ok=True)
            
            with open(os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(demo_transitions, f)
            demo_transitions = []

        timer.tock("total")

        # Log stats
        if step % config.log_period == 0:
            client.request("send-stats", {"timer": timer.get_average_times()})

def learner(agent: BCAgent, demo_buffer, wandb_logger=None):
    """Learner loop implementation."""
    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", demo_buffer)
    server.start(threaded=True)

    # Send initial network to actor
    server.publish_network(agent.state.model.state_dict())
    print_green("sent initial network to actor")

    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=device,
    )

    # Pretrain BC policy
    update_step = 0
    if FLAGS.pretrain_steps:
        checkpoint_path = os.path.join(FLAGS.checkpoint_path, f"checkpoint_{FLAGS.pretrain_steps}.pt")
        
        if os.path.exists(checkpoint_path):
            print_green(f"BC checkpoint at {FLAGS.pretrain_steps} steps found, restoring BC checkpoint")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            agent.state.model.load_state_dict(checkpoint['model_state_dict'])
            agent.state.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            update_step = FLAGS.pretrain_steps
        else:
            update_step = 0
            print_yellow(f"No BC checkpoint at {FLAGS.pretrain_steps} steps found, starting from scratch")
            
            for step in tqdm.tqdm(range(FLAGS.pretrain_steps), dynamic_ncols=True, desc="bc_pretraining"):
                update_step += 1
                batch = next(demo_iterator)
                update_info = agent.update(batch)
                
                if update_step % config.log_period == 0 and wandb_logger:
                    wandb_logger.log({"bc": update_info}, step=update_step)
                    
            # Save checkpoint
            torch.save({
                'step': update_step,
                'model_state_dict': agent.state.model.state_dict(),
                'optimizer_state_dict': agent.state.optimizer.state_dict(),
            }, checkpoint_path)
            print_green("bc pretraining done and saved checkpoint")

    # Send updated network to actor
    server.publish_network(agent.state.model.state_dict())

    # Training loop
    timer = Timer()
    for step in tqdm.tqdm(range(FLAGS.pretrain_steps + 1, config.max_steps), dynamic_ncols=True, desc="learner"):
        with timer.context("train"):
            batch = next(demo_iterator)
            update_info = agent.update(batch)

        # Publish updated network
        if step > 0 and step % config.steps_per_update == 0:
            server.publish_network(agent.state.model.state_dict())

        # Log stats
        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        # Save checkpoint
        if step > 0 and config.checkpoint_period and step % config.checkpoint_period == 0:
            torch.save({
                'step': step,
                'model_state_dict': agent.state.model.state_dict(),
                'optimizer_state_dict': agent.state.optimizer.state_dict(),
            }, os.path.join(FLAGS.checkpoint_path, f"checkpoint_{step}.pt"))

def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'

    # Set random seeds
    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Create environment
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=False,
        classifier=FLAGS.actor
    )
    env = RecordEpisodeStatistics(env)

    # Create agent
    agent: BCAgent = make_bc_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        device=device,
    )

    if FLAGS.learner:
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )
        
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=50000,
            image_keys=config.image_keys,
        )

        assert FLAGS.demo_path is not None or FLAGS.demo_buffer_path is not None

        # Load demo buffer data
        if FLAGS.demo_buffer_path:
            for file in glob.glob(os.path.join(FLAGS.demo_buffer_path, "*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}")

        # Load additional demos
        if FLAGS.demo_path:
            for path in FLAGS.demo_path:
                with open(path, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print(f"demo buffer size: {len(demo_buffer)}")

        # Start learner loop
        print_green("starting learner loop")
        learner(agent, demo_buffer, wandb_logger)

    elif FLAGS.actor:
        data_store = QueuedDataStore(50000)  # Queue size on actor
        actor(agent, data_store, env)

    else:
        raise NotImplementedError("Must be either a learner or an actor")

if __name__ == "__main__":
    app.run(main)
