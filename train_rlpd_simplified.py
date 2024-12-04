#!/usr/bin/env python3

# # For learner mode
# python train_rlpd_sim_torch.py --exp_name my_experiment --learner --demo_path path1 path2 --checkpoint_path ./checkpoints

# # For actor mode
# python train_rlpd_sim_torch.py --exp_name my_experiment --actor --ip localhost

# # For evaluation
# python train_rlpd_sim_torch.py --exp_name my_experiment --eval_checkpoint_step 1000 --eval_n_trajs 10
import glob
import time
import torch
import numpy as np
import tqdm
import argparse
import os
import copy
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

from sac_torch import SACAgent
from serl_launcher.utils.timer_utils import Timer
from train_utils_torch import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from launcher_torch import (
    make_sac_state_agent,
    make_sac_pixel_agent,
    make_trainer_config,
    make_wandb_logger,
)
from data_store_torch import MemoryEfficientReplayBufferDataStore

from mappings import CONFIG_MAPPING
import mujoco.viewer

def parse_args():
    parser = argparse.ArgumentParser(description='Train RLPD with PyTorch')
    
    # Basic settings
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Name of experiment corresponding to folder')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Role settings
    parser.add_argument('--learner', action='store_true',
                        help='Whether this is a learner')
    parser.add_argument('--actor', action='store_true', 
                        help='Whether this is an actor')
    
    # Network settings
    parser.add_argument('--ip', type=str, default='localhost',
                        help='IP address of the learner')
    
    # Data settings
    parser.add_argument('--demo_path', type=str, nargs='+',
                        help='Path(s) to the demo data')
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to save checkpoints')
    
    # Evaluation settings
    parser.add_argument('--eval_checkpoint_step', type=int, default=0,
                        help='Step to evaluate the checkpoint')
    parser.add_argument('--eval_n_trajs', type=int, default=0,
                        help='Number of trajectories to evaluate')
    
    # Misc settings
    parser.add_argument('--save_video', action='store_true',
                        help='Save video')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode (will disable wandb logging)')
    
    args = parser.parse_args()
    return args

def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

def actor(agent, data_store, intvn_data_store, env, args, device="cuda"):
    """Actor loop implementation"""
    if args.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        # Load checkpoint
        checkpoint = torch.load(
            os.path.join(args.checkpoint_path, f"checkpoint_{args.eval_checkpoint_step}.pt")
        )
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()

        for episode in range(args.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            
            while not done:
                with torch.no_grad():
                    actions = agent.sample_actions(
                        observations={k: torch.as_tensor(v, device=device) for k, v in obs.items()},
                        argmax=False
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
                    print(f"{success_counter}/{episode + 1}")

        print(f"Success rate: {success_counter / args.eval_n_trajs}")
        print(f"Average time: {np.mean(time_list)}")
        return

    # Training actor setup
    start_step = (
        int(os.path.basename(natsorted(glob.glob(os.path.join(args.checkpoint_path, "buffer/*.pkl")))[-1])[12:-4]) + 1
        if args.checkpoint_path and os.path.exists(args.checkpoint_path)
        else 0
    )

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        args.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    def update_params(params):
        """Update agent parameters from server"""
        agent.load_state_dict({k: torch.as_tensor(v, device=device) for k, v in params.items()})

    client.recv_network_callback(update_params)

    transitions = []
    demo_transitions = []
    obs, _ = env.reset()
    done = False

    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < config.random_steps:
                actions = env.action_space.sample()
            else:
                with torch.no_grad():
                    actions = agent.sample_actions(
                        observations={k: torch.as_tensor(v, device=device) for k, v in obs.items()},
                        argmax=False
                    )
                actions = actions.cpu().numpy()

        # Environment step
        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)

            if "left" in info: info.pop("left")
            if "right" in info: info.pop("right")

            if "intervene_action" in info:
                actions = info.pop("intervene_action")
                print("intervened!!!")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False

            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            if 'grasp_penalty' in info:
                transition['grasp_penalty'] = info['grasp_penalty']

            data_store.insert(transition)
            transitions.append(copy.deepcopy(transition))
            if already_intervened:
                intvn_data_store.insert(transition)
                demo_transitions.append(copy.deepcopy(transition))

            obs = next_obs
            if done or truncated:
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                stats = {"environment": info}
                client.request("send-stats", stats)
                pbar.set_description(f"last return: {running_return}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                client.update()
                obs, _ = env.reset()

        # Save buffer periodically
        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            buffer_path = os.path.join(args.checkpoint_path, "buffer")
            demo_buffer_path = os.path.join(args.checkpoint_path, "demo_buffer")
            os.makedirs(buffer_path, exist_ok=True)
            os.makedirs(demo_buffer_path, exist_ok=True)
            
            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(transitions, f)
                transitions = []
            with open(os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(demo_transitions, f)
                demo_transitions = []

        timer.tock("total")

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats) 

def learner(agent, replay_buffer, demo_buffer, wandb_logger=None, args=None, device="cuda"):
    """Learner loop implementation"""
    start_step = (
        int(os.path.basename(natsorted(glob.glob(os.path.join(args.checkpoint_path, "checkpoint_*.pt")))[-1])[11:-3]) + 1
        if args.checkpoint_path and os.path.exists(args.checkpoint_path)
        else 0
    )
    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    # Wait for replay buffer to fill
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)
    pbar.close()

    # Send initial network to actor
    server.publish_network({k: v.cpu().numpy() for k, v in agent.state_dict().items()})
    print_green("sent initial network to actor")

    # Setup iterators for 50/50 sampling from demo and online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=device,
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=device,
    )

    timer = Timer()
    
    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = {"critic"}
        train_networks_to_update = {"critic", "actor", "temperature"}
    else:
        train_critic_networks_to_update = {"critic", "grasp_critic"}
        train_networks_to_update = {"critic", "grasp_critic", "actor", "temperature"}

    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        # Run n-1 critic updates and 1 critic + actor update
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )

        # Publish updated network
        if step > 0 and step % config.steps_per_update == 0:
            torch.cuda.synchronize()  # Ensure all operations are complete
            server.publish_network({k: v.cpu().numpy() for k, v in agent.state_dict().items()})

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if step > 0 and config.checkpoint_period and step % config.checkpoint_period == 0:
            torch.save({
                'step': step,
                'model_state_dict': agent.state_dict(),
            }, os.path.join(args.checkpoint_path, f'checkpoint_{step}.pt'))

def main():
    # Parse arguments
    args = parse_args()
    global config
    config = CONFIG_MAPPING[args.exp_name]()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    assert args.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=False,
        save_video=args.save_video,
        classifier=False,
    )
    env = RecordEpisodeStatistics(env)
    
    if config.setup_mode == 'single-arm-fixed-gripper' or config.setup_mode == 'dual-arm-fixed-gripper':   
        agent = make_sac_pixel_agent(
            seed=args.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            device=device
        )
        include_grasp_penalty = False
    elif config.setup_mode == 'state_agent':
        agent = make_sac_state_agent(
            seed=args.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            discount=config.discount,
            device=device
        )
        include_grasp_penalty = False
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    agent = agent.to(device)

    if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
        input("Checkpoint path already exists. Press Enter to resume training.")
        checkpoint = torch.load(
            os.path.join(args.checkpoint_path, "latest_checkpoint.pt"),
            map_location=device
        )
        agent.load_state_dict(checkpoint['model_state_dict'])
        ckpt_number = checkpoint['step']
        print_green(f"Loaded previous checkpoint at step {ckpt_number}.")

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
            device=device
        )
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=args.exp_name,
            debug=args.debug,
        )
        return replay_buffer, wandb_logger

    if args.learner:
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
            device=device
        )

        # Load demo data
        assert args.demo_path is not None
        for path in args.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                        transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        # Load previous buffers if they exist
        if args.checkpoint_path is not None:
            if os.path.exists(os.path.join(args.checkpoint_path, "buffer")):
                for file in glob.glob(os.path.join(args.checkpoint_path, "buffer/*.pkl")):
                    with open(file, "rb") as f:
                        transitions = pkl.load(f)
                        for transition in transitions:
                            replay_buffer.insert(transition)
                print_green(f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}")

            if os.path.exists(os.path.join(args.checkpoint_path, "demo_buffer")):
                for file in glob.glob(os.path.join(args.checkpoint_path, "demo_buffer/*.pkl")):
                    with open(file, "rb") as f:
                        transitions = pkl.load(f)
                        for transition in transitions:
                            demo_buffer.insert(transition)
                print_green(f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}")

        print_green("starting learner loop")
        learner(agent, replay_buffer, demo_buffer, wandb_logger, args, device)

    elif args.actor:
        data_store = QueuedDataStore(50000)
        intvn_data_store = QueuedDataStore(50000)

        print_green("starting actor loop")
        actor(agent, data_store, intvn_data_store, env, args, device)

    else:
        raise NotImplementedError("Must be either a learner or an actor")

if __name__ == "__main__":
    main()