# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
# from utils import prepare_inference_batch_pi0, load_pi0_policy


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-FrankaOmron-Camera-v0", help="Name of the task.")
# parser.add_argument("--task", type=str, default="Isaac-Cartpole-RGB-v0", help="Name of the task.")

parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)


args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
args_cli.enable_cameras = True # for vla training
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    num_steps_per_env = agent_cfg.to_dict()["num_steps_per_env"]


    # # create runner from rsl-rl
    # runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # # write git state to logs
    # runner.add_git_repo_to_log(__file__)
    # # load the checkpoint
    # if agent_cfg.resume:
    #     print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    #     # load previously trained model
    #     runner.load(resume_path)

    # # dump the configuration into log-directory
    # dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    # dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    # dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    # dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    # runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    
    import pandas as pd
    # all_actions = pd.read_csv('/home/shaoze.yang/IsaacLab/logs/rsl_rl/franka_lift/2025-04-14_13-17-44/action_trajectories/actions_env_0.csv')

    
    for step in range(1):
        # step to initial state
        # actions = all_actions.iloc[1, :].to_numpy()
        actions = torch.zeros(env.num_envs, env.num_actions)
        # actions = torch.from_numpy(actions)
        # actions = actions.expand(env.num_envs, -1)
        obs, rewards, dones, infos = env.step(actions.to(env.device))

    all_states = []

    step = 1
    env.reset()
    while step < 230:
        # start = time.time()
        # Rollout
        with torch.inference_mode():
            for i in range(num_steps_per_env):
                step += 1
                # actions = runner.alg.act(obs, critic_obs)
                # pesudo actions
                actions = torch.zeros(env.num_envs, env.num_actions)

                # actions = all_actions.iloc[step, :].to_numpy()
                # actions = torch.from_numpy(actions)
                # actions = actions.expand(env.num_envs, -1)
                obs, rewards, dones, infos = env.step(actions.to(env.device))

                joint_pos = obs['joint_pos'][0, :]
                joint_pos[0], joint_pos[1] = joint_pos[1].clone(), joint_pos[0].clone()
                all_states.append(joint_pos.cpu().numpy())


                # breakpoint()

                # # prepare batch for pi0
                # batch = prepare_inference_batch_pi0(obs, rewards, dones, infos)

                if isinstance(obs, dict):
                    import cv2
                    import numpy as np
                    
                    # Process all environments instead of just env_id=0
                    for env_id in range(env.num_envs):
                        # Collect images and their keys
                        images_with_keys = []
                        for key, value in obs.items():
                            if 'rgb' in key:
                                img = value[env_id].cpu().numpy()
                                # Convert from [C, H, W] to [H, W, C] if needed
                                if img.shape[0] == 3:
                                    img = img.transpose(1, 2, 0)
                                # Convert to uint8 range if in [0,1] float range
                                if img.max() <= 1.0:
                                    img = (img * 255).astype(np.uint8)
                                images_with_keys.append((key, img))
                        
                        if images_with_keys:
                            # Define display parameters
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.8
                            font_color = (255, 255, 255)  # White
                            font_thickness = 2
                            caption_height = 30  # Height for the caption area
                            
                            # Get sample image dimensions
                            sample_img = images_with_keys[0][1]
                            img_height, img_width = sample_img.shape[0], sample_img.shape[1]
                            
                            # Define a reasonable display width per image
                            target_width = min(img_width, 400)  # Max 400px width per image
                            scale_factor = target_width / img_width
                            display_height = int(img_height * scale_factor)
                            display_width = target_width
                            
                            # Create a canvas for all images
                            total_width = display_width * len(images_with_keys)
                            canvas = np.zeros((display_height + caption_height, total_width, 3), dtype=np.uint8)
                            
                            # Place each image with its caption
                            for idx, (key, img) in enumerate(images_with_keys):
                                # Resize image
                                resized_img = cv2.resize(img, (display_width, display_height))
                                
                                # Calculate position
                                x_offset = idx * display_width
                                
                                # Place image on canvas
                                canvas[0:display_height, x_offset:x_offset+display_width] = resized_img
                                
                                # Add caption
                                text_size = cv2.getTextSize(key, font, font_scale, font_thickness)[0]
                                text_x = x_offset + (display_width - text_size[0]) // 2  # Center text
                                text_y = display_height + caption_height - 10  # Position at bottom of caption area
                                
                                # Add background for text
                                cv2.rectangle(canvas, 
                                             (x_offset, display_height), 
                                             (x_offset + display_width, display_height + caption_height), 
                                             (0, 0, 0), 
                                             -1)  # Filled rectangle
                                
                                # Draw text
                                cv2.putText(canvas, key, (text_x, text_y), font, font_scale, font_color, font_thickness)
                            
                            # Save the canvas to video
                            # You need to initialize VideoWriter outside this loop
                            if 'video_writers' not in locals():
                                video_writers = {}
                            
                            # Create video writer for each environment if doesn't exist yet
                            if env_id not in video_writers:
                                os.makedirs(os.path.join(log_dir, "videos", "train"), exist_ok=True)
                                video_path = os.path.join(log_dir, "videos", "train", f"env_{env_id}.mp4")
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                video_writers[env_id] = cv2.VideoWriter(
                                    video_path, fourcc, 10.0, (total_width, display_height + caption_height)
                                )
                            
                            # Write frame to video
                            video_writers[env_id].write(canvas)
                


                # # move to the right device
                # obs, rewards, dones = (
                #     obs.to(env.device),
                #     rewards.to(env.device),
                #     dones.to(env.device),
                # )
                # perform normalization
                # obs = runner.obs_normalizer(obs)

                # # process the step
                # runner.alg.process_env_step(rewards, dones, infos)

            #stop = time.time()
            #collection_time = stop - start

            # Learning step
            # start = stop
            # runner.alg.compute_returns(critic_obs)


    # After the training loop, before env.close()
    if 'video_writers' in locals():
        for writer in video_writers.values():
            writer.release()
        
        print("Video writers released")

    # close the simulator
    env.close()

    # save all states to csv
    all_states = np.array(all_states)
    pd.DataFrame(all_states).to_csv(os.path.join(log_dir, "states.csv"), index=False)
    # all_actions.to_csv(os.path.join(log_dir, "actions.csv"), index=False)

    from lerobot.common.policies.pi0.isaac_utils import plot_action_trajectories
    # plot_action_trajectories(os.path.join(log_dir, "actions.csv"), os.path.join(log_dir, "states.csv"), os.path.join(log_dir))


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
