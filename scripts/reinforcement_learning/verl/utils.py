from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
import torch
import time


def load_pi0_policy(policy_path, batch, meta_path):
    import pickle

    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = policy_path

    ds_meta = pickle.load(open(meta_path, "rb"))

    start = time.time()
    policy = make_policy(cfg, ds_meta=ds_meta)
    end = time.time()
    print(f"Time taken to create policy: {end - start} seconds")

    # policy = torch.compile(policy, mode="reduce-overhead")
    warmup_iters = 10
    benchmark_iters = 30

    # Warmup
    for _ in range(warmup_iters):
        torch.cuda.synchronize()
        policy.select_action(batch)
        policy.reset()
        torch.cuda.synchronize()
    
    return policy

    

def prepare_inference_batch_pi0(obs, rewards, dones, infos):
    """
    Prepare a batch of observations, rewards, dones, and infos for inference.
    """
    
    batch = {}

    
    batch["observation.images.agentview_left"] = obs["agentview_left_rgb"].permute(0, 3, 1, 2) / 255.0
    batch["observation.images.agentview_right"] = obs["agentview_right_rgb"].permute(0, 3, 1, 2) / 255.0
    batch["observation.images.eye_in_hand"] = obs["eye_in_hand_rgb"].permute(0, 3, 1, 2) / 255.0

    # images = images.float() / 255.0
    # mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
    # images -= mean_tensor

    # !handle gripper mismatch between robocasa and pi0

    joint_pos = obs['joint_pos']
    joint_pos[..., -1] = joint_pos[..., -1] * -1
    # joint_vel = obs['joint_vel']
    # joint_vel[:, -1] = joint_vel[:, -1] * -1

    joint_pos[..., 0], joint_pos[..., 1] = joint_pos[..., 1].clone(), joint_pos[..., 0].clone()
    # joint_vel[:, 0], joint_vel[:, 1] = joint_vel[:, 1].clone(), joint_vel[:, 0].clone()

   
    # batch['observation.state'] = torch.cat([joint_pos, joint_vel], dim=1)
    batch['observation.state'] = joint_pos

    # batch["rewards"] = rewards
    # batch["dones"] = dones
    # batch["infos"] = infos
    batch["task"] = ["Pick up the cube"] * obs['joint_pos'].shape[0]


    return batch



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_action_trajectories(csv_path, state_csv_path=None, save_dir=None):
    """
    Plot action trajectories from CSV files where each column is a joint's trajectory.
    
    Args:
        csv_path: Path to the CSV file containing gt_actions
        state_csv_path: Path to the CSV file containing joint positions (optional)
        save_dir: Directory to save plots (if None, will display instead)
    """
    # Load the action CSV file
    df = pd.read_csv(csv_path, header=None)
    
    # Load the state CSV file if provided
    df_state = None
    if state_csv_path:
        df_state = pd.read_csv(state_csv_path, header=None)

        # get after first 10
        df_state = df_state.iloc[10:]
        df = df.iloc[10:]
        
        # Ensure both dataframes have the same length
        min_length = min(len(df), len(df_state))
        df = df.iloc[:min_length]
        df_state = df_state.iloc[:min_length]
    
    # Extract timesteps from the index or first column
    if 'timestep' in df.columns:
        timesteps = df['timestep'].values
        # Remove timestep column to keep only action data
        df = df.drop('timestep', axis=1)
        if df_state is not None and 'timestep' in df_state.columns:
            df_state = df_state.drop('timestep', axis=1)
    else:
        # If no timestep column, use row indices
        timesteps = np.arange(len(df))
    
    # Get number of joints
    num_joints = len(df.columns)
    
    # Determine plot layout
    if num_joints <= 3:
        fig, axes = plt.subplots(num_joints, 1, figsize=(10, 3*num_joints), sharex=True)
        if num_joints == 1:
            axes = [axes]  # Make it iterable
    else:
        # Calculate a reasonable grid layout
        cols = min(3, num_joints)
        rows = (num_joints + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3*rows), sharex=True)
        axes = axes.flatten()
    
    # Create individual plots for each joint
    for i, joint_name in enumerate(df.columns):
        ax = axes[i]
        
        # Plot action trajectory
        action_line = ax.plot(timesteps, df[joint_name], linewidth=2, color='blue', 
                              label='GT Action')
        
        # Plot state trajectory if available
        if df_state is not None and i < len(df_state.columns):
            state_line = ax.plot(timesteps, df_state[i], linewidth=2, color='green', 
                                 linestyle='--', label='Joint Position')
        
        ax.set_title(f'{joint_name} Trajectory')
        ax.set_ylabel('Value')
        ax.grid(True)
        
        # Add horizontal line at y=0 for reference
        ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Add legend
        ax.legend()
        
        # Set reasonable y-axis limits with some padding
        max_val = df[joint_name].abs().max()
        # y_limit = max(1.0, max_val * 1.1)  # At least ±1.0 or 10% larger than max value
        # ax.set_ylim(-y_limit, y_limit)
        # ax.set_ylim(-0.05, 0.05)
    
    # Hide unused subplots if we have a grid
    for j in range(num_joints, len(axes)):
        fig.delaxes(axes[j])
    
    # Add common x-axis label
    fig.text(0.5, 0.04, 'Timestep', ha='center', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.1)
    
    # Add title with filename
    filename = os.path.basename(csv_path)
    title = f'Action Trajectories - {filename}'
    if state_csv_path:
        state_filename = os.path.basename(state_csv_path)
        title += f' vs {state_filename}'
    fig.suptitle(title, fontsize=16, y=1.02)
    
    # Save or display
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(filename)[0]
        save_path = os.path.join(save_dir, f"{base_name}_trajectories.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

