from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
import torch



def load_pi0_policy(policy_path, batch, meta_path):
    import pickle

    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = policy_path

    ds_meta = pickle.load(open(meta_path, "rb"))
    policy = make_policy(cfg, ds_meta=ds_meta)

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

    
    batch["observation.images.agentview_left"] = obs["agentview_left_rgb"].permute(0, 3, 1, 2)
    batch["observation.images.agentview_right"] = obs["agentview_right_rgb"].permute(0, 3, 1, 2)
    batch["observation.images.eye_in_hand"] = obs["eye_in_hand_rgb"].permute(0, 3, 1, 2)

    # !handle gripper mismatch between robocasa and pi0
    joint_pos = obs['joint_pos']
    joint_pos[:, -1] = joint_pos[:, -1] * -1
    joint_vel = obs['joint_vel']
    joint_vel[:, -1] = joint_vel[:, -1] * -1

    batch['observation.state'] = torch.cat([joint_pos, joint_vel], dim=1)


    # batch["rewards"] = rewards
    # batch["dones"] = dones
    # batch["infos"] = infos
    batch["task"] = ["Lift the cube"]

    return batch

