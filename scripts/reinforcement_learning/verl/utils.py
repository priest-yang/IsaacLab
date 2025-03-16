

def load_pi0_policy(policy_path, batch, meta_path):
    import pickle

    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = policy_path

    ds_meta = pickle.load(open(meta_path, "rb"))
    policy = make_policy(cfg, ds_meta=meta_path)

    # policy = torch.compile(policy, mode="reduce-overhead")
    warmup_iters = 10
    benchmark_iters = 30

    # Warmup
    for _ in range(warmup_iters):
        torch.cuda.synchronize()
        policy.select_action(batch)
        policy.reset()
        torch.cuda.synchronize()

    

def prepare_inference_batch_pi0(obs, rewards, dones, infos):
    """
    Prepare a batch of observations, rewards, dones, and infos for inference.
    """
    
    batch = {}
    
    batch["observation.image.agentview_left"] = obs["agentview_left_rgb"]
    batch["observation.image.agentview_right"] = obs["agentview_right_rgb"]
    batch["observation.image.eye_in_hand"] = obs["eye_in_hand_rgb"]
    batch['observation.state'] = torch.cat([obs['joint_pos'], obs['joint_vel'], obs['object_position_in_robot_frame'], obs['target_object_position_in_robot_frame']], dim=1)
    
    batch['task'] = "Lift the cube"

    batch["rewards"] = rewards
    batch["dones"] = dones
    batch["infos"] = infos

