from stable_baselines3 import DQN
import torch

def make_dqn_model(env, dataset_size, device="cpu", verbose=0):
    policy_kwargs = dict(
        net_arch=[256, 256, 128],
        activation_fn=torch.nn.ReLU,
        dueling=True
    )

    if dataset_size > 100_000:
        hyperparams = {
            "learning_rate": 1e-3, 
            "buffer_size": 100_000,  
            "learning_starts": 1000,
            "batch_size": 256,
            "train_freq": 8,
            "gradient_steps": 4,
            "exploration_initial_eps": 1.0,
            "exploration_fraction": 0.2,
            "exploration_final_eps": 0.05,
            "target_update_interval": 500, 
            "policy_kwargs": policy_kwargs,
            "double_q": True,
        }
    else:
        hyperparams = {
            "learning_rate": 7e-4,
            "buffer_size": 100_000,
            "learning_starts": 2_000,
            "batch_size": 128,
            "train_freq": 4,
            "gradient_steps": 4,
            "exploration_initial_eps": 1.0,
            "exploration_fraction": 0.3,
            "exploration_final_eps": 0.05,
            "target_update_interval": 750,
            "policy_kwargs": policy_kwargs,
            "double_q": True,
        }

    return DQN("MlpPolicy", env, verbose=verbose, device=device, **hyperparams)
