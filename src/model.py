from stable_baselines3 import DQN
import torch

def make_dqn_model(env, dataset_size, device="cpu"):
    policy_kwargs = dict(
        net_arch=[256, 256, 128],
        activation_fn=torch.nn.ReLU
    )

    if dataset_size > 100_000:
        hyperparams = {
            "learning_rate": 5e-4,
            "buffer_size": 200_000,
            "learning_starts": 5_000,
            "batch_size": 128,
            "train_freq": 4,
            "gradient_steps": 8,
            "exploration_initial_eps": 1.0,
            "exploration_fraction": 0.3,
            "exploration_final_eps": 0.05,
            "target_update_interval": 1000,
            "policy_kwargs": policy_kwargs,
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
        }

    return DQN("MlpPolicy", env, verbose=1, device=device, **hyperparams)
