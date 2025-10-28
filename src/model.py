from stable_baselines3 import DQN
import torch

def make_dqn_model(env, dataset_size, device="cpu", verbose=0):
    policy_kwargs = dict(
        net_arch=[256, 256, 128],
        activation_fn=torch.nn.ReLU
    )

    if dataset_size > 100_000:
        hyperparams = {
            "learning_rate": 1e-3,  # Higher: was 5e-4—faster updates
            "buffer_size": 100_000,  # Halve: was 200k—less memory, still effective
            "learning_starts": 1000,  # Lower: was 5k—start training sooner
            "batch_size": 256,  # Double: was 128—better GPU utilization
            "train_freq": 8,  # Less frequent: was 4—tradeoff for speed
            "gradient_steps": 4,  # Halve: was 8—fewer updates per collect
            "exploration_initial_eps": 1.0,
            "exploration_fraction": 0.2,  # Shorter: was 0.3—less random steps
            "exploration_final_eps": 0.05,
            "target_update_interval": 500,  # More frequent: was 1000—stabilizes faster
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

    return DQN("MlpPolicy", env, verbose=verbose, device=device, **hyperparams)
