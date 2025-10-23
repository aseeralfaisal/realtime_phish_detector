from stable_baselines3 import DQN

def make_dqn_model(env, dataset_size, device="cpu"):
    if dataset_size > 100_000: 
        return DQN(
            "MlpPolicy", env, verbose=1, device=device,
            learning_rate=5e-4,
            buffer_size=150_000,
            learning_starts=10_000,
            batch_size=256, 
            train_freq=4, gradient_steps=4,
            exploration_initial_eps=1.0,
            exploration_fraction=0.2, 
            exploration_final_eps=0.1,
            target_update_interval=500,
        )
    else:  
        return DQN(
            "MlpPolicy", env, verbose=1, device=device,
            learning_rate=1e-3, 
            buffer_size=50_000, 
            learning_starts=2_000, 
            batch_size=64, 
            train_freq=2, gradient_steps=2,
            exploration_initial_eps=1.0,
            exploration_fraction=0.1, 
            exploration_final_eps=0.2,
            target_update_interval=250, 
        )
