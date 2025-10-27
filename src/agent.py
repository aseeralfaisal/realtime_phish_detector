import os
import pandas as pd
from src.phishing_env import PhishEnv
from stable_baselines3.common.env_checker import check_env
import argparse
from src.model import make_dqn_model

def load_process_data(mode):
    csv_path = f"./data/{mode}_content.csv"
    df = pd.read_csv(csv_path)
    df.drop(columns=["FILENAME"], inplace=True) if mode == "url" else None
    print(f"Loaded Data: {csv_path}")
    process_data = PhishEnv(df, mode)  
    return process_data

def training_process(timesteps, mode, verbose, device):
    env = load_process_data(mode=mode)
    print(f"Training Mode: {mode}\n")
    check_env(env, warn=True)
    
    model = make_dqn_model(env, dataset_size=len(env.data), verbose=verbose, device=device)
    model.learn(total_timesteps=timesteps, progress_bar=True)

    total_rewards = 0
    correct_predictions = 0
    false_positives = 0
    false_negatives = 0
    total_steps = 0
    obs, _ = env.reset()
    terminated = truncated = False
    
    while not (terminated or truncated):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_rewards += reward
        true_label = info["true_label"]
        
        if action == true_label:
            correct_predictions += 1
        elif action == 1 and true_label == 0:
            false_positives += 1
        elif action == 0 and true_label == 1:
            false_negatives += 1
            
        total_steps += 1

    save_dir = f"./trained_models/{mode}_dqn_model"
    os.makedirs("./trained_models", exist_ok=True)
    model.save(save_dir)
    print(f"Trained Model Saved to {save_dir}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["dom", "html", "url"], required=True, help="Choose training mode: dom, html, or url")
    parser.add_argument("--verbose", action="store_const", const=1, default=0, help="Enable verbosity for each step")
    args = parser.parse_args()
    mode = args.mode
    verbose = args.verbose
    
    steps = {"url": 206_000, "html": 150_000}
    training_process(timesteps=steps[mode], mode=mode, verbose=verbose, device="cuda")
