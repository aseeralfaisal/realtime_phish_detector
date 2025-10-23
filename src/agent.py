import os
import pandas as pd
from src.phishing_env import PhishEnv
from stable_baselines3.common.env_checker import check_env
import time
import argparse
from src.model import make_dqn_model

def load_process_data(mode):
    csv_path = f"./data/{mode}_content.csv"
    df = pd.read_csv(csv_path)
    df.drop(columns=["FILENAME"], inplace=True) if mode == "url" else None
    
    print(f"Loaded Data: {csv_path}\n")
    time.sleep(1)
    
    process_data = PhishEnv(df, mode)  
    return process_data

def training_process(timesteps, mode):
    print(f"Training Mode: {mode}\n")
    time.sleep(1)
    
    env = load_process_data(mode=mode)
    check_env(env, warn=True)
    
    model = make_dqn_model(env, dataset_size=len(env.data))
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
        
        print(f"Action: {action} ({'Phishing' if action == 1 else 'Legitimate'})")
        print(f"True Label: {true_label} ({'Phishing' if true_label == 1 else 'Legitimate'})")
        print(f"Reward: {reward}")
        print(f"Current Accuracy: {correct_predictions / total_steps * 100:.2f}%")
        print(f"False Positive Rate: {false_positives / total_steps * 100:.2f}%")
        print(f"False Negative Rate: {false_negatives / total_steps * 100:.2f}%")
        print("\n=====================================")

    accuracy = correct_predictions / total_steps * 100
    false_positive_rate = false_positives / total_steps * 100
    false_negative_rate = false_negatives / total_steps * 100
    
    print("\nFinal Evaluation Metrics:")
    print(f"Total Episodes: {total_steps}")
    print(f"Total Reward: {total_rewards}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"False Positive Rate: {false_positive_rate:.2f}%")
    print(f"False Negative Rate: {false_negative_rate:.2f}%")
    print(f"True Positives: {correct_predictions}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")

    save_dir = f"./models/{mode}_dqn_model"
        
    os.makedirs("./models", exist_ok=True)
    model.save(save_dir)
    print(f"Trained Model Saved to {save_dir}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["dom", "html", "url"], required=True, help="Choose training mode: dom, html, or url")
    args = parser.parse_args()
    steps = {"dom": 30_000, "html": 40_000, "url": 250_000}
    training_process(timesteps=steps[args.mode], mode=args.mode)
