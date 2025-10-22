import os
import pandas as pd
from env import PhishEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import time

def make_dqn_model(env, dataset_size, device="cuda"):
    if dataset_size > 100_000: 
        return DQN(
            "MlpPolicy", env, verbose=1, device=device,
            learning_rate=1e-4, buffer_size=120_000,
            learning_starts=5000, batch_size=512,
            train_freq=4, gradient_steps=4,
            exploration_initial_eps=1.0,
            exploration_fraction=0.15,
            exploration_final_eps=0.05,
            target_update_interval=1000,
        )
    else:  
        return DQN(
            "MlpPolicy", env, verbose=1, device=device,
            learning_rate=3e-4, buffer_size=20_000,
            learning_starts=1000, batch_size=128,
            train_freq=2, gradient_steps=2,
            exploration_initial_eps=1.0,
            exploration_fraction=0.05,
            exploration_final_eps=0.2,
            target_update_interval=500,
        )

def load_process_data(mode="url"):
    if mode == "url":
        csv_path = "./data/url_content.csv"
    elif mode == "html":
        csv_path = "./data/html_content.csv"
    elif mode == "dom":
        csv_path = "./data/dom_content.csv"
    
    df = pd.read_csv(csv_path)
    df.drop(columns=["FILENAME"], inplace=True) if mode == "url" else None
    
    print(f"Loaded dataset: {csv_path}\n")
    time.sleep(2)
    
    process_data = PhishEnv(df, mode)  
    return process_data

def training_process(timesteps, type):
    env = load_process_data(mode=type)
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

    if type == "url":
        save_dir = "./models/url_dqn_model"
    elif type == "html":
        save_dir = "./models/html_dqn_model"
    elif type == "dom":
        save_dir = "./models/dom_dqn_model"
        
    os.makedirs("./models", exist_ok=True)
    model.save(save_dir)
    print(f"Trained Model Saved to {save_dir}")
    
if __name__ == "__main__":
    type = "url"
    steps = 250_000
    training_process(timesteps=steps, type=type)