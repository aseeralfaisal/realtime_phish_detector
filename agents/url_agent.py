import os
import pandas as pd
from env import Environment
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

def load_process_data():
    df = pd.read_csv("./data_set/url_content.csv")
    df.drop(columns=["FILENAME"], inplace=True)
    process_data = Environment(df)
    return process_data

def training_process(timesteps):
    env = load_process_data()
    check_env(env, warn=True)
    
    model = DQN("MlpPolicy", env, verbose=1, device="cuda", exploration_final_eps=0.05)
    model.learn(total_timesteps=timesteps)

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
        
        env.render()
        
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

    save_dir = "./models/dqn_model"
    os.makedirs("./models", exist_ok=True)
    model.save(save_dir)
    print(f"Model saved to {save_dir}")
    
if __name__ == "__main__":
    training_process(timesteps=250_000)