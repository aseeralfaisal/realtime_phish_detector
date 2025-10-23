import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from src.phishing_env import PhishEnv
import argparse

args = argparse.ArgumentParser()
args.add_argument("--mode", type=str, choices=["url", "html", "dom"], help="Choose the model type to test", required=True)
args = args.parse_args()
mode = args.mode

data = pd.read_csv(f"./data/{mode}_content.csv")
env = PhishEnv(data, mode=mode)
model = DQN.load(f"./trained_models/{mode}_dqn_model.zip", env=env, device="cpu")

results = []
range_len = range(len(env.data))

correct = 0
for idx in range_len:
    if mode == "url":
        obs = env.data.iloc[idx].values.astype(np.int64)
    else:
        obs = env.data.iloc[idx].values.astype(np.float32)
    
    true_label = env.labels[idx]
    action, _ = model.predict(obs, deterministic=True)
    correct += 1 if action == true_label else 0

    results.append({
        "Index": idx,
        "Predicted Action": action,
        "True Label": true_label,
    })

df_results = pd.DataFrame(results)
print(f"{df_results}\n")
print(f"Correct Predictions: {correct} / {len(results)}")