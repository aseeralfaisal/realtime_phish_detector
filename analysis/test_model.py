import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from src.phishing_env import PhishEnv
import argparse

args = argparse.ArgumentParser()
args.add_argument("--mode", type=str, choices=["url", "html", "dom"], help="Choose url or html", required=True)
args.add_argument("--test-type", type=str, choices=["external", "internal"], help="Choose external or internal", required=True)
args = args.parse_args()
parser = argparse.ArgumentParser()
mode = args.mode 
test_type = args.test_type

csv_path = f"./data/{mode}_content.csv" if test_type == "internal" else f"./data/extracted.csv"

df = pd.read_csv(csv_path)
env = PhishEnv(df, mode=mode)
model = DQN.load(f"./trained_models/{mode}_dqn_model.zip", env=env, device="cuda")

results = []
len_data = len(env.data)
range_len = range(len_data)

correct = 0
for idx in range_len:
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