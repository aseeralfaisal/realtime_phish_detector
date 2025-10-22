import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from env import PhishEnv

data = pd.read_csv("./data/dom_content.csv")
env = PhishEnv(data)

model = DQN.load("./models/dom_dqn_model", env=env, device="cuda")

index = 15_000
obs = env.data.iloc[index].values.astype(np.float32)
true_label = env.labels[index]

action, _ = model.predict(obs, deterministic=True)

print(f"Feature Vector (normalized):\n{obs}")
print(f"\nPredicted Action: {action} ({'Phishing' if action == 0 else 'Legitimate'})")
print(f"True Label: {true_label} ({'Phishing' if true_label == 0 else 'Legitimate'})")
print("\n===============================")
