import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from env import Environment

data = pd.read_csv("./data_set/phiusiil.csv")
env = Environment(data)

model = DQN.load("./models/dqn_model", env=env, device="cpu")

index = 31
obs = env.data.iloc[index].values.astype(np.float32)
true_label = env.labels[index]

action, _ = model.predict(obs, deterministic=True)

print("\n=== Testing the 10th URL ===")
print(f"Feature Vector (normalized):\n{obs}")
print(f"\nPredicted Action: {action} ({'Phishing' if action == 1 else 'Legitimate'})")
print(f"True Label: {true_label} ({'Phishing' if true_label == 1 else 'Legitimate'})")
print("\n===============================")
