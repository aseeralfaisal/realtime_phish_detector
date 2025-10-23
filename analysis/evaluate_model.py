import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from src.phishing_env import PhishEnv
from sklearn.metrics import accuracy_score

train_data = pd.read_csv("./data/phiusiil.csv")
train_env = PhishEnv(train_data)

test_data = pd.read_csv("data/openphish.csv")  
test_env = PhishEnv(test_data)

model = DQN.load("./trained_models/dqn_model", env=train_env, device="cuda")

def evaluate_model(env, data, model):
    int_columns = env.int_columns
    obs = data[int_columns]
    obs = ((obs - env.feature_min) / (env.feature_max - env.feature_min)).astype(np.float32)
    obs = obs.values

    true_labels = data["label"].values
    predictions = []

    for idx, item in enumerate(obs):
        action, _ = model.predict(item.reshape(1, -1), deterministic=True)
        predicted_action = action[0] if isinstance(action, np.ndarray) else action
        print(f"Step {idx+1}: {'Phishing' if predicted_action == 0 else 'Legitimate'}")
        predictions.append(predicted_action)

    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

train_accuracy = evaluate_model(train_env, train_data, model)
print(f"Training Accuracy: {train_accuracy:.2f}")

test_accuracy = evaluate_model(test_env, test_data, model)
print(f"Test Accuracy: {test_accuracy:.2f}")

if train_accuracy > test_accuracy + 0.1:  
    print("The model might be overfitting to the training data.")
else:
    print("The model does not appear to be overfitting.")