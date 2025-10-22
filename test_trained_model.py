import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from env import Environment
from sklearn.metrics import accuracy_score

train_data = pd.read_csv("./data_set/phiusiil.csv")
train_env = Environment(train_data)

test_data = pd.read_csv("output.csv")  
test_env = Environment(test_data)

model = DQN.load("./models/dqn_model", env=train_env, device="cuda")

def evaluate_model(env, data, model):
    int_columns = env.int_columns
    obs = data[int_columns]
    obs = ((obs - env.feature_min) / (env.feature_max - env.feature_min)).astype(np.float32)
    obs = obs.values

    true_labels = data["label"].values
    predictions = []

    for observation in obs:
        action, _ = model.predict(observation.reshape(1, -1), deterministic=True)
        predicted_action = action[0] if isinstance(action, np.ndarray) else action
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