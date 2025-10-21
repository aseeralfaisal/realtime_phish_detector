import gymnasium as gym
import numpy as np


class Environment(gym.Env):
    def __init__(self, data):
        super(Environment, self).__init__()
        
        self.labels = data["label"].values.astype(np.int64)
        self.data = data.drop(columns=["label"])
        self.int_columns = self.data.select_dtypes(include=[np.integer]).columns
        self.data = self.data[self.int_columns]
        
        self.feature_max = self.data.max()
        self.feature_min = self.data.min()
        self.data = ((self.data - self.feature_min) / (self.feature_max - self.feature_min)).astype(np.float32)
        self.n_features = len(self.int_columns)
        
        self.observation_space = gym.spaces.Box(
            low=np.zeros(self.n_features, dtype=np.float32),
            high=np.ones(self.n_features, dtype=np.float32),
            dtype=np.float32
        )
        
        self.action_space = gym.spaces.Discrete(2)
        self.current_state = 0

    def get_state(self):
        state = self.data.iloc[self.current_state].values.astype(np.float32)
        return state

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_state = np.random.randint(0, len(self.data))
        return self.get_state(), {}

    def step(self, action):
        true_label = self.labels[self.current_state]

        if action == true_label:
            reward = 1
        else:
            reward = -2 if true_label == 1 else -0.5  

        terminated = True
        truncated = False

        next_state = self.get_state()

        info = {
            "current_state": self.current_state,
            "reward": reward,
            "terminated": terminated,
            "true_label": true_label
        }

        return next_state, reward, terminated, truncated, info

    def render(self, mode="human"):
        if mode == "human":
            if self.current_state < len(self.data):
                state = self.get_state()
                label = self.labels[self.current_state]
                print("\n=== Environment State ===")
                print(f"Step: {self.current_state + 1}/{len(self.data)}")
                print(f"Values: {state}")
                print(f"True Label: {label} ({'Phishing' if label == 1 else 'Legitimate'})")
            else:
                print("\n[!] End of data. No more states to render.\n")