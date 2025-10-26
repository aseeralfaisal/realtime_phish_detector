import gymnasium as gym
import numpy as np

class PhishEnv(gym.Env):
    def __init__(self, data, mode="url"):
        super(PhishEnv, self).__init__()
        
        self.mode = mode
        self.labels = data["label"].values.astype(np.int32)
        self.data = data.drop(columns=["label", "DegitRatioInURL"])
        
        if mode == "url":
            self.int_columns = self.data.select_dtypes(include=[np.integer]).columns
        else:
            self.int_columns = self.data.select_dtypes(include=[np.number]).columns
            
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
        
        if np.random.rand() < 0.5:
            indices = np.where(self.labels == 0)[0]
        else:
            indices = np.where(self.labels == 1)[0]
            
        self.current_state = np.random.choice(indices)
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
