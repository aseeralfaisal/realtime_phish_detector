import gymnasium as gym
import numpy as np
from src.bert import get_bert_embedding
from tqdm import tqdm

BERT_DIM = 768

class PhishEnv(gym.Env):
    def __init__(self, data, mode="url"):
        super(PhishEnv, self).__init__()
        
        self.mode = mode
        self.labels = data["label"].values.astype(np.int32)
        
        text_col_name = "URL"

        if text_col_name not in data.columns:
            raise ValueError(f"Required column '{text_col_name}' not found in input data.")
            
        self.raw_text_data = data[text_col_name].values
        
        columns_to_drop = ["label", "DegitRatioInURL", text_col_name]
        self.data = data.drop(columns=columns_to_drop, errors='ignore')
        
        if mode == "url":
            self.int_columns = self.data.select_dtypes(include=[np.integer]).columns
        else:
            self.int_columns = self.data.select_dtypes(include=[np.number]).columns
            
        self.data = self.data[self.int_columns]
        
        self.n_numerical_features = len(self.int_columns)
        self.TOTAL_OBS_DIM = self.n_numerical_features + BERT_DIM
        
        self.feature_max = self.data.max()
        self.feature_min = self.data.min()
        self.data = ((self.data - self.feature_min) / (self.feature_max - self.feature_min)).fillna(0).astype(np.float32)
        self.n_features = len(self.int_columns)
        
        print("Precomputing BERT embeddings...")
        self.bert_embeddings = np.array([get_bert_embedding(text) for text in tqdm(self.raw_text_data)])
        print(f"Precomputed {len(self.bert_embeddings)} embeddings (shape: {self.bert_embeddings.shape})")
        
        self.observation_space = gym.spaces.Box(
            low=np.full(self.TOTAL_OBS_DIM, -np.inf, dtype=np.float32),
            high=np.full(self.TOTAL_OBS_DIM, np.inf, dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)
        self.current_state = 0
    
    def get_state_for_testing(self, index):
        numerical_state = self.data.iloc[index].values.astype(np.float32)
        bert_embedding = self.bert_embeddings[index]
        state = np.concatenate([numerical_state, bert_embedding])
        return state

    def get_state(self):
        numerical_state = self.data.iloc[self.current_state].values.astype(np.float32)
        bert_embedding = self.bert_embeddings[self.current_state]
        state = np.concatenate([numerical_state, bert_embedding])
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
