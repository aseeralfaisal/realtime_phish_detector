import torch
import numpy as np
from transformers import BertTokenizer, BertModel

try:
    TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
    BERT_MODEL = BertModel.from_pretrained("bert-base-uncased")
    BERT_MODEL.eval()
    DEVICE = torch.device("cuda")
    BERT_MODEL.to(DEVICE)
    
except Exception as e:
    print(f"Warning: BERT loading failed. Using mock features. Error: {e}")
    class MockTokenizer:
        def __call__(self, text, **kwargs): return {'input_ids': torch.zeros(1, 10), 'attention_mask': torch.zeros(1, 10)}
    class MockModel:
        def __init__(self): 
            self.DEVICE = 'cuda'
        def eval(self): pass
        def to(self, device): pass
        def __call__(self, **kwargs): 
            return type('MockOutputs', (object,), {'last_hidden_state': torch.rand(1, 10, 768)})()
    TOKENIZER = MockTokenizer()
    BERT_MODEL = MockModel()
    DEVICE = 'cuda'

def get_bert_embedding(text: str) -> np.ndarray:
    if not isinstance(text, str) or not text.strip():
        return np.zeros(768, dtype=np.float32)

    inputs = TOKENIZER(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding='max_length', 
        max_length=128
    )

    with torch.no_grad():
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = BERT_MODEL(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
    
    if cls_embedding.ndim == 1 and cls_embedding.shape[0] == 768:
        return cls_embedding
    else:
        return np.zeros(768, dtype=np.float32)