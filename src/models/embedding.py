from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class SemanticVectorizer:
    def __init__(self, model_name='intfloat/multilingual-e5-large-instruct', device="cpu"):
        self.device = device
        print(f"Loading {model_name} into {device} (float32 for CPU stability)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Using float32. float16 can cause RuntimeError on standard CPUs.
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to(self.device)
        self.model.eval()

    def average_pool(self, last_hidden_states, attention_mask):
        """Mean pooling operation neutralizing padding tokens."""
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode(self, text: str):
        # Strict adherence to E5 prefixing mandate
        formatted_text = f"query: {text}"
        
        # Tokenize and strictly truncate to 512 context window
        batch_dict = self.tokenizer(
            [formatted_text], 
            max_length=512, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**batch_dict)
            
        # Apply mean pooling
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        
        # Normalize the embedding
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Return as a flat Python list of floats
        return embeddings[0].numpy().tolist()