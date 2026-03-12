import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
from src.models.mtl_network import ComplaintMTLNetwork
from src.models.embedding import SemanticVectorizer

def prepare_dataloaders(json_path, vectorizer, batch_size=32):
    print(f"Loading synthetic data from: {json_path}")
    df = pd.read_json(json_path)
    
    print("Generating 1024-d embeddings for the training set...")
    print("WARNING: On a CPU, embedding 500 records will take a few minutes. Please wait...")
    embeddings = [vectorizer.encode(text) for text in df['raw_text']]
    
    X = torch.tensor(embeddings, dtype=torch.float32)
    y_priority = torch.tensor(df['predicted_priority'].values - 1, dtype=torch.long)
    y_eta = torch.tensor(df['estimated_eta'].values, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X, y_priority, y_eta)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_network(epochs=15):
    # Force CPU device
    device = torch.device("cpu")
    print(f"Training on: {device}")
    
    # 1. Initialize models (CPU optimized)
    vectorizer = SemanticVectorizer(device=device)
    model = ComplaintMTLNetwork().to(device)
    
    # 2. Dynamic Optimizer
    optimizer = optim.Adam([
        {'params': model.shared_layers.parameters()},
        {'params': model.classifier_head.parameters()},
        {'params': model.regression_head.parameters()},
        {'params': model.log_var_class, 'lr': 1e-3},
        {'params': model.log_var_reg, 'lr': 1e-3}
    ], lr=1e-4)
    
    # 3. Resolve paths dynamically
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, "data", "synthetic", "training_data.json")
    model_output = os.path.join(project_root, "data", "processed", "mtl_weights.pth")
    os.makedirs(os.path.dirname(model_output), exist_ok=True)

    # 4. Load Data
    dataloader = prepare_dataloaders(data_path, vectorizer)
    
    # 5. Execution Loop
    print("Starting PyTorch training loop...")
    model.train()
    for epoch in range(epochs):
        total_epoch_loss = 0
        for batch_x, batch_y_priority, batch_y_eta in dataloader:
            batch_x = batch_x.to(device)
            batch_y_priority = batch_y_priority.to(device)
            batch_y_eta = batch_y_eta.to(device)
            
            optimizer.zero_grad()
            pred_priority, pred_eta = model(batch_x)
            loss = model.compute_joint_loss(pred_priority, batch_y_priority, pred_eta, batch_y_eta)
            
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_epoch_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), model_output)
    print(f"Training complete. Weights saved to {model_output}")

if __name__ == "__main__":
    train_network()