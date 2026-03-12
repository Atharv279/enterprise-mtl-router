import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    mean_absolute_error, 
    mean_squared_error, 
    r2_score
)

from src.models.embedding import SemanticVectorizer
from src.models.mtl_network import ComplaintMTLNetwork

def evaluate_model():
    device = torch.device("cpu")
    print("Initializing Comprehensive Evaluation Framework...")

    # 1. Resolve Paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, "data", "synthetic", "training_data.json")
    weights_path = os.path.join(project_root, "data", "processed", "mtl_weights.pth")

    # 2. Load Data (Using a subset for quick evaluation)
    print(f"Loading data from: {data_path}")
    df = pd.read_json(data_path).sample(n=200, random_state=42) # Evaluate on 200 random samples
    
    # Extract Ground Truth
    y_true_priority = df['predicted_priority'].values - 1 # 0-indexed
    y_true_eta = df['estimated_eta'].values

    # 3. Load Models
    print("Loading Semantic Vectorizer and MTL Weights...")
    vectorizer = SemanticVectorizer(device=device)
    mtl_net = ComplaintMTLNetwork().to(device)
    mtl_net.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    mtl_net.eval()

    # 4. Generate Predictions
    print("Generating predictions (This will take a moment on CPU)...")
    y_pred_priority = []
    y_pred_eta = []

    with torch.no_grad():
        for text in df['raw_text']:
            # Vectorize
            vec = vectorizer.encode(text)
            tensor_vec = torch.tensor([vec], dtype=torch.float32).to(device)
            
            # Predict
            priority_logits, eta_pred = mtl_net(tensor_vec)
            
            # Store
            y_pred_priority.append(torch.argmax(priority_logits, dim=1).item())
            y_pred_eta.append(eta_pred.item())

    # 5. Calculate Classification Metrics [cite: 179, 180, 184]
    print("\n" + "="*40)
    print("CLASSIFICATION METRICS: PRIORITY HEAD")
    print("="*40)
    acc = accuracy_score(y_true_priority, y_pred_priority)
    # Using macro avg to ensure performance on minority classes is weighted equally [cite: 186, 187]
    precision = precision_score(y_true_priority, y_pred_priority, average='macro', zero_division=0)
    recall = recall_score(y_true_priority, y_pred_priority, average='macro', zero_division=0)
    f1 = f1_score(y_true_priority, y_pred_priority, average='macro', zero_division=0)
    
    print(f"Accuracy:      {acc:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1-Score:  {f1:.4f}")

    # 6. Calculate Regression Metrics [cite: 190, 191]
    print("\n" + "="*40)
    print("REGRESSION METRICS: ETA HEAD")
    print("="*40)
    mae = mean_absolute_error(y_true_eta, y_pred_eta)
    rmse = np.sqrt(mean_squared_error(y_true_eta, y_pred_eta))
    r2 = r2_score(y_true_eta, y_pred_eta)

    print(f"Mean Absolute Error (MAE): {mae:.4f} days")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} days") # Disproportionately penalizes large failures 
    print(f"R-squared (R2): {r2:.4f}") # Percentage of variance successfully explained [cite: 198]
    print("="*40 + "\n")

if __name__ == "__main__":
    evaluate_model()