import torch
import torch.nn as nn

class ComplaintMTLNetwork(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512):
        super(ComplaintMTLNetwork, self).__init__()
        
        # Shared Representation Block
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Priority Classification Head (Multi-class: High, Medium, Low)
        self.classifier_head = nn.Linear(hidden_dim // 2, 3) 
        
        # ETA Regression Head (Continuous days)
        self.regression_head = nn.Linear(hidden_dim // 2, 1)
        self.softplus = nn.Softplus() # Strictly enforces non-negativity

        # Learnable parameters for Homoscedastic Task Uncertainty (Log Variances)
        self.log_var_class = nn.Parameter(torch.zeros(1))
        self.log_var_reg = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        shared_rep = self.shared_layers(x) #
        
        # Task 1: Priority Logits
        priority_logits = self.classifier_head(shared_rep)
        
        # Task 2: ETA Prediction
        eta_pred = self.softplus(self.regression_head(shared_rep))
        
        return priority_logits, eta_pred
        
    def compute_joint_loss(self, priority_logits, priority_targets, eta_pred, eta_targets):
        """Calculates dynamic weighted loss."""
        criterion_class = nn.CrossEntropyLoss() #
        criterion_reg = nn.HuberLoss() # Robust against outliers
        
        loss_class = criterion_class(priority_logits, priority_targets)
        loss_reg = criterion_reg(eta_pred, eta_targets)
        
        # Mathematical formulation for dynamic weighting
        precision_class = torch.exp(-self.log_var_class)
        precision_reg = torch.exp(-self.log_var_reg)
        
        total_loss = (precision_class * loss_class + self.log_var_class) + \
                     (precision_reg * loss_reg + self.log_var_reg) #
                     
        return total_loss