import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

class HarmonicLayer(nn.Module):
    """
    Harmonic Loss Layer - Direct implementation of the paper's harmonic loss.
    
    Instead of: logits = W @ x (cross-entropy)
    Uses: harmonic_logits = ||x - W_i||² (distance to class centers)
    
    Key properties:
    - Scale invariant
    - Finite convergence point
    - Weights correspond to class centers
    - Better interpretability
    """
    def __init__(self, in_features: int, out_features: int, harmonic_exponent: Optional[float] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize class center weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
        # Set harmonic exponent following paper: n ~ sqrt(D) where D is embedding dimension
        if harmonic_exponent is None:
            import math
            # Clamp harmonic exponent to prevent numerical issues
            self.harmonic_exponent = min(math.sqrt(in_features), 10.0)  # Cap at 10
        else:
            self.harmonic_exponent = min(harmonic_exponent, 10.0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute harmonic distances and return logits using paper-exact HarMax formula
        
        Args:
            x: Input tensor [batch_size, in_features]
            
        Returns:
            logits: Harmonic logits [batch_size, out_features] 
        """
        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(1)  # [batch_size, 1, in_features]
        weight_expanded = self.weight.unsqueeze(0)  # [1, out_features, in_features]
        
        # Compute squared distances: di = ||wi - x||²
        distances = torch.sum((x_expanded - weight_expanded) ** 2, dim=2)
        
        # Paper-exact HarMax formula: pi = 1/di^n / Σj(1/dj^n)
        # Use log-space computation to prevent numerical overflow
        epsilon = 1e-8
        distances = torch.clamp(distances, min=epsilon)
        
        # Compute log(1/di^n) = -n * log(di) for numerical stability
        log_inv_distances_n = -self.harmonic_exponent * torch.log(distances)
        
        # Use log-sum-exp trick for numerical stability
        # log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
        max_log_inv = torch.max(log_inv_distances_n, dim=1, keepdim=True)[0]
        log_sum_exp = max_log_inv + torch.log(torch.sum(torch.exp(log_inv_distances_n - max_log_inv), dim=1, keepdim=True))
        
        # Compute log probabilities: log(pi) = log(1/di^n) - log(sum(1/dj^n))
        log_probabilities = log_inv_distances_n - log_sum_exp
        
        return log_probabilities

class HarmonicLoss(nn.Module):
    """
    Complete harmonic loss implementation matching the paper
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, harmonic_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-entropy loss to harmonic logits
        (The loss function itself is unchanged - only the logit computation differs)
        """
        return F.cross_entropy(harmonic_logits, targets)

class HarmonicMLP(nn.Module):
    """
    Simple MLP with harmonic output layer for demonstration
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, use_harmonic: bool = True):
        super().__init__()
        
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.use_harmonic = use_harmonic
        if use_harmonic:
            self.output = HarmonicLayer(hidden_dim, num_classes)
            self.loss_fn = HarmonicLoss()
        else:
            self.output = nn.Linear(hidden_dim, num_classes)
            self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.hidden(x)
        return self.output(features)
    
    def compute_loss(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return self.loss_fn(logits, targets)

if __name__ == "__main__":
    print("Harmonic Loss Paper Implementation")
    print("Based on: https://arxiv.org/html/2502.01628v1")
    
    # Test the implementation
    model = HarmonicMLP(8, 32, 3, use_harmonic=True)
    test_data = torch.randn(10, 8)
    test_labels = torch.randint(0, 3, (10,))
    
    output = model(test_data)
    loss = model.compute_loss(test_data, test_labels)
    
    print(f"Test successful - Output shape: {output.shape}, Loss: {loss.item():.4f}")
    print(f"Class centers shape: {model.output.weight.shape}")
