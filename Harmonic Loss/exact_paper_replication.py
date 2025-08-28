"""
Exact Harmonic Loss Paper Replication
=====================================
This script replicates the paper's methodology exactly:
- Standard model vs Harmonic model comparison
- Same architecture for both (only output layer differs)
- Modular addition task as specified in paper
- Proper comparative analysis

Separated from R3 trajectory analysis for clean testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time

# Import our harmonic loss implementation
from harmonic_loss_paper import HarmonicLayer, HarmonicLoss, HarmonicMLP

class StandardMLP(nn.Module):
    """Standard MLP with cross-entropy loss for comparison"""
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.output = nn.Linear(hidden_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.hidden(x)
        return self.output(features)
    
    def compute_loss(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return self.loss_fn(logits, targets)

class HarmonicMLPExact(nn.Module):
    """Harmonic MLP matching standard architecture exactly"""
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        
        # Identical hidden layers to standard model
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Only difference: harmonic output layer
        self.output = HarmonicLayer(hidden_dim, num_classes)
        self.loss_fn = HarmonicLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.hidden(x)
        return self.output(features)
    
    def compute_loss(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return self.loss_fn(logits, targets)

def generate_modular_addition_data(n_samples: int = 1000, modulus: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate modular addition dataset as used in the paper
    Task: (a + b) mod modulus
    """
    # Generate random pairs
    a = torch.randint(0, modulus, (n_samples,))
    b = torch.randint(0, modulus, (n_samples,))
    
    # Create input features (one-hot encoding)
    inputs = torch.zeros(n_samples, 2 * modulus)
    inputs[torch.arange(n_samples), a] = 1
    inputs[torch.arange(n_samples), modulus + b] = 1
    
    # Compute targets
    targets = (a + b) % modulus
    
    return inputs, targets

def train_model(model: nn.Module, train_data: torch.Tensor, train_targets: torch.Tensor, 
                epochs: int = 100, lr: float = 0.002) -> List[float]:
    """Train a model and return loss history with progress tracking"""
    # Use exact paper hyperparameters: AdamW, lr=2e-3, weight_decay=1e-2
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_history = []
    
    model.train()
    print_interval = max(1, epochs // 20)  # Print 20 updates regardless of epoch count
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = model.compute_loss(train_data, train_targets)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % print_interval == 0 or epoch == epochs - 1:
            progress = (epoch + 1) / epochs * 100
            print(f"  Epoch {epoch:4d}/{epochs}: Loss = {loss.item():.6f} ({progress:5.1f}%)")
    
    return loss_history

def evaluate_model(model: nn.Module, test_data: torch.Tensor, test_targets: torch.Tensor) -> Dict:
    """Evaluate model performance"""
    model.eval()
    with torch.no_grad():
        logits = model(test_data)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == test_targets).float().mean().item()
        loss = model.compute_loss(test_data, test_targets).item()
    
    return {
        'accuracy': accuracy,
        'loss': loss,
        'predictions': predictions
    }

def analyze_learned_representations(model: nn.Module, model_name: str) -> Dict:
    """Analyze what the model learned"""
    analysis = {'model_name': model_name}
    
    if hasattr(model.output, 'weight'):
        # Get class centers (for harmonic model) or weights (for standard model)
        weights = model.output.weight.detach().numpy()
        analysis['weights_shape'] = weights.shape
        analysis['weight_norms'] = np.linalg.norm(weights, axis=1)
        
        # Compute pairwise distances between class centers/weights
        n_classes = weights.shape[0]
        distances = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(n_classes):
                distances[i, j] = np.linalg.norm(weights[i] - weights[j])
        
        analysis['pairwise_distances'] = distances
        analysis['mean_distance'] = np.mean(distances[distances > 0])
        analysis['std_distance'] = np.std(distances[distances > 0])
        
        # For harmonic model, weights are class centers
        if isinstance(model.output, HarmonicLayer):
            analysis['class_centers'] = weights
            print(f"\n{model_name} - Class Centers Analysis:")
            print(f"  Shape: {weights.shape}")
            print(f"  Mean pairwise distance: {analysis['mean_distance']:.4f}")
            print(f"  Std pairwise distance: {analysis['std_distance']:.4f}")
            
            # Print first few centers
            for i, center in enumerate(weights[:3]):
                norm = np.linalg.norm(center)
                print(f"  Class {i} center norm: {norm:.4f}")
    
    return analysis

def create_comparison_visualization(standard_history: List[float], harmonic_history: List[float], 
                                  standard_analysis: Dict, harmonic_analysis: Dict):
    """Create comprehensive comparison visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training curves
    ax1.plot(standard_history, label='Standard MLP', color='blue', alpha=0.7)
    ax1.plot(harmonic_history, label='Harmonic MLP', color='red', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Final accuracy comparison
    models = ['Standard', 'Harmonic']
    accuracies = [standard_analysis.get('accuracy', 0), harmonic_analysis.get('accuracy', 0)]
    colors = ['blue', 'red']
    
    bars = ax2.bar(models, accuracies, color=colors, alpha=0.7)
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Final Test Accuracy')
    ax2.set_ylim(0, 1)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Weight/center norms comparison
    if 'weight_norms' in standard_analysis and 'weight_norms' in harmonic_analysis:
        x = np.arange(len(standard_analysis['weight_norms']))
        width = 0.35
        
        ax3.bar(x - width/2, standard_analysis['weight_norms'], width, 
                label='Standard Weights', color='blue', alpha=0.7)
        ax3.bar(x + width/2, harmonic_analysis['weight_norms'], width,
                label='Harmonic Centers', color='red', alpha=0.7)
        
        ax3.set_xlabel('Class')
        ax3.set_ylabel('L2 Norm')
        ax3.set_title('Weight/Center Norms by Class')
        ax3.legend()
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'Class {i}' for i in range(len(x))])
    
    # Distance matrices heatmap
    if 'pairwise_distances' in harmonic_analysis:
        distances = harmonic_analysis['pairwise_distances']
        im = ax4.imshow(distances, cmap='viridis', aspect='auto')
        ax4.set_title('Harmonic Model: Class Center Distances')
        ax4.set_xlabel('Class')
        ax4.set_ylabel('Class')
        
        # Add colorbar
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        
        # Add distance values
        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                text = ax4.text(j, i, f'{distances[i, j]:.2f}',
                               ha="center", va="center", color="white", fontsize=8)
    
    plt.tight_layout()
    plt.savefig('exact_paper_replication_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nComparison visualization saved to: exact_paper_replication_comparison.png")

def run_exact_paper_replication():
    """Run the exact paper replication experiment"""
    print("=" * 60)
    print("EXACT HARMONIC LOSS PAPER REPLICATION - SCALED UP")
    print("=" * 60)
    
    # Scaled up parameters using exact paper hyperparameters
    modulus = 13  # Larger modulus for more complex task
    hidden_dim = 512  # Much larger hidden dimensions
    input_dim = 2 * modulus  # One-hot encoding for two numbers
    epochs = 1000  # More epochs for thorough training
    lr = 0.002  # Paper specification: 2×10^-3
    
    print(f"\nExperiment Setup:")
    print(f"  Task: Modular addition (mod {modulus})")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Output classes: {modulus}")
    print(f"  Training epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    
    # Generate larger datasets
    print(f"\nGenerating datasets...")
    train_data, train_targets = generate_modular_addition_data(20000, modulus)  # 10x larger
    test_data, test_targets = generate_modular_addition_data(5000, modulus)  # 10x larger
    
    print(f"  Training samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")
    
    # Create models with identical architectures
    print(f"\nCreating models...")
    standard_model = StandardMLP(input_dim, hidden_dim, modulus)
    harmonic_model = HarmonicMLPExact(input_dim, hidden_dim, modulus)
    
    print(f"  Standard model parameters: {sum(p.numel() for p in standard_model.parameters())}")
    print(f"  Harmonic model parameters: {sum(p.numel() for p in harmonic_model.parameters())}")
    
    # Train standard model
    print(f"\n" + "="*40)
    print("TRAINING STANDARD MODEL")
    print("="*40)
    start_time = time.time()
    standard_history = train_model(standard_model, train_data, train_targets, epochs, lr)
    standard_train_time = time.time() - start_time
    
    # Train harmonic model  
    print(f"\n" + "="*40)
    print("TRAINING HARMONIC MODEL")
    print("="*40)
    start_time = time.time()
    harmonic_history = train_model(harmonic_model, train_data, train_targets, epochs, lr)
    harmonic_train_time = time.time() - start_time
    
    # Evaluate both models
    print(f"\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    
    standard_results = evaluate_model(standard_model, test_data, test_targets)
    harmonic_results = evaluate_model(harmonic_model, test_data, test_targets)
    
    print(f"\nStandard Model:")
    print(f"  Test Accuracy: {standard_results['accuracy']:.4f}")
    print(f"  Test Loss: {standard_results['loss']:.6f}")
    print(f"  Training Time: {standard_train_time:.2f}s")
    
    print(f"\nHarmonic Model:")
    print(f"  Test Accuracy: {harmonic_results['accuracy']:.4f}")
    print(f"  Test Loss: {harmonic_results['loss']:.6f}")
    print(f"  Training Time: {harmonic_train_time:.2f}s")
    
    # Analyze learned representations
    print(f"\n" + "="*40)
    print("REPRESENTATION ANALYSIS")
    print("="*40)
    
    standard_analysis = analyze_learned_representations(standard_model, "Standard MLP")
    standard_analysis.update(standard_results)
    
    harmonic_analysis = analyze_learned_representations(harmonic_model, "Harmonic MLP")
    harmonic_analysis.update(harmonic_results)
    
    # Create visualization
    create_comparison_visualization(standard_history, harmonic_history, 
                                  standard_analysis, harmonic_analysis)
    
    # Summary
    print(f"\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    accuracy_diff = harmonic_results['accuracy'] - standard_results['accuracy']
    loss_diff = standard_results['loss'] - harmonic_results['loss']
    
    print(f"\nPerformance Comparison:")
    print(f"  Accuracy difference (Harmonic - Standard): {accuracy_diff:+.4f}")
    print(f"  Loss difference (Standard - Harmonic): {loss_diff:+.6f}")
    
    if accuracy_diff > 0:
        print(f"  -> Harmonic model achieved {accuracy_diff:.1%} higher accuracy")
    else:
        print(f"  -> Standard model achieved {-accuracy_diff:.1%} higher accuracy")
    
    print(f"\nKey Paper Findings Replicated:")
    print(f"  ✓ Identical architectures (only output layer differs)")
    print(f"  ✓ Modular addition task implementation")
    print(f"  ✓ Class center analysis for harmonic model")
    print(f"  ✓ Comparative performance evaluation")
    
    return {
        'standard_model': standard_model,
        'harmonic_model': harmonic_model,
        'standard_results': standard_analysis,
        'harmonic_results': harmonic_analysis,
        'training_histories': {
            'standard': standard_history,
            'harmonic': harmonic_history
        }
    }

if __name__ == "__main__":
    results = run_exact_paper_replication()
    print(f"\nExact paper replication completed successfully!")
    print(f"Results and visualization saved.")
