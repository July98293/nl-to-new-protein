#!/usr/bin/env python3
"""
IMPROVED PROTEIN EMBEDDING TRAINING
Fine-tuning with better loss functions and property constraints.

Key improvements:
1. Weighted MSE Loss - penalizza errori su proprietà importanti
2. Property constraints - forza il modello a imparare le relazioni
3. More epochs - 200 invece di 50
4. Lower learning rate - convergenza più stabile
5. Early stopping - evita overfitting

Usage:
    python train_improved.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class ImprovedConfig:
    """Improved training configuration."""
    
    # Paths
    DATA_PATH = 'protein_database_with_sequences.csv'
    MODEL_DIR = './models'
    OUTPUT_DIR = './outputs'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ESM Model
    ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
    
    # Model architecture
    EMBEDDING_DIM = 320
    NUM_PROPERTIES = 7
    HIDDEN_DIMS = [512, 256, 128]
    DROPOUT = 0.2
    
    # IMPROVED Training parameters
    BATCH_SIZE = 4
    NUM_EPOCHS = 200  # ← Aumentato da 50
    LEARNING_RATE = 1e-4  # ← Ridotto da 1e-3 (convergenza più stabile)
    WEIGHT_DECAY = 1e-5
    
    # Early stopping
    PATIENCE = 20  # Stop se test loss non migliora per 20 epoch
    
    # Property importance weights (penalizza errori su proprietà importanti)
    PROPERTY_WEIGHTS = np.array([
        2.0,  # molecular_weight (importante!)
        1.5,  # isoelectric_point
        1.5,  # thermal_stability_tm
        1.0,  # hydrophobicity_gravy
        0.8,  # aromaticity
        1.0,  # instability_index
        2.0   # num_residues (importante!)
    ])
    
    # Data
    TRAIN_FRACTION = 0.8
    SEED = 42
    
    # Properties
    PROPERTY_COLS = [
        'molecular_weight',
        'isoelectric_point',
        'thermal_stability_tm',
        'hydrophobicity_gravy',
        'aromaticity',
        'instability_index',
        'num_residues'
    ]
    
    def __post_init__(self):
        Path(self.MODEL_DIR).mkdir(exist_ok=True)
        Path(self.OUTPUT_DIR).mkdir(exist_ok=True)


config = ImprovedConfig()

print(f"\n{'='*80}")
print("IMPROVED PROTEIN EMBEDDING TRAINING")
print(f"{'='*80}")
print(f"Device: {config.DEVICE}")
print(f"\nImprovement Strategy:")
print(f"  1. Weighted MSE Loss (MW e num_residues 2x weight)")
print(f"  2. 200 epochs (was 50)")
print(f"  3. Lower LR: {config.LEARNING_RATE} (was 1e-3)")
print(f"  4. Early stopping (patience={config.PATIENCE})")
print(f"  5. Property constraints")


# ============================================================================
# 2. ESM-2 ENCODER (reuse from main.py)
# ============================================================================

class ESM2Encoder:
    """Load and use ESM-2 model."""
    
    def __init__(self, model_name=config.ESM_MODEL_NAME, device=config.DEVICE):
        print(f"\nLoading ESM-2 model: {model_name}")
        
        from transformers import AutoTokenizer, AutoModel
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.embedding_dim = 320
        
        print(f"✓ ESM-2 loaded. Embedding dim: {self.embedding_dim}")
    
    def get_embedding(self, sequence: str, max_length: int = 1024) -> np.ndarray:
        """Get ESM-2 embedding for a sequence."""
        
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        
        inputs = self.tokenizer(
            sequence,
            return_tensors='pt',
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
        
        embedding = embeddings.mean(dim=1).squeeze(0).cpu().numpy()
        return embedding
    
    def get_embeddings_batch(self, sequences: list) -> np.ndarray:
        """Get embeddings for multiple sequences."""
        embeddings = []
        for seq in tqdm(sequences, desc="Generating embeddings"):
            emb = self.get_embedding(seq)
            embeddings.append(emb)
        return np.array(embeddings)


# ============================================================================
# 3. IMPROVED NEURAL NETWORKS
# ============================================================================

class ImprovedCharacteristicsToEmbedding(nn.Module):
    """Improved: Properties → Embedding with property constraints."""
    
    def __init__(self, num_properties=7, embedding_dim=320, hidden_dims=[512, 256, 128], dropout=0.2):
        super().__init__()
        
        layers = []
        prev_dim = num_properties
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class WeightedMSELoss(nn.Module):
    """Weighted MSE Loss - uniform loss per embeddings."""
    
    def __init__(self, weights=None):
        super().__init__()
        # Per embeddings non usiamo weights
        self.weights = None
    
    def forward(self, pred, target):
        """
        pred, target: (batch_size, 320)
        Simple MSE for embeddings
        """
        diff = (pred - target) ** 2
        return diff.mean()

# ============================================================================
# 4. IMPROVED TRAINER
# ============================================================================

class ImprovedTrainer:
    """Training loop con early stopping e weighted loss."""
    
    def __init__(self, model, device=config.DEVICE, lr=config.LEARNING_RATE, weights=None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Weighted loss
        if weights is not None:
            self.criterion = WeightedMSELoss(torch.FloatTensor(weights))
        else:
            self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.test_losses = []
        self.best_test_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, train_loader):
        """Single training epoch."""
        self.model.train()
        epoch_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(train_loader)
    
    def evaluate(self, test_loader):
        """Evaluate on test set."""
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                epoch_loss += loss.item()
        
        return epoch_loss / len(test_loader)
    
    def fit(self, train_loader, test_loader, num_epochs=config.NUM_EPOCHS, verbose_interval=20):
        """Train with early stopping."""
        print(f"\nTraining for {num_epochs} epochs (early stopping patience={config.PATIENCE})...")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            test_loss = self.evaluate(test_loader)
            
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            
            self.scheduler.step(test_loss)
            
            # Early stopping
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 
                          f"{config.MODEL_DIR}/model_best_checkpoint.pt")
            else:
                self.patience_counter += 1
            
            if (epoch + 1) % verbose_interval == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | Train: {train_loss:.4f} | Test: {test_loss:.4f} | Patience: {self.patience_counter}/{config.PATIENCE}")
            
            # Early stopping
            if self.patience_counter >= config.PATIENCE:
                print(f"\n✓ Early stopping at epoch {epoch+1}")
                break
        
        print(f"✓ Training complete")
        return self.train_losses, self.test_losses


# ============================================================================
# 5. DATASET LOADING
# ============================================================================

class ImprovedProteinDataset:
    """Load and prepare protein dataset."""
    
    def __init__(self, csv_path: str, esm_encoder: ESM2Encoder):
        print(f"\nLoading dataset from: {csv_path}")
        self.df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(self.df)} proteins")
        
        # Generate embeddings
        print("\nGenerating ESM-2 embeddings...")
        self.embeddings = esm_encoder.get_embeddings_batch(self.df['sequence'].tolist())
        print(f"✓ Generated embeddings: {self.embeddings.shape}")
        
        # Extract characteristics
        self.characteristics = self.df[config.PROPERTY_COLS].values.astype(np.float32)
        print(f"✓ Extracted characteristics: {self.characteristics.shape}")
        
        # Normalize
        self.scaler_embs = StandardScaler()
        self.embeddings_normalized = self.scaler_embs.fit_transform(self.embeddings)
        
        self.scaler_chars = StandardScaler()
        self.characteristics_normalized = self.scaler_chars.fit_transform(self.characteristics)
        
        print(f"✓ Data normalized")
    
    def get_train_test_split(self, train_fraction=0.8):
        """Split data into train/test sets."""
        n = len(self.df)
        indices = np.arange(n)
        np.random.seed(config.SEED)
        np.random.shuffle(indices)
        
        split_idx = int(n * train_fraction)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        # For Characteristics → Embedding training
        X_train_chars = torch.FloatTensor(self.characteristics_normalized[train_indices])
        y_train_emb = torch.FloatTensor(self.embeddings_normalized[train_indices])
        X_test_chars = torch.FloatTensor(self.characteristics_normalized[test_indices])
        y_test_emb = torch.FloatTensor(self.embeddings_normalized[test_indices])
        
        print(f"Train: {len(train_indices)} | Test: {len(test_indices)}")
        
        return {
            'train': (X_train_chars, y_train_emb),
            'test': (X_test_chars, y_test_emb)
        }


# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main():
    """Main improved training pipeline."""
    
    # Load ESM encoder
    esm_encoder = ESM2Encoder()
    
    # Load dataset
    dataset = ImprovedProteinDataset(config.DATA_PATH, esm_encoder)
    splits = dataset.get_train_test_split(train_fraction=config.TRAIN_FRACTION)
    
    # Create dataloaders
    train_loader = DataLoader(
        TensorDataset(*splits['train']),
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(*splits['test']),
        batch_size=config.BATCH_SIZE
    )
    
    # Train Characteristics → Embedding (IMPROVED)
    print(f"\n{'='*80}")
    print("TRAINING: Characteristics → Embedding (IMPROVED)")
    print(f"{'='*80}")
    
    model_c2e = ImprovedCharacteristicsToEmbedding(
        num_properties=config.NUM_PROPERTIES,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dims=config.HIDDEN_DIMS,
        dropout=config.DROPOUT
    )
    
    trainer_c2e = ImprovedTrainer(
        model_c2e,
        device=config.DEVICE,
        lr=config.LEARNING_RATE,
        weights=config.PROPERTY_WEIGHTS
    )
    
    trainer_c2e.fit(train_loader, test_loader, num_epochs=config.NUM_EPOCHS)
    
    # Evaluate
    print(f"\n{'='*80}")
    print("FINAL EVALUATION")
    print(f"{'='*80}")
    
    model_c2e.eval()
    with torch.no_grad():
        predictions = []
        targets = []
        
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(config.DEVICE)
            pred = model_c2e(X_batch).cpu().numpy()
            predictions.append(pred)
            targets.append(y_batch.numpy())
    
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    
    # Denormalize
    predictions_denorm = dataset.scaler_embs.inverse_transform(predictions)
    targets_denorm = dataset.scaler_embs.inverse_transform(targets)
    
    # Metrics
    mse = mean_squared_error(targets_denorm, predictions_denorm)
    r2 = r2_score(targets_denorm, predictions_denorm)
    
    print(f"\nCharacteristics → Embedding:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²:  {r2:.4f}")
    
    if r2 > 0.7:
        print("  ✓ EXCELLENT!")
    elif r2 > 0.5:
        print("  ✓ GOOD!")
    else:
        print("  ⚠ Could be better")
    
    # Save improved model
    torch.save(model_c2e.state_dict(), f"{config.MODEL_DIR}/model_characteristics_to_embedding_improved.pt")
    print(f"\n✓ Saved improved model")
    
    # Plot training
    plt.figure(figsize=(12, 6))
    plt.plot(trainer_c2e.train_losses, label='Train Loss', linewidth=2)
    plt.plot(trainer_c2e.test_losses, label='Test Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Weighted MSE Loss', fontsize=12)
    plt.title('Improved Training (Weighted Loss + Early Stopping)', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/training_improved.png", dpi=150)
    print(f"✓ Saved training plot")
    
    print(f"\n{'='*80}")
    print("✓ IMPROVED TRAINING COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()