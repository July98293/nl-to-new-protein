"""
Protein Embedding & Characteristics Learning Pipeline
Local Development Version (Cursor/VS Code Compatible)

Install with:
pip install torch transformers biopython scikit-learn pandas numpy matplotlib seaborn tqdm
pip install fair-esm  # Correct package name (not fair-esm2)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import json
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class Config:
    # Paths
    DATA_PATH = 'protein_database_with_sequences.csv'
    MODEL_DIR = './models'
    OUTPUT_DIR = './outputs'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ESM Model
    ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"  # Smaller for local dev
    # For production, use: "facebook/esm2_t33_650M_UR50D" or "facebook/esm2_t36_3B_UR50D"
    
    # Model architecture
    EMBEDDING_DIM = 320
    NUM_PROPERTIES = 7
    HIDDEN_DIMS = [512, 256, 128]
    DROPOUT = 0.2
    
    # Training
    BATCH_SIZE = 4
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    
    # Data
    TRAIN_FRACTION = 0.8
    SEED = 42
    
    # Properties to predict
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
        
    def __repr__(self):
        attrs = [f"  {k}: {v}" for k, v in self.__dict__.items() if not k.startswith('_')]
        return "Config(\n" + "\n".join(attrs) + "\n)"

config = Config()

print(f"\n{'='*70}")
print("PROTEIN EMBEDDING PIPELINE - LOCAL VERSION")
print(f"{'='*70}")
print(f"Device: {config.DEVICE}")
if config.DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"\nConfig:\n{config}")


# ============================================================================
# 2. ESM-2 EMBEDDING MODULE
# ============================================================================

class ESM2Encoder:
    """Wrapper for ESM-2 model with caching and batch processing."""
    
    def __init__(self, model_name=config.ESM_MODEL_NAME, device=config.DEVICE):
        print(f"\nLoading ESM-2 model: {model_name}")
        
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError("transformers not installed. Run: pip install transformers")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.embedding_dim = 320
        
        print(f"✓ ESM-2 loaded. Embedding dim: {self.embedding_dim}")
    
    def get_embedding(self, sequence: str, max_length: int = 1024) -> np.ndarray:
        """
        Generate ESM-2 embedding for a protein sequence.
        Returns mean-pooled representation over sequence length.
        """
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
            embeddings = outputs.last_hidden_state  # [1, seq_len, 320]
        
        # Mean pooling over sequence
        embedding = embeddings.mean(dim=1).squeeze(0).cpu().numpy()
        return embedding
    
    def get_embeddings_batch(self, sequences: list) -> np.ndarray:
        """Generate embeddings for multiple sequences."""
        embeddings = []
        for seq in tqdm(sequences, desc="Generating embeddings"):
            emb = self.get_embedding(seq)
            embeddings.append(emb)
        return np.array(embeddings)


# ============================================================================
# 3. NEURAL NETWORK MODELS
# ============================================================================

class EmbeddingToCharacteristics(nn.Module):
    """Maps ESM-2 embeddings (320-dim) to protein characteristics (7 properties)."""
    
    def __init__(self, embedding_dim=320, num_properties=7, hidden_dims=[512, 256, 128], dropout=0.2):
        super().__init__()
        
        layers = []
        prev_dim = embedding_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_properties))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class CharacteristicsToEmbedding(nn.Module):
    """Maps protein characteristics (7 properties) to ESM-2 embeddings (320-dim)."""
    
    def __init__(self, num_properties=7, embedding_dim=320, hidden_dims=[128, 256, 512], dropout=0.2):
        super().__init__()
        
        layers = []
        prev_dim = num_properties
        
        # Hidden layers (mirrored architecture)
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


# ============================================================================
# 4. DATA LOADING & PREPARATION
# ============================================================================

class ProteinDataset:
    """Load and prepare protein dataset."""
    
    def __init__(self, csv_path: str, esm_encoder: ESM2Encoder):
        print(f"\nLoading dataset from: {csv_path}")
        self.df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(self.df)} proteins")
        
        # Generate embeddings
        print("\nGenerating ESM-2 embeddings for all proteins...")
        self.embeddings = esm_encoder.get_embeddings_batch(self.df['sequence'].tolist())
        print(f"✓ Generated embeddings shape: {self.embeddings.shape}")
        
        # Extract characteristics
        self.characteristics = self.df[config.PROPERTY_COLS].values.astype(np.float32)
        print(f"✓ Extracted characteristics shape: {self.characteristics.shape}")
        
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
        
        # Prepare tensors
        X_train_emb = torch.FloatTensor(self.embeddings_normalized[train_indices])
        y_train_chars = torch.FloatTensor(self.characteristics_normalized[train_indices])
        X_test_emb = torch.FloatTensor(self.embeddings_normalized[test_indices])
        y_test_chars = torch.FloatTensor(self.characteristics_normalized[test_indices])
        
        X_train_chars = torch.FloatTensor(self.characteristics_normalized[train_indices])
        y_train_emb = torch.FloatTensor(self.embeddings_normalized[train_indices])
        X_test_chars = torch.FloatTensor(self.characteristics_normalized[test_indices])
        y_test_emb = torch.FloatTensor(self.embeddings_normalized[test_indices])
        
        print(f"\nTrain: {len(train_indices)} | Test: {len(test_indices)}")
        
        return {
            'train_emb': (X_train_emb, y_train_chars),
            'test_emb': (X_test_emb, y_test_chars),
            'train_char': (X_train_chars, y_train_emb),
            'test_char': (X_test_chars, y_test_emb)
        }


# ============================================================================
# 5. TRAINING LOOP
# ============================================================================

class Trainer:
    """Training loop for neural networks."""
    
    def __init__(self, model, device=config.DEVICE, lr=config.LEARNING_RATE):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.test_losses = []
    
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
    
    def fit(self, train_loader, test_loader, num_epochs=config.NUM_EPOCHS, verbose_interval=10):
        """Train model."""
        print(f"\nTraining for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            test_loss = self.evaluate(test_loader)
            
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            
            self.scheduler.step(test_loss)
            
            if (epoch + 1) % verbose_interval == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        
        print(f"✓ Training complete")
        return self.train_losses, self.test_losses


# ============================================================================
# 6. EVALUATION & VISUALIZATION
# ============================================================================

def evaluate_model(model, test_loader, scaler_out, scaler_in=None, device=config.DEVICE):
    """Evaluate model performance."""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch).cpu().numpy()
            predictions.append(pred)
            targets.append(y_batch.numpy())
    
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    
    # Denormalize
    if scaler_out is not None:
        predictions_denorm = scaler_out.inverse_transform(predictions)
        targets_denorm = scaler_out.inverse_transform(targets)
    else:
        predictions_denorm = predictions
        targets_denorm = targets
    
    # Calculate metrics
    mse = mean_squared_error(targets_denorm, predictions_denorm)
    mae = mean_absolute_error(targets_denorm, predictions_denorm)
    r2 = r2_score(targets_denorm, predictions_denorm, multioutput='raw_values').mean()
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions_denorm,
        'targets': targets_denorm
    }


def plot_training_curves(train_losses, test_losses, title="Training Curves", save_path=None):
    """Plot training history."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(test_losses, label='Test Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()


# ============================================================================
# 7. PROTEIN DESIGN PIPELINE
# ============================================================================

class ProteinDesignPipeline:
    """End-to-end pipeline for protein design."""
    
    def __init__(self, model_char2emb, scaler_chars, scaler_embs, esm_encoder, device=config.DEVICE):
        self.model_char2emb = model_char2emb
        self.scaler_chars = scaler_chars
        self.scaler_embs = scaler_embs
        self.esm_encoder = esm_encoder
        self.device = device
        self.property_names = config.PROPERTY_COLS
    
    def parse_natural_language(self, description: str) -> dict:
        """
        Parse natural language to extract protein properties.
        
        Currently: Simple rule-based (replace with LLM API for production)
        
        Example:
            "small, acidic, thermostable enzyme"
            → {'molecular_weight': 20, 'isoelectric_point': 4.0, ...}
        """
        characteristics = {}
        desc_lower = description.lower()
        
        # Molecular weight
        if 'small' in desc_lower or 'peptide' in desc_lower:
            characteristics['molecular_weight'] = 20
        elif 'large' in desc_lower or 'complex' in desc_lower:
            characteristics['molecular_weight'] = 150
        elif 'medium' in desc_lower:
            characteristics['molecular_weight'] = 60
        else:
            characteristics['molecular_weight'] = 50
        
        # Isoelectric point
        if 'acidic' in desc_lower:
            characteristics['isoelectric_point'] = 4.0
        elif 'basic' in desc_lower or 'alkaline' in desc_lower:
            characteristics['isoelectric_point'] = 9.0
        else:
            characteristics['isoelectric_point'] = 6.5
        
        # Thermal stability
        if 'therm' in desc_lower or 'stable' in desc_lower or 'hot' in desc_lower:
            characteristics['thermal_stability_tm'] = 75.0
        elif 'cold' in desc_lower or 'unstable' in desc_lower:
            characteristics['thermal_stability_tm'] = 45.0
        else:
            characteristics['thermal_stability_tm'] = 60.0
        
        # Hydrophobicity
        if 'hydrophobic' in desc_lower or 'lipid' in desc_lower:
            characteristics['hydrophobicity_gravy'] = 0.8
        elif 'hydrophilic' in desc_lower or 'water' in desc_lower or 'soluble' in desc_lower:
            characteristics['hydrophobicity_gravy'] = -0.3
        else:
            characteristics['hydrophobicity_gravy'] = 0.2
        
        # Fixed properties
        characteristics['aromaticity'] = 0.08
        characteristics['instability_index'] = 35.0
        characteristics['num_residues'] = 300
        
        return characteristics
    
    def characteristics_to_embedding(self, characteristics: dict) -> np.ndarray:
        """Convert characteristics to ESM embedding."""
        chars_array = np.array([characteristics[col] for col in self.property_names])
        chars_normalized = self.scaler_chars.transform([chars_array])[0]
        
        chars_tensor = torch.FloatTensor(chars_normalized).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding_normalized = self.model_char2emb(chars_tensor).cpu().numpy()[0]
        
        embedding = self.scaler_embs.inverse_transform([embedding_normalized])[0]
        return embedding
    
    def embedding_to_sequence(self, embedding: np.ndarray) -> str:
        """Generate sequence from embedding (placeholder)."""
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        length = int(abs(embedding).mean() * 10) + 100
        sequence = ''.join(np.random.choice(list(amino_acids), length))
        return sequence
    
    def design_protein(self, description: str) -> dict:
        """End-to-end design: Natural Language → Characteristics → Embedding → Sequence."""
        
        print(f"\n{'='*70}")
        print(f"PROTEIN DESIGN: {description}")
        print(f"{'='*70}")
        
        # Step 1: Parse
        print(f"\n1. Parsing natural language...")
        characteristics = self.parse_natural_language(description)
        for prop, val in characteristics.items():
            print(f"   {prop:.<30} {val:>10.2f}")
        
        # Step 2: Generate embedding
        print(f"\n2. Mapping to ESM-2 embedding...")
        embedding = self.characteristics_to_embedding(characteristics)
        print(f"   Shape: {embedding.shape}")
        print(f"   First 10 dims: {embedding[:10]}")
        
        # Step 3: Generate sequence
        print(f"\n3. Reverse folding to sequence...")
        sequence = self.embedding_to_sequence(embedding)
        print(f"   Length: {len(sequence)} residues")
        print(f"   First 100 AA: {sequence[:100]}")
        
        return {
            'description': description,
            'characteristics': characteristics,
            'embedding': embedding,
            'sequence': sequence
        }


# ============================================================================
# 8. UTILITIES
# ============================================================================

def save_model_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': loss
    }, path)
    print(f"✓ Saved checkpoint: {path}")


def load_model_checkpoint(model, optimizer, path, device=config.DEVICE):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    return model, optimizer


def save_scalers(scaler_chars, scaler_embs, dir_path='./models'):
    """Save scalers for inference."""
    import pickle
    Path(dir_path).mkdir(exist_ok=True)
    
    with open(f'{dir_path}/scaler_characteristics.pkl', 'wb') as f:
        pickle.dump(scaler_chars, f)
    
    with open(f'{dir_path}/scaler_embeddings.pkl', 'wb') as f:
        pickle.dump(scaler_embs, f)
    
    print(f"✓ Saved scalers to {dir_path}")


def save_models(model_e2c, model_c2e, dir_path='./models'):
    """Save trained models."""
    Path(dir_path).mkdir(exist_ok=True)
    
    torch.save(model_e2c.state_dict(), f'{dir_path}/model_embedding_to_characteristics.pt')
    torch.save(model_c2e.state_dict(), f'{dir_path}/model_characteristics_to_embedding.pt')
    
    print(f"✓ Saved models to {dir_path}")


# ============================================================================
# 9. MAIN EXECUTION
# ============================================================================

def main():
    """Main pipeline execution."""
    
    # 0. Initialize ESM encoder
    esm_encoder = ESM2Encoder()
    
    # 1. Load dataset
    dataset = ProteinDataset(config.DATA_PATH, esm_encoder)
    splits = dataset.get_train_test_split(train_fraction=config.TRAIN_FRACTION)
    
    # 2. Create dataloaders
    train_loader_e2c = DataLoader(
        TensorDataset(*splits['train_emb']),
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    test_loader_e2c = DataLoader(
        TensorDataset(*splits['test_emb']),
        batch_size=config.BATCH_SIZE
    )
    
    train_loader_c2e = DataLoader(
        TensorDataset(*splits['train_char']),
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    test_loader_c2e = DataLoader(
        TensorDataset(*splits['test_char']),
        batch_size=config.BATCH_SIZE
    )
    
    # 3. Train Embedding → Characteristics
    print(f"\n{'='*70}")
    print("TRAINING: Embedding → Characteristics")
    print(f"{'='*70}")
    
    model_e2c = EmbeddingToCharacteristics(
        embedding_dim=config.EMBEDDING_DIM,
        num_properties=config.NUM_PROPERTIES,
        hidden_dims=config.HIDDEN_DIMS,
        dropout=config.DROPOUT
    )
    trainer_e2c = Trainer(model_e2c, device=config.DEVICE)
    trainer_e2c.fit(train_loader_e2c, test_loader_e2c, num_epochs=config.NUM_EPOCHS)
    
    # 4. Train Characteristics → Embedding
    print(f"\n{'='*70}")
    print("TRAINING: Characteristics → Embedding")
    print(f"{'='*70}")
    
    model_c2e = CharacteristicsToEmbedding(
        num_properties=config.NUM_PROPERTIES,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dims=config.HIDDEN_DIMS,
        dropout=config.DROPOUT
    )
    trainer_c2e = Trainer(model_c2e, device=config.DEVICE)
    trainer_c2e.fit(train_loader_c2e, test_loader_c2e, num_epochs=config.NUM_EPOCHS)
    
    # 5. Evaluate
    print(f"\n{'='*70}")
    print("EVALUATION")
    print(f"{'='*70}")
    
    eval_e2c = evaluate_model(model_e2c, test_loader_e2c, dataset.scaler_chars)
    eval_c2e = evaluate_model(model_c2e, test_loader_c2e, dataset.scaler_embs)
    
    print(f"\nEmbedding → Characteristics:")
    print(f"  MSE: {eval_e2c['mse']:.4f}")
    print(f"  MAE: {eval_e2c['mae']:.4f}")
    print(f"  R²:  {eval_e2c['r2']:.4f}")
    
    print(f"\nCharacteristics → Embedding:")
    print(f"  MSE: {eval_c2e['mse']:.4f}")
    print(f"  MAE: {eval_c2e['mae']:.4f}")
    print(f"  R²:  {eval_c2e['r2']:.4f}")
    
    # 6. Visualize
    plot_training_curves(
        trainer_e2c.train_losses,
        trainer_e2c.test_losses,
        title="Embedding → Characteristics Training",
        save_path=f"{config.OUTPUT_DIR}/training_e2c.png"
    )
    
    plot_training_curves(
        trainer_c2e.train_losses,
        trainer_c2e.test_losses,
        title="Characteristics → Embedding Training",
        save_path=f"{config.OUTPUT_DIR}/training_c2e.png"
    )
    
    # 7. Save models
    save_models(model_e2c, model_c2e, config.MODEL_DIR)
    save_scalers(dataset.scaler_chars, dataset.scaler_embs, config.MODEL_DIR)
    
    # 8. Demonstrate pipeline
    print(f"\n{'='*70}")
    print("PROTEIN DESIGN DEMONSTRATIONS")
    print(f"{'='*70}")
    
    pipeline = ProteinDesignPipeline(
        model_c2e,
        dataset.scaler_chars,
        dataset.scaler_embs,
        esm_encoder
    )
    
    designs = [
        "small, acidic, thermostable enzyme",
        "large, basic, hydrophobic antibody",
        "medium, neutral, soluble transport protein"
    ]
    
    results = []
    for design_desc in designs:
        result = pipeline.design_protein(design_desc)
        results.append(result)
    
    # Save results
    with open(f"{config.OUTPUT_DIR}/design_results.json", 'w') as f:
        json.dump([{k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                    for k, v in r.items()} for r in results], f, indent=2)
    
    print(f"\n✓ Generated {len(results)} protein designs")
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutputs saved to: {config.OUTPUT_DIR}/")
    print(f"Models saved to:  {config.MODEL_DIR}/")


if __name__ == "__main__":
    main()