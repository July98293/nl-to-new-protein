#!/usr/bin/env python3
"""
Protein Generation Script
Generate new protein sequences from natural language descriptions
using trained ESM-2 embeddings and neural network mappers.

Usage:
    python generate_proteins.py
"""

import torch
import pickle
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class GenerationConfig:
    """Configuration for protein generation."""
    
    # Model paths
    MODEL_DIR = './models'
    CHAR2EMB_MODEL = f'{MODEL_DIR}/model_characteristics_to_embedding.pt'
    EMB2CHAR_MODEL = f'{MODEL_DIR}/model_embedding_to_characteristics.pt'
    SCALER_CHARS_PATH = f'{MODEL_DIR}/scaler_characteristics.pkl'
    SCALER_EMBS_PATH = f'{MODEL_DIR}/scaler_embeddings.pkl'
    
    # Output
    OUTPUT_DIR = './outputs'
    DESIGN_OUTPUT = f'{OUTPUT_DIR}/designed_proteins.json'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Property ranges (for validation)
    PROPERTY_RANGES = {
        'molecular_weight': (1, 900),           # kDa
        'isoelectric_point': (2, 11.3),         # pH
        'thermal_stability_tm': (20, 100),      # °C
        'hydrophobicity_gravy': (-1, 2),        # dimensionless
        'aromaticity': (0, 0.25),               # fraction
        'instability_index': (0, 100),          # 0-100 score
        'num_residues': (10, 5000)              # residues
    }
    
    PROPERTY_NAMES = [
        'molecular_weight',
        'isoelectric_point',
        'thermal_stability_tm',
        'hydrophobicity_gravy',
        'aromaticity',
        'instability_index',
        'num_residues'
    ]


config = GenerationConfig()

print(f"\n{'='*80}")
print("PROTEIN GENERATION PIPELINE")
print(f"{'='*80}")
print(f"Device: {config.DEVICE}")
print(f"Model directory: {config.MODEL_DIR}")


# ============================================================================
# 2. LOAD TRAINED MODELS AND SCALERS
# ============================================================================

class ModelLoader:
    """Load trained models and scalers."""
    
    @staticmethod
    def load_models_and_scalers():
        """Load all trained models and normalization scalers."""
        
        print("\n" + "="*80)
        print("LOADING TRAINED MODELS")
        print("="*80)
        
        # Import model classes
        from main import CharacteristicsToEmbedding, EmbeddingToCharacteristics
        
        # Check if model files exist
        if not Path(config.CHAR2EMB_MODEL).exists():
            raise FileNotFoundError(
                f"Model not found: {config.CHAR2EMB_MODEL}\n"
                f"Please run: python main.py (to train models first)"
            )
        
        # Load models
        print("\n1. Loading Characteristics→Embedding model...")
        model_c2e = CharacteristicsToEmbedding(
            num_properties=7,
            embedding_dim=320,
            hidden_dims=[512, 256, 128],  
            dropout=0.2
        )

        
        model_c2e.load_state_dict(torch.load(config.CHAR2EMB_MODEL, map_location=config.DEVICE))
        model_c2e.to(config.DEVICE)
        model_c2e.eval()
        print("   ✓ Characteristics→Embedding model loaded")
        
        print("\n2. Loading Embedding→Characteristics model...")
        model_e2c = EmbeddingToCharacteristics(
            embedding_dim=320,
            num_properties=7
        )
        model_e2c.load_state_dict(torch.load(config.EMB2CHAR_MODEL, map_location=config.DEVICE))
        model_e2c.to(config.DEVICE)
        model_e2c.eval()
        print("   ✓ Embedding→Characteristics model loaded")
        
        # Load scalers
        print("\n3. Loading normalization scalers...")
        with open(config.SCALER_CHARS_PATH, 'rb') as f:
            scaler_chars = pickle.load(f)
        print("   ✓ Characteristics scaler loaded")
        
        with open(config.SCALER_EMBS_PATH, 'rb') as f:
            scaler_embs = pickle.load(f)
        print("   ✓ Embeddings scaler loaded")
        
        print("\n✓ All models and scalers loaded successfully!")
        
        return model_c2e, model_e2c, scaler_chars, scaler_embs


# ============================================================================
# 3. NATURAL LANGUAGE PARSER
# ============================================================================

class NLParser:
    """Parse natural language descriptions to protein characteristics."""
    
    @staticmethod
    def parse(description: str) -> Dict[str, float]:
        """
        Parse natural language description into 7 protein properties.
        
        Keywords:
        - Size: small, tiny, mini, large, big, huge, giant, medium
        - Charge: acidic, basic, alkaline, neutral, positive, negative
        - Stability: stable, thermostable, unstable, labile, robust, fragile
        - Hydrophobicity: hydrophobic, hydrophilic, lipophilic, polar, nonpolar
        - Domain: enzyme, antibody, protein, peptide, carrier
        - Activity: active, inactive, catalytic, binding
        """
        
        desc_lower = description.lower()
        
        print(f"\n{'─'*80}")
        print(f"INPUT: {description}")
        print(f"{'─'*80}")
        
        characteristics = {}
        
        # ─────────────────────────────────────────────────────────────────
        # 1. Molecular Weight (Size)
        # ─────────────────────────────────────────────────────────────────
        if any(word in desc_lower for word in ['small', 'tiny', 'mini', 'peptide']):
            characteristics['molecular_weight'] = 15.0
            print("🔹 Size: SMALL (15 kDa)")
        elif any(word in desc_lower for word in ['medium', 'moderate']):
            characteristics['molecular_weight'] = 50.0
            print("🔹 Size: MEDIUM (50 kDa)")
        elif any(word in desc_lower for word in ['large', 'big', 'huge', 'giant']):
            characteristics['molecular_weight'] = 150.0
            print("🔹 Size: LARGE (150 kDa)")
        else:
            characteristics['molecular_weight'] = 50.0
            print("🔹 Size: DEFAULT (50 kDa)")
        
        # ─────────────────────────────────────────────────────────────────
        # 2. Isoelectric Point (Charge)
        # ─────────────────────────────────────────────────────────────────
        if any(word in desc_lower for word in ['acidic', 'acid', 'negatively charged', 'anionic']):
            characteristics['isoelectric_point'] = 4.0
            print("🔹 Charge: ACIDIC (pI = 4.0)")
        elif any(word in desc_lower for word in ['basic', 'alkaline', 'positively charged', 'cationic']):
            characteristics['isoelectric_point'] = 9.0
            print("🔹 Charge: BASIC (pI = 9.0)")
        elif any(word in desc_lower for word in ['neutral', 'neutral charge']):
            characteristics['isoelectric_point'] = 6.5
            print("🔹 Charge: NEUTRAL (pI = 6.5)")
        else:
            characteristics['isoelectric_point'] = 6.5
            print("🔹 Charge: DEFAULT (pI = 6.5)")
        
        # ─────────────────────────────────────────────────────────────────
        # 3. Thermal Stability (Melting Temperature)
        # ─────────────────────────────────────────────────────────────────
        if any(word in desc_lower for word in ['thermostable', 'stable', 'robust', 'heat', 'hot', 'high temp']):
            characteristics['thermal_stability_tm'] = 80.0
            print("🔹 Stability: THERMOSTABLE (Tm = 80°C)")
        elif any(word in desc_lower for word in ['unstable', 'labile', 'fragile', 'cold', 'cryogenic']):
            characteristics['thermal_stability_tm'] = 40.0
            print("🔹 Stability: UNSTABLE (Tm = 40°C)")
        else:
            characteristics['thermal_stability_tm'] = 60.0
            print("🔹 Stability: DEFAULT (Tm = 60°C)")
        
        # ─────────────────────────────────────────────────────────────────
        # 4. Hydrophobicity
        # ─────────────────────────────────────────────────────────────────
        if any(word in desc_lower for word in ['hydrophobic', 'lipophilic', 'nonpolar', 'fat', 'lipid']):
            characteristics['hydrophobicity_gravy'] = 0.8
            print("🔹 Hydrophobicity: HIGH (GRAVY = 0.8)")
        elif any(word in desc_lower for word in ['hydrophilic', 'polar', 'soluble', 'water']):
            characteristics['hydrophobicity_gravy'] = -0.3
            print("🔹 Hydrophobicity: LOW (GRAVY = -0.3)")
        else:
            characteristics['hydrophobicity_gravy'] = 0.2
            print("🔹 Hydrophobicity: DEFAULT (GRAVY = 0.2)")
        
        # ─────────────────────────────────────────────────────────────────
        # 5. Aromaticity (Fixed - can be extended)
        # ─────────────────────────────────────────────────────────────────
        characteristics['aromaticity'] = 0.08
        print("🔹 Aromaticity: DEFAULT (0.08)")
        
        # ─────────────────────────────────────────────────────────────────
        # 6. Instability Index
        # ─────────────────────────────────────────────────────────────────
        if any(word in desc_lower for word in ['stable', 'robust']):
            characteristics['instability_index'] = 30.0
            print("🔹 Instability: LOW (II = 30)")
        else:
            characteristics['instability_index'] = 40.0
            print("🔹 Instability: DEFAULT (II = 40)")
        
        # ─────────────────────────────────────────────────────────────────
        # 7. Sequence Length
        # ─────────────────────────────────────────────────────────────────
        if any(word in desc_lower for word in ['short', 'small', 'peptide']):
            characteristics['num_residues'] = 100.0
            print("🔹 Length: SHORT (100 residues)")
        elif any(word in desc_lower for word in ['long', 'extended']):
            characteristics['num_residues'] = 500.0
            print("🔹 Length: LONG (500 residues)")
        else:
            characteristics['num_residues'] = 300.0
            print("🔹 Length: DEFAULT (300 residues)")
        
        return characteristics


# ============================================================================
# 4. PROTEIN GENERATION ENGINE
# ============================================================================

class ProteinGenerator:
    """Generate proteins from characteristics using trained models."""
    
    def __init__(self, model_c2e, model_e2c, scaler_chars, scaler_embs):
        """Initialize with trained models and scalers."""
        self.model_c2e = model_c2e
        self.model_e2c = model_e2c
        self.scaler_chars = scaler_chars
        self.scaler_embs = scaler_embs
        self.device = config.DEVICE
        self.property_names = config.PROPERTY_NAMES
    
    def characteristics_to_embedding(self, characteristics: Dict[str, float]) -> np.ndarray:
        """
        Convert characteristics dictionary to ESM-2 embedding via neural network.
        
        STAGE 2B: Properties → Embedding Mapper
        Input: 7 properties (normalized)
        Output: 320-dim ESM-2 embedding (normalized)
        """
        
        print("\n" + "─"*80)
        print("STAGE 2: Characteristics → Embedding (Neural Network)")
        print("─"*80)
        
        # Convert to array in correct order
        chars_array = np.array([characteristics[name] for name in self.property_names], dtype=np.float32)
        
        print("\nInput Properties (raw):")
        for name, val in zip(self.property_names, chars_array):
            print(f"  {name:.<30} {val:>10.2f}")
        
        # Normalize using scaler
        chars_normalized = self.scaler_chars.transform([chars_array])[0]
        
        print("\nProperties (normalized):")
        for name, val in zip(self.property_names, chars_normalized):
            print(f"  {name:.<30} {val:>10.4f}")
        
        # Convert to tensor
        chars_tensor = torch.FloatTensor(chars_normalized).unsqueeze(0).to(self.device)
        
        # Forward pass through neural network
        with torch.no_grad():
            embedding_normalized = self.model_c2e(chars_tensor).cpu().numpy()[0]
        
        print(f"\nEmbedding (normalized, 320-dim):")
        print(f"  Shape: {embedding_normalized.shape}")
        print(f"  Mean: {embedding_normalized.mean():.4f}")
        print(f"  Std: {embedding_normalized.std():.4f}")
        print(f"  First 10 dims: {embedding_normalized[:10]}")
        
        # Denormalize embedding
        embedding = self.scaler_embs.inverse_transform([embedding_normalized])[0]
        
        print(f"\nEmbedding (denormalized):")
        print(f"  Shape: {embedding.shape}")
        print(f"  Mean: {embedding.mean():.4f}")
        print(f"  Std: {embedding.std():.4f}")
        
        print("✓ Embedding generated successfully!")
        
        return embedding
    
    def embedding_to_sequence(self, embedding: np.ndarray, length: int = None) -> str:
        """
        Convert embedding to amino acid sequence (reverse folding).
        
        STAGE 3: Embedding → Sequence
        Uses probabilistic generation based on embedding features.
        """
        
        print("\n" + "─"*80)
        print("STAGE 3: Reverse Folding (Embedding → Sequence)")
        print("─"*80)
        
        if length is None:
            length = int(abs(embedding).mean() * 10) + 100
        
        print(f"\nGenerating sequence of length: {length} residues")
        
        # Amino acids with different properties
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        
        # Map embedding features to amino acid probabilities
        # This is a simplified approach - in production, use ProteinMPNN or similar
        
        # Use embedding statistics to bias amino acid selection
        embedding_mean = embedding.mean()
        embedding_std = embedding.std()
        embedding_energy = np.abs(embedding).sum()
        
        print(f"\nEmbedding Statistics:")
        print(f"  Mean: {embedding_mean:.4f}")
        print(f"  Std: {embedding_std:.4f}")
        print(f"  Total Energy: {embedding_energy:.4f}")
        
        # Generate sequence with bias from embedding
        sequence = []
        np.random.seed(42)  # For reproducibility
        
        for i in range(length):
            # Use embedding values to bias amino acid selection
            position_idx = i % len(embedding)
            bias = embedding[position_idx]
            
            # Create probability distribution
            if bias > 0.5:
                # Hydrophobic: AILMFVPW
                weights = [1, 0.5, 1, 1, 0.8, 0.5, 0.5, 0.5, 1, 0, 1, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 1, 0.8]
            elif bias < -0.5:
                # Hydrophilic: STNQ, charged: DE, KRH
                weights = [0.5, 0.8, 0.8, 0.8, 0.5, 1, 1, 0.5, 0.5, 1, 0.5, 0, 0.8, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            else:
                # Mixed
                weights = np.ones(len(amino_acids))
            
            weights = np.array(weights, dtype=float)
            weights /= weights.sum()
            
            aa = np.random.choice(list(amino_acids), p=weights)
            sequence.append(aa)
        
        sequence = ''.join(sequence)
        
        print(f"\n✓ Sequence generated!")
        print(f"\nGenerated Sequence (first 100 residues):")
        print(f"  {sequence[:100]}")
        
        if len(sequence) > 100:
            print(f"\nGenerated Sequence (last 100 residues):")
            print(f"  {sequence[-100:]}")
        
        print(f"\nFull length: {len(sequence)} residues")
        
        return sequence
    
    def verify_embedding(self, embedding: np.ndarray, original_chars: Dict[str, float]) -> Dict[str, float]:
        """
        Verify the generated embedding by predicting properties (for validation).
        
        This shows how well the model can recover characteristics from embedding.
        """
        
        print("\n" + "─"*80)
        print("VERIFICATION: Embedding → Properties (Quality Check)")
        print("─"*80)
        
        embedding_normalized = self.scaler_embs.transform([embedding])[0]
        embedding_tensor = torch.FloatTensor(embedding_normalized).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            chars_pred_norm = self.model_e2c(embedding_tensor).cpu().numpy()[0]
        
        chars_pred = self.scaler_chars.inverse_transform([chars_pred_norm])[0]
        
        print("\nProperty Comparison (Original vs Predicted):")
        print(f"{'Property':<30} {'Original':>12} {'Predicted':>12} {'Error':>12}")
        print("─" * 70)
        
        errors = {}
        for name, orig_val, pred_val in zip(self.property_names, original_chars.values(), chars_pred):
            error = abs(orig_val - pred_val)
            error_pct = (error / max(abs(orig_val), 1e-6)) * 100
            errors[name] = error
            
            print(f"{name:<30} {orig_val:>12.2f} {pred_val:>12.2f} {error_pct:>11.1f}%")
        
        # Calculate R² equivalently
        from sklearn.metrics import r2_score
        r2 = r2_score(list(original_chars.values()), chars_pred)
        
        print(f"\nOverall R² Score: {r2:.4f}")
        
        if r2 > 0.7:
            print("✓ HIGH QUALITY embedding (R² > 0.7)")
        elif r2 > 0.5:
            print("✓ GOOD embedding (R² > 0.5)")
        else:
            print("⚠ MODERATE embedding (R² < 0.5)")
        
        return dict(zip(self.property_names, chars_pred))
    
    def design_protein(self, description: str) -> Dict:
        """
        Complete pipeline: Natural Language → Design → Verification
        
        FULL PIPELINE:
        [1] Natural Language → Properties
        [2] Properties → Embedding
        [3] Embedding → Sequence
        [4] Verify by: Embedding → Properties
        """
        
        print("\n" + "="*80)
        print("COMPLETE PROTEIN DESIGN PIPELINE")
        print("="*80)
        
        # STAGE 1: Parse natural language
        characteristics = NLParser.parse(description)
        
        # STAGE 2: Properties → Embedding
        embedding = self.characteristics_to_embedding(characteristics)
        
        # STAGE 3: Embedding → Sequence
        sequence = self.embedding_to_sequence(embedding)
        
        # STAGE 4: Verify quality
        predicted_chars = self.verify_embedding(embedding, characteristics)
        
        result = {
            'description': description,
            'characteristics_input': characteristics,
            'characteristics_predicted': predicted_chars,
            'embedding': embedding.tolist(),
            'sequence': sequence,
            'sequence_length': len(sequence)
        }
        
        return result


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution: generate proteins from descriptions."""
    
    # Load models
    model_c2e, model_e2c, scaler_chars, scaler_embs = ModelLoader.load_models_and_scalers()
    
    # Initialize generator
    generator = ProteinGenerator(model_c2e, model_e2c, scaler_chars, scaler_embs)
    
    # Design prompts
    design_prompts = [
        "small, acidic, thermostable enzyme",
        "large, basic, hydrophobic antibody domain",
        "medium, neutral, soluble transport protein",
        "short, peptide, hydrophilic, unstable",
        "big, thermostable, robust carrier protein"
    ]
    
    print("\n" + "="*80)
    print(f"GENERATING {len(design_prompts)} PROTEIN DESIGNS")
    print("="*80)
    
    results = []
    
    for i, prompt in enumerate(design_prompts, 1):
        print(f"\n\n{'#'*80}")
        print(f"# DESIGN {i}/{len(design_prompts)}")
        print(f"{'#'*80}\n")
        
        try:
            result = generator.design_protein(prompt)
            results.append(result)
            
            print("\n" + "="*80)
            print(f"✓ DESIGN {i} COMPLETE")
            print("="*80)
            print(f"Sequence: {result['sequence'][:50]}...")
            print(f"Length: {result['sequence_length']} residues")
            
        except Exception as e:
            print(f"\n❌ Error designing protein {i}: {str(e)}")
            continue
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
    
    with open(config.DESIGN_OUTPUT, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = []
        for r in results:
            json_r = r.copy()
            json_r['embedding'] = [float(x) for x in r['embedding']]
            json_results.append(json_r)
        
        json.dump(json_results, f, indent=2)
    
    print(f"✓ Results saved to: {config.DESIGN_OUTPUT}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nGenerated proteins: {len(results)}")
    print(f"\nOutput files:")
    print(f"  - {config.DESIGN_OUTPUT}")
    
    print(f"\nDesigned proteins:")
    for i, result in enumerate(results, 1):
        print(f"\n  {i}. {result['description']}")
        print(f"     Length: {result['sequence_length']} residues")
        print(f"     Sequence: {result['sequence'][:60]}...")
    
    print("\n" + "="*80)
    print("✓ PROTEIN GENERATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()