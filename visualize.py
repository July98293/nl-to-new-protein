#!/usr/bin/env python3
"""
Protein Embedding & Feature Vector Visualization
Visualize ESM-2 embeddings (320-dim) and property vectors (7-dim)
using PCA and UMAP dimensionality reduction techniques.

Usage:
    python visualize_embeddings.py
"""

import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class VisualizationConfig:
    """Configuration for visualization."""
    
    # Data paths
    DATA_PATH = 'protein_database_with_sequences.csv'
    MODEL_DIR = './models'
    OUTPUT_DIR = './outputs'
    VIZ_OUTPUT_DIR = f'{OUTPUT_DIR}/visualizations'
    
    # Model paths
    EMB2CHAR_MODEL = f'{MODEL_DIR}/model_embedding_to_characteristics.pt'
    SCALER_CHARS_PATH = f'{MODEL_DIR}/scaler_characteristics.pkl'
    SCALER_EMBS_PATH = f'{MODEL_DIR}/scaler_embeddings.pkl'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Visualization parameters
    FIGURE_SIZE = (14, 10)
    DPI = 150
    
    # Dimensionality reduction (t-SNE disabled - too slow)
    USE_PCA = True
    USE_UMAP = True
    
    # Colors for categories
    CATEGORY_COLORS = {
        'enzyme': '#FF6B6B',
        'antibody': '#4ECDC4',
        'transport': '#45B7D1',
        'structural': '#FFA07A',
        'hormone': '#98D8C8',
        'immune': '#F7DC6F',
        'viral': '#BB8FCE',
        'other': '#95A5A6'
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


config = VisualizationConfig()

print(f"\n{'='*80}")
print("PROTEIN EMBEDDING & FEATURE VISUALIZATION")
print(f"{'='*80}")
print(f"Device: {config.DEVICE}")


# ============================================================================
# 2. LOAD DATA AND MODELS
# ============================================================================

class DataLoader:
    """Load data and trained models."""
    
    @staticmethod
    def load_dataset() -> pd.DataFrame:
        """Load protein dataset."""
        
        print("\n" + "="*80)
        print("LOADING DATA")
        print("="*80)
        
        df = pd.read_csv(config.DATA_PATH)
        
        print(f"\n✓ Loaded {len(df)} proteins")
        print(f"✓ Columns: {list(df.columns)}")
        
        return df
    
    @staticmethod
    def load_models_and_scalers():
        """Load trained models and scalers."""
        
        print("\n" + "="*80)
        print("LOADING TRAINED MODELS")
        print("="*80)
        
        from main import EmbeddingToCharacteristics
        
        # Check if model exists
        if not Path(config.EMB2CHAR_MODEL).exists():
            raise FileNotFoundError(
                f"Model not found: {config.EMB2CHAR_MODEL}\n"
                f"Please run: python main.py (to train models first)"
            )
        
        # Load model
        print("\nLoading Embedding→Characteristics model...")
        model = EmbeddingToCharacteristics(
            embedding_dim=320,
            num_properties=7
        )
        model.load_state_dict(torch.load(config.EMB2CHAR_MODEL, map_location=config.DEVICE))
        model.to(config.DEVICE)
        model.eval()
        print("✓ Model loaded")
        
        # Load scalers
        print("\nLoading scalers...")
        with open(config.SCALER_CHARS_PATH, 'rb') as f:
            scaler_chars = pickle.load(f)
        print("✓ Characteristics scaler loaded")
        
        with open(config.SCALER_EMBS_PATH, 'rb') as f:
            scaler_embs = pickle.load(f)
        print("✓ Embeddings scaler loaded")
        
        return model, scaler_chars, scaler_embs


# ============================================================================
# 3. GENERATE EMBEDDINGS
# ============================================================================

class EmbeddingGenerator:
    """Generate ESM-2 embeddings from sequences."""
    
    @staticmethod
    def generate_embeddings(df: pd.DataFrame) -> np.ndarray:
        """Generate ESM-2 embeddings for all proteins."""
        
        print("\n" + "="*80)
        print("GENERATING ESM-2 EMBEDDINGS")
        print("="*80)
        
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError("transformers not installed")
        
        model_name = "facebook/esm2_t6_8M_UR50D"
        print(f"\nLoading ESM-2 model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        esm_model = AutoModel.from_pretrained(model_name).to(config.DEVICE)
        esm_model.eval()
        
        print("✓ ESM-2 model loaded")
        
        embeddings = []
        
        print(f"\nGenerating embeddings for {len(df)} proteins...")
        
        for idx, row in df.iterrows():
            sequence = row['sequence']
            
            # Truncate if too long
            if len(sequence) > 1024:
                sequence = sequence[:1024]
            
            # Tokenize
            inputs = tokenizer(
                sequence,
                return_tensors='pt',
                truncation=True,
                max_length=1024
            ).to(config.DEVICE)
            
            # Get embedding
            with torch.no_grad():
                outputs = esm_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            
            embeddings.append(embedding)
            
            if (idx + 1) % max(1, len(df) // 5) == 0:
                print(f"  {idx + 1}/{len(df)} embeddings generated")
        
        embeddings = np.array(embeddings)
        print(f"\n✓ Generated embeddings shape: {embeddings.shape}")
        
        return embeddings


# ============================================================================
# 4. DIMENSIONALITY REDUCTION
# ============================================================================

class DimensionalityReducer:
    """Reduce high-dimensional data for visualization."""
    
    @staticmethod
    def apply_pca(data: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Apply PCA reduction."""
        
        print(f"\nApplying PCA ({len(data)} samples → {n_components}D)...")
        
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError("scikit-learn not installed")
        
        pca = PCA(n_components=n_components)
        data_reduced = pca.fit_transform(data)
        
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"✓ PCA: {explained_var*100:.1f}% variance explained")
        
        return data_reduced
    
    @staticmethod
    def apply_umap(data: np.ndarray, n_components: int = 2, n_neighbors: int = 15) -> np.ndarray:
        """Apply UMAP reduction."""
        
        print(f"\nApplying UMAP ({len(data)} samples → {n_components}D)...")
        
        try:
            import umap
        except ImportError:
            print("⚠ UMAP not installed. Install with: pip install umap-learn")
            return None
        
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            random_state=42
        )
        data_reduced = reducer.fit_transform(data)
        
        print(f"✓ UMAP completed")
        
        return data_reduced


# ============================================================================
# 5. VISUALIZATION
# ============================================================================

class Visualizer:
    """Create visualizations of embeddings."""
    
    @staticmethod
    def plot_embeddings_by_category(
        data_2d: np.ndarray,
        df: pd.DataFrame,
        title: str,
        filename: str
    ):
        """Plot embeddings colored by protein category."""
        
        fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.DPI)
        
        # Get categories
        categories = df['category'].unique() if 'category' in df else ['enzyme']
        
        # Plot each category
        for category in categories:
            if 'category' in df:
                mask = df['category'] == category
                indices = df[mask].index
            else:
                indices = range(len(df))
            
            if len(indices) == 0:
                continue
            
            color = config.CATEGORY_COLORS.get(category, '#95A5A6')
            
            ax.scatter(
                data_2d[indices, 0],
                data_2d[indices, 1],
                c=color,
                label=category,
                s=100,
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5
            )
        
        ax.set_xlabel('Component 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('Component 2', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        Path(config.VIZ_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        filepath = f"{config.VIZ_OUTPUT_DIR}/{filename}"
        plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    @staticmethod
    def plot_embeddings_by_property(
        data_2d: np.ndarray,
        properties: np.ndarray,
        property_name: str,
        title: str,
        filename: str
    ):
        """Plot embeddings colored by property value."""
        
        fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.DPI)
        
        scatter = ax.scatter(
            data_2d[:, 0],
            data_2d[:, 1],
            c=properties,
            cmap='viridis',
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )
        
        ax.set_xlabel('Component 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('Component 2', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(property_name, fontsize=11, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        
        Path(config.VIZ_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        filepath = f"{config.VIZ_OUTPUT_DIR}/{filename}"
        plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    @staticmethod
    def plot_feature_space(
        features: np.ndarray,
        df: pd.DataFrame,
        filename: str = "07_feature_space_pairs.png"
    ):
        """Plot 7D feature space (protein properties)."""
        
        fig = plt.figure(figsize=(16, 12), dpi=config.DPI)
        
        property_names = config.PROPERTY_NAMES
        
        # Create subplots for each pair of properties
        for idx, (prop1, prop2) in enumerate(
            [(property_names[i], property_names[j]) 
             for i in range(len(property_names)) 
             for j in range(i+1, len(property_names))][:6]
        ):
            ax = plt.subplot(2, 3, idx + 1)
            
            prop1_idx = property_names.index(prop1)
            prop2_idx = property_names.index(prop2)
            
            # Get categories for coloring
            if 'category' in df:
                categories = df['category'].unique()
                for category in categories:
                    mask = df['category'] == category
                    color = config.CATEGORY_COLORS.get(category, '#95A5A6')
                    ax.scatter(
                        features[mask, prop1_idx],
                        features[mask, prop2_idx],
                        c=color,
                        label=category,
                        s=50,
                        alpha=0.6,
                        edgecolors='black',
                        linewidth=0.5
                    )
            else:
                ax.scatter(
                    features[:, prop1_idx],
                    features[:, prop2_idx],
                    c='#3498DB',
                    s=50,
                    alpha=0.6,
                    edgecolors='black',
                    linewidth=0.5
                )
            
            ax.set_xlabel(prop1, fontsize=10, fontweight='bold')
            ax.set_ylabel(prop2, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{prop1} vs {prop2}", fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        Path(config.VIZ_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        filepath = f"{config.VIZ_OUTPUT_DIR}/{filename}"
        plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    @staticmethod
    def plot_correlation_heatmap(
        features: np.ndarray,
        filename: str = "08_correlation_heatmap.png"
    ):
        """Plot correlation matrix of properties."""
        
        property_names = config.PROPERTY_NAMES
        
        # Calculate correlation
        correlation_matrix = np.corrcoef(features.T)
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=config.DPI)
        
        im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(property_names)))
        ax.set_yticks(range(len(property_names)))
        ax.set_xticklabels(property_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(property_names, fontsize=9)
        
        # Add correlation values
        for i in range(len(property_names)):
            for j in range(len(property_names)):
                text = ax.text(
                    j, i,
                    f'{correlation_matrix[i, j]:.2f}',
                    ha="center", va="center", color="black", fontsize=8
                )
        
        ax.set_title('Protein Property Correlation Matrix', fontsize=13, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation', fontsize=11)
        
        plt.tight_layout()
        
        Path(config.VIZ_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        filepath = f"{config.VIZ_OUTPUT_DIR}/{filename}"
        plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()


# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main():
    """Main visualization pipeline."""
    
    # Load data
    df = DataLoader.load_dataset()
    
    # Load models
    model, scaler_chars, scaler_embs = DataLoader.load_models_and_scalers()
    
    # Generate embeddings
    print("\n" + "="*80)
    print("GENERATING EMBEDDINGS")
    print("="*80)
    
    embeddings = EmbeddingGenerator.generate_embeddings(df)
    
    # Extract features (properties)
    print("\n" + "="*80)
    print("EXTRACTING FEATURES")
    print("="*80)
    
    features = df[config.PROPERTY_NAMES].values.astype(np.float32)
    print(f"\n✓ Features shape: {features.shape}")
    
    # ────────────────────────────────────────────────────────────────────
    # EMBEDDING VISUALIZATIONS (320D → 2D)
    # ────────────────────────────────────────────────────────────────────
    
    print("\n" + "="*80)
    print("DIMENSIONALITY REDUCTION: EMBEDDINGS (320D → 2D)")
    print("="*80)
    
    reducer = DimensionalityReducer()
    
    # PCA on embeddings
    if config.USE_PCA:
        print("\n1. PCA REDUCTION")
        embeddings_pca = reducer.apply_pca(embeddings, n_components=2)
        
        Visualizer.plot_embeddings_by_category(
            embeddings_pca, df,
            "Protein Embeddings (PCA) - By Category",
            "01_embeddings_pca_category.png"
        )
        
        Visualizer.plot_embeddings_by_property(
            embeddings_pca,
            features[:, 0],
            "Molecular Weight (kDa)",
            "Protein Embeddings (PCA) - By MW",
            "02_embeddings_pca_mw.png"
        )
        
        Visualizer.plot_embeddings_by_property(
            embeddings_pca,
            features[:, 1],
            "Isoelectric Point (pH)",
            "Protein Embeddings (PCA) - By pI",
            "03_embeddings_pca_pi.png"
        )
    
    # UMAP on embeddings
    if config.USE_UMAP:
        print("\n2. UMAP REDUCTION")
        embeddings_umap = reducer.apply_umap(embeddings, n_components=2, n_neighbors=15)
        
        if embeddings_umap is not None:
            Visualizer.plot_embeddings_by_category(
                embeddings_umap, df,
                "Protein Embeddings (UMAP) - By Category",
                "04_embeddings_umap_category.png"
            )
            
            Visualizer.plot_embeddings_by_property(
                embeddings_umap,
                features[:, 2],
                "Thermal Stability (°C)",
                "Protein Embeddings (UMAP) - By Tm",
                "05_embeddings_umap_tm.png"
            )
            
            Visualizer.plot_embeddings_by_property(
                embeddings_umap,
                features[:, 3],
                "Hydrophobicity (GRAVY)",
                "Protein Embeddings (UMAP) - By Hydrophobicity",
                "06_embeddings_umap_hydro.png"
            )
    
    # ────────────────────────────────────────────────────────────────────
    # FEATURE SPACE VISUALIZATIONS (7D)
    # ────────────────────────────────────────────────────────────────────
    
    print("\n" + "="*80)
    print("FEATURE SPACE VISUALIZATION (7D → 2D × 6 Pairs)")
    print("="*80)
    
    Visualizer.plot_feature_space(features, df)
    
    # ────────────────────────────────────────────────────────────────────
    # CORRELATION ANALYSIS
    # ────────────────────────────────────────────────────────────────────
    
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    Visualizer.plot_correlation_heatmap(features)
    
    # ────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ────────────────────────────────────────────────────────────────────
    
    print("\n" + "="*80)
    print("VISUALIZATION SUMMARY")
    print("="*80)
    
    print(f"\n✓ Generated visualizations saved to: {config.VIZ_OUTPUT_DIR}/")
    print(f"\nVisualization files created:")
    print(f"  01. PCA: Embeddings colored by category")
    print(f"  02. PCA: Embeddings colored by molecular weight")
    print(f"  03. PCA: Embeddings colored by isoelectric point")
    print(f"  04. UMAP: Embeddings colored by category")
    print(f"  05. UMAP: Embeddings colored by thermal stability")
    print(f"  06. UMAP: Embeddings colored by hydrophobicity")
    print(f"  07. Feature space: 2D projections of all property pairs")
    print(f"  08. Correlation heatmap: Property relationships")
    
    print(f"\n" + "="*80)
    print("✓ VISUALIZATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
    