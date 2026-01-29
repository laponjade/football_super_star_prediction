"""
MLOps Phase 1: Data Versioning with Weights & Biases
This script versions the football player dataset as a W&B artifact.
"""

import pandas as pd
import wandb
from pathlib import Path
from mlops_config import ENTITY, PROJECT, DATA_PATH, OUTPUT_DIR

# Convert string paths to Path objects
DATA_PATH = Path(DATA_PATH)
OUTPUT_DIR = Path(OUTPUT_DIR)

def prepare_and_version_data():
    """
    Prepare the dataset with temporal splits and version it in W&B.
    """
    # Initialize W&B run for data preparation
    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        job_type="data-preparation",
        notes="Cleaned and versioned Football Player dataset with temporal splits (FIFA 17-21)."
    )
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nFIFA version distribution:")
    print(df['fifa_version'].value_counts().sort_index())
    print(f"\nTarget distribution:")
    print(df['big_potential'].value_counts())
    
    # Define temporal splits (FIFA versions)
    train_versions = [17.0, 18.0, 19.0, 20.0]
    val_version = 21.0
    test_version = 21.0
    
    # Split data by FIFA version
    df_train = df[df['fifa_version'].isin(train_versions)].copy()
    df_val = df[df['fifa_version'] == val_version].copy()
    df_test = df[df['fifa_version'] == test_version].copy()
    
    print(f"\nTrain (FIFA 17-20): {len(df_train)} samples")
    print(f"Validation (FIFA 21): {len(df_val)} samples")
    print(f"Test (FIFA 21): {len(df_test)} samples")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Save splits
    df_train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    df_val.to_csv(OUTPUT_DIR / "val.csv", index=False)
    df_test.to_csv(OUTPUT_DIR / "test.csv", index=False)
    
    print(f"\nData prepared and saved in '{OUTPUT_DIR}' directory.")
    
    # Calculate metadata
    feature_columns = [col for col in df.columns if col not in ['fifa_version', 'big_potential']]
    
    metadata = {
        "source": "feature_engineered_data_v2.csv",
        "splits": ["train", "val", "test"],
        "train_versions": train_versions,
        "val_version": val_version,
        "test_version": test_version,
        "num_features": len(feature_columns),
        "features": feature_columns,
        "train_size": len(df_train),
        "val_size": len(df_val),
        "test_size": len(df_test),
        "train_class_distribution": {
            "class_0": int((df_train['big_potential'] == 0).sum()),
            "class_1": int((df_train['big_potential'] == 1).sum())
        },
        "val_class_distribution": {
            "class_0": int((df_val['big_potential'] == 0).sum()),
            "class_1": int((df_val['big_potential'] == 1).sum())
        },
        "test_class_distribution": {
            "class_0": int((df_test['big_potential'] == 0).sum()),
            "class_1": int((df_test['big_potential'] == 1).sum())
        }
    }
    
    # Create W&B artifact
    artifact = wandb.Artifact(
        name="football-player-dataset",
        type="dataset",
        description="Football player dataset with temporal splits (FIFA 17-21). Features engineered for big potential prediction.",
        metadata=metadata
    )
    
    # Add prepared data directory to artifact
    artifact.add_dir(str(OUTPUT_DIR))
    
    # Log artifact to W&B
    run.log_artifact(artifact)
    run.finish()
    
    print("\n[SUCCESS] Dataset artifact logged to W&B!")
    print(f"View at: https://wandb.ai/{ENTITY}/{PROJECT}/artifacts/dataset/football-player-dataset")
    
    return metadata

if __name__ == "__main__":
    # Login to W&B (if not already logged in)
    # wandb.login()
    
    prepare_and_version_data()
