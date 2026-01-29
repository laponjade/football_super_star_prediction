"""
MLOps Phase 4: Model Registration
This script selects the best model from a sweep and registers it in W&B.
"""

import pandas as pd
import numpy as np
import wandb
import joblib
import json
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss
)
from xgboost import XGBClassifier

# Configuration - Import from centralized config
from mlops_config import ENTITY, PROJECT

def get_best_run_from_sweep(sweep_id, use_local_cache=False):
    """
    Get the best run from a completed sweep.
    
    Args:
        sweep_id: W&B sweep ID
        use_local_cache: If True, use local cache instead of API (for offline mode)
    
    Returns:
        Best run object or dict with best run info
    """
    if use_local_cache:
        return get_best_run_from_local_cache(sweep_id)
    
    try:
        api = wandb.Api()
        sweep = api.sweep(f"{ENTITY}/{PROJECT}/{sweep_id}")
        best_run = sweep.best_run()
        
        print(f"Best run: {best_run.name}")
        print(f"Best config: {best_run.config}")
        print(f"Best F1 score: {best_run.summary.get('val/f1', 'N/A')}")
        
        return best_run
    except Exception as e:
        print(f"[WARN] Could not connect to W&B API: {e}")
        print("[INFO] Falling back to local cache...")
        return get_best_run_from_local_cache(sweep_id)

def get_best_run_from_local_cache(sweep_id):
    """
    Get the best run from local W&B cache without API connection.
    
    Args:
        sweep_id: W&B sweep ID
    
    Returns:
        dict with best run info (config, summary, etc.)
    """
    import json
    import yaml
    
    wandb_dir = Path("wandb")
    sweep_dir = wandb_dir / f"sweep-{sweep_id}"
    
    if not sweep_dir.exists():
        raise ValueError(f"Sweep directory not found: {sweep_dir}")
    
    # Get all run configs from sweep
    config_files = list(sweep_dir.glob("config-*.yaml"))
    
    if not config_files:
        raise ValueError(f"No run configs found in sweep directory: {sweep_dir}")
    
    # Find all runs from this sweep and their summaries
    runs = []
    for config_file in config_files:
        run_id = config_file.stem.replace("config-", "")
        
        # Find the run directory
        run_dirs = list(wandb_dir.glob(f"run-*-{run_id}"))
        if not run_dirs:
            continue
        
        run_dir = run_dirs[0]
        summary_file = run_dir / "files" / "wandb-summary.json"
        config_file_full = run_dir / "files" / "config.yaml"
        
        if summary_file.exists() and config_file_full.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                
                with open(config_file_full, 'r', encoding='utf-8') as f:
                    config_raw = yaml.safe_load(f)
                
                # Extract actual values from config (W&B stores them as {'value': ...})
                config = {}
                for key, value in config_raw.items():
                    if key == '_wandb':
                        continue  # Skip W&B metadata
                    if isinstance(value, dict) and 'value' in value:
                        config[key] = value['value']
                    else:
                        config[key] = value
                
                val_f1 = summary.get('val/f1', 0)
                runs.append({
                    'id': run_id,
                    'name': run_dir.name,
                    'config': config,
                    'summary': summary,
                    'val_f1': val_f1
                })
            except Exception as e:
                print(f"[WARN] Could not read run {run_id}: {e}")
                continue
    
    if not runs:
        raise ValueError(f"No valid runs found in sweep {sweep_id}")
    
    # Find best run by val/f1
    best_run = max(runs, key=lambda r: r['val_f1'])
    
    print(f"[OK] Found {len(runs)} runs in local cache")
    print(f"Best run: {best_run['name']}")
    print(f"Best config: {best_run['config']}")
    print(f"Best F1 score: {best_run['val_f1']:.4f}")
    
    # Return a dict that mimics the wandb Run object interface
    class LocalRun:
        def __init__(self, run_data):
            self.id = run_data['id']
            self.name = run_data['name']
            self.config = run_data['config']
            self.summary = run_data['summary']
    
    return LocalRun(best_run)

def register_best_model(sweep_id=None, best_run=None, use_offline=False):
    """
    Register the best model from a sweep as a W&B artifact.
    
    Args:
        sweep_id: Optional sweep ID to get best run from
        best_run: Optional best run object (if already retrieved)
        use_offline: If True, use local cache only (no W&B API calls)
    """
    # Get best run
    if best_run is None:
        if sweep_id is None:
            raise ValueError("Either sweep_id or best_run must be provided")
        best_run = get_best_run_from_sweep(sweep_id, use_local_cache=use_offline)
    
    # Initialize W&B run for model registration (with offline mode support)
    try:
        run = wandb.init(
            project=PROJECT,
            entity=ENTITY,
            job_type="register-model",
            notes=f"Registering best model from run: {getattr(best_run, 'name', 'local-cache')}"
        )
    except Exception as e:
        if use_offline:
            print(f"[WARN] Could not initialize W&B run: {e}")
            print("[INFO] Continuing in offline mode - model will be saved locally only")
            run = None
        else:
            raise
    
    # Load versioned dataset from W&B (with fallback to local cache)
    print("Loading dataset from W&B artifact...")
    data_dir = None
    
    try:
        if run is not None:
            artifact = run.use_artifact(f'{ENTITY}/{PROJECT}/football-player-dataset:latest')
            data_dir = Path(artifact.download())
            print(f"[OK] Dataset downloaded from W&B artifact")
        else:
            raise ConnectionError("W&B run not initialized (offline mode)")
    except Exception as e:
        print(f"[WARN] Could not download artifact from W&B: {e}")
        print("[INFO] Trying to use local cached artifact...")
        
        # Try to find local artifact cache (from Phase 1)
        from mlops_config import OUTPUT_DIR
        local_data_dir = Path(OUTPUT_DIR)
        
        # Also check wandb artifact cache
        wandb_cache = Path("wandb") / "artifacts" / f"{ENTITY}__{PROJECT}__football-player-dataset"
        
        if local_data_dir.exists() and (local_data_dir / "train.csv").exists():
            print(f"[OK] Using local dataset from: {local_data_dir}")
            data_dir = local_data_dir
        elif wandb_cache.exists():
            # Find the latest version in cache
            versions = sorted([d for d in wandb_cache.iterdir() if d.is_dir()], reverse=True)
            if versions:
                latest_version = versions[0]
                data_files = list(latest_version.rglob("*.csv"))
                if data_files:
                    print(f"[OK] Using cached artifact from: {latest_version}")
                    data_dir = data_files[0].parent
        else:
            print("[FAIL] No local dataset found.")
            print("[INFO] Options:")
            print("  1. Check your internet connection and try again")
            print("  2. Run Phase 1 first to create local dataset")
            print("  3. Wait for network to recover and retry")
            raise ConnectionError(f"Could not load dataset: {e}")
    
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    # Prepare features and target
    feature_columns = [col for col in train_df.columns if col not in ['fifa_version', 'big_potential']]
    target = 'big_potential'
    
    X_train = train_df[feature_columns]
    y_train = train_df[target]
    X_val = val_df[feature_columns]
    y_val = val_df[target]
    X_test = test_df[feature_columns]
    y_test = test_df[target]
    
    # Get best hyperparameters
    best_config = best_run.config
    
    # Create semi-supervised split with best labeled ratio
    labeled_ratio = best_config.get('labeled_ratio', 0.2)
    
    X_labeled, X_unlabeled, y_labeled, y_unlabeled_true = train_test_split(
        X_train, y_train,
        test_size=1 - labeled_ratio,
        stratify=y_train,
        random_state=best_config.get('random_state', 42)
    )
    
    y_unlabeled = np.full(len(y_unlabeled_true), -1)
    
    # Scale features
    scaler = StandardScaler()
    X_labeled_scaled = scaler.fit_transform(X_labeled)
    X_unlabeled_scaled = scaler.transform(X_unlabeled)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Combine for self-training
    X_train_ssl = np.vstack([X_labeled_scaled, X_unlabeled_scaled])
    y_train_ssl = np.concatenate([y_labeled.values, y_unlabeled])
    
    # Create XGBoost base estimator with best hyperparameters
    xgb_base = XGBClassifier(
        n_estimators=int(best_config.get('n_estimators', 250)),
        max_depth=int(best_config.get('max_depth', 3)),
        learning_rate=best_config.get('learning_rate', 0.03),
        subsample=best_config.get('subsample', 0.7),
        colsample_bytree=best_config.get('colsample_bytree', 0.7),
        reg_alpha=best_config.get('reg_alpha', 0.5),
        reg_lambda=best_config.get('reg_lambda', 2.0),
        scale_pos_weight=best_config.get('scale_pos_weight', 3),
        eval_metric="logloss",
        tree_method="hist",
        random_state=best_config.get('random_state', 42),
        n_jobs=-1
    )
    
    # Create self-training classifier
    self_training = SelfTrainingClassifier(
        xgb_base,
        threshold=0.9,
        criterion='threshold',
        k_best=10,
        verbose=True
    )
    
    # Train best model on full training set
    print("Training best model on full training set...")
    self_training.fit(X_train_ssl, y_train_ssl)
    
    # Evaluate on validation and test sets
    y_val_pred = self_training.predict(X_val_scaled)
    y_val_proba = self_training.predict_proba(X_val_scaled)[:, 1]
    y_test_pred = self_training.predict(X_test_scaled)
    y_test_proba = self_training.predict_proba(X_test_scaled)[:, 1]
    
    val_f1 = f1_score(y_val, y_val_pred)
    val_roc_auc = roc_auc_score(y_val, y_val_proba)
    test_f1 = f1_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"\nFinal Model Performance:")
    print(f"Validation F1: {val_f1:.4f}, ROC-AUC: {val_roc_auc:.4f}")
    print(f"Test F1: {test_f1:.4f}, ROC-AUC: {test_roc_auc:.4f}")
    
    # Save model and scaler
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "best_self_training_xgb_model.joblib"
    scaler_path = model_dir / "best_self_training_xgb_scaler.joblib"
    
    joblib.dump(self_training, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    
    # Create model artifact
    model_artifact = wandb.Artifact(
        name="football-superstar-predictor",
        type="model",
        description="XGBoost self-training classifier for football player big potential prediction",
        metadata={
            "model_type": "XGBoost + SelfTraining",
            "hyperparameters": dict(best_config),
            "val_f1": float(val_f1),
            "val_roc_auc": float(val_roc_auc),
            "test_f1": float(test_f1),
            "test_roc_auc": float(test_roc_auc),
            "best_run_id": getattr(best_run, 'id', 'local-cache'),
            "best_run_name": getattr(best_run, 'name', 'local-cache'),
            "num_features": len(feature_columns),
            "features": feature_columns
        }
    )
    
    # Add model files to artifact
    model_artifact.add_file(str(model_path))
    model_artifact.add_file(str(scaler_path))
    
    # Log artifact (if W&B is available)
    if run is not None:
        try:
            run.log_artifact(model_artifact)
            run.finish()
            print("\n[SUCCESS] Best model registered in W&B!")
            print(f"View at: https://wandb.ai/{ENTITY}/{PROJECT}/artifacts/model/football-superstar-predictor")
        except Exception as e:
            print(f"[WARN] Could not upload to W&B: {e}")
            print("[INFO] Model saved locally and will be uploaded when connection is restored")
    else:
        print("\n[SUCCESS] Best model saved locally!")
        print(f"[INFO] Model files: {model_path}, {scaler_path}")
        print(f"[INFO] Upload to W&B when connection is restored: wandb sync")
    
    return self_training, scaler

if __name__ == "__main__":
    # Login to W&B (if not already logged in)
    # wandb.login()
    
    # Option 1: Register best model from a specific sweep
    # sweep_id = "your-sweep-id-here"
    # register_best_model(sweep_id=sweep_id)
    
    # Option 2: Get best run manually and register
    # api = wandb.Api()
    # sweep = api.sweep(f"{ENTITY}/{PROJECT}/your-sweep-id")
    # best_run = sweep.best_run()
    # register_best_model(best_run=best_run)
    
    print("Please provide a sweep_id or best_run to register the model.")
