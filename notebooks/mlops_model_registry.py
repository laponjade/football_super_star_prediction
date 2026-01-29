"""
MLOps Phase 4: Model Registration
This script selects the best model from a sweep and registers it in W&B.
"""

import pandas as pd
import numpy as np
import wandb
import joblib
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

def get_best_run_from_sweep(sweep_id):
    """
    Get the best run from a completed sweep.
    
    Args:
        sweep_id: W&B sweep ID
    
    Returns:
        Best run object
    """
    api = wandb.Api()
    sweep = api.sweep(f"{ENTITY}/{PROJECT}/{sweep_id}")
    best_run = sweep.best_run()
    
    print(f"Best run: {best_run.name}")
    print(f"Best config: {best_run.config}")
    print(f"Best F1 score: {best_run.summary.get('val/f1', 'N/A')}")
    
    return best_run

def register_best_model(sweep_id=None, best_run=None):
    """
    Register the best model from a sweep as a W&B artifact.
    
    Args:
        sweep_id: Optional sweep ID to get best run from
        best_run: Optional best run object (if already retrieved)
    """
    # Get best run
    if best_run is None:
        if sweep_id is None:
            raise ValueError("Either sweep_id or best_run must be provided")
        best_run = get_best_run_from_sweep(sweep_id)
    
    # Initialize W&B run for model registration
    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        job_type="register-model",
        notes=f"Registering best model from run: {best_run.name}"
    )
    
    # Load versioned dataset from W&B
    print("Loading dataset from W&B artifact...")
    artifact = run.use_artifact(f'{ENTITY}/{PROJECT}/football-player-dataset:latest')
    data_dir = Path(artifact.download())
    
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
            "best_run_id": best_run.id,
            "best_run_name": best_run.name,
            "num_features": len(feature_columns),
            "features": feature_columns
        }
    )
    
    # Add model files to artifact
    model_artifact.add_file(str(model_path))
    model_artifact.add_file(str(scaler_path))
    
    # Log artifact
    run.log_artifact(model_artifact)
    
    # Optionally link to model registry (if available)
    # run.link_artifact(
    #     model_artifact,
    #     f"{ENTITY}/model-registry/football-superstar-predictor",
    #     aliases=["production", "best"]
    # )
    
    run.finish()
    
    print("\n[SUCCESS] Best model registered in W&B!")
    print(f"View at: https://wandb.ai/{ENTITY}/{PROJECT}/artifacts/model/football-superstar-predictor")
    
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
