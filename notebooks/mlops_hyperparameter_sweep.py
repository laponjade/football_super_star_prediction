"""
MLOps Phase 3: Hyperparameter Optimization with W&B Sweeps
This script sets up and runs a hyperparameter sweep for XGBoost.
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
from mlops_config import ENTITY, PROJECT, SWEEP_CONFIG

# Use the sweep config from mlops_config
sweep_config = SWEEP_CONFIG

def train():
    """
    Training function for the sweep.
    This function is called by W&B sweep agent for each trial.
    """
    # Initialize W&B run (automatically part of sweep)
    run = wandb.init()
    
    # Load versioned dataset from W&B
    print("Loading dataset from W&B artifact...")
    artifact = run.use_artifact(f'{ENTITY}/{PROJECT}/football-player-dataset:latest')
    data_dir = Path(artifact.download())
    
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    
    # Prepare features and target
    feature_columns = [col for col in train_df.columns if col not in ['fifa_version', 'big_potential']]
    target = 'big_potential'
    
    X_train = train_df[feature_columns]
    y_train = train_df[target]
    X_val = val_df[feature_columns]
    y_val = val_df[target]
    
    # Create semi-supervised split
    labeled_ratio = wandb.config.labeled_ratio
    
    X_labeled, X_unlabeled, y_labeled, y_unlabeled_true = train_test_split(
        X_train, y_train,
        test_size=1 - labeled_ratio,
        stratify=y_train,
        random_state=wandb.config.random_state
    )
    
    y_unlabeled = np.full(len(y_unlabeled_true), -1)
    
    # Scale features
    scaler = StandardScaler()
    X_labeled_scaled = scaler.fit_transform(X_labeled)
    X_unlabeled_scaled = scaler.transform(X_unlabeled)
    X_val_scaled = scaler.transform(X_val)
    
    # Combine for self-training
    X_train_ssl = np.vstack([X_labeled_scaled, X_unlabeled_scaled])
    y_train_ssl = np.concatenate([y_labeled.values, y_unlabeled])
    
    # Create XGBoost base estimator with sweep hyperparameters
    xgb_base = XGBClassifier(
        n_estimators=int(wandb.config.n_estimators),
        max_depth=int(wandb.config.max_depth),
        learning_rate=wandb.config.learning_rate,
        subsample=wandb.config.subsample,
        colsample_bytree=wandb.config.colsample_bytree,
        reg_alpha=wandb.config.reg_alpha,
        reg_lambda=wandb.config.reg_lambda,
        scale_pos_weight=wandb.config.scale_pos_weight,
        eval_metric="logloss",
        tree_method="hist",
        random_state=wandb.config.random_state,
        n_jobs=-1
    )
    
    # Create self-training classifier
    self_training = SelfTrainingClassifier(
        xgb_base,
        threshold=0.9,
        criterion='threshold',
        k_best=10,
        verbose=False
    )
    
    # Train model
    print(f"Training with config: {wandb.config}")
    self_training.fit(X_train_ssl, y_train_ssl)
    
    # Evaluate on validation set
    y_val_pred = self_training.predict(X_val_scaled)
    y_val_proba = self_training.predict_proba(X_val_scaled)[:, 1]
    
    # Calculate metrics
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_roc_auc = roc_auc_score(y_val, y_val_proba)
    val_log_loss = log_loss(y_val, y_val_proba)
    
    # Log all metrics to W&B
    wandb.log({
        "val/accuracy": val_accuracy,
        "val/precision": val_precision,
        "val/recall": val_recall,
        "val/f1": val_f1,
        "val/roc_auc": val_roc_auc,
        "val/log_loss": val_log_loss
    })
    
    print(f"Validation F1: {val_f1:.4f}, ROC-AUC: {val_roc_auc:.4f}")
    
    run.finish()

def run_sweep(num_trials=None):
    """
    Initialize and run the hyperparameter sweep.
    
    Args:
        num_trials: Number of trials to run (defaults to SWEEP_TRIALS from config)
    
    Returns:
        sweep_id: The ID of the created sweep
    """
    from mlops_config import SWEEP_TRIALS
    if num_trials is None:
        num_trials = SWEEP_TRIALS
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project=PROJECT, entity=ENTITY)
    
    print(f"Created sweep with ID: {sweep_id}")
    print(f"Sweep URL: https://wandb.ai/{ENTITY}/{PROJECT}/sweeps/{sweep_id}")
    
    # Run sweep agent
    wandb.agent(sweep_id, train, count=num_trials)
    
    print(f"\n[SUCCESS] Sweep completed! View results at:")
    print(f"https://wandb.ai/{ENTITY}/{PROJECT}/sweeps/{sweep_id}")
    
    return sweep_id  # Return sweep_id for use in Phase 4

if __name__ == "__main__":
    # Login to W&B (if not already logged in)
    # wandb.login()
    
    # Run sweep with 20 trials (adjust as needed)
    run_sweep(num_trials=20)
