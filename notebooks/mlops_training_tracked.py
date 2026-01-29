"""
MLOps Phase 2: Training with Experiment Tracking
This script trains the XGBoost model with W&B experiment tracking.
"""

import pandas as pd
import numpy as np
import wandb
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration - Import from centralized config
from mlops_config import ENTITY, PROJECT, DEFAULT_CONFIG

def train_with_tracking(config=None):
    """
    Train XGBoost model with self-training and track experiment in W&B.
    
    Args:
        config: Optional dict with hyperparameters. If None, uses default values.
    """
    # Initialize W&B run
    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        job_type="training",
        config=config or DEFAULT_CONFIG
    )
    
    config = wandb.config
    
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
    
    # Log dataset info
    wandb.log({
        "dataset/train_size": len(X_train),
        "dataset/val_size": len(X_val),
        "dataset/test_size": len(X_test),
        "dataset/num_features": len(feature_columns),
        "dataset/train_class_0": int((y_train == 0).sum()),
        "dataset/train_class_1": int((y_train == 1).sum()),
        "dataset/val_class_0": int((y_val == 0).sum()),
        "dataset/val_class_1": int((y_val == 1).sum()),
    })
    
    # Create semi-supervised split
    from sklearn.model_selection import train_test_split
    labeled_ratio = config.get("labeled_ratio", 0.2)
    
    X_labeled, X_unlabeled, y_labeled, y_unlabeled_true = train_test_split(
        X_train, y_train,
        test_size=1 - labeled_ratio,
        stratify=y_train,
        random_state=config.random_state
    )
    
    y_unlabeled = np.full(len(y_unlabeled_true), -1)
    
    print(f"Labeled samples: {len(y_labeled)} ({len(y_labeled)/len(y_train)*100:.1f}%)")
    print(f"Unlabeled samples: {len(y_unlabeled)} ({len(y_unlabeled)/len(y_train)*100:.1f}%)")
    
    # Scale features
    scaler = StandardScaler()
    X_labeled_scaled = scaler.fit_transform(X_labeled)
    X_unlabeled_scaled = scaler.transform(X_unlabeled)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Combine for self-training
    X_train_ssl = np.vstack([X_labeled_scaled, X_unlabeled_scaled])
    y_train_ssl = np.concatenate([y_labeled.values, y_unlabeled])
    
    # Create XGBoost base estimator
    xgb_base = XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        scale_pos_weight=config.scale_pos_weight,
        eval_metric="logloss",
        tree_method="hist",
        random_state=config.random_state,
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
    
    # Train model
    print("\nTraining self-training model...")
    import time
    start_time = time.time()
    
    self_training.fit(X_train_ssl, y_train_ssl)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    wandb.log({"training_time": training_time})
    
    # Log self-training iterations
    if hasattr(self_training, 'labeled_iter_'):
        for i, n_labeled in enumerate(self_training.labeled_iter_):
            wandb.log({f"self_training/labels_added_iter_{i}": n_labeled})
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    y_val_pred = self_training.predict(X_val_scaled)
    y_val_proba = self_training.predict_proba(X_val_scaled)[:, 1]
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_roc_auc = roc_auc_score(y_val, y_val_proba)
    val_log_loss = log_loss(y_val, y_val_proba)
    
    # Log validation metrics
    wandb.log({
        "val/accuracy": val_accuracy,
        "val/precision": val_precision,
        "val/recall": val_recall,
        "val/f1": val_f1,
        "val/roc_auc": val_roc_auc,
        "val/log_loss": val_log_loss
    })
    
    print(f"\nValidation Metrics:")
    print(f"Accuracy:  {val_accuracy:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall:    {val_recall:.4f}")
    print(f"F1 Score:  {val_f1:.4f}")
    print(f"ROC-AUC:   {val_roc_auc:.4f}")
    print(f"Log Loss:  {val_log_loss:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_test_pred = self_training.predict(X_test_scaled)
    y_test_proba = self_training.predict_proba(X_test_scaled)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    test_log_loss = log_loss(y_test, y_test_proba)
    
    # Log test metrics
    wandb.log({
        "test/accuracy": test_accuracy,
        "test/precision": test_precision,
        "test/recall": test_recall,
        "test/f1": test_f1,
        "test/roc_auc": test_roc_auc,
        "test/log_loss": test_log_loss
    })
    
    # Create and log confusion matrix
    cm_val = confusion_matrix(y_val, y_val_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['No BP', 'BP'], yticklabels=['No BP', 'BP'])
    axes[0].set_title('Validation Set Confusion Matrix')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['No BP', 'BP'], yticklabels=['No BP', 'BP'])
    axes[1].set_title('Test Set Confusion Matrix')
    axes[1].set_ylabel('Actual')
    axes[1].set_xlabel('Predicted')
    
    plt.tight_layout()
    wandb.log({"confusion_matrices": wandb.Image(fig)})
    plt.close()
    
    # Log feature importance - access through the wrapped estimator
    try:
        # Get the base estimator from self-training
        base_estimator = self_training.base_estimator
        if hasattr(base_estimator, 'feature_importances_'):
            feature_importance = base_estimator.feature_importances_
        else:
            # If wrapped, try to get from the underlying model
            feature_importance = np.zeros(len(feature_columns))  # Fallback
    except Exception as e:
        print(f"Warning: Could not get feature importance: {e}")
        feature_importance = np.zeros(len(feature_columns))  # Fallback
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # Create feature importance plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df.head(15), y='feature', x='importance')
    plt.title('Top 15 Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    wandb.log({"feature_importance": wandb.Image(plt)})
    plt.close()
    
    # Log classification reports as tables
    val_report = classification_report(y_val, y_val_pred, output_dict=True, target_names=['No BP', 'BP'])
    test_report = classification_report(y_test, y_test_pred, output_dict=True, target_names=['No BP', 'BP'])
    
    wandb.log({
        "val/classification_report": wandb.Table(
            dataframe=pd.DataFrame(val_report).transpose()
        ),
        "test/classification_report": wandb.Table(
            dataframe=pd.DataFrame(test_report).transpose()
        )
    })
    
    # Save model and scaler locally (will be registered later)
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    joblib.dump(self_training, model_dir / "self_training_xgb_model.joblib")
    joblib.dump(scaler, model_dir / "self_training_xgb_scaler.joblib")
    
    print("\n[SUCCESS] Model training completed and tracked in W&B!")
    
    run.finish()
    
    return self_training, scaler

if __name__ == "__main__":
    # Login to W&B (if not already logged in)
    # wandb.login()
    
    # Train with default config
    model, scaler = train_with_tracking()
    
    # Or train with custom config
    # custom_config = {
    #     "n_estimators": 300,
    #     "max_depth": 4,
    #     "learning_rate": 0.05,
    #     "subsample": 0.8,
    #     "colsample_bytree": 0.8,
    #     "reg_alpha": 0.1,
    #     "reg_lambda": 1.0,
    #     "scale_pos_weight": 3,
    #     "labeled_ratio": 0.2,
    #     "random_state": 42
    # }
    # model, scaler = train_with_tracking(custom_config)
