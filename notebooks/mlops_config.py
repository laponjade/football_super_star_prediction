"""
MLOps Configuration File
Centralized configuration for all MLOps scripts.
Modify the ENTITY and PROJECT here to update all scripts at once.
"""

# ============================================================================
# W&B Configuration
# ============================================================================
# Your W&B team/entity name
# Change this line to update ENTITY in all MLOps scripts
ENTITY = "abdoubendaia7-cole-sup-rieure-en-informatique-sidi-bel-abbes"

# Your W&B project name
PROJECT = "football-superstar-prediction"

# ============================================================================
# Data Configuration
# ============================================================================
DATA_PATH = "../data/feature_engineered_data_v2.csv"
OUTPUT_DIR = "football_data_prepared"

# ============================================================================
# Model Configuration
# ============================================================================
# Default hyperparameters (can be overridden in training scripts)
DEFAULT_CONFIG = {
    "n_estimators": 250,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "scale_pos_weight": 3,
    "labeled_ratio": 0.2,
    "random_state": 42
}

# ============================================================================
# Sweep Configuration
# ============================================================================
SWEEP_CONFIG = {
    'method': 'bayes',  # Bayesian optimization
    'metric': {
        'name': 'val/f1',  # Optimize for F1 score
        'goal': 'maximize'
    },
    'parameters': {
        'n_estimators': {'min': 100, 'max': 300, 'distribution': 'int_uniform'},
        'max_depth': {'min': 3, 'max': 7, 'distribution': 'int_uniform'},
        'learning_rate': {'min': 0.01, 'max': 0.1, 'distribution': 'uniform'},
        'subsample': {'min': 0.6, 'max': 0.9, 'distribution': 'uniform'},
        'colsample_bytree': {'min': 0.6, 'max': 0.9, 'distribution': 'uniform'},
        'reg_alpha': {'min': 0.1, 'max': 1.0, 'distribution': 'uniform'},
        'reg_lambda': {'min': 1.0, 'max': 3.0, 'distribution': 'uniform'},
        'scale_pos_weight': {'min': 2, 'max': 4, 'distribution': 'uniform'},
        'labeled_ratio': {'values': [0.1, 0.2, 0.3]},
        'random_state': {'value': 42}
    }
}

# Number of sweep trials
SWEEP_TRIALS = 20
