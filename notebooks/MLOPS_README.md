# MLOps Pipeline - Football Superstar Prediction

This directory contains the complete MLOps implementation for the Football Superstar Prediction project using Weights & Biases (W&B).

## 📁 File Structure

### Core MLOps Scripts

#### Configuration
- **`mlops_config.py`** - Centralized configuration file
  - W&B entity and project settings
  - Data paths
  - Model hyperparameters
  - Sweep configuration
  - **⚠️ IMPORTANT**: Update `ENTITY` here to change it in all scripts

#### Phase 1: Data Versioning
- **`mlops_data_versioning.py`** - Versions datasets as W&B artifacts
  - Loads raw data
  - Creates temporal train/val/test splits
  - Logs dataset artifact to W&B

#### Phase 2: Experiment Tracking
- **`mlops_training_tracked.py`** - Tracks training runs with metrics
  - Fetches versioned dataset from W&B
  - Trains XGBoost model with self-training
  - Logs metrics, visualizations, and artifacts

#### Phase 3: Hyperparameter Optimization
- **`mlops_hyperparameter_sweep.py`** - Runs W&B hyperparameter sweeps
  - Bayesian optimization
  - Tunes XGBoost parameters
  - Returns best configuration

#### Phase 4: Model Registration
- **`mlops_model_registry.py`** - Registers best model as W&B artifact
  - Retrieves best run from sweep
  - Retrains with optimal hyperparameters
  - Registers model artifact

#### Phase 5: Production Monitoring
- **`mlops_production_monitoring.py`** - Production monitoring script
  - Model health checks
  - Data drift detection
  - Prediction monitoring
  - Performance metrics tracking
- **`backend/api/monitoring.py`** - Django API monitoring integration
  - Real-time prediction tracking
  - Latency and error monitoring
  - Automatic W&B logging

### Pipeline Management

- **`run_mlops_pipeline.py`** - Main pipeline runner
  - Runs all phases sequentially
  - Pre-flight checks (requirements, W&B login, config)
  - Interactive phase selection
  - Automatic flow between phases

- **`MLOPS_COMPLETE_PIPELINE.ipynb`** - Jupyter notebook version
  - Interactive step-by-step workflow
  - Same functionality as scripts

### Utilities

- **`verify_mlops_structure.py`** - Static verification of MLOps implementation
  - Checks file existence
  - Verifies imports and W&B usage
  - Validates MLOps principles

- **`check_model_registration.py`** - Check if model is registered in W&B
  - Queries W&B API
  - Displays model artifact details

- **`test_all_phases.py`** - Test script for all 4 phases
  - Runs complete pipeline with reduced trials
  - Useful for quick verification

- **`mlops_production_monitoring.py`** - Production monitoring
  - Model health checks
  - Data drift detection
  - Prediction monitoring
  - Performance metrics tracking

---

## 🚀 Quick Start

### 1. Configure W&B

Edit `mlops_config.py`:
```python
ENTITY = "your-wandb-entity"  # Your W&B team/user entity
PROJECT = "football-superstar-prediction"
```

### 2. Login to W&B

```bash
wandb login
```

### 3. Run Complete Pipeline

```bash
cd notebooks
python run_mlops_pipeline.py
# Choose option 5: All phases (1, 2, 3, 4)
```

### 4. Run Individual Phases

```bash
# Phase 1: Data Versioning
python mlops_data_versioning.py

# Phase 2: Experiment Tracking
python mlops_training_tracked.py

# Phase 3: Hyperparameter Sweep
python mlops_hyperparameter_sweep.py

# Phase 4: Model Registration (requires sweep_id)
python mlops_model_registry.py

# Phase 5: Production Monitoring
python mlops_production_monitoring.py
```

### 5. Production Monitoring (Automatic)

Monitoring is **automatic** when using the Django API:
```bash
cd backend
python manage.py runserver
# All predictions are automatically monitored!
```

View metrics in W&B dashboard under `job_type="production-monitoring"`

---

## 📊 MLOps Principles Implemented

✅ **Data Versioning** - Datasets versioned as W&B artifacts  
✅ **Experiment Tracking** - All runs tracked with metrics and configs  
✅ **Hyperparameter Optimization** - Bayesian sweeps for systematic search  
✅ **Model Registration** - Best models registered as versioned artifacts  
✅ **Reproducibility** - Random seeds, configs logged  
✅ **Metrics Logging** - Comprehensive metrics (F1, ROC-AUC, etc.)  
✅ **Visualization** - Confusion matrices, feature importance plots  

---

## 🔍 Verification

### Check Model Registration
```bash
python check_model_registration.py
```

### Verify MLOps Structure
```bash
python verify_mlops_structure.py
```

### Test All Phases
```bash
python test_all_phases.py
```

---

## 📝 Notes

- **Models**: Saved locally in `models/` but primarily use W&B artifacts
- **Data Splits**: Generated in `football_data_prepared/` but use W&B artifacts
- **W&B Runs**: Stored locally in `wandb/` (ignored by git)
- **Configuration**: All settings in `mlops_config.py` (single source of truth)

---

## 🔗 W&B Dashboard

View your project at:
```
https://wandb.ai/{ENTITY}/{PROJECT}
```

Replace `{ENTITY}` and `{PROJECT}` with your values from `mlops_config.py`.

---

**For detailed documentation, see the main project README.md**
