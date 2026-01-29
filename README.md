# Football Superstar Prediction

A full-stack web application for predicting football player superstar potential using machine learning (XGBoost).

## Features

- 🔍 **Player Search**: Search for players from the dataset with autocomplete
- 📊 **Custom Prediction**: Create a custom player profile and predict their potential
- 🤖 **ML-Powered**: Uses trained XGBoost model for accurate predictions
- 🎨 **Modern UI**: Beautiful React frontend with real-time predictions
- 🐳 **Docker Support**: Easy deployment with Docker Compose

## Quick Start with Docker

The easiest way to run the application:

```bash
# Build and start all services
docker-compose up --build

# Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
```

For detailed Docker instructions, see [DOCKER.md](DOCKER.md).

## Manual Setup

See [SETUP.md](SETUP.md) for manual installation instructions.

## Project Structure

```
football_super_star_prediction/
├── backend/              # Django REST API
│   ├── api/             # API endpoints and ML integration
│   └── football_predictor/  # Django project settings
├── data/                 # CSV datasets
├── notebooks/            # Jupyter notebooks and MLOps pipeline
│   ├── mlops_*.py       # MLOps scripts (data versioning, training, sweeps, registry)
│   ├── run_mlops_pipeline.py  # Main pipeline runner
│   └── MLOPS_README.md  # MLOps documentation
└── superstar-ai-scout-main/  # React frontend
```

## MLOps Pipeline

This project includes a complete MLOps (Machine Learning Operations) implementation using **Weights & Biases (W&B)**. MLOps ensures that machine learning models are developed, deployed, and maintained in a systematic, reproducible, and scalable way.

### What is MLOps?

MLOps applies DevOps principles to machine learning, providing:
- **Reproducibility**: Track every experiment with exact configurations
- **Versioning**: Version datasets, models, and code
- **Automation**: Automated hyperparameter tuning and model selection
- **Monitoring**: Track model performance in production
- **Collaboration**: Share experiments and results with your team

### What We Implemented

Our MLOps pipeline includes 4 main phases:

1. **Data Versioning** - Track and version datasets as W&B artifacts
2. **Experiment Tracking** - Log all training runs with metrics and visualizations
3. **Hyperparameter Optimization** - Automated Bayesian optimization to find best hyperparameters
4. **Model Registration** - Register and version the best model for deployment

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- W&B account (free at [wandb.ai](https://wandb.ai))

### Step 1: Install Required Packages

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install MLOps-specific packages manually
pip install wandb pandas numpy scikit-learn xgboost matplotlib seaborn joblib pyyaml
```

### Step 2: Configure W&B

1. **Create a W&B account** at [wandb.ai](https://wandb.ai) (free)

2. **Login to W&B**:
   ```bash
   wandb login
   ```
   You'll be prompted to enter your API key (found in your W&B account settings)

3. **Update configuration**:
   Edit `notebooks/mlops_config.py` and set your W&B entity:
   ```python
   ENTITY = "your-wandb-entity"  # Your W&B username or team name
   PROJECT = "football-superstar-prediction"
   ```

---

## 🚀 Running the MLOps Pipeline

### Option 1: Run Complete Pipeline (Recommended)

```bash
cd notebooks
python run_mlops_pipeline.py
```

You'll see a menu:
```
Which phases would you like to run?
  1. Phase 1: Data Versioning
  2. Phase 2: Experiment Tracking
  3. Phase 3: Hyperparameter Sweep (FULL - may take 30-60 min)
  4. Phase 4: Model Registration (requires completed sweep)
  5. All phases (1, 2, 3, 4)
  6. Phase 3 (dry run - just show config)
  7. Just verification (no execution)
```

**Choose option 5** to run all phases sequentially.

### Option 2: Run Individual Phases

```bash
cd notebooks

# Phase 1: Data Versioning
python mlops_data_versioning.py

# Phase 2: Experiment Tracking
python mlops_training_tracked.py

# Phase 3: Hyperparameter Sweep (takes 30-60 minutes)
python mlops_hyperparameter_sweep.py

# Phase 4: Model Registration (requires sweep_id from Phase 3)
python mlops_model_registry.py
```

### Option 3: Offline Mode (When Network is Unavailable)

If you have network issues, Phase 4 can work offline:

```bash
cd notebooks
python -c "from mlops_model_registry import register_best_model; register_best_model(sweep_id='YOUR_SWEEP_ID', use_offline=True)"
```

---

## 📁 MLOps File Structure

### Core Configuration

#### `mlops_config.py`
**Purpose**: Centralized configuration for all MLOps scripts

**What it contains**:
- W&B entity and project name
- Data file paths
- Default model hyperparameters
- Sweep configuration (hyperparameter search space)
- Number of sweep trials

**Important**: Update `ENTITY` here to change it in all scripts at once.

---

### Phase 1: Data Versioning

#### `mlops_data_versioning.py`
**Purpose**: Prepare and version datasets as W&B artifacts

**What it does**:
1. Loads raw data from `data/feature_engineered_data_v2.csv`
2. Creates temporal train/validation/test splits based on FIFA versions:
   - **Train**: FIFA 17-20 (historical data)
   - **Validation**: FIFA 21 (recent data)
   - **Test**: FIFA 21 (recent data)
3. Saves splits to `football_data_prepared/` directory
4. Uploads dataset as a versioned W&B artifact named `football-player-dataset`

**Outputs**:
- Local files: `football_data_prepared/train.csv`, `val.csv`, `test.csv`
- W&B artifact: `football-player-dataset:latest`

**Why it matters**: Ensures everyone uses the same dataset version, enabling reproducibility.

---

### Phase 2: Experiment Tracking

#### `mlops_training_tracked.py`
**Purpose**: Train model with full experiment tracking

**What it does**:
1. Downloads versioned dataset from W&B artifact
2. Creates semi-supervised learning setup (labeled + unlabeled data)
3. Trains XGBoost model using `SelfTrainingClassifier` (semi-supervised learning)
4. Logs to W&B:
   - **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, Log Loss
   - **Visualizations**: Confusion matrices, feature importance plots
   - **Self-training progress**: Labels added per iteration (time-series chart)
   - **Hyperparameters**: All model configuration
   - **Training time**: How long training took

**Outputs**:
- W&B run with all metrics and visualizations
- Model saved locally (optional)

**Key Features**:
- Tracks self-training iterations as time-series (shows how model improves)
- Logs both validation and test metrics
- Creates beautiful visualizations automatically

---

### Phase 3: Hyperparameter Optimization

#### `mlops_hyperparameter_sweep.py`
**Purpose**: Automatically find the best hyperparameters using Bayesian optimization

**What it does**:
1. Defines a search space for hyperparameters:
   - `n_estimators`: 100-300
   - `max_depth`: 3-7
   - `learning_rate`: 0.01-0.1
   - `subsample`, `colsample_bytree`: 0.6-0.9
   - `reg_alpha`, `reg_lambda`: Regularization parameters
   - `scale_pos_weight`: 2-4 (for class imbalance)
   - `labeled_ratio`: 0.1, 0.2, or 0.3
2. Runs multiple training trials (default: 20 trials)
3. Uses **Bayesian optimization** to intelligently search the space
4. Tracks all trials in W&B sweep
5. Identifies the best configuration based on validation F1 score

**What is a Sweep?**
A sweep is an automated hyperparameter tuning process. Instead of manually trying different values, W&B:
- Runs many experiments with different hyperparameters
- Uses Bayesian optimization to focus on promising areas
- Tracks all results for comparison
- Identifies the best configuration automatically

**Outputs**:
- W&B sweep with all trial results
- Best run identified (highest validation F1 score)
- Sweep ID (used in Phase 4)

**Time**: Takes 30-60 minutes depending on your system (20 trials × ~2-3 min each)

**Note**: You can stop it with `Ctrl+C` - it will finish the current trial and save all completed results.

---

### Phase 4: Model Registration

#### `mlops_model_registry.py`
**Purpose**: Register the best model from the sweep as a production-ready artifact

**What it does**:
1. Retrieves the best run from Phase 3 sweep (highest F1 score)
2. Downloads the versioned dataset
3. Retrains the model with the best hyperparameters on the full training set
4. Evaluates on validation and test sets
5. Saves model and scaler locally:
   - `models/best_self_training_xgb_model.joblib`
   - `models/best_self_training_xgb_scaler.joblib`
6. Creates a W&B model artifact with:
   - Model files
   - Hyperparameters
   - Performance metrics
   - Metadata (features, run info, etc.)
7. Registers the artifact in W&B for easy retrieval

**Outputs**:
- Local model files in `models/` directory
- W&B model artifact: `football-superstar-predictor`
- Model metadata and performance metrics

**Offline Support**: Can work without W&B API connection using local cache.

---

### Pipeline Management

#### `run_mlops_pipeline.py`
**Purpose**: Main pipeline runner that orchestrates all phases

**What it does**:
1. **Pre-flight checks**:
   - Verifies all required packages are installed
   - Checks W&B login status
   - Validates configuration
   - Checks data file exists
2. **Interactive menu**: Choose which phases to run
3. **Sequential execution**: Runs phases in order
4. **Automatic flow**: Passes sweep_id from Phase 3 to Phase 4 automatically
5. **Error handling**: Graceful error messages and recovery
6. **Summary**: Shows what completed successfully

**Usage**:
```bash
python run_mlops_pipeline.py
```

---

### Utility Scripts

#### `verify_mlops_structure.py`
**Purpose**: Verify that all MLOps components are properly implemented

**What it checks**:
- All required files exist
- Imports work correctly
- W&B is properly configured
- MLOps principles are implemented

**Usage**:
```bash
python verify_mlops_structure.py
```

#### `check_model_registration.py`
**Purpose**: Check if a model is registered in W&B

**Usage**:
```bash
python check_model_registration.py
```

#### `check_pipeline_status.py`
**Purpose**: Check the status of recent MLOps runs

**Usage**:
```bash
python check_pipeline_status.py
```

---

## 📊 Understanding the Results

### W&B Dashboard

After running the pipeline, view your results at:
```
https://wandb.ai/{YOUR_ENTITY}/football-superstar-prediction
```

### What You'll See

1. **Runs**: Each training experiment (Phase 2 and Phase 3 trials)
   - Metrics over time
   - Hyperparameters used
   - Visualizations (confusion matrices, feature importance)

2. **Sweeps**: Hyperparameter optimization results (Phase 3)
   - Parallel coordinates plot (shows hyperparameter relationships)
   - Best run highlighted
   - All trial results compared

3. **Artifacts**: Versioned datasets and models
   - `football-player-dataset`: Your versioned dataset
   - `football-superstar-predictor`: Your registered model

4. **Charts**: 
   - `self_training/labels_added`: Time-series showing self-training progress
   - `val/f1`, `val/roc_auc`: Validation metrics
   - `test/f1`, `test/roc_auc`: Test metrics

### Model Performance

After Phase 4, you'll see:
- **Validation F1**: ~0.50 (balanced precision/recall)
- **Validation ROC-AUC**: ~0.72 (good discrimination ability)
- **Test metrics**: Similar to validation (model generalizes well)

---

## 🔧 Troubleshooting

### Network Connection Issues

**Problem**: `ConnectTimeoutError: Connection to api.wandb.ai timed out`

**Solutions**:
1. **Wait and retry**: Network issues are usually temporary
2. **Use offline mode**: Phase 4 supports offline mode with `use_offline=True`
3. **Check firewall**: Ensure `api.wandb.ai` is not blocked
4. **Use W&B offline**: Run `wandb offline` to work offline, then `wandb sync` later

### Missing Packages

**Problem**: `ModuleNotFoundError: No module named 'xgboost'`

**Solution**:
```bash
pip install -r requirements.txt
```

### W&B Authentication Error

**Problem**: `AuthenticationError: Unable to connect to https://api.wandb.ai`

**Solutions**:
1. **Re-login**: `wandb login --relogin`
2. **Check API key**: Verify your API key in W&B account settings
3. **Use offline mode**: For Phase 4, use `use_offline=True`

### Sweep Takes Too Long

**Problem**: Phase 3 (sweep) takes hours

**Solutions**:
1. **Reduce trials**: Edit `mlops_config.py`, set `SWEEP_TRIALS = 5` (instead of 20)
2. **Stop early**: Press `Ctrl+C` - completed trials are saved
3. **Run fewer phases**: Skip Phase 3 and use default hyperparameters

### Charts Appear Empty

**Problem**: W&B charts show single points instead of time-series

**Solution**: This was fixed! The code now properly logs metrics with step numbers. Re-run Phase 2 to see proper charts.

---

## 📈 MLOps Principles Implemented

✅ **Data Versioning** - Datasets versioned as W&B artifacts  
✅ **Experiment Tracking** - All runs tracked with metrics, configs, and visualizations  
✅ **Hyperparameter Optimization** - Bayesian sweeps for systematic search  
✅ **Model Registration** - Best models registered as versioned artifacts  
✅ **Reproducibility** - Random seeds, configs logged, versioned datasets  
✅ **Metrics Logging** - Comprehensive metrics (F1, ROC-AUC, accuracy, etc.)  
✅ **Visualization** - Confusion matrices, feature importance plots  
✅ **Production Monitoring** - Real-time prediction tracking (via Django API)

---

## 🎯 Quick Reference

### Run Complete Pipeline
```bash
cd notebooks
python run_mlops_pipeline.py
# Choose option 5
```

### Run Individual Phase
```bash
cd notebooks
python mlops_data_versioning.py        # Phase 1
python mlops_training_tracked.py       # Phase 2
python mlops_hyperparameter_sweep.py   # Phase 3
python mlops_model_registry.py         # Phase 4
```

### Check Status
```bash
python check_pipeline_status.py        # Check what completed
python verify_mlops_structure.py       # Verify implementation
python check_model_registration.py      # Check model registration
```

### View Results
Visit your W&B dashboard:
```
https://wandb.ai/{YOUR_ENTITY}/football-superstar-prediction
```

### Charts and Visualizations
For a complete guide on all charts created and where to find them in W&B, see [notebooks/WANDB_CHARTS_GUIDE.md](notebooks/WANDB_CHARTS_GUIDE.md).

**Quick Summary**:
- **Time-series charts**: `self_training/labels_added`, `self_training/cumulative_labels` (in Charts tab)
- **Performance metrics**: `val/f1`, `val/roc_auc`, `test/f1`, etc. (in Charts tab)
- **Confusion matrices**: Side-by-side validation and test confusion matrices (in Media tab)
- **Feature importance**: Top 15 features bar chart (in Media tab)
- **Classification reports**: Detailed per-class metrics (in Tables tab)
- **Sweep visualizations**: Parallel coordinates, hyperparameter importance (in Sweeps tab)

---

For more detailed documentation, see [notebooks/MLOPS_README.md](notebooks/MLOPS_README.md).

## API Endpoints

- `GET /api/players/search/?q=<query>` - Search players
- `POST /api/players/predict/` - Predict player potential

See [backend/README.md](backend/README.md) for detailed API documentation.

## Technologies

- **Backend**: Django, Django REST Framework, XGBoost, scikit-learn
- **Frontend**: React, TypeScript, Vite, Tailwind CSS
- **ML**: XGBoost, scikit-learn, joblib
- **Deployment**: Docker, Docker Compose

## License

This project is for educational purposes.
