# W&B Charts and Visualizations Guide

This guide explains all the charts and visualizations created by the MLOps pipeline and where to find them in your W&B dashboard.

## 📊 Charts Created

### Phase 2: Experiment Tracking (`mlops_training_tracked.py`)

When you run Phase 2, the following charts and visualizations are automatically created:

---

## 1. Time-Series Charts (Metrics Over Time)

### Location in W&B:
**Dashboard → Your Run → Charts Tab → Metrics Section**

### Charts Created:

#### a) `self_training/labels_added`
- **What it shows**: Number of new labels added in each self-training iteration
- **Type**: Line chart (time-series)
- **X-axis**: Iteration number (step)
- **Y-axis**: Number of labels added
- **Why it matters**: Shows how the semi-supervised learning process progresses - you'll see it start high and decrease as the model becomes more confident

#### b) `self_training/cumulative_labels`
- **What it shows**: Total cumulative labels added across all iterations
- **Type**: Line chart (time-series, increasing)
- **X-axis**: Iteration number (step)
- **Y-axis**: Cumulative total labels
- **Why it matters**: Shows the total growth of labeled data during self-training

#### c) `self_training/labels_added_iter_1`, `labels_added_iter_2`, etc.
- **What it shows**: Individual iteration metrics (for reference)
- **Type**: Individual data points
- **Why it matters**: Detailed breakdown per iteration

---

## 2. Performance Metrics Charts

### Location in W&B:
**Dashboard → Your Run → Charts Tab → Metrics Section**

### Charts Created:

#### Validation Metrics:
- **`val/accuracy`**: Classification accuracy on validation set
- **`val/precision`**: Precision score (true positives / (true positives + false positives))
- **`val/recall`**: Recall score (true positives / (true positives + false negatives))
- **`val/f1`**: F1 score (harmonic mean of precision and recall)
- **`val/roc_auc`**: ROC-AUC score (area under ROC curve)
- **`val/log_loss`**: Log loss (lower is better)

#### Test Metrics:
- **`test/accuracy`**: Classification accuracy on test set
- **`test/precision`**: Precision on test set
- **`test/recall`**: Recall on test set
- **`test/f1`**: F1 score on test set
- **`test/roc_auc`**: ROC-AUC on test set
- **`test/log_loss`**: Log loss on test set

**Note**: These appear as single points (not time-series) because they're final evaluation metrics.

---

## 3. Image Visualizations

### Location in W&B:
**Dashboard → Your Run → Media Tab** or **Charts Tab → Media Section**

### Visualizations Created:

#### a) Confusion Matrices (`confusion_matrices`)
- **What it shows**: Two side-by-side confusion matrices
  - Left: Validation set confusion matrix
  - Right: Test set confusion matrix
- **Type**: Heatmap (seaborn)
- **Colors**: Blue gradient (darker = higher count)
- **Labels**: 
  - Rows: Actual labels (No BP, BP)
  - Columns: Predicted labels (No BP, BP)
- **Why it matters**: Shows exactly where the model makes mistakes:
  - True Negatives (top-left): Correctly predicted "No Big Potential"
  - False Positives (top-right): Incorrectly predicted "Big Potential" (Type I error)
  - False Negatives (bottom-left): Missed "Big Potential" predictions (Type II error)
  - True Positives (bottom-right): Correctly predicted "Big Potential"

#### b) Feature Importance (`feature_importance`)
- **What it shows**: Top 15 most important features for the model
- **Type**: Horizontal bar chart (seaborn)
- **X-axis**: Feature importance score
- **Y-axis**: Feature names (sorted by importance)
- **Why it matters**: Shows which player attributes (e.g., age, overall rating, potential) are most important for predicting superstar potential

---

## 4. Data Tables

### Location in W&B:
**Dashboard → Your Run → Tables Tab** or **Charts Tab → Tables Section**

### Tables Created:

#### a) `val/classification_report`
- **What it shows**: Detailed classification report for validation set
- **Columns**: 
  - `precision`: Precision per class
  - `recall`: Recall per class
  - `f1-score`: F1 score per class
  - `support`: Number of samples per class
- **Rows**: 
  - `No BP`: Metrics for "No Big Potential" class
  - `BP`: Metrics for "Big Potential" class
  - `accuracy`: Overall accuracy
  - `macro avg`: Average across classes
  - `weighted avg`: Weighted average (by support)

#### b) `test/classification_report`
- **What it shows**: Same as above, but for test set
- **Why it matters**: Compare validation vs test performance to check for overfitting

---

## 5. Dataset Information (Static Metrics)

### Location in W&B:
**Dashboard → Your Run → Charts Tab → Metrics Section**

### Metrics Logged:
- **`dataset/train_size`**: Number of training samples
- **`dataset/val_size`**: Number of validation samples
- **`dataset/test_size`**: Number of test samples
- **`dataset/num_features`**: Number of input features
- **`dataset/train_class_0`**: Number of "No BP" samples in training
- **`dataset/train_class_1`**: Number of "BP" samples in training
- **`dataset/val_class_0`**: Number of "No BP" samples in validation
- **`dataset/val_class_1`**: Number of "BP" samples in validation

**Note**: These appear as single points (logged at step 0).

---

## 6. Training Metadata

### Location in W&B:
**Dashboard → Your Run → Charts Tab → Metrics Section**

### Metrics Logged:
- **`training_time`**: Total training time in seconds
- **Hyperparameters**: All model configuration (in Config tab)

---

## Phase 3: Hyperparameter Sweep Charts

### Location in W&B:
**Dashboard → Sweeps Tab → Your Sweep**

### Charts Created:

#### 1. Parallel Coordinates Plot
- **What it shows**: Relationship between hyperparameters and performance
- **Type**: Parallel coordinates visualization
- **Why it matters**: See which hyperparameter combinations lead to best F1 scores

#### 2. Hyperparameter Importance
- **What it shows**: Which hyperparameters matter most for performance
- **Type**: Bar chart
- **Why it matters**: Understand what to tune for better results

#### 3. Best Run Highlighted
- **What it shows**: The run with highest validation F1 score
- **Type**: Highlighted in all visualizations
- **Why it matters**: Easy to identify the best configuration

#### 4. All Trial Metrics
- **What it shows**: All validation metrics from each trial
- **Type**: Scatter plots, line charts
- **Metrics**: Same as Phase 2 (val/f1, val/roc_auc, etc.)

---

## 📍 How to Access Charts in W&B

### Step 1: Go to Your W&B Dashboard
```
https://wandb.ai/{YOUR_ENTITY}/football-superstar-prediction
```

Replace `{YOUR_ENTITY}` with your W&B entity name (from `mlops_config.py`).

### Step 2: Navigate to Your Run

**For Phase 2 (Experiment Tracking)**:
1. Click on **"Runs"** in the left sidebar
2. Find your run (usually named something like "ancient-wave-31" or "woven-capybara-32")
3. Click on the run name

**For Phase 3 (Sweep)**:
1. Click on **"Sweeps"** in the left sidebar
2. Find your sweep (ID like "0qoli38u")
3. Click on the sweep name

### Step 3: View Charts

Once in a run or sweep:

#### Option A: Charts Tab (Default View)
- **Location**: Top of the page, "Charts" tab
- **Shows**: All metrics as interactive charts
- **Features**: 
  - Hover to see exact values
  - Zoom in/out
  - Toggle metrics on/off
  - Change chart type (line, scatter, bar)

#### Option B: Media Tab
- **Location**: Top of the page, "Media" tab
- **Shows**: Images (confusion matrices, feature importance)
- **Features**: Click to view full-size images

#### Option C: Tables Tab
- **Location**: Top of the page, "Tables" tab
- **Shows**: Classification reports as interactive tables
- **Features**: Sort, filter, export

#### Option D: Config Tab
- **Location**: Top of the page, "Config" tab
- **Shows**: All hyperparameters used
- **Features**: See exact configuration for reproducibility

---

## 🎯 Quick Navigation Guide

### To See Self-Training Progress:
1. Go to your Phase 2 run
2. Charts tab → Find `self_training/labels_added`
3. You'll see a line chart showing labels added per iteration

### To See Model Performance:
1. Go to your Phase 2 run
2. Charts tab → Scroll to `val/` and `test/` metrics
3. Compare F1, ROC-AUC, accuracy, etc.

### To See Confusion Matrices:
1. Go to your Phase 2 run
2. Media tab → Click on `confusion_matrices` image
3. See validation and test confusion matrices side-by-side

### To See Feature Importance:
1. Go to your Phase 2 run
2. Media tab → Click on `feature_importance` image
3. See top 15 most important features

### To Compare Sweep Trials:
1. Go to your Phase 3 sweep
2. Sweeps tab → See parallel coordinates plot
3. Hover over points to see hyperparameters and F1 scores

### To See Best Model:
1. Go to your Phase 3 sweep
2. Best run is highlighted
3. Click on best run to see its detailed metrics

---

## 📸 Example Chart Locations

```
W&B Dashboard
├── Runs
│   └── [Your Run Name]
│       ├── Charts Tab
│       │   ├── Metrics Section
│       │   │   ├── self_training/labels_added (line chart)
│       │   │   ├── self_training/cumulative_labels (line chart)
│       │   │   ├── val/f1, val/roc_auc, etc. (single points)
│       │   │   └── test/f1, test/roc_auc, etc. (single points)
│       │   └── Media Section
│       │       ├── confusion_matrices (image)
│       │       └── feature_importance (image)
│       ├── Media Tab
│       │   ├── confusion_matrices
│       │   └── feature_importance
│       ├── Tables Tab
│       │   ├── val/classification_report
│       │   └── test/classification_report
│       └── Config Tab
│           └── All hyperparameters
│
└── Sweeps
    └── [Your Sweep ID]
        ├── Parallel Coordinates Plot
        ├── Hyperparameter Importance
        ├── Best Run Highlighted
        └── All Trial Metrics
```

---

## 💡 Tips for Reading Charts

1. **Time-Series Charts**: Look for trends - is the model improving over iterations?
2. **Confusion Matrices**: Focus on the diagonal (correct predictions) vs off-diagonal (errors)
3. **Feature Importance**: Higher bars = more important features for predictions
4. **Sweep Charts**: Look for patterns - do certain hyperparameter ranges perform better?
5. **Compare Runs**: Use the "Compare" feature in W&B to overlay multiple runs

---

## 🔍 Troubleshooting

### Charts Not Showing?
- **Wait a few seconds**: W&B needs time to process and display charts
- **Refresh the page**: Sometimes charts need a refresh to appear
- **Check run status**: Make sure the run completed successfully (green checkmark)

### Charts Show Single Points?
- **For metrics like val/f1**: This is normal - they're final evaluation metrics, not time-series
- **For self_training/labels_added**: Should show a line chart - if not, re-run Phase 2 with the latest code

### Can't Find a Chart?
- **Use search**: W&B has a search bar to find specific metrics
- **Check all tabs**: Charts, Media, Tables, Config
- **Check run vs sweep**: Some charts are in runs, others in sweeps

---

**Need Help?** Check the main README.md for more information about the MLOps pipeline.
