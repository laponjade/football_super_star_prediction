"""
MLOps Phase 5: Production Monitoring
This script implements production monitoring for the deployed model:
- Model performance tracking
- Data drift detection
- Prediction monitoring (latency, throughput, errors)
- Alerting for performance degradation
"""

import wandb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import joblib
from mlops_config import ENTITY, PROJECT

class ProductionMonitor:
    """Monitor model performance in production."""
    
    def __init__(self, model_artifact_name: str = "football-superstar-predictor:latest"):
        """
        Initialize production monitor.
        
        Args:
            model_artifact_name: Name of the registered model artifact in W&B
        """
        self.model_artifact_name = model_artifact_name
        self.model = None
        self.scaler = None
        self.reference_data = None
        self.load_model_from_registry()
        
    def load_model_from_registry(self):
        """Load model and scaler from W&B registry."""
        try:
            # Download model artifact
            artifact = wandb.use_artifact(f"{ENTITY}/{PROJECT}/{self.model_artifact_name}")
            artifact_dir = artifact.download()
            
            # Load model and scaler
            model_path = Path(artifact_dir) / "best_self_training_xgb_model.joblib"
            scaler_path = Path(artifact_dir) / "best_self_training_xgb_scaler.joblib"
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            print(f"[OK] Model loaded from registry: {self.model_artifact_name}")
            
        except Exception as e:
            print(f"[WARN] Could not load from registry: {str(e)}")
            print("   Falling back to local model files...")
            # Fallback to local files
            model_path = Path("models") / "best_self_training_xgb_model.joblib"
            scaler_path = Path("models") / "best_self_training_xgb_scaler.joblib"
            
            if model_path.exists() and scaler_path.exists():
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                print("[OK] Model loaded from local files")
            else:
                raise FileNotFoundError("Model files not found locally or in registry")
    
    def load_reference_data(self, data_path: str = "../data/feature_engineered_data_v2.csv"):
        """Load reference dataset for drift detection."""
        try:
            df = pd.read_csv(data_path)
            # Remove target and non-feature columns
            feature_columns = [col for col in df.columns 
                             if col not in ['fifa_version', 'big_potential']]
            self.reference_data = df[feature_columns]
            print(f"[OK] Reference data loaded: {len(self.reference_data)} samples")
        except Exception as e:
            print(f"[WARN] Could not load reference data: {str(e)}")
            self.reference_data = None
    
    def detect_data_drift(self, production_data: pd.DataFrame) -> Dict:
        """
        Detect data drift between reference and production data.
        
        Args:
            production_data: DataFrame with production features
            
        Returns:
            Dictionary with drift metrics
        """
        if self.reference_data is None:
            return {"drift_detected": False, "message": "No reference data available"}
        
        drift_metrics = {}
        drift_detected = False
        
        # Ensure same columns
        common_cols = set(self.reference_data.columns) & set(production_data.columns)
        if len(common_cols) != len(self.reference_data.columns):
            return {"drift_detected": True, "message": "Column mismatch detected"}
        
        for col in common_cols:
            ref_mean = self.reference_data[col].mean()
            prod_mean = production_data[col].mean()
            ref_std = self.reference_data[col].std()
            
            # Z-score based drift detection
            if ref_std > 0:
                z_score = abs(prod_mean - ref_mean) / ref_std
                drift_metrics[f"drift_zscore_{col}"] = z_score
                
                # Alert if z-score > 3 (significant drift)
                if z_score > 3:
                    drift_detected = True
                    drift_metrics[f"drift_alert_{col}"] = True
        
        drift_metrics["drift_detected"] = drift_detected
        drift_metrics["features_checked"] = len(common_cols)
        
        return drift_metrics
    
    def monitor_predictions(
        self,
        predictions: List[Dict],
        actuals: Optional[List[int]] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Monitor prediction performance.
        
        Args:
            predictions: List of prediction dictionaries with 'prediction', 'probability', etc.
            actuals: Optional list of actual labels for performance calculation
            metadata: Optional metadata (latency, timestamp, etc.)
            
        Returns:
            Dictionary with monitoring metrics
        """
        metrics = {}
        
        # Prediction statistics
        probs = [p.get('probability', 0) for p in predictions]
        preds = [p.get('prediction', 0) for p in predictions]
        
        metrics['predictions/count'] = len(predictions)
        metrics['predictions/mean_probability'] = np.mean(probs) if probs else 0
        metrics['predictions/std_probability'] = np.std(probs) if probs else 0
        metrics['predictions/positive_rate'] = np.mean(preds) if preds else 0
        
        # Performance metrics (if actuals available)
        if actuals and len(actuals) == len(predictions):
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics['performance/accuracy'] = accuracy_score(actuals, preds)
            metrics['performance/precision'] = precision_score(actuals, preds, zero_division=0)
            metrics['performance/recall'] = recall_score(actuals, preds, zero_division=0)
            metrics['performance/f1'] = f1_score(actuals, preds, zero_division=0)
        
        # Latency and throughput (from metadata)
        if metadata:
            if 'latency_ms' in metadata:
                metrics['system/latency_ms'] = metadata['latency_ms']
            if 'throughput_per_sec' in metadata:
                metrics['system/throughput_per_sec'] = metadata['throughput_per_sec']
            if 'error_rate' in metadata:
                metrics['system/error_rate'] = metadata['error_rate']
        
        # Log to W&B
        wandb.log(metrics)
        
        return metrics
    
    def check_model_health(self) -> Dict:
        """
        Check overall model health.
        
        Returns:
            Dictionary with health status
        """
        health = {
            "status": "healthy",
            "model_loaded": self.model is not None,
            "scaler_loaded": self.scaler is not None,
            "reference_data_loaded": self.reference_data is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        if not health["model_loaded"] or not health["scaler_loaded"]:
            health["status"] = "unhealthy"
            health["issues"] = []
            if not health["model_loaded"]:
                health["issues"].append("Model not loaded")
            if not health["scaler_loaded"]:
                health["issues"].append("Scaler not loaded")
        
        # Log health status
        wandb.log({
            "health/status": 1 if health["status"] == "healthy" else 0,
            "health/model_loaded": 1 if health["model_loaded"] else 0,
            "health/scaler_loaded": 1 if health["scaler_loaded"] else 0,
            "health/reference_data_loaded": 1 if health["reference_data_loaded"] else 0
        })
        
        return health


def monitor_production(
    predictions_file: Optional[str] = None,
    actuals_file: Optional[str] = None,
    reference_data_path: str = "../data/feature_engineered_data_v2.csv"
):
    """
    Main function to monitor production model.
    
    Args:
        predictions_file: Optional CSV file with production predictions
        actuals_file: Optional CSV file with actual labels
        reference_data_path: Path to reference dataset for drift detection
    """
    print("=" * 60)
    print("PRODUCTION MONITORING")
    print("=" * 60)
    
    # Initialize W&B
    wandb.init(
        project=PROJECT,
        entity=ENTITY,
        job_type="monitoring",
        name=f"production-monitoring-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    
    # Initialize monitor
    monitor = ProductionMonitor()
    monitor.load_reference_data(reference_data_path)
    
    # Check model health
    print("\n[1] Checking Model Health...")
    health = monitor.check_model_health()
    print(f"   Status: {health['status']}")
    print(f"   Model loaded: {health['model_loaded']}")
    print(f"   Scaler loaded: {health['scaler_loaded']}")
    print(f"   Reference data loaded: {health['reference_data_loaded']}")
    
    # Monitor predictions if file provided
    if predictions_file:
        print(f"\n[2] Monitoring Predictions from {predictions_file}...")
        pred_df = pd.read_csv(predictions_file)
        
        # Convert to prediction format
        predictions = []
        for _, row in pred_df.iterrows():
            predictions.append({
                'prediction': row.get('prediction', 0),
                'probability': row.get('probability', 0.0)
            })
        
        # Load actuals if available
        actuals = None
        if actuals_file:
            actuals_df = pd.read_csv(actuals_file)
            actuals = actuals_df['actual'].tolist() if 'actual' in actuals_df.columns else None
        
        metrics = monitor.monitor_predictions(predictions, actuals)
        print(f"   Predictions monitored: {metrics.get('predictions/count', 0)}")
        print(f"   Mean probability: {metrics.get('predictions/mean_probability', 0):.4f}")
        if 'performance/f1' in metrics:
            print(f"   F1 Score: {metrics['performance/f1']:.4f}")
    
    # Data drift detection (if production features available)
    if predictions_file:
        print(f"\n[3] Checking Data Drift...")
        print("   [INFO] Data drift detection requires feature data")
        print("   [INFO] In production, compare incoming features with reference")
    
    print("\n[SUCCESS] Production monitoring completed!")
    print(f"   View monitoring dashboard: https://wandb.ai/{ENTITY}/{PROJECT}")
    
    wandb.finish()


if __name__ == "__main__":
    # Example usage
    monitor_production()
