"""
Production Monitoring for Django Backend
Integrates monitoring into the prediction API endpoints.
"""

import time
import logging
from typing import Dict, Optional
from datetime import datetime
import wandb
from django.conf import settings

logger = logging.getLogger(__name__)

# Initialize W&B for monitoring (only if configured)
WANDB_MONITORING_ENABLED = getattr(settings, 'WANDB_MONITORING_ENABLED', False)
WANDB_ENTITY = getattr(settings, 'WANDB_ENTITY', None)
WANDB_PROJECT = getattr(settings, 'WANDB_PROJECT', 'football-superstar-prediction')

if WANDB_MONITORING_ENABLED:
    try:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            job_type="production-monitoring",
            name=f"api-monitoring-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            mode="online" if WANDB_MONITORING_ENABLED else "disabled"
        )
    except Exception as e:
        logger.warning(f"W&B monitoring initialization failed: {str(e)}")
        WANDB_MONITORING_ENABLED = False


class PredictionMonitor:
    """Monitor predictions in the Django API."""
    
    def __init__(self):
        self.prediction_count = 0
        self.total_latency = 0.0
        self.error_count = 0
        self.predictions_history = []
    
    def log_prediction(
        self,
        prediction: Dict,
        latency_ms: float,
        error: Optional[Exception] = None,
        features: Optional[Dict] = None
    ):
        """
        Log a prediction for monitoring.
        
        Args:
            prediction: Prediction result dictionary
            latency_ms: Prediction latency in milliseconds
            error: Optional error that occurred
            features: Optional input features
        """
        self.prediction_count += 1
        self.total_latency += latency_ms
        
        if error:
            self.error_count += 1
        
        # Store prediction history (keep last 100)
        self.predictions_history.append({
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction.get('prediction', 0),
            'probability': prediction.get('probability', 0.0),
            'latency_ms': latency_ms,
            'error': str(error) if error else None
        })
        
        if len(self.predictions_history) > 100:
            self.predictions_history.pop(0)
        
        # Log to W&B if enabled
        if WANDB_MONITORING_ENABLED:
            try:
                init_wandb_if_needed()
                metrics = {
                    'production/prediction_count': self.prediction_count,
                    'production/latency_ms': latency_ms,
                    'production/avg_latency_ms': self.total_latency / self.prediction_count,
                    'production/error_count': self.error_count,
                    'production/error_rate': self.error_count / self.prediction_count,
                    'production/prediction': prediction.get('prediction', 0),
                    'production/probability': prediction.get('probability', 0.0),
                }
                
                # Calculate throughput (predictions per second)
                if len(self.predictions_history) >= 2:
                    time_diff = (
                        datetime.fromisoformat(self.predictions_history[-1]['timestamp']) -
                        datetime.fromisoformat(self.predictions_history[0]['timestamp'])
                    ).total_seconds()
                    if time_diff > 0:
                        metrics['production/throughput_per_sec'] = len(self.predictions_history) / time_diff
                
                wandb.log(metrics)
            except Exception as e:
                logger.warning(f"Failed to log to W&B: {str(e)}")
        
        # Log to Django logger
        logger.info(
            f"Prediction logged: pred={prediction.get('prediction')}, "
            f"prob={prediction.get('probability', 0):.4f}, "
            f"latency={latency_ms:.2f}ms"
        )
    
    def get_metrics(self) -> Dict:
        """Get current monitoring metrics."""
        return {
            'prediction_count': self.prediction_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / self.prediction_count if self.prediction_count > 0 else 0,
            'avg_latency_ms': self.total_latency / self.prediction_count if self.prediction_count > 0 else 0,
            'total_latency_ms': self.total_latency
        }


# Global monitor instance
monitor = PredictionMonitor()


def monitor_prediction_decorator(func):
    """Decorator to monitor prediction functions."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        error = None
        prediction = None
        
        try:
            prediction = func(*args, **kwargs)
            return prediction
        except Exception as e:
            error = e
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000
            if prediction:
                monitor.log_prediction(prediction, latency_ms, error)
    
    return wrapper
