import logging
import time
from typing import Dict, Any, List, Optional
import numpy as np
from models.arima import ARIMAModel

logger = logging.getLogger(__name__)

class TrafficPredictor:
    def __init__(self, prometheus_client):
        """
        Initialize the traffic prediction system.
        
        Args:
            prometheus_client: Client for fetching metrics from Prometheus
        """
        self.prometheus_client = prometheus_client
        self.models = {
            'requests_per_second': ARIMAModel(order=(5,1,0)),
            'cpu_usage': ARIMAModel(order=(5,1,0)),
            'memory_usage': ARIMAModel(order=(2,1,0)),
        }
        self.latest_prediction = None
        self.training_window = 60  # Number of historical data points to use for training
        
    def predict(self, metrics: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Run predictions on the provided metrics.
        
        Args:
            metrics: Dictionary of metric name to list of values
            
        Returns:
            Dictionary with prediction results
        """
        prediction_results = {}
        spike_detected = False
        confidence_scores = []
        
        # Process each metric type with its corresponding model
        for metric_name, values in metrics.items():
            if metric_name not in self.models:
                continue
                
            model = self.models[metric_name]
            
            # Train model if we have enough data
            if len(values) >= self.training_window:
                model.train(values[-self.training_window:])
            
            # Make prediction
            if model.is_trained:
                forecast, intervals, confidence = model.predict(steps=10)
                
                # Detect potential spike (simple heuristic)
                current_avg = np.mean(values[-5:]) if len(values) >= 5 else np.mean(values)
                forecast_max = max(forecast)
                
                is_spike = forecast_max > current_avg * 1.5  # 50% increase
                
                prediction_results[metric_name] = {
                    'forecast': forecast,
                    'confidence': confidence,
                    'is_spike': is_spike,
                }
                
                if is_spike:
                    spike_detected = True
                    confidence_scores.append(confidence)
                
                # Update the model with the latest data point
                if values:
                    model.update(values[-1])
        
        # Determine overall confidence and recommendations
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        result = {
            'timestamp': time.time(),
            'spike_detected': spike_detected,
            'confidence': overall_confidence,
            'metrics': prediction_results,
            'recommended_replicas': self._calculate_recommended_replicas(prediction_results),
            'recommended_warm_pool_percent': self._calculate_warm_pool_percent(overall_confidence),
        }
        
        self.latest_prediction = result
        return result
    
    def get_latest_prediction(self) -> Optional[Dict[str, Any]]:
        """Return the most recent prediction results"""
        return self.latest_prediction
    
    def _calculate_recommended_replicas(self, prediction_results: Dict[str, Any]) -> int:
        """
        Calculate recommended number of replicas based on predictions.
        
        A more sophisticated implementation would use the forecast values
        and current resource usage to estimate needed capacity.
        """
        # Simple placeholder logic - in production this would be more complex
        base_replicas = 1
        
        if 'requests_per_second' in prediction_results:
            metric = prediction_results['requests_per_second']
            if metric.get('is_spike', False):
                # Get maximum value in forecast
                forecast_max = max(metric.get('forecast', [0]))
                # Scale linearly with request rate, assuming 10 RPS per replica
                rps_based_replicas = max(1, int(forecast_max / 10))
                return max(base_replicas, rps_based_replicas)
        
        return base_replicas
    
    def _calculate_warm_pool_percent(self, confidence: float) -> int:
        """
        Calculate what percentage of traffic should go to warm pool.
        
        Args:
            confidence: Prediction confidence (0-1)
            
        Returns:
            Integer percentage (0-100)
        """
        # Higher confidence = more traffic to warm pool
        # This is a simple linear function - could be more sophisticated
        if confidence < 0.3:
            return 0  # Not confident enough
        elif confidence > 0.8:
            return 20  # Very confident, route 20% to warm pool
        else:
            # Linear scaling between 5% and 15%
            return int(5 + (confidence - 0.3) * (15 - 5) / (0.8 - 0.3))