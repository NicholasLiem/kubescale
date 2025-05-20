import logging
import numpy as np
import time
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from models.arima import ARIMAModel
from services.metrics_service import MetricsService
from services.metrics_transformer_service import MetricsTransformerService

logger = logging.getLogger("mltps")

class PredictionService:
    def __init__(self, metrics_service, 
                 prediction_interval_minutes: int = 5,
                 window_size: int = 12,
                 confidence_threshold: float = 0.7,
                 model_update_interval_minutes: int = 10,
                 min_training_points: int = 15):
        """
        Service for traffic prediction based on CPU usage
        
        Args:
            metrics_service: Service for fetching metrics
            prediction_interval_minutes: Interval between predictions
            window_size: Number of intervals to predict ahead
            confidence_threshold: Minimum confidence threshold for spike detection
            model_update_interval_minutes: How often to update the model
            min_training_points: Minimum number of data points required for initial training
        """
        self.metrics_service = metrics_service
        self.transformer = MetricsTransformerService()
        self.prediction_interval_minutes = prediction_interval_minutes
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.model_update_interval_minutes = model_update_interval_minutes
        self.min_training_points = min_training_points
        
        # Model state
        self.model = ARIMAModel(order=(5, 1, 0))
        self.is_initialized = False
        self.history = pd.DataFrame()
        self.last_update = 0
        self.last_prediction = None
        self.prediction_history = []
        
        logger.info("PredictionService initialized with CPU usage metrics. Waiting for sufficient data collection...")
    
    def initialize_model(self) -> bool:
        """
        Initialize the model with sufficient CPU usage training data.
        Returns True if initialization was successful.
        """
        if self.is_initialized:
            return True
            
        # Get historical CPU usage data from Prometheus
        raw_data = self.metrics_service.get_prometheus_data(
            query='sum(rate(container_cpu_usage_seconds_total{namespace="default", pod=~"s[0-2].*|gw-nginx.*"}[3m])) by (pod)',
            # Use appropriate time range for initialization
            step='1m'
        )
        
        if not raw_data or 'data' not in raw_data or 'result' not in raw_data['data']:
            logger.warning("No CPU usage data available for model initialization")
            return False
            
        # Transform Prometheus data to DataFrame
        try:
            # Convert Prometheus data to DataFrame
            raw_df = self.transformer.prometheus_to_dataframe(raw_data)
            
            if raw_df.empty:
                logger.warning("Transformed CPU data is empty")
                return False
                
            # Prepare for ARIMA modeling - aggregate across all pods
            prepared_df = self.transformer.prepare_for_arima(
                df=raw_df,
                metric_type="cpu",
                resample_freq="1min",
                fillna_method='ffill',
                aggregate=True  # Aggregate all pods
            )
            
            # Store in history
            self.history = prepared_df
            
            # Check if we have enough data points for training
            if len(prepared_df) < self.min_training_points:
                logger.info(f"Collecting initial CPU training data: {len(prepared_df)}/{self.min_training_points} points")
                return False
                
            # We have enough data, train the model
            try:
                # Get the aggregated values for model training
                if 'total' in prepared_df.columns:
                    training_values = prepared_df['total'].values
                else:
                    # If no 'total' column, use the first available column
                    col = prepared_df.columns[0]
                    training_values = prepared_df[col].values
                    logger.info(f"Using pod {col} for CPU model training")
                
                self.model.train(training_values)
                self.is_initialized = True
                self.last_update = time.time()
                logger.info(f"CPU model successfully initialized with {len(prepared_df)} data points")
                return True
            except Exception as e:
                logger.error(f"Error initializing CPU ARIMA model: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing CPU metrics data for model initialization: {e}")
            return False

    def update_model(self) -> bool:
        """Update ARIMA model with latest CPU usage data"""
        # First ensure model is initialized
        if not self.is_initialized:
            return self.initialize_model()
            
        current_time = time.time()
        
        # Only update model if it's been long enough since last update
        # TODO: Temporary change to 30 seconds
        if current_time - self.last_update < 30:
            return True  # Model is up-to-date
            
        # Get latest CPU usage data from Prometheus
        raw_data = self.metrics_service.get_prometheus_data(
            query='sum(rate(container_cpu_usage_seconds_total{namespace="default", pod=~"s[0-2].*|gw-nginx.*"}[3m])) by (pod)',
            # Use appropriate time range for updates
            step='1m'
        )
        
        if not raw_data or 'data' not in raw_data or 'result' not in raw_data['data']:
            logger.warning("No CPU data available to update model")
            return False
        
        try:
            # Transform data
            raw_df = self.transformer.prometheus_to_dataframe(raw_data)
            
            if raw_df.empty:
                logger.warning("Transformed CPU update data is empty")
                return False
                
            # Prepare for ARIMA modeling - aggregate across all pods
            prepared_df = self.transformer.prepare_for_arima(
                df=raw_df,
                metric_type="cpu",
                resample_freq="1min",
                fillna_method='ffill',
                aggregate=True  # Aggregate all pods
            )
            
            # Store in history
            self.history = prepared_df
            
            # Update existing ARIMA model
            try:
                # Get the aggregated values for model update
                if 'total' in prepared_df.columns:
                    latest_value = prepared_df['total'].values[-1]
                else:
                    # If no 'total' column, use the first available column
                    col = prepared_df.columns[0]
                    latest_value = prepared_df[col].values[-1]
                
                self.model.update(latest_value)
                logger.info("CPU ARIMA model updated successfully with latest data point")
                self.last_update = current_time
                return True
            except Exception as e:
                logger.error(f"Error updating CPU ARIMA model: {e}")
                
                # If update fails, try full retraining
                try:
                    # Get the aggregated values for model retraining
                    if 'total' in prepared_df.columns:
                        training_values = prepared_df['total'].values
                    else:
                        # If no 'total' column, use the first available column
                        col = prepared_df.columns[0]
                        training_values = prepared_df[col].values
                    
                    self.model.train(training_values)
                    logger.info("CPU ARIMA model retrained successfully after update failure")
                    self.last_update = current_time
                    return True
                except Exception as e2:
                    logger.error(f"Error retraining CPU ARIMA model: {e2}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error processing CPU metrics data for model update: {e}")
            return False
    
    def predict_traffic(self) -> Tuple[Optional[np.ndarray], float]:
        """Predict CPU usage for next n intervals"""
        if not self.is_initialized:
            initialized = self.initialize_model()
            if not initialized:
                logger.warning("Cannot predict CPU usage: model not initialized and initialization failed")
                return None, 0.0
                
        try:
            forecast, intervals, confidence = self.model.predict(steps=self.window_size)
            
            self.last_prediction = {
                "timestamp": time.time(),
                "forecast": forecast,
                "confidence": confidence
            }
            
            return np.array(forecast), confidence
        except Exception as e:
            logger.error(f"Error predicting CPU usage: {e}")
            return None, 0.0
    
    def detect_spike(self) -> Tuple[bool, float, int]:
        """Detect if a CPU usage spike is predicted"""
        if not self.is_initialized:
            return False, 0, 0
            
        if self.last_prediction is None:
            return False, 0, 0
            
        forecast = np.array(self.last_prediction["forecast"])
        confidence = self.last_prediction["confidence"]
        
        # Simple spike detection: max forecast exceeds current average by 50%
        if not self.history.empty:
            # Get current average from the aggregated column or first available column
            if 'total' in self.history.columns:
                current_series = self.history['total']
            else:
                current_series = self.history[self.history.columns[0]]
                
            current_avg = np.mean(current_series.astype(float).tail(10))
            max_forecast = np.max(forecast)
            max_forecast_idx = np.argmax(forecast)
            
            spike_threshold = current_avg * 1.5
            
            if max_forecast > spike_threshold and confidence > self.confidence_threshold - 10:
                # When spike is predicted to occur (in minutes)
                spike_time = max_forecast_idx * self.prediction_interval_minutes
                
                logger.info(f"🔥 CPU Spike detected! Current avg: {current_avg:.2f}, "
                           f"Predicted max: {max_forecast:.2f}, "
                           f"Expected in {spike_time} minutes, "
                           f"Confidence: {confidence:.2f}")
                
                # Save prediction for later evaluation
                self.prediction_history.append({
                    "timestamp": time.time(),
                    "current_avg": current_avg,
                    "predicted_max": max_forecast,
                    "predicted_time": spike_time,
                    "confidence": confidence
                })
                
                return True, float(max_forecast), int(spike_time)
            else:
                if max_forecast > current_avg * 1.2:
                    # Log increased CPU usage that doesn't qualify as a spike
                    logger.info(f"📈 CPU increase detected but below spike threshold. "
                               f"Current: {current_avg:.2f}, Predicted: {max_forecast:.2f}")
        
        return False, 0, 0