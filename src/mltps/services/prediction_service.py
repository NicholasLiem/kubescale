import logging
import numpy as np
import time
import pandas as pd
from typing import Optional, Tuple
from models.arima import ARIMAModel
from services.metrics_transformer_service import MetricsTransformerService

logger = logging.getLogger("mltps")

class PredictionService:
    def __init__(self, metrics_service, 
                 update_interval_seconds: int = 60,
                 window_size: int = 12,
                 confidence_threshold: float = 0.7,
                 model_update_interval_minutes: int = 10,
                 min_training_points: int = 15):
        """
        Service for traffic prediction based on CPU usage
        
        Args:
            metrics_service: Service for fetching metrics
            update_interval_seconds: Interval between predictions
            window_size: Number of intervals to predict ahead
            confidence_threshold: Minimum confidence threshold for spike detection
            model_update_interval_minutes: How often to update the model
            min_training_points: Minimum number of data points required for initial training
        """
        self.metrics_service = metrics_service
        self.transformer = MetricsTransformerService()
        self.update_interval_seconds = update_interval_seconds
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.model_update_interval_minutes = model_update_interval_minutes
        self.min_training_points = min_training_points
        
        # Model state
        self.model = ARIMAModel(order=(3, 1, 1))
        self.is_initialized = False
        self.history = pd.DataFrame()
        self.last_update = 0
        self.last_prediction = None
        self.prediction_history = []
        
        logger.info("PredictionService initialized with CPU usage metrics. Waiting for sufficient data collection...")
    
    def update_settings(self, 
                        update_interval_seconds: Optional[int] = None,
                        window_size: Optional[int] = None,
                        confidence_threshold: Optional[float] = None,
                        model_update_interval_minutes: Optional[int] = None,
                        min_training_points: Optional[int] = None):
        """
        Update service settings at runtime
        
        Args:
            update_interval_seconds: Interval between predictions
            window_size: Number of intervals to predict ahead
            confidence_threshold: Minimum confidence threshold for spike detection
            model_update_interval_minutes: How often to update the model
            min_training_points: Minimum number of data points required for initial training
        """
        if update_interval_seconds is not None:
            self.update_interval_seconds = update_interval_seconds
        if window_size is not None:
            self.window_size = window_size
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        if model_update_interval_minutes is not None:
            self.model_update_interval_minutes = model_update_interval_minutes
        if min_training_points is not None:
            self.min_training_points = min_training_points
            
        logger.info(f"PredictionService settings updated: interval={self.update_interval_seconds}m, "
                    f"window={self.window_size}, confidence={self.confidence_threshold}, "
                    f"update={self.model_update_interval_minutes}m, min_points={self.min_training_points}")
    
    def initialize_model(self) -> bool:
        """
        Initialize the model with sufficient CPU usage training data.
        Returns True if initialization was successful.
        """
        if self.is_initialized:
            return True
        
        # Get historical CPU usage data from Prometheus
        raw_data = self.metrics_service.get_prometheus_data(
            query='sum(rate(container_cpu_usage_seconds_total{namespace="default", pod=~"s[0-2].*|gw-nginx.*"}[1m])) by (pod)',
            step='15s'
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


    def update_model(self) -> bool:
        """Update ARIMA model with latest CPU usage data"""
        # First ensure model is initialized
        if not self.is_initialized:
            return self.initialize_model()
            
        current_time = time.time()
        
        # Only update if enough time has passed
        if current_time - self.last_update < self.update_interval_seconds:
            return True  # Model is up-to-date
    
        # Get historical data (15s resolution to better capture 30s spikes)
        raw_data = self.metrics_service.get_prometheus_data(
            query='sum(rate(container_cpu_usage_seconds_total{namespace="default", pod=~"s[0-2].*|gw-nginx.*"}[1m])) by (pod)',
            step='15s'
        )
        
        if not raw_data or 'data' not in raw_data or 'result' not in raw_data['data']:
            logger.warning("No CPU data available to update model")
            return False
        
        try:
            # Transform and prepare data
            raw_df = self.transformer.prometheus_to_dataframe(raw_data)
            if raw_df.empty:
                return False
                
            prepared_df = self.transformer.prepare_for_arima(
                df=raw_df,
                metric_type="cpu",
                resample_freq="15s",
                fillna_method='ffill',
                aggregate=True
            )
            
            # Store in history
            self.history = prepared_df
            
            # Get training values
            training_values = self._get_training_values(prepared_df)
            
            if len(training_values) >= 80:  # Enough data for full retrain
                self.model.train(training_values)
                logger.info(f"Model retrained with {len(training_values)} data points")
            elif len(training_values) > 0:  # Just update with latest point
                self.model.update(training_values[-1])
                logger.info("Model updated with latest data point")
            else:
                logger.warning("No data available for model update")
                return False
                
            self.last_update = current_time
            return True
            
        except Exception as e:
            logger.error(f"Error updating model: {e}")
            return False
    
    def detect_traffic_anomaly(self) -> dict:
        """Unified method to detect both spikes and spike endings"""
        result = {
            "spike_detected": False,
            "spike_ending": False,
            "predicted_value": 0.0,
            "current_value": 0.0,
            "time_to_spike": 0,
            "confidence": 0.0
        }
        
        if not self.is_initialized or self.last_prediction is None or self.history.empty:
            return result
            
        forecast = np.array(self.last_prediction["forecast"])
        confidence = self.last_prediction["confidence"]
        result["confidence"] = confidence
        
        # Get current CPU metrics
        current_series = self._get_current_series()
        if current_series is None:
            return result
            
        current_avg = np.mean(current_series.astype(float).tail(10))
        result["current_value"] = current_avg
        
        # Calculate baseline (normal traffic level)
        baseline = np.percentile(current_series.astype(float).tail(60), 25)
        max_forecast = np.max(forecast)
        result["predicted_value"] = max_forecast
        
        # Configurable thresholds
        SPIKE_THRESHOLD = 1.5  # 50% above current
        ELEVATED_THRESHOLD = 1.3  # 30% above baseline
        NORMAL_THRESHOLD = 1.2  # 20% above baseline
        
        # Check for spike
        if (max_forecast > current_avg * SPIKE_THRESHOLD and 
                confidence > self.confidence_threshold):
            result["spike_detected"] = True
            return result
            
        # Check for spike ending
        current_is_elevated = current_avg > (baseline * ELEVATED_THRESHOLD)
        forecast_trend = forecast[0] > forecast[-1]  # Decreasing trend
        returning_to_normal = forecast[-1] < (baseline * NORMAL_THRESHOLD)
        
        if (current_is_elevated and forecast_trend and returning_to_normal and 
                confidence > self.confidence_threshold * 0.8):
            result["spike_ending"] = True
            
        return result

    def _get_current_series(self):
        """Get the appropriate data series from history"""
        if self.history.empty:
            return None
            
        if 'total' in self.history.columns:
            return self.history['total']
        elif len(self.history.columns) > 0:
            return self.history[self.history.columns[0]]
        return None