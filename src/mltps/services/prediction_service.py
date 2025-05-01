import logging
import numpy as np
import time
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from models.arima import ARIMAModel
from services.metrics_service import MetricsService

logger = logging.getLogger("mltps")

class PredictionService:
    def __init__(self, metrics_service: MetricsService, 
                 prediction_interval_minutes: int = 5,
                 window_size: int = 12,
                 confidence_threshold: float = 0.7,
                 model_update_interval_minutes: int = 10):
        """
        Service for traffic prediction
        
        Args:
            metrics_service: Service for fetching metrics
            prediction_interval_minutes: Interval between predictions
            window_size: Number of intervals to predict ahead
            confidence_threshold: Minimum confidence threshold for spike detection
            model_update_interval_minutes: How often to update the model
        """
        self.metrics_service = metrics_service
        self.prediction_interval_minutes = prediction_interval_minutes
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.model_update_interval_minutes = model_update_interval_minutes
        
        self.model = None
        self.history = pd.DataFrame()
        self.last_update = 0
        self.last_prediction = None
        self.prediction_history = []
        
    def update_model(self) -> bool:
        """Update ARIMA model with latest data"""
        current_time = time.time()
        
        # Only update model if it's been long enough since last update
        if current_time - self.last_update < self.model_update_interval_minutes * 60:
            return False
            
        # Get latest data
        df = self.metrics_service.get_request_rate_data()
        
        if df.empty:
            logger.warning("No data available to update model")
            return False
        
        # Store in history
        self.history = df
        
        # Fit ARIMA model
        try:
            # A simple ARIMA model with parameters (p=5, d=1, q=0)
            # You'll want to tune these parameters based on your data
            model = ARIMAModel(order=(5, 1, 0))
            model.train(df["value"].values.tolist())
            self.model = model
            logger.info("ARIMA model updated successfully")
            self.last_update = current_time
            return True
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            return False
    
    def predict_traffic(self) -> Tuple[Optional[np.ndarray], float]:
        """Predict traffic for next n intervals"""
        if self.model is None or not self.model.is_trained:
            self.update_model()
            if self.model is None or not self.model.is_trained:
                logger.error("Cannot make prediction: no model available")
                return None, 0
                
        try:
            # Generate forecast
            forecast, intervals, confidence = self.model.predict(steps=self.window_size)
            
            prediction = {
                "forecast": forecast,
                "confidence": confidence,
                "timestamp": time.time()
            }
            
            self.last_prediction = prediction
            self.prediction_history.append(prediction)
            
            return np.array(forecast), confidence
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None, 0
    
    def detect_spike(self) -> Tuple[bool, float, int]:
        """Detect if a traffic spike is predicted"""
        if self.last_prediction is None:
            return False, 0, 0
            
        forecast = np.array(self.last_prediction["forecast"])
        confidence = self.last_prediction["confidence"]
        
        # Simple spike detection: max forecast exceeds current average by 50%
        if len(self.history) > 0:
            current_avg = np.mean(self.history["value"].astype(float).tail(10))  # avg of last 10 points
            max_forecast = np.max(forecast)
            
            if max_forecast > current_avg * 1.5 and confidence > self.confidence_threshold:
                # When spike is predicted to occur (in minutes)
                spike_time = np.argmax(forecast) * self.prediction_interval_minutes
                return True, float(max_forecast), int(spike_time)
        
        return False, 0, 0