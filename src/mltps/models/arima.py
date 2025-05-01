import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ARIMAModel:
    def __init__(self, order=(5,1,0)):
        """
        Initialize ARIMA model with specified order.
        
        Args:
            order: ARIMA order tuple (p,d,q)
                p: The number of lag observations
                d: The degree of differencing
                q: The size of the moving average window
        """
        self.order = order
        self.model = None
        self.history = []
        self.is_trained = False
        
    def train(self, time_series_data: List[float]) -> None:
        """
        Train the ARIMA model with historical time series data.
        
        Args:
            time_series_data: List of historical values
        """
        if len(time_series_data) < 10:  # Need minimum data points
            logger.warning(f"Insufficient data for ARIMA training: {len(time_series_data)} points")
            return
            
        try:
            # Convert to pandas Series for ARIMA
            ts = pd.Series(time_series_data)
            
            # Fit the model
            self.model = ARIMA(ts, order=self.order).fit()
            self.history = time_series_data.copy()
            self.is_trained = True
            logger.info(f"ARIMA model trained with {len(time_series_data)} data points")
        except Exception as e:
            logger.error(f"Error training ARIMA model: {e}")
            self.is_trained = False
            
    def predict(self, steps=10, confidence_interval=0.95) -> Tuple[List[float], List[Tuple[float, float]], float]:
        """
        Generate predictions for future values.
        
        Args:
            steps: Number of steps ahead to predict
            confidence_interval: Confidence level for prediction intervals
            
        Returns:
            Tuple containing:
                - List of predicted values
                - List of confidence intervals as (lower, upper) tuples
                - Model confidence score
        """
        if not self.is_trained or self.model is None:
            logger.warning("Cannot predict: model not trained")
            return [], [], 0.0
            
        try:
            # Make forecasts
            forecast_result = self.model.get_forecast(steps=steps)
            predicted_mean = forecast_result.predicted_mean.tolist()
            
            # Get confidence intervals
            conf_int = forecast_result.conf_int(alpha=1-confidence_interval)
            confidence_intervals = [(lower, upper) for lower, upper in zip(conf_int.iloc[:, 0], conf_int.iloc[:, 1])]
            
            # Calculate a simple confidence score based on the width of the prediction intervals
            # Narrower intervals suggest higher confidence
            conf_widths = [upper - lower for lower, upper in confidence_intervals]
            avg_width = np.mean(conf_widths)
            max_value = max(abs(max(predicted_mean)), abs(min(predicted_mean))) 
            if max_value == 0:
                confidence_score = 0.5
            else:
                # Normalize to 0-1 range (closer to 1 is better)
                confidence_score = max(0, min(1, 1 - (avg_width / (2 * max_value))))
            
            return predicted_mean, confidence_intervals, confidence_score
            
        except Exception as e:
            logger.error(f"Error making ARIMA prediction: {e}")
            return [], [], 0.0
            
    def update(self, new_data_point: float) -> None:
        """
        Update the model with a new data point.
        
        Args:
            new_data_point: The new observation to add to the model
        """
        self.history.append(new_data_point)
        
        # Periodically retrain the model with updated data
        if len(self.history) % 10 == 0:  # Retrain every 10 new data points
            self.train(self.history[-100:] if len(self.history) > 100 else self.history)