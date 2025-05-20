import numpy as np
import pandas as pd
import logging
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple, List, Optional, Union

logger = logging.getLogger("mltps")

class ARIMAModel:
    """ARIMA time series forecasting model"""
    
    def __init__(self, order=(5, 1, 0)):
        """
        Initialize ARIMA model
        
        Args:
            order: ARIMA model order (p, d, q)
        """
        self.order = order
        self.model = None
        self.is_trained = False
        self.data = []  # Store training data
        
    def train(self, data):
        """
        Train the ARIMA model with time series data
        
        Args:
            data: Time series data for training
        """
        if len(data) == 0:
            logger.error("Empty data provided for ARIMA training")
            return False
            
        # Store data for future updates
        self.data = list(data)  # Convert to list to ensure it's appendable
        
        try:
            self.model = ARIMA(self.data, order=self.order).fit()
            self.is_trained = True
            logger.info(f"ARIMA model trained with {len(data)} data points")
            return True
        except Exception as e:
            logger.error(f"Error training ARIMA model: {e}")
            try:
                # Try alternative model if original fails
                alt_order = (2, 1, 0)
                logger.info(f"Trying alternative ARIMA order: {alt_order}")
                self.model = ARIMA(self.data, order=alt_order).fit()
                self.order = alt_order
                self.is_trained = True
                logger.info(f"ARIMA model trained with alternative order")
                return True
            except Exception as e2:
                logger.error(f"All ARIMA model attempts failed: {e2}")
                self.is_trained = False
                return False
    
    def update(self, new_value):
        """
        Update the model with a new observation
        
        Args:
            new_value: New observation to add to the model
        """
        if self.model is None:
            logger.error("Model must be trained before updating")
            return False
        
        # Append the new value to the data
        self.data.append(new_value)
        
        # Retrain the model with the updated data
        try:
            self.model = ARIMA(self.data, order=self.order).fit()
            logger.debug(f"ARIMA model updated with new value: {new_value}")
            return True
        except Exception as e:
            logger.error(f"Error updating ARIMA model: {e}")
            return False
    
    def predict(self, steps=10) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Make predictions with the trained model
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Tuple of (forecast values, confidence intervals, confidence score)
        """
        if not self.is_trained or self.model is None:
            logger.error("Model must be trained before predicting")
            return np.array([]), np.array([]), 0.0
            
        try:
            # Get forecast
            forecast_result = self.model.forecast(steps=steps)
            
            # Get confidence intervals (alpha=0.05 for 95% confidence)
            pred = self.model.get_forecast(steps=steps)
            conf_int = pred.conf_int(alpha=0.05)
            
            # Convert to arrays
            forecast = np.array(forecast_result)
            intervals = np.array(conf_int)
            
            # Calculate crude confidence score based on interval width
            interval_width = np.mean(intervals[:, 1] - intervals[:, 0])
            mean_value = np.mean(np.abs(forecast))
            confidence_score = 1.0 - min(1.0, (interval_width / (mean_value * 2 + 1e-10)))
            
            return forecast, intervals, confidence_score
            
        except Exception as e:
            logger.error(f"Error making ARIMA predictions: {e}")
            return np.array([]), np.array([]), 0.0