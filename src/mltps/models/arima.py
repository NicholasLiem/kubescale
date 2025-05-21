import numpy as np
import pandas as pd
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Tuple, List, Optional, Union

logger = logging.getLogger("mltps")

class ARIMAModel:
    """SARIMA time series forecasting model for seasonal patterns"""
    
    def __init__(self, order=(3, 1, 1), seasonal_order=(1, 0, 1, 12)):
        """
        Initialize SARIMA model
        
        Args:
            order: ARIMA model order (p, d, q)
            seasonal_order: Seasonal component of the model (P, D, Q, s)
                            where s is the number of steps in a season
                            (12 for 3-minute cycle with 15s intervals)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.is_trained = False
        self.data = []  # Store training data
        
    def train(self, data):
        """
        Train the SARIMA model with time series data
        
        Args:
            data: Time series data for training
        """
        if len(data) == 0:
            logger.error("Empty data provided for SARIMA training")
            return False
            
        # Store data for future updates
        self.data = list(data)  # Convert to list to ensure it's appendable
        
        try:
            self.model = SARIMAX(
                self.data, 
                order=self.order,
                seasonal_order=self.seasonal_order
            ).fit(disp=False)
            self.is_trained = True
            logger.info(f"SARIMA model trained with {len(data)} data points")
            return True
        except Exception as e:
            logger.error(f"Error training SARIMA model: {e}")
            try:
                # Try alternative model if original fails
                alt_order = (2, 1, 0)
                alt_seasonal = (1, 0, 0, 12)
                logger.info(f"Trying alternative SARIMA order: {alt_order}, seasonal: {alt_seasonal}")
                self.model = SARIMAX(
                    self.data, 
                    order=alt_order,
                    seasonal_order=alt_seasonal
                ).fit(disp=False)
                self.order = alt_order
                self.seasonal_order = alt_seasonal
                self.is_trained = True
                logger.info(f"SARIMA model trained with alternative order")
                return True
            except Exception as e2:
                logger.error(f"All SARIMA model attempts failed: {e2}")
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
            self.model = SARIMAX(
                self.data, 
                order=self.order,
                seasonal_order=self.seasonal_order
            ).fit(disp=False)
            logger.debug(f"SARIMA model updated with new value: {new_value}")
            return True
        except Exception as e:
            logger.error(f"Error updating SARIMA model: {e}")
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
            
            # Calculate confidence score based on interval width
            interval_width = np.mean(intervals[:, 1] - intervals[:, 0])
            mean_value = np.mean(np.abs(forecast))
            confidence_score = 1.0 - min(1.0, (interval_width / (mean_value * 2 + 1e-10)))
            
            return forecast, intervals, confidence_score
            
        except Exception as e:
            logger.error(f"Error making SARIMA predictions: {e}")
            return np.array([]), np.array([]), 0.0