import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from typing import Dict, Tuple, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('arima.model')

class ARIMAModeler:
    """
    Class for creating and using ARIMA models for time series forecasting
    with Prometheus metrics
    """
    
    def __init__(self):
        logger.info("Initializing ARIMAModeler")
    
    def find_best_parameters(self, ts_data: pd.Series) -> Tuple[int, int, int]:
        """
        Determine the best p, d, q parameters for ARIMA model using grid search
        
        Args:
            ts_data: Time series data as pandas Series
            
        Returns:
            Tuple of (p, d, q) parameters
        """
        try:
            logger.info("Finding best ARIMA parameters...")
            
            # Define parameter grid
            p_values = range(0, 3)
            d_values = range(0, 2)
            q_values = range(0, 3)
            
            best_aic = float('inf')
            best_order = (0, 0, 0)
            
            # Try different combinations (limited to avoid excessive computation)
            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        try:
                            model = ARIMA(ts_data, order=(p, d, q))
                            model_fit = model.fit()
                            aic = model_fit.aic
                            
                            if aic < best_aic:
                                best_aic = aic
                                best_order = (p, d, q)
                                logger.info(f"New best parameters (p,d,q)={best_order} with AIC={best_aic}")
                        except:
                            continue
            
            logger.info(f"Best ARIMA parameters: {best_order}")
            return best_order
        except Exception as e:
            logger.error(f"Error finding best parameters: {e}")
            # Return default parameters
            return (1, 1, 1)
    
    def train_model(self, ts_data: pd.Series, order: Optional[Tuple[int, int, int]] = None) -> ARIMA:
        """
        Train an ARIMA model on the provided time series data
        
        Args:
            ts_data: Time series data as pandas Series
            order: ARIMA order parameters (p, d, q)
            
        Returns:
            Trained ARIMA model
        """
        if ts_data.empty:
            raise ValueError("Cannot train model on empty data")
            
        # Find best parameters if not provided
        if order is None:
            order = self.find_best_parameters(ts_data)
            
        try:
            logger.info(f"Training ARIMA model with order {order}...")
            model = ARIMA(ts_data, order=order)
            model_fit = model.fit()
            logger.info("Model training completed")
            return model_fit
        except Exception as e:
            logger.error(f"Error training ARIMA model: {e}")
            raise
    
    def forecast(self, model, steps: int = 10, return_conf_int: bool = True) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """
        Generate forecasts from a trained ARIMA model
        
        Args:
            model: Trained ARIMA model
            steps: Number of steps to forecast
            return_conf_int: Whether to return confidence intervals
            
        Returns:
            Tuple of (forecast Series, confidence intervals DataFrame)
        """
        try:
            logger.info(f"Forecasting {steps} steps ahead...")
            forecast = model.forecast(steps=steps)
            
            if return_conf_int:
                pred_ci = model.get_forecast(steps=steps).conf_int()
                return forecast, pred_ci
            else:
                return forecast, None
                
        except Exception as e:
            logger.error(f"Error forecasting: {e}")
            raise
    
    def evaluate_model(self, model, test_data: pd.Series) -> Dict[str, float]:
        """
        Evaluate the performance of an ARIMA model on test data
        
        Args:
            model: Trained ARIMA model
            test_data: Test data to evaluate on
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Get predictions for the test period
            predictions = model.forecast(steps=len(test_data))
            
            # Calculate metrics
            mse = np.mean((test_data - predictions) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(test_data - predictions))
            mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape
            }
            
            logger.info(f"Model evaluation metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def plot_forecast(self, history: pd.Series, forecast: pd.Series, 
                     conf_int: Optional[pd.DataFrame] = None,
                     figsize=(12, 6), title="ARIMA Forecast",
                     save_path: Optional[str] = None):
        """
        Plot the forecast results
        
        Args:
            history: Historical data
            forecast: Forecast data
            conf_int: Confidence intervals for the forecast
            figsize: Figure size
            title: Plot title
            save_path: If provided, save the plot to this path
        """
        plt.figure(figsize=figsize)
        
        # Plot history
        plt.plot(history.index, history.values, label='Historical Data')
        
        # Create forecast index continuing from history
        forecast_index = pd.date_range(
            start=history.index[-1] + (history.index[1] - history.index[0]), 
            periods=len(forecast),
            freq=history.index.freq or pd.infer_freq(history.index)
        )
        
        # Plot forecast
        plt.plot(forecast_index, forecast.values, 'r', label='Forecast')
        
        # Plot confidence intervals if available
        if conf_int is not None:
            plt.fill_between(forecast_index,
                            conf_int.iloc[:, 0],
                            conf_int.iloc[:, 1],
                            color='pink', alpha=0.3)
        
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved forecast plot to {save_path}")
            
        plt.show()