import logging
import numpy as np
import time
from scipy import stats
import pandas as pd
from typing import Optional, Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
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
        
        # Enhanced model state
        self.is_initialized = False
        self.history = pd.DataFrame()
        self.last_update = 0
        self.last_prediction = None
        self.prediction_history = []
        
        # SARIMAX model state
        self.raw_df = None
        self.transformed_df = None
        self.transform_method = None
        self.boxcox_lambda = None
        self.best_model = None
        self.best_params = None
        self.data_characteristics = None
        
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
    
    def analyze_data_characteristics(self):
        """Analyze data to determine if transformation is needed"""
        if self.raw_df is None or self.raw_df.empty:
            logger.warning("No data available for characteristics analysis")
            return None
            
        data = self.raw_df['total'] if 'total' in self.raw_df.columns else self.raw_df.iloc[:, 0]
        
        # Calculate various statistics
        stats_dict = {
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'skewness': data.skew(),
            'kurtosis': data.kurtosis(),
            'cv': data.std() / data.mean(),  # Coefficient of variation
            'range_ratio': (data.max() - data.min()) / data.mean(),
            'spike_ratio': len(data[data > data.mean() + 2*data.std()]) / len(data),
            'zero_ratio': len(data[data == 0]) / len(data)
        }
        
        # Determine if transformation is beneficial
        needs_transformation = False
        recommended_method = 'none'
        
        # Check for high skewness (> 1 or < -1)
        if abs(stats_dict['skewness']) > 1:
            needs_transformation = True
            if stats_dict['skewness'] > 0:
                recommended_method = 'log' if stats_dict['zero_ratio'] < 0.1 else 'sqrt'
            else:
                recommended_method = 'sqrt'
        
        # Check for high coefficient of variation (> 1)
        elif stats_dict['cv'] > 1:
            needs_transformation = True
            recommended_method = 'sqrt'
        
        # Check for extreme range ratio
        elif stats_dict['range_ratio'] > 10:
            needs_transformation = True
            recommended_method = 'log' if stats_dict['zero_ratio'] < 0.1 else 'sqrt'
        
        self.data_characteristics = {
            'stats': stats_dict,
            'needs_transformation': needs_transformation,
            'recommended_method': recommended_method
        }
        
        logger.info(f"Data Analysis Results:")
        logger.info(f"  Skewness: {stats_dict['skewness']:.3f}")
        logger.info(f"  Coefficient of Variation: {stats_dict['cv']:.3f}")
        logger.info(f"  Range Ratio: {stats_dict['range_ratio']:.3f}")
        logger.info(f"  Spike Ratio: {stats_dict['spike_ratio']:.3f}")
        logger.info(f"  Needs Transformation: {needs_transformation}")
        logger.info(f"  Recommended Method: {recommended_method}")
        
        return self.data_characteristics
    
    def transform_data(self, series, method='log'):
        """Transform data to make it more suitable for ARIMA modeling"""
        if method == 'log':
            return np.log1p(series)
        elif method == 'sqrt':
            return np.sqrt(series)
        elif method == 'boxcox':
            transformed, lmbda = stats.boxcox(series + 1)
            return transformed, lmbda
        return series
    
    def inverse_transform(self, series, method=None, boxcox_lambda=None):
        """Inverse transform the predictions"""
        method = method or self.transform_method
        boxcox_lambda = boxcox_lambda or self.boxcox_lambda
        
        if method == 'none':
            return series
        elif method == 'log':
            return np.expm1(series)
        elif method == 'sqrt':
            return series ** 2
        elif method == 'boxcox':
            return stats.inv_boxcox(series, boxcox_lambda) - 1
        return series
    
    def prepare_transformed_data(self, method='auto'):
        """Prepare dataset with optional transformation"""
        if self.raw_df is None or self.raw_df.empty:
            logger.warning("No raw data available for transformation")
            return False
            
        if method == 'auto':
            if not self.data_characteristics:
                self.analyze_data_characteristics()
            
            if not self.data_characteristics['needs_transformation']:
                method = 'none'
                logger.info("Data analysis suggests no transformation needed - using original data")
            else:
                method = self.data_characteristics['recommended_method']
                logger.info(f"Data analysis recommends: {method} transformation")
        
        self.transform_method = method
        
        # Get the data series to transform
        data_series = self.raw_df['total'] if 'total' in self.raw_df.columns else self.raw_df.iloc[:, 0]
        
        if method == 'none':
            # Use original data without transformation
            transformed_data = data_series
            self.boxcox_lambda = None
        elif method == 'boxcox':
            transformed_data, self.boxcox_lambda = self.transform_data(
                data_series, method=method
            )
        else:
            transformed_data = self.transform_data(data_series, method=method)
            self.boxcox_lambda = None
            
        self.transformed_df = pd.DataFrame(
            {'value': transformed_data}, 
            index=self.raw_df.index
        )
        
        logger.info(f"Data transformed using method: {method}")
        return True
    
    def fit_sarima_model(self, train_data, order, seasonal_order):
        """Fit SARIMA model with given parameters"""
        model = SARIMAX(
            train_data['value'],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        return model.fit(disp=False, maxiter=50)
    
    def calculate_metrics_improved(self, forecast_orig, test_actual):
        """Enhanced metrics calculation that preserves spike characteristics"""
        rmse = np.sqrt(np.mean((forecast_orig - test_actual) ** 2))
        mae = np.mean(np.abs(forecast_orig - test_actual))
        
        # Use adaptive thresholds based on data characteristics
        original_data = self.raw_df['total'] if 'total' in self.raw_df.columns else self.raw_df.iloc[:, 0]
        data_stats = {
            'median': original_data.median(),
            'mean': original_data.mean(),
            'std': original_data.std(),
            'q75': np.percentile(original_data, 75),
            'q90': np.percentile(original_data, 90),
            'q95': np.percentile(original_data, 95),
            'q99': np.percentile(original_data, 99)
        }
        
        # Multiple threshold strategies
        threshold_strategies = {
            'conservative': data_stats['q99'],
            'moderate': data_stats['q95'],
            'liberal': data_stats['q90'],
            'statistical': data_stats['mean'] + 2 * data_stats['std'],
            'robust': data_stats['median'] + 2 * data_stats['std']
        }
        
        # best_strategy = None
        best_accuracy = 0
        best_result = None
        
        for strategy_name, threshold in threshold_strategies.items():
            # Find spikes with current threshold
            test_spikes = self._find_spikes_adaptive(test_actual.values, threshold)
            pred_spikes = self._find_spikes_adaptive(forecast_orig, threshold)
            
            if len(test_spikes) > 0:  # Only evaluate if there are actual spikes
                spike_matches = self._count_spike_matches_improved(pred_spikes, test_spikes, tolerance=3)
                spike_accuracy = spike_matches / len(test_spikes)
                
                if spike_accuracy > best_accuracy:
                    best_accuracy = spike_accuracy
                    best_strategy = strategy_name
                    best_result = {
                        'threshold': threshold,
                        'actual_spikes': len(test_spikes),
                        'predicted_spikes': len(pred_spikes),
                        'spike_accuracy': spike_accuracy
                    }
        
        if best_result is None:
            # Fallback if no spikes found
            threshold = data_stats['q95']
            test_spikes = self._find_spikes_adaptive(test_actual.values, threshold)
            pred_spikes = self._find_spikes_adaptive(forecast_orig, threshold)
            best_result = {
                'threshold': threshold,
                'actual_spikes': len(test_spikes),
                'predicted_spikes': len(pred_spikes),
                'spike_accuracy': 0
            }
        
        return {
            'rmse': rmse,
            'mae': mae,
            'spike_accuracy': best_result['spike_accuracy'],
            'actual_spikes': best_result['actual_spikes'],
            'predicted_spikes': best_result['predicted_spikes'],
            'best_threshold': best_result['threshold'],
            'transform_method': self.transform_method
        }
    
    def _find_spikes_adaptive(self, data, threshold):
        """Adaptive spike detection with context awareness"""
        spikes = []
        min_spike_duration = 1  # Minimum duration for a spike
        min_gap = 2  # Minimum gap between separate spikes
        
        i = 0
        while i < len(data):
            if data[i] > threshold:
                spike_start = i
                # Find the end of the spike
                while i < len(data) and data[i] > threshold:
                    i += 1
                spike_duration = i - spike_start
                
                # Only consider spikes that meet minimum duration
                if spike_duration >= min_spike_duration:
                    # Check if this is far enough from the last spike
                    if not spikes or spike_start - spikes[-1] >= min_gap:
                        spikes.append(spike_start)
            else:
                i += 1
        
        return spikes
    
    def _count_spike_matches_improved(self, pred_spikes, actual_spikes, tolerance=3):
        """Improved spike matching with better tolerance handling"""
        if not pred_spikes or not actual_spikes:
            return 0
        
        matches = 0
        used_actual_spikes = set()
        
        for pred_spike in pred_spikes:
            best_match = None
            best_distance = float('inf')
            
            for j, actual_spike in enumerate(actual_spikes):
                if j not in used_actual_spikes:
                    distance = abs(pred_spike - actual_spike)
                    if distance <= tolerance and distance < best_distance:
                        best_match = j
                        best_distance = distance
            
            if best_match is not None:
                matches += 1
                used_actual_spikes.add(best_match)
        
        return matches
    
    def optimize_parameters_for_spikes(self):
        """Quick parameter optimization with minimal combinations for faster initialization"""
        logger.info("Starting quick parameter optimization...")
        
        if self.transformed_df is None or self.transformed_df.empty:
            logger.error("No transformed data available for optimization")
            return None
            
        # Split data for training and testing
        train_size = int(len(self.transformed_df) * 0.8)
        train_data = self.transformed_df.iloc[:train_size]
        test_data = self.transformed_df.iloc[train_size:]
        
        # Minimal parameter combinations for quick initialization (only 8 combinations)
        quick_param_combinations = [
            # Simple models first (faster convergence)
            ((1, 1, 1), (0, 1, 0, 12)),  # Basic ARIMA with seasonal
            ((2, 1, 1), (1, 1, 0, 12)),  # Slightly more complex
            ((1, 1, 2), (0, 1, 1, 12)),  # Different MA structure
            ((2, 1, 2), (1, 1, 1, 12)),  # Balanced approach
            
            # Spike-focused patterns
            ((3, 1, 1), (0, 1, 0, 20)),  # Higher AR for spike dependencies
            ((1, 1, 3), (0, 1, 1, 20)),  # Higher MA for smoothing
            ((2, 1, 2), (1, 1, 1, 20)),  # Balanced with longer seasonality
            ((3, 1, 2), (1, 1, 0, 40)),  # Complex for detailed patterns
        ]
        
        best_score = float('-inf')
        results = []
        
        logger.info(f"Testing {len(quick_param_combinations)} optimized parameter combinations...")
        
        start_time = time.time()
        timeout_seconds = 30  # Maximum 30 seconds for optimization
        
        for i, (order, seasonal_order) in enumerate(quick_param_combinations):
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                logger.warning(f"Parameter optimization timeout after {timeout_seconds}s")
                break
                
            try:
                # Quick fit with reduced iterations
                fitted = self.fit_sarima_model_quick(train_data, order, seasonal_order)
                if fitted is None:
                    logger.debug(f"Failed to fit {order}x{seasonal_order}")
                    continue
                    
                forecast = fitted.forecast(steps=len(test_data))
                forecast_orig = self.inverse_transform(forecast)
                
                # Get original scale test data
                original_test_start = train_size
                original_test_end = original_test_start + len(test_data)
                original_data = self.raw_df['total'] if 'total' in self.raw_df.columns else self.raw_df.iloc[:, 0]
                test_actual = original_data.iloc[original_test_start:original_test_end]
                
                # Quick metrics calculation (simplified)
                rmse = np.sqrt(np.mean((forecast_orig - test_actual) ** 2))
                mae = np.mean(np.abs(forecast_orig - test_actual))
                
                # Simple scoring for quick evaluation
                combined_score = 1 / (1 + rmse + mae)  # Simple inverse error score
                
                result = {
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'aic': fitted.aic,
                    'combined_score': combined_score,
                    'rmse': rmse,
                    'mae': mae
                }
                results.append(result)
                
                if combined_score > best_score:
                    best_score = combined_score
                    self.best_params = (order, seasonal_order)
                    self.best_model = fitted
                    
                logger.debug(f"✓ {i+1}/{len(quick_param_combinations)}: {order}x{seasonal_order} "
                            f"(score: {combined_score:.3f}, rmse: {rmse:.3f})")
                    
            except Exception as e:
                logger.debug(f"✗ Error with {order}x{seasonal_order}: {e}")
                continue
        
        # Fallback to simplest model if nothing worked
        if not results or self.best_params is None:
            logger.warning("No parameters worked, using simple fallback")
            self.best_params = ((1, 1, 1), (0, 1, 0, 12))
            return None
        
        if results:
            results_df = pd.DataFrame(results).sort_values('combined_score', ascending=False)
            logger.info("Top 3 Quick Parameter Results:")
            for i, row in results_df.head(3).iterrows():
                logger.info(f"  {row['order']}x{row['seasonal_order']}: "
                           f"Score={row['combined_score']:.3f}, RMSE={row['rmse']:.3f}")
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Quick optimization complete in {elapsed_time:.1f}s")
            logger.info(f"Best parameters: SARIMA{self.best_params[0]}x{self.best_params[1]}")
            
            return results_df
        else:
            logger.warning("No valid parameter combinations found")
            return None

    def fit_sarima_model_quick(self, train_data, order, seasonal_order):
        """Quick SARIMA model fitting with timeout and reduced iterations"""
        try:
            model = SARIMAX(
                train_data['value'],
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            # Reduced iterations for faster fitting
            fitted = model.fit(disp=False, maxiter=20, method='lbfgs')
            return fitted
        except Exception as e:
            logger.debug(f"Quick fit failed for {order}x{seasonal_order}: {e}")
            return None

    def fit_sarima_model(self, train_data, order, seasonal_order):
        """Standard SARIMA model fitting (for regular updates)"""
        model = SARIMAX(
            train_data['value'],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        return model.fit(disp=False, maxiter=50)

    def calculate_metrics_simple(self, forecast_orig, test_actual):
        """Simplified metrics calculation for quick optimization"""
        rmse = np.sqrt(np.mean((forecast_orig - test_actual) ** 2))
        mae = np.mean(np.abs(forecast_orig - test_actual))
        
        # Quick spike detection using simple threshold
        threshold = np.percentile(test_actual, 90)
        actual_spikes = len(test_actual[test_actual > threshold])
        pred_spikes = len(forecast_orig[forecast_orig > threshold])
        
        spike_accuracy = 1.0 if actual_spikes == 0 else min(pred_spikes, actual_spikes) / actual_spikes
        
        return {
            'rmse': rmse,
            'mae': mae,
            'spike_accuracy': spike_accuracy,
            'actual_spikes': actual_spikes,
            'predicted_spikes': pred_spikes
        }

    # Update the main update_model method to use quick optimization for initialization
    def update_model(self, force_refresh=False) -> bool:
        """Update SARIMAX model with latest CPU usage data"""
        current_time = time.time()
        
        # Only update if enough time has passed (unless not initialized or forced)
        if (self.is_initialized and 
            current_time - self.last_update < self.update_interval_seconds and 
            not force_refresh):
            return True  # Model is up-to-date

        logger.info(f"🔄 Fetching latest CPU data from Prometheus...")
        
        # Get historical data (15s resolution to better capture 30s spikes)
        raw_data = self.metrics_service.get_prometheus_data(
            query='sum(rate(container_cpu_usage_seconds_total{namespace="default", pod=~"s[0-2].*|gw-nginx.*"}[1m])) by (pod)',
            step='15s'
        )
        
        if not raw_data or 'data' not in raw_data or 'result' not in raw_data['data']:
            logger.warning("No CPU data available to update model")
            return False
        
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
                resample_freq="15s",  # Use 15s for better spike detection
                fillna_method='ffill',
                aggregate=True
            )
            
            # Store raw data for analysis
            self.raw_df = prepared_df
            self.history = prepared_df
            
            logger.info(f"📊 Updated with {len(prepared_df)} data points")
            
            # Check if we have enough data points for training
            if len(prepared_df) < self.min_training_points:
                logger.info(f"Collecting initial CPU training data: {len(prepared_df)}/{self.min_training_points} points")
                return False
            
            # For initialization, perform QUICK optimization
            if not self.is_initialized:
                logger.info("Performing quick model initialization...")
                
                # Analyze data characteristics
                self.analyze_data_characteristics()
                
                # Prepare transformed data
                if not self.prepare_transformed_data(method='auto'):
                    logger.error("Failed to prepare transformed data")
                    return False
                
                # QUICK optimization for spike detection (only 8 combinations, 30s max)
                optimization_results = self.optimize_parameters_for_spikes()
                
                if self.best_params is None:
                    logger.warning("Quick optimization failed, using simple default")
                    self.best_params = ((1, 1, 1), (0, 1, 0, 12))
                
                # Train the final model with best parameters
                try:
                    order, seasonal_order = self.best_params
                    final_model = SARIMAX(
                        self.transformed_df['value'],
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    self.best_model = final_model.fit(disp=False, maxiter=30)  # Quick fit
                    
                    self.is_initialized = True
                    self.last_update = current_time
                    
                    logger.info(f"✅ Quick CPU model initialization complete:")
                    logger.info(f"  - Data points: {len(prepared_df)}")
                    logger.info(f"  - Transform method: {self.transform_method}")
                    logger.info(f"  - Parameters: SARIMA{order}x{seasonal_order}")
                    
                    return True
                    
                except Exception as e:
                    logger.error(f"Error training quick SARIMA model: {e}")
                    return False
            
            else:
                # For updates, retrain with existing parameters on new data
                logger.info("📈 Updating model with fresh data...")
                
                # Update transformed data with same method
                if not self.prepare_transformed_data(method=self.transform_method):
                    logger.error("Failed to update transformed data")
                    return False
                
                # Retrain model with existing best parameters
                try:
                    order, seasonal_order = self.best_params
                    updated_model = SARIMAX(
                        self.transformed_df['value'],
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    self.best_model = updated_model.fit(disp=False, maxiter=30)  # Quick refit
                    
                    self.last_update = current_time
                    logger.info(f"✅ Model updated with {len(prepared_df)} data points")
                    
                    return True
                    
                except Exception as e:
                    logger.error(f"Error updating SARIMA model: {e}")
                    return False
            
        except Exception as e:
            logger.error(f"Error processing CPU metrics data for model update: {e}")
            return False

    def initialize_model(self) -> bool:
        """
        Initialize model by calling update_model
        """
        if self.is_initialized:
            return True
        
        logger.info("Initializing CPU prediction model...")
        return self.update_model()
        
    def predict_traffic(self) -> Tuple[Optional[np.ndarray], float]:
        """Enhanced predict method using the optimized SARIMA model"""
        if not self.is_initialized:
            initialized = self.initialize_model()
            if not initialized:
                logger.warning("Cannot predict CPU usage: model not initialized and initialization failed")
                return None, 0.0
                
        try:
            if self.best_model is None:
                logger.error("Best model is not available")
                return None, 0.0
            
            # Generate forecast using the optimized model
            forecast = self.best_model.forecast(steps=self.window_size)
            
            # Get prediction intervals for confidence
            pred = self.best_model.get_forecast(steps=self.window_size)
            confidence_intervals = pred.conf_int(alpha=0.05)  # 95% confidence
            
            # Transform back to original scale
            forecast_orig = self.inverse_transform(forecast)
            
            # Calculate confidence based on interval width
            ci_width = np.mean(confidence_intervals.iloc[:, 1] - confidence_intervals.iloc[:, 0])
            max_possible_width = np.std(self.transformed_df['value']) * 4  # Rough estimate
            confidence = max(0.0, min(1.0, 1 - (ci_width / max_possible_width)))
            
            self.last_prediction = {
                "timestamp": time.time(),
                "forecast": forecast_orig,
                "confidence": confidence
            }
            
            return np.array(forecast_orig), confidence
            
        except Exception as e:
            logger.error(f"Error predicting CPU usage: {e}")
            return None, 0.0
        
    def predict_traffic_extended(self, steps: int = 40) -> Tuple[Optional[np.ndarray], float]:
        """
        Extended predict method for custom number of steps
        
        Args:
            steps: Number of future time steps to predict
            
        Returns:
            Tuple of (forecast array, confidence score)
        """
        if not self.is_initialized:
            logger.warning("Cannot predict: model not initialized")
            return None, 0.0
            
        try:
            if self.best_model is None:
                logger.error("Best model is not available")
                return None, 0.0
            
            # Generate forecast using the optimized model
            forecast = self.best_model.forecast(steps=steps)
            
            # Get prediction intervals for confidence
            pred = self.best_model.get_forecast(steps=steps)
            confidence_intervals = pred.conf_int(alpha=0.05)  # 95% confidence
            
            # Transform back to original scale
            forecast_orig = self.inverse_transform(forecast)
            
            # Calculate confidence based on interval width
            ci_width = np.mean(confidence_intervals.iloc[:, 1] - confidence_intervals.iloc[:, 0])
            max_possible_width = np.std(self.transformed_df['value']) * 4  # Rough estimate
            confidence = max(0.0, min(1.0, 1 - (ci_width / max_possible_width)))
            
            # Store prediction for reference
            self.last_prediction = {
                "timestamp": time.time(),
                "forecast": forecast_orig,
                "confidence": confidence,
                "steps": steps
            }
            
            return np.array(forecast_orig), confidence
            
        except Exception as e:
            logger.error(f"Error predicting traffic for {steps} steps: {e}")
            return None, 0.0
        
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

    def _get_training_values(self, prepared_df):
        """Get training values from prepared DataFrame"""
        if 'total' in prepared_df.columns:
            return prepared_df['total'].values
        elif len(prepared_df.columns) > 0:
            return prepared_df[prepared_df.columns[0]].values
        return np.array([])