import base64
from datetime import timedelta
from io import BytesIO
from itertools import product
import logging
import numpy as np
import time
from scipy import stats
import pandas as pd
from typing import Optional, Tuple
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
        
        # SARIMAX model state
        self.raw_df = None
        self.transformed_df = None
        self.transform_method = None
        self.boxcox_lambda = None
        self.best_model = None
        self.best_params = None
        self.data_characteristics = None
        
        logger.info("PredictionService initialized with CPU usage metrics. Waiting for sufficient data collection...")
    
    #
    # DATA ANALYTICS
    # 

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

    # 
    # DATA MANIPULATION
    # 

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
            {'total': transformed_data}, 
            index=self.raw_df.index
        )
        
        logger.info(f"Data transformed using method: {method}")
        return True
    
    def split_data(self, train_ratio=0.8):
        """Split data into training and testing sets"""
        train_size = int(len(self.transformed_df) * train_ratio)
        train_data = self.transformed_df.iloc[:train_size]
        test_data = self.transformed_df.iloc[train_size:]
        return train_data, test_data

    # 
    # MODEL FITTING
    # 

    def fit_sarima_model(self, train_data, order, seasonal_order):
        """Fit SARIMA model with given parameters"""
        model = SARIMAX(
            train_data['total'],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        return model.fit(disp=False, maxiter=25)
    
    # 
    # FORECAST AND SPIKE DETECTION IN FORECAST
    # 
    
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
    
    def forecast_future(self, steps=100):
        """Generate future forecasts using the best model"""
        if not self.best_params:
            raise ValueError("No optimal parameters found. Run optimize_parameters() first.")
        
        order, seasonal_order = self.best_params
        
        # Retrain on full dataset
        full_model = SARIMAX(
            self.transformed_df['total'],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        full_results = full_model.fit(disp=False)
        
        # Generate forecast
        forecast_index = pd.date_range(
            start=self.raw_df.index[-1] + timedelta(seconds=15),
            periods=steps,
            freq='15S'
        )
        
        forecast = full_results.forecast(steps=steps)
        pred = full_results.get_forecast(steps=steps)
        ci = pred.conf_int(alpha=0.05)
        
        # Transform back to original scale
        forecast_orig = self.inverse_transform(forecast)
        ci_orig_lower = self.inverse_transform(ci.iloc[:, 0])
        ci_orig_upper = self.inverse_transform(ci.iloc[:, 1])
        
        return {
            'forecast_index': forecast_index,
            'forecast': forecast_orig,
            'ci_lower': ci_orig_lower,
            'ci_upper': ci_orig_upper,
            'model_results': full_results
        }

    def detect_spikes_in_forecast_improved(self, forecast_data, grace_period_seconds=60):
        """Enhanced spike detection with better threshold separation"""
        logger.info("DEBUG: Starting spike detection with grace period...")
        
        # Calculate grace period in data points (15-second intervals)
        grace_period_points = int(grace_period_seconds / 15)
        logger.info(f"DEBUG: Grace period: {grace_period_seconds} seconds ({grace_period_points} data points)")
        
        # Safety check for raw_df
        if self.raw_df is None or self.raw_df.empty:
            logger.warning("No raw data available for spike detection")
            return []
        
        baseline_value = self.raw_df['total'].median()
        
        small_spike_threshold = baseline_value * 1.1
        big_spike_threshold = baseline_value * 1.8
        
        logger.info(f"DEBUG: Small threshold: {small_spike_threshold:.2f}, Big threshold: {big_spike_threshold:.2f}")
        
        all_spikes = []
        
        # Safety check for forecast data
        if 'forecast' not in forecast_data or forecast_data['forecast'] is None:
            logger.warning("No forecast data available for spike detection")
            return []
        
        # Find all peaks above small spike threshold, but exclude grace period
        peak_indices = np.where(forecast_data['forecast'] > small_spike_threshold)[0]
        valid_peak_indices = peak_indices[peak_indices >= grace_period_points]
        
        logger.info(f"DEBUG: Found {len(peak_indices)} total peaks, {len(valid_peak_indices)} peaks after grace period")
        
        if len(valid_peak_indices) > 0:
            spike_groups = self._group_consecutive_peaks(valid_peak_indices, max_gap=3)
            
            for i, group in enumerate(spike_groups):
                peak_idx = group[np.argmax(forecast_data['forecast'][group])]
                spike_value = forecast_data['forecast'][peak_idx]
                spike_time = forecast_data['forecast_index'][peak_idx]
                
                time_from_now = (spike_time - self.raw_df.index[-1]).total_seconds()
                if time_from_now < grace_period_seconds:
                    continue
                
                # Better spike classification based on your pattern
                if spike_value > big_spike_threshold:
                    spike_type = "BIG"
                elif spike_value > small_spike_threshold:
                    spike_type = "SMALL"
                else:
                    continue  # Skip if below small threshold
                
                spike_info = {
                    'index': peak_idx,
                    'time': spike_time,
                    'value': spike_value,
                    'spike_id': len(all_spikes) + 1,
                    'type': spike_type,
                    'time_from_now': time_from_now
                }
                all_spikes.append(spike_info)
                logger.info(f"DEBUG: {spike_type} Spike {spike_info['spike_id']} - "
                      f"Value: {spike_value:.2f}, Threshold ratio: {spike_value/baseline_value:.2f}x")
    
        if not all_spikes:
            logger.info("DEBUG: No absolute spikes found, using relative peak detection")
            try:
                from scipy.signal import find_peaks
                
                peaks, properties = find_peaks(
                    forecast_data['forecast'], 
                    height=np.percentile(forecast_data['forecast'], 75),
                    prominence=np.std(forecast_data['forecast']) * 0.3,
                    distance=2
                )
                
                # Apply grace period to relative peaks
                valid_relative_peaks = peaks[peaks >= grace_period_points]
                
                for i, peak_idx in enumerate(valid_relative_peaks):
                    spike_value = forecast_data['forecast'][peak_idx]
                    spike_time = forecast_data['forecast_index'][peak_idx]
                    time_from_now = (spike_time - self.raw_df.index[-1]).total_seconds()
                    
                    # Double-check grace period
                    if time_from_now < grace_period_seconds:
                        continue
                    
                    spike_info = {
                        'index': peak_idx,
                        'time': spike_time,
                        'value': spike_value,
                        'spike_id': len(all_spikes) + 1,
                        'type': "SMALL",  # Default type
                        'time_from_now': time_from_now
                    }
                    all_spikes.append(spike_info)
                    logger.info(f"DEBUG: Relative Spike {len(all_spikes)} - Index: {peak_idx}, "
                        f"Value: {spike_value:.2f}, Time from now: {time_from_now:.1f}s")
                
            except ImportError:
                logger.info("DEBUG: scipy not available for relative peak detection")
            except Exception as e:
                logger.warning(f"DEBUG: Error in relative peak detection: {e}")
    
        logger.info(f"DEBUG: Final spike count: {len(all_spikes)}")
        return all_spikes
    
    def _group_consecutive_peaks(self, peak_indices, max_gap=2):
        """Group consecutive peaks together"""
        if len(peak_indices) == 0:
            return []
        
        groups = []
        current_group = [peak_indices[0]]
        
        for i in range(1, len(peak_indices)):
            if peak_indices[i] - peak_indices[i-1] <= max_gap:
                current_group.append(peak_indices[i])
            else:
                groups.append(current_group)
                current_group = [peak_indices[i]]
        
        groups.append(current_group)
        return groups

    def create_spike_features(self):
        """Create additional features to help detect spike patterns"""
        if self.raw_df is None or self.raw_df.empty:
            logger.warning("No raw data available for spike features")
            return pd.DataFrame()
        
        df = self.raw_df.copy()
        
        # Rolling statistics to capture local patterns
        df['rolling_mean_5'] = df['total'].rolling(window=5).mean()
        df['rolling_std_5'] = df['total'].rolling(window=5).std()
        df['rolling_max_5'] = df['total'].rolling(window=5).max()
        
        # Detect previous spikes to create pattern features
        threshold = df['total'].quantile(0.95)
        df['is_spike'] = (df['total'] > threshold).astype(int)
        
        # Create lag features for spike pattern
        df['prev_spike_1'] = df['is_spike'].shift(1)
        df['prev_spike_2'] = df['is_spike'].shift(2)
        df['prev_spike_3'] = df['is_spike'].shift(3)
        
        # Create alternating pattern feature
        df['spike_magnitude'] = np.where(df['is_spike'] == 1, 
                                       np.where(df['total'] > df['total'].quantile(0.98), 2, 1), 0)
        
        # Pattern state: 0=normal, 1=small spike expected, 2=big spike expected
        df['pattern_state'] = 0
        for i in range(3, len(df)):
            if df.iloc[i-1]['spike_magnitude'] == 1:  # Previous was small spike
                df.iloc[i, df.columns.get_loc('pattern_state')] = 2  # Expect big spike
            elif df.iloc[i-1]['spike_magnitude'] == 2:  # Previous was big spike
                df.iloc[i, df.columns.get_loc('pattern_state')] = 1  # Expect small spike
    
        return df

    def predict_spike_pattern(self, forecast_data, enhanced_df):
        """Enhanced spike prediction with proper alternating pattern"""
        if 'forecast' not in forecast_data or forecast_data['forecast'] is None:
            logger.warning("No forecast data available for spike pattern prediction")
            return np.array([])
        
        if enhanced_df is None or enhanced_df.empty:
            logger.warning("No enhanced features available, returning original forecast")
            return forecast_data['forecast'].copy()
        
        base_forecast = forecast_data['forecast'].copy()
        
        # Analyze recent spike pattern
        recent_data = enhanced_df.tail(40)  # Look at more history
        spike_pattern = recent_data['spike_magnitude'].values
        
        # Find the most recent spike type
        last_spike_type = 0
        last_spike_index = -1
        
        for i in range(len(spike_pattern)-1, -1, -1):
            if spike_pattern[i] > 0:
                last_spike_type = spike_pattern[i]
                last_spike_index = i
                break

        logger.info(f"DEBUG: Last spike type: {last_spike_type} at index {last_spike_index}")
        
        # Don't over-amplify - use subtle adjustments
        enhanced_forecast = base_forecast.copy()
        
        if self.raw_df is None or self.raw_df.empty:
            logger.warning("No raw data for baseline calculation")
            return enhanced_forecast
        
        baseline = self.raw_df['total'].median()
        
        # Look for potential spike locations in forecast
        for i, value in enumerate(base_forecast):
            # Only adjust if value is already elevated
            if value > baseline * 1.2:  # 20% above baseline
                if last_spike_type == 1:  # Last was small, next should be big
                    # Slightly increase prediction for big spike
                    enhanced_forecast[i] = value * 1.2
                    logger.info(f"DEBUG: Enhancing prediction at {i} for expected BIG spike (1.2x)")
                    last_spike_type = 2
                elif last_spike_type == 2:  # Last was big, next should be small  
                    # Slightly decrease prediction for small spike
                    enhanced_forecast[i] = value * 0.9
                    logger.info(f"DEBUG: Reducing prediction at {i} for expected SMALL spike (0.9x)")
                    last_spike_type = 1
                else:
                    # First spike, assume small
                    enhanced_forecast[i] = value * 0.95
                    last_spike_type = 1
                break  # Only adjust first spike found

        return enhanced_forecast
    
    #
    # PARAMETER OPTIMIZING FOR SPIKES
    # 

    def optimize_parameters_for_spikes(self):
        """Parameter optimization with minimal combinations for faster initialization"""
        logger.info("Starting parameter optimization...")
        
        if self.transformed_df is None or self.transformed_df.empty:
            logger.error("No transformed data available for optimization")
            return None
            
        # Split data for training and testing
        train_size = int(len(self.transformed_df) * 0.8)
        train_data = self.transformed_df.iloc[:train_size]
        test_data = self.transformed_df.iloc[train_size:]
        
        param_ranges = {
            'p': [3, 5, 7],  # Higher AR terms to capture spike dependencies
            'd': [0, 1],
            'q': [1, 2],  # Higher MA terms for pattern smoothing
            'P': [0, 1],     # Seasonal AR for pattern repetition
            'D': [1],
            'Q': [1],     # Seasonal MA
            's': [40]
        }

        param_combinations = list(product(
            param_ranges['p'], param_ranges['d'], param_ranges['q'],
            param_ranges['P'], param_ranges['D'], param_ranges['Q'], param_ranges['s']
        ))
        
        best_score = float('-inf')
        results = []
        
        for params in tqdm(param_combinations[:50], desc="Testing spike-focused parameters"):  # Limit for speed
            p, d, q, P, D, Q, s = params
            order = (p, d, q)
            seasonal_order = (P, D, Q, s)
            
            try:
                fitted = self.fit_sarima_model(train_data, order, seasonal_order)
                forecast = fitted.forecast(steps=len(test_data))
                forecast_orig = self.inverse_transform(forecast)
                test_actual = self.raw_df.iloc[len(train_data):len(train_data)+len(test_data)]['total']
                
                # Enhanced metrics that heavily weight pattern detection
                metrics = self.calculate_pattern_metrics(forecast_orig, test_actual)
                
                # Score heavily weighted towards pattern accuracy
                combined_score = (
                    0.8 * metrics['pattern_accuracy'] +
                    0.2 * metrics['spike_accuracy'] - 
                    0.1 * (metrics['rmse'] / 100)
                )
                
                result = {
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'aic': fitted.aic,
                    'combined_score': combined_score,
                    **metrics
                }
                results.append(result)
                
                if combined_score > best_score:
                    best_score = combined_score
                    self.best_params = (order, seasonal_order)
                    self.best_model = fitted
                    
            except Exception as e:
                continue
    
        return pd.DataFrame(results).sort_values('combined_score', ascending=False)
    
    def calculate_pattern_metrics(self, forecast_orig, test_actual):
        """Calculate metrics specifically for alternating spike patterns"""
        rmse = np.sqrt(np.mean((forecast_orig - test_actual) ** 2))
        mae = np.mean(np.abs(forecast_orig - test_actual))
        
        # Detect spike patterns in both series
        threshold = self.raw_df['total'].quantile(0.95)
        
        actual_spikes = self._classify_spikes(test_actual, threshold)
        pred_spikes = self._classify_spikes(forecast_orig, threshold)
        
        # Calculate pattern accuracy (small-big-small-big sequence)
        pattern_accuracy = self._calculate_pattern_accuracy(actual_spikes, pred_spikes)
        
        # Regular spike accuracy
        spike_matches = self._count_spike_matches_improved(
            [i for i, _ in pred_spikes], 
            [i for i, _ in actual_spikes], 
            tolerance=3
        )
        spike_accuracy = spike_matches / max(len(actual_spikes), 1) if actual_spikes else 0
        
        return {
            'rmse': rmse,
            'mae': mae,
            'spike_accuracy': spike_accuracy,
            'pattern_accuracy': pattern_accuracy,
            'actual_spikes': len(actual_spikes),
            'predicted_spikes': len(pred_spikes)
        }

    def _classify_spikes(self, data, threshold):
        """Classify spikes as small (1) or big (2)"""
        spikes = []
        big_threshold = threshold * 1.2  # 20% higher for "big" spikes
        
        for i, value in enumerate(data):
            if value > big_threshold:
                spikes.append((i, 2))  # Big spike
            elif value > threshold:
                spikes.append((i, 1))  # Small spike
        
        return spikes

    def _calculate_pattern_accuracy(self, actual_spikes, pred_spikes):
        """Calculate how well the alternating pattern is predicted"""
        if len(actual_spikes) < 2 or len(pred_spikes) < 2:
            return 0
        
        # Get pattern sequences
        actual_pattern = [magnitude for _, magnitude in actual_spikes]
        pred_pattern = [magnitude for _, magnitude in pred_spikes]
        
        # Check for alternating pattern in actual data
        actual_alternates = self._check_alternating_pattern(actual_pattern)
        pred_alternates = self._check_alternating_pattern(pred_pattern)
        
        if actual_alternates and pred_alternates:
            # Both have alternating patterns - calculate similarity
            min_len = min(len(actual_pattern), len(pred_pattern))
            matches = sum(1 for i in range(min_len) if actual_pattern[i] == pred_pattern[i])
            return matches / min_len
        elif actual_alternates or pred_alternates:
            # Only one has alternating pattern
            return 0.5
        else:
            # Neither has clear alternating pattern
            return 0.3

    def _check_alternating_pattern(self, pattern):
        """Check if pattern alternates between small(1) and big(2) spikes"""
        if len(pattern) < 3:
            return False
        
        alternating_count = 0
        for i in range(len(pattern) - 1):
            if pattern[i] != pattern[i + 1]:
                alternating_count += 1
        
        # Pattern alternates if at least 60% of transitions are different
        return alternating_count / (len(pattern) - 1) >= 0.6
    
    # 
    # 
    # 

    def detect_spikes_in_forecast_improved(self, forecast_data, grace_period_seconds=60):
        """Enhanced spike detection with better threshold separation"""
        logger.info("DEBUG: Starting spike detection with grace period...")
        
        # Calculate grace period in data points (15-second intervals)
        grace_period_points = int(grace_period_seconds / 15)
        logger.info(f"DEBUG: Grace period: {grace_period_seconds} seconds ({grace_period_points} data points)")
        
        # Safety check for raw_df
        if self.raw_df is None or self.raw_df.empty:
            logger.warning("No raw data available for spike detection")
            return []
        
        baseline_value = self.raw_df['total'].median()
        
        small_spike_threshold = baseline_value * 1.1
        big_spike_threshold = baseline_value * 1.8
        
        logger.info(f"DEBUG: Small threshold: {small_spike_threshold:.2f}, Big threshold: {big_spike_threshold:.2f}")
        
        all_spikes = []
        
        # Safety check for forecast data
        if 'forecast' not in forecast_data or forecast_data['forecast'] is None:
            logger.warning("No forecast data available for spike detection")
            return []
        
        # Find all peaks above small spike threshold, but exclude grace period
        peak_indices = np.where(forecast_data['forecast'] > small_spike_threshold)[0]
        valid_peak_indices = peak_indices[peak_indices >= grace_period_points]
        
        logger.info(f"DEBUG: Found {len(peak_indices)} total peaks, {len(valid_peak_indices)} peaks after grace period")
        
        if len(valid_peak_indices) > 0:
            spike_groups = self._group_consecutive_peaks(valid_peak_indices, max_gap=3)
            
            for i, group in enumerate(spike_groups):
                peak_idx = group[np.argmax(forecast_data['forecast'][group])]
                spike_value = forecast_data['forecast'][peak_idx]
                spike_time = forecast_data['forecast_index'][peak_idx]
                
                time_from_now = (spike_time - self.raw_df.index[-1]).total_seconds()
                if time_from_now < grace_period_seconds:
                    continue
                
                # Better spike classification based on your pattern
                if spike_value > big_spike_threshold:
                    spike_type = "BIG"
                elif spike_value > small_spike_threshold:
                    spike_type = "SMALL"
                else:
                    continue  # Skip if below small threshold
                
                spike_info = {
                    'index': peak_idx,
                    'time': spike_time,
                    'value': spike_value,
                    'spike_id': len(all_spikes) + 1,
                    'type': spike_type,
                    'time_from_now': time_from_now
                }
                all_spikes.append(spike_info)
                logger.info(f"DEBUG: {spike_type} Spike {spike_info['spike_id']} - "
                      f"Value: {spike_value:.2f}, Threshold ratio: {spike_value/baseline_value:.2f}x")
    
    #
    # MODEL INITIALIZATION AND UPDATE
    #

    def initialize_model(self) -> bool:
        """
        Initialize model by calling update_model
        """
        if self.is_initialized:
            return True
        
        logger.info("Initializing CPU prediction model...")
        return self.update_model()

    def update_model(self, force_refresh=False) -> bool:
        """Update SARIMAX model with latest CPU usage data"""
        current_time = time.time()
        
        # Only update if enough time has passed (unless not initialized or forced)
        if (self.is_initialized and 
            current_time - self.last_update < self.update_interval_seconds and 
            not force_refresh):
            return True  # Model is up-to-date

        logger.info(f"🔄 Fetching latest CPU data from Prometheus...")
        
        # Load Data
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
            
            if not self.is_initialized:
                logger.info("Performing model initialization...")
                
                self.analyze_data_characteristics()
                
                self.prepare_transformed_data(method='auto')
                
                self.optimize_parameters_for_spikes()
                
                if self.best_params is None:
                    logger.warning("Optimization failed, using simple default")
                    self.best_params = ((1, 1, 1), (0, 1, 0, 12))
                
                forecast_data = self.forecast_future(steps=100)

                enhanced_df = self.create_spike_features()

                enhanced_forecast = self.predict_spike_pattern(forecast_data, enhanced_df)
                forecast_data['forecast'] = enhanced_forecast
                
                grace_period = 60  # seconds
                self.detect_spikes_in_forecast_improved(forecast_data, grace_period_seconds=grace_period)

                self.is_initialized = True
                self.last_update = current_time
                return True
            else:
                # For updates, retrain with existing parameters on new data
                logger.info("📈 Updating model with fresh data...")
                
                # Update transformed data with same method
                self.prepare_transformed_data(method=self.transform_method)
                
                # Retrain model with existing best parameters
                try:
                    order, seasonal_order = self.best_params
                    updated_model = SARIMAX(
                        self.transformed_df['total'],
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    self.best_model = updated_model.fit(disp=False, maxiter=25)
                    
                    self.last_update = current_time
                    logger.info(f"✅ Model updated with {len(prepared_df)} data points")
                    
                    return True
                    
                except Exception as e:
                    logger.error(f"Error updating SARIMA model: {e}")
                    return False
            
        except Exception as e:
            logger.error(f"Error processing CPU metrics data for model update: {e}")
            return False
    
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
            max_possible_width = np.std(self.transformed_df['total']) * 4  # Rough estimate
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
        

    # PLOTTING USES

    def generate_forecast_plot(self, steps=100, figsize=(15, 10), include_confidence=True):
        """Generate a comprehensive forecast visualization"""
        if not self.is_initialized:
            logger.warning("Model not initialized, cannot generate plot")
            return None
            
        try:
            # Generate forecast data
            forecast_data = self.forecast_future(steps=steps)
            if not forecast_data or 'forecast' not in forecast_data:
                logger.error("Failed to generate forecast data")
                return None
            
            # Create enhanced features for spike detection
            enhanced_df = self.create_spike_features()
            
            # Get enhanced forecast
            enhanced_forecast = self.predict_spike_pattern(forecast_data, enhanced_df)
            
            # Ensure enhanced_forecast is valid
            if enhanced_forecast is None or len(enhanced_forecast) == 0:
                logger.warning("Enhanced forecast is empty, using original forecast")
                enhanced_forecast = forecast_data['forecast']
            
            # Update forecast data with enhanced predictions
            forecast_data['forecast'] = enhanced_forecast
            
            # Detect spikes in forecast
            spikes = self.detect_spikes_in_forecast_improved(
                forecast_data, grace_period_seconds=60
            )
            
            # Ensure spikes is a list
            if spikes is None:
                spikes = []
            
            # Create the plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
            
            # Main forecast plot
            self._plot_main_forecast(ax1, forecast_data, spikes, include_confidence)
            
            # Spike detection details
            self._plot_spike_details(ax2, spikes)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)  # Clean up
            
            return {
                'plot_base64': plot_data,
                'forecast_data': {
                    'forecast_values': forecast_data['forecast'].tolist(),
                    'forecast_times': [t.isoformat() for t in forecast_data['forecast_index']],
                    'confidence_lower': forecast_data['ci_lower'].tolist() if include_confidence else None,
                    'confidence_upper': forecast_data['ci_upper'].tolist() if include_confidence else None
                },
                'spikes': spikes,
                'model_info': {
                    'best_params': self.best_params,
                    'transform_method': self.transform_method,
                    'data_points': len(self.raw_df) if self.raw_df is not None else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating forecast plot: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
        
    def _plot_main_forecast(self, ax, forecast_data, spikes, include_confidence):
        """Plot the main forecast with historical data"""
        # Plot historical data
        historical_data = self.raw_df.tail(50)  # Last 50 points for context
        ax.plot(historical_data.index, historical_data['total'], 
                'b-', label='Historical CPU Usage', linewidth=2, alpha=0.8)
        
        # Plot forecast
        ax.plot(forecast_data['forecast_index'], forecast_data['forecast'], 
                'r-', label='Forecast', linewidth=2)
        
        # Plot confidence intervals if requested
        if include_confidence:
            ax.fill_between(forecast_data['forecast_index'], 
                          forecast_data['ci_lower'], 
                          forecast_data['ci_upper'],
                          alpha=0.3, color='red', label='95% Confidence Interval')
        
        # Mark spikes
        for spike in spikes:
            color = 'orange' if spike['type'] == 'SMALL' else 'red'
            marker = 'o' if spike['type'] == 'SMALL' else '^'
            size = 8 if spike['type'] == 'SMALL' else 12
            
            ax.scatter(spike['time'], spike['value'], 
                      color=color, marker=marker, s=size**2, 
                      label=f"{spike['type']} Spike" if spike['spike_id'] == 1 else "",
                      zorder=5, edgecolors='black', linewidth=1)
            
            # Add spike labels
            ax.annotate(f"S{spike['spike_id']}", 
                       (spike['time'], spike['value']),
                       xytext=(0, 15), textcoords='offset points',
                       ha='center', va='bottom', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
        
        # Add baseline and threshold lines
        baseline = self.raw_df['total'].median()
        ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        ax.axhline(y=baseline * 1.1, color='orange', linestyle=':', alpha=0.7, label='Small Spike Threshold')
        ax.axhline(y=baseline * 1.8, color='red', linestyle=':', alpha=0.7, label='Big Spike Threshold')
        
        # Add vertical line to separate historical from forecast
        if len(self.raw_df) > 0:
            ax.axvline(x=self.raw_df.index[-1], color='black', linestyle='--', alpha=0.5, label='Prediction Start')
        
        ax.set_ylabel('CPU Usage Rate')
        ax.set_title('CPU Usage Forecast with Spike Detection')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_spike_details(self, ax, spikes):
        """Plot spike timing and type details"""
        if not spikes:
            ax.text(0.5, 0.5, 'No spikes detected in forecast', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Spike Detection Summary')
            return
        
        # Create timeline of spikes
        spike_times = [spike['time_from_now'] for spike in spikes]
        spike_types = [1 if spike['type'] == 'SMALL' else 2 for spike in spikes]
        spike_colors = ['orange' if spike['type'] == 'SMALL' else 'red' for spike in spikes]
        
        ax.scatter(spike_times, spike_types, c=spike_colors, s=100, alpha=0.7)
        
        for i, spike in enumerate(spikes):
            ax.annotate(f"S{spike['spike_id']}\n{spike['time_from_now']:.0f}s", 
                       (spike['time_from_now'], spike_types[i]),
                       xytext=(0, 20), textcoords='offset points',
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Time from Now (seconds)')
        ax.set_ylabel('Spike Type')
        ax.set_yticks([1, 2])
        ax.set_yticklabels(['SMALL', 'BIG'])
        ax.set_title('Spike Detection Timeline')
        ax.grid(True, alpha=0.3)
        
        # Add summary text
        small_count = sum(1 for s in spikes if s['type'] == 'SMALL')
        big_count = sum(1 for s in spikes if s['type'] == 'BIG')
        summary_text = f"Total: {len(spikes)} spikes | Small: {small_count} | Big: {big_count}"
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7),
               verticalalignment='top')