import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats
from itertools import product
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class SARIMAForecaster:
    """SARIMA-based forecasting for CPU usage with spike detection"""
    
    def __init__(self, data_path='data/training_data_big_small_idle.csv'):
        """Initialize the forecaster with data"""
        self.data_path = data_path
        self.df = None
        self.transformed_df = None
        self.transform_method = None
        self.boxcox_lambda = None
        self.best_model = None
        self.best_params = None
        self.data_characteristics = None

    def load_data(self):
        """Load and prepare the time series data"""
        self.df = pd.read_csv(self.data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.set_index('timestamp')
        print(f"Loaded {len(self.df)} data points")
        
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
        data = self.df['value']
        
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
        
        print(f"Data Analysis Results:")
        print(f"  Skewness: {stats_dict['skewness']:.3f}")
        print(f"  Coefficient of Variation: {stats_dict['cv']:.3f}")
        print(f"  Range Ratio: {stats_dict['range_ratio']:.3f}")
        print(f"  Spike Ratio: {stats_dict['spike_ratio']:.3f}")
        print(f"  Needs Transformation: {needs_transformation}")
        print(f"  Recommended Method: {recommended_method}")
        
        return self.data_characteristics

    def prepare_transformed_data(self, method='auto'):
        """Prepare dataset with optional transformation"""
        if method == 'auto':
            if not self.data_characteristics:
                self.analyze_data_characteristics()
            
            if not self.data_characteristics['needs_transformation']:
                method = 'none'
                print("Data analysis suggests no transformation needed - using original data")
            else:
                method = self.data_characteristics['recommended_method']
                print(f"Data analysis recommends: {method} transformation")
        
        self.transform_method = method
        
        if method == 'none':
            # Use original data without transformation
            transformed_data = self.df['value']
            self.boxcox_lambda = None
        elif method == 'boxcox':
            transformed_data, self.boxcox_lambda = self.transform_data(
                self.df['value'], method=method
            )
        else:
            transformed_data = self.transform_data(self.df['value'], method=method)
            self.boxcox_lambda = None
            
        self.transformed_df = pd.DataFrame(
            {'value': transformed_data}, 
            index=self.df.index
        )
    
    def split_data(self, train_ratio=0.8):
        """Split data into training and testing sets"""
        train_size = int(len(self.transformed_df) * train_ratio)
        train_data = self.transformed_df.iloc[:train_size]
        test_data = self.transformed_df.iloc[train_size:]
        return train_data, test_data
    
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
            self.transformed_df['value'],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        full_results = full_model.fit(disp=False)
        
        # Generate forecast
        forecast_index = pd.date_range(
            start=self.df.index[-1] + timedelta(seconds=15),
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
        print("DEBUG: Starting spike detection with grace period...")
        
        # Calculate grace period in data points (15-second intervals)
        grace_period_points = int(grace_period_seconds / 15)
        print(f"DEBUG: Grace period: {grace_period_seconds} seconds ({grace_period_points} data points)")
        
        baseline_value = self.df['value'].median()
        
        small_spike_threshold = baseline_value * 1.1
        big_spike_threshold = baseline_value * 1.8
        
        # print(f"DEBUG: Baseline: {baseline_value:.2f}, Std: {std_value:.2f}")
        print(f"DEBUG: Small threshold: {small_spike_threshold:.2f}, Big threshold: {big_spike_threshold:.2f}")
        
        all_spikes = []
        
        # Find all peaks above small spike threshold, but exclude grace period
        peak_indices = np.where(forecast_data['forecast'] > small_spike_threshold)[0]
        valid_peak_indices = peak_indices[peak_indices >= grace_period_points]
        
        print(f"DEBUG: Found {len(peak_indices)} total peaks, {len(valid_peak_indices)} peaks after grace period")
        
        if len(valid_peak_indices) > 0:
            spike_groups = self._group_consecutive_peaks(valid_peak_indices, max_gap=3)
            
            for i, group in enumerate(spike_groups):
                peak_idx = group[np.argmax(forecast_data['forecast'][group])]
                spike_value = forecast_data['forecast'][peak_idx]
                spike_time = forecast_data['forecast_index'][peak_idx]
                
                time_from_now = (spike_time - self.df.index[-1]).total_seconds()
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
                print(f"DEBUG: {spike_type} Spike {spike_info['spike_id']} - "
                      f"Value: {spike_value:.2f}, Threshold ratio: {spike_value/baseline_value:.2f}x")
                
        if not all_spikes:
            print("DEBUG: No absolute spikes found, using relative peak detection")
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
                    time_from_now = (spike_time - self.df.index[-1]).total_seconds()
                    
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
                    print(f"DEBUG: Relative Spike {len(all_spikes)} - Index: {peak_idx}, "
                        f"Value: {spike_value:.2f}, Time from now: {time_from_now:.1f}s")
                    
            except ImportError:
                print("DEBUG: scipy not available for relative peak detection")
        
        print(f"DEBUG: Final spike count: {len(all_spikes)}")
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
        df = self.df.copy()
        
        # Rolling statistics to capture local patterns
        df['rolling_mean_5'] = df['value'].rolling(window=5).mean()
        df['rolling_std_5'] = df['value'].rolling(window=5).std()
        df['rolling_max_5'] = df['value'].rolling(window=5).max()
        
        # Detect previous spikes to create pattern features
        threshold = df['value'].quantile(0.95)
        df['is_spike'] = (df['value'] > threshold).astype(int)
        
        # Create lag features for spike pattern
        df['prev_spike_1'] = df['is_spike'].shift(1)
        df['prev_spike_2'] = df['is_spike'].shift(2)
        df['prev_spike_3'] = df['is_spike'].shift(3)
        
        # Create alternating pattern feature
        df['spike_magnitude'] = np.where(df['is_spike'] == 1, 
                                       np.where(df['value'] > df['value'].quantile(0.98), 2, 1), 0)
        
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
    
        print(f"DEBUG: Last spike type: {last_spike_type} at index {last_spike_index}")
        
        # Don't over-amplify - use subtle adjustments
        enhanced_forecast = base_forecast.copy()
        baseline = self.df['value'].median()
        
        # Look for potential spike locations in forecast
        for i, value in enumerate(base_forecast):
            # Only adjust if value is already elevated
            if value > baseline * 1.2:  # 20% above baseline
                if last_spike_type == 1:  # Last was small, next should be big
                    # Slightly increase prediction for big spike
                    enhanced_forecast[i] = value * 1.2
                    print(f"DEBUG: Enhancing prediction at {i} for expected BIG spike (1.2x)")
                    last_spike_type = 2
                elif last_spike_type == 2:  # Last was big, next should be small  
                    # Slightly decrease prediction for small spike
                    enhanced_forecast[i] = value * 0.9
                    print(f"DEBUG: Reducing prediction at {i} for expected SMALL spike (0.9x)")
                    last_spike_type = 1
                else:
                    # First spike, assume small
                    enhanced_forecast[i] = value * 0.95
                    last_spike_type = 1
                break  # Only adjust first spike found
    
        return enhanced_forecast

    def optimize_parameters_for_spikes(self):
        """Modified parameter optimization focused on spike patterns"""
        print("Starting spike-focused parameter optimization...")
        
        train_data, test_data = self.split_data()
        
        # Focus on parameters that better capture short-term patterns
        param_ranges = {
            'p': [3, 5, 7],  # Higher AR terms to capture spike dependencies
            'd': [0, 1],
            'q': [1, 2, 3],  # Higher MA terms for pattern smoothing
            'P': [1, 2],     # Seasonal AR for pattern repetition
            'D': [0, 1],
            'Q': [1, 2],     # Seasonal MA
            's': [20, 40]
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
                test_actual = self.df.iloc[len(train_data):len(train_data)+len(test_data)]['value']
                
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
        threshold = self.df['value'].quantile(0.95)
        
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

class SARIMAVisualizer:
    """Visualization class for SARIMA forecasting results"""
    
    def __init__(self, forecaster):
        self.forecaster = forecaster
        sns.set(style="whitegrid")
    
    def plot_data_transformation(self):
        """Plot original and transformed data"""
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Original data
        axes[0].plot(self.forecaster.df.index, self.forecaster.df['value'], 'b-', label='Original CPU Usage')
        axes[0].set_title('Original CPU Usage Data', fontsize=14)
        axes[0].set_ylabel('CPU Usage', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Transformed data
        axes[1].plot(self.forecaster.transformed_df.index, self.forecaster.transformed_df['value'], 
                    'g-', label=f'{self.forecaster.transform_method.capitalize()}-Transformed Data')
        axes[1].set_title(f'{self.forecaster.transform_method.capitalize()}-Transformed CPU Usage Data', fontsize=14)
        axes[1].set_ylabel('Transformed Value', fontsize=12)
        axes[1].set_xlabel('Time', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data_transformation.png', dpi=300)
        plt.show()
    
    def plot_forecast_results(self, forecast_data, spikes=None, grace_period_seconds=60):
        """Plot the final forecast results with grace period visualization and threshold lines"""
        plt.figure(figsize=(16, 10))
        
        # Calculate thresholds (same as in detect_spikes_in_forecast_improved)
        baseline_value = self.forecaster.df['value'].median()
        small_spike_threshold = baseline_value * 1.1
        big_spike_threshold = baseline_value * 1.8
        
        # Plot historical data
        plt.plot(self.forecaster.df.index, self.forecaster.df['value'], 
                'b-', alpha=0.7, label='Historical CPU Usage')
        
        # Plot forecast
        order, seasonal_order = self.forecaster.best_params
        plt.plot(forecast_data['forecast_index'], forecast_data['forecast'], 
                'r-', linewidth=2, label=f'SARIMA{order}x{seasonal_order} Forecast')
        
        # Plot confidence intervals
        plt.fill_between(
            forecast_data['forecast_index'],
            forecast_data['ci_lower'],
            forecast_data['ci_upper'],
            color='r', alpha=0.2, label='95% Confidence Interval'
        )
        
        # Add baseline and threshold lines
        x_range = [self.forecaster.df.index[0], forecast_data['forecast_index'][-1]]
        
        # Baseline (median)
        plt.axhline(y=baseline_value, color='green', linestyle='-', linewidth=2, 
                    alpha=0.8, label=f'Baseline (Median: {baseline_value:.1f}%)')
        
        # Small spike threshold
        plt.axhline(y=small_spike_threshold, color='orange', linestyle='--', linewidth=2, 
                    alpha=0.8, label=f'Small Spike Threshold ({small_spike_threshold:.1f}%)')
        
        # Big spike threshold
        plt.axhline(y=big_spike_threshold, color='red', linestyle='--', linewidth=2, 
                    alpha=0.8, label=f'Big Spike Threshold ({big_spike_threshold:.1f}%)')
        
        # Add threshold range shading
        plt.fill_between(x_range, baseline_value, small_spike_threshold, 
                        color='lightgreen', alpha=0.1, label='Normal Range')
        plt.fill_between(x_range, small_spike_threshold, big_spike_threshold, 
                        color='yellow', alpha=0.1, label='Small Spike Range')
        plt.fill_between(x_range, big_spike_threshold, plt.ylim()[1], 
                        color='red', alpha=0.1, label='Big Spike Range')
        
        # Add grace period shading
        grace_end_time = self.forecaster.df.index[-1] + timedelta(seconds=grace_period_seconds)
        grace_mask = forecast_data['forecast_index'] <= grace_end_time
        
        if np.any(grace_mask):
            plt.fill_between(
                forecast_data['forecast_index'][grace_mask],
                forecast_data['ci_lower'][grace_mask],
                forecast_data['ci_upper'][grace_mask],
                color='gray', alpha=0.3, label=f'Grace Period ({grace_period_seconds}s)'
            )
        
        # Mark predicted spikes (only those outside grace period)
        if spikes:
            for spike in spikes:
                # Use different colors for visual distinction but same annotation
                if spike.get('type') == 'BIG':
                    color = 'darkred'
                    marker = '^'
                    size = 150
                else:
                    color = 'orange' 
                    marker = 'o'
                    size = 100
                    
                plt.scatter(spike['time'], spike['value'], color=color, s=size, 
                        zorder=5, marker=marker, edgecolors='black', linewidth=1)
                plt.annotate(f'Spike {spike["spike_id"]}\n{spike["value"]:.1f}\n(+{spike["time_from_now"]:.0f}s)', 
                        xy=(spike['time'], spike['value']),
                        xytext=(0, 20), textcoords='offset points',
                        ha='center', fontsize=10, color=color, weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9, edgecolor=color),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        
        plt.title(f'CPU Usage Forecast with Spike Detection (Grace Period: {grace_period_seconds}s) - SARIMA{order}x{seasonal_order}', fontsize=15)
        plt.ylabel('CPU Usage (%)', fontsize=14)
        plt.xlabel('Time', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left', fontsize=10, ncol=2)  # Use 2 columns for better space usage
        
        # Add model info with threshold values
        plt.text(0.02, 0.95, 
                f"Model: SARIMA{order}x{seasonal_order}\n"
                f"Transform: {self.forecaster.transform_method}\n"
                f"Grace Period: {grace_period_seconds}s\n"
                f"Baseline: {baseline_value:.1f}%\n"
                f"Small Threshold: {small_spike_threshold:.1f}%\n"
                f"Big Threshold: {big_spike_threshold:.1f}%",
                transform=plt.gca().transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
        
        # Enhanced spike timing info
        if spikes:
            next_spike = spikes[0]
            time_diff = next_spike['time_from_now']
            
            plt.figtext(0.5, 0.02, 
                    f"Valid Spikes: {len(spikes)} (after {grace_period_seconds}s grace period) | "
                    f"Next spike in {time_diff:.1f} seconds at {next_spike['time'].strftime('%H:%M:%S')}",
                    ha="center", fontsize=11, 
                    bbox=dict(facecolor='lightblue', alpha=0.8, edgecolor='navy'))
        else:
            plt.figtext(0.5, 0.02, 
                    f"No spikes predicted after {grace_period_seconds}s grace period",
                    ha="center", fontsize=11, 
                    bbox=dict(facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('outputs/spike_forecast_with_grace.png', dpi=300, bbox_inches='tight')
        plt.show()

# Enhanced main function
def main():
    forecaster = SARIMAForecaster()
    forecaster.load_data()
    forecaster.analyze_data_characteristics()
    forecaster.prepare_transformed_data(method='auto')
    
    # Use spike-focused optimization
    forecaster.optimize_parameters_for_spikes()
    
    if forecaster.best_params:
        # Generate forecast with pattern awareness
        forecast_data = forecaster.forecast_future(steps=100)
        
        # Create enhanced features
        enhanced_df = forecaster.create_spike_features()
        
        # Apply pattern-aware prediction
        enhanced_forecast = forecaster.predict_spike_pattern(forecast_data, enhanced_df)
        forecast_data['forecast'] = enhanced_forecast
        
        grace_period = 60  # seconds
        spikes = forecaster.detect_spikes_in_forecast_improved(forecast_data, grace_period_seconds=grace_period)
        
        # Visualize results with grace period
        visualizer = SARIMAVisualizer(forecaster)
        visualizer.plot_forecast_results(forecast_data, spikes, grace_period_seconds=grace_period)
        
        print(f"\nPattern-Aware SARIMA Model: SARIMA{forecaster.best_params[0]}x{forecaster.best_params[1]}")
        print(f"Grace Period: {grace_period} seconds")
        
        if spikes:
            print(f"\nDetected Spike Pattern (after {grace_period}s grace period):")
            small_count = sum(1 for spike in spikes if spike['type'] == 'SMALL')
            big_count = sum(1 for spike in spikes if spike['type'] == 'BIG')
            
            print(f"Valid Spikes: {len(spikes)} (Small: {small_count}, Big: {big_count})")
            
            for spike in spikes:
                print(f"{spike['type']} Spike {spike['spike_id']}: "
                      f"{spike['time'].strftime('%H:%M:%S')} (+{spike['time_from_now']:.1f}s) - "
                      f"Value: {spike['value']:.2f}")
            
            # Analyze pattern
            pattern_sequence = [spike['type'] for spike in spikes]
            if len(pattern_sequence) > 1:
                alternating = all(pattern_sequence[i] != pattern_sequence[i+1] 
                                for i in range(len(pattern_sequence)-1))
                if alternating:
                    print("\n✓ Detected alternating SMALL-BIG pattern!")
                else:
                    print(f"\n Pattern: {' → '.join(pattern_sequence)}")
        else:
            print(f"No spikes predicted after {grace_period}s grace period.")
    else:
        print("No optimal parameters found!")

if __name__ == "__main__":
    main()