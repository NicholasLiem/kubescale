import logging
import pandas as pd
from typing import Dict, Optional

logger = logging.getLogger("mltps")

class MetricsTransformerService:
    """Service for transforming raw Prometheus metrics into formats suitable for modeling"""
    
    def prometheus_to_dataframe(self, prometheus_data: Dict) -> pd.DataFrame:
        """
        Convert Prometheus query response to a pandas DataFrame with proper formatting
        
        Args:
            prometheus_data: Raw Prometheus response JSON
            
        Returns:
            DataFrame with timestamp index and columns for each metric/pod
        """
        if not prometheus_data or "data" not in prometheus_data or "result" not in prometheus_data["data"]:
            logger.warning("Empty or invalid Prometheus data provided")
            return pd.DataFrame()
            
        # Create an empty dictionary to store series for each pod/metric
        series_dict = {}
        
        # Process each result (typically one per pod/metric)
        for result in prometheus_data["data"]["result"]:
            # Get the name/label for this series (using pod name if available)
            if "metric" in result and "pod" in result["metric"]:
                name = result["metric"]["pod"]
            elif "metric" in result and "instance" in result["metric"]:
                name = result["metric"]["instance"]
            else:
                name = str(result.get("metric", "unknown"))
                
            # Extract timestamp and value pairs
            timestamps = []
            values = []
            
            for point in result.get("values", []):
                if len(point) >= 2:
                    timestamps.append(pd.to_datetime(point[0], unit='s'))
                    values.append(float(point[1]))
                
            # Create a series for this pod/metric
            if timestamps and values:
                series_dict[name] = pd.Series(values, index=timestamps)
        
        # Combine all series into a DataFrame
        df = pd.DataFrame(series_dict)
        
        return df
    
    def prepare_for_arima(self, df: pd.DataFrame, 
                     metric_type: str = "cpu", 
                     resample_freq: str = '1min', 
                     fillna_method: str = 'ffill', 
                     pod_name: Optional[str] = None,
                     aggregate: bool = False) -> pd.DataFrame:
        """
        Prepare metrics DataFrame for ARIMA modeling with appropriate unit conversion
        
        Args:
            df: Input DataFrame with timestamp index
            metric_type: Type of metric ('cpu' or 'memory')
            resample_freq: Frequency to resample the time series
            fillna_method: Method to fill missing values ('ffill', 'bfill', or numeric value)
            pod_name: If provided, extract only this pod's data
            aggregate: If True, aggregate all pod metrics into a single 'total' column
            
        Returns:
            Processed DataFrame ready for ARIMA modeling
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for ARIMA preparation")
            return df
            
        # Handle case where we want data for a specific pod only
        if pod_name and pod_name in df.columns:
            df = df[[pod_name]]
        
        # Make sure index is datetime type
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not DatetimeIndex, metrics processing may fail")
            return df
            
        # Resample to regular time intervals
        df_resampled = df.resample(resample_freq).mean()
        
        # Fill missing values
        if fillna_method == 'ffill':
            df_processed = df_resampled.fillna(method='ffill')
            # Fill any remaining NaNs at the beginning with backfill
            df_processed = df_processed.fillna(method='bfill')
        elif fillna_method == 'bfill':
            df_processed = df_resampled.fillna(method='bfill')
            # Fill any remaining NaNs at the end with forward fill
            df_processed = df_processed.fillna(method='ffill')
        else:
            # Fill with a specific numeric value
            try:
                fill_value = float(fillna_method)
                df_processed = df_resampled.fillna(fill_value)
            except ValueError:
                logger.error(f"Invalid fillna_method: {fillna_method}, using forward fill")
                df_processed = df_resampled.fillna(method='ffill')
                
        # Apply unit conversions based on metric type
        if metric_type == "memory":
            # Convert memory from bytes to MB for better scale
            for col in df_processed.columns:
                df_processed[col] = df_processed[col] / (1024 * 1024)  # Convert to MB
            logger.info(f"Converted memory values from bytes to MB for {len(df_processed.columns)} pods")
        elif metric_type == "cpu" or metric_type == "cpu_usage":
            # CPU values are already in core fractions, optionally convert to percentage 
            for col in df_processed.columns:
                df_processed[col] = df_processed[col] * 100  # Convert to percentage
            logger.info(f"Converted CPU values from core fractions to percentages for {len(df_processed.columns)} pods")
        
        # Check for and handle any remaining NaNs
        if df_processed.isna().any().any():
            logger.warning("Some NaN values remain after filling, replacing with zeros")
            df_processed = df_processed.fillna(0)
        
        # Aggregate if requested
        if aggregate and len(df_processed.columns) > 0:
            # Sum across all pods and create a single 'total' column
            total_series = df_processed.sum(axis=1)
            df_processed = pd.DataFrame({'total': total_series}, index=df_processed.index)
            logger.info(f"Aggregated metrics from {len(df.columns)} pods into a single 'total' column")
                
        return df_processed