import base64
import io
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils.logging_config import setup_logging

logger = setup_logging()

def generate_forecast_plot(historical_data, forecast, forecast_timestamps, confidence_interval=None):
    """Generate a matplotlib plot of historical data and forecast"""
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare historical data timestamps
        hist_timestamps = []
        current_time = datetime.datetime.now()
        for i in range(len(historical_data)):
            hist_time = current_time - datetime.timedelta(seconds=(len(historical_data) - i) * 15)
            hist_timestamps.append(hist_time)
        
        # Plot historical data
        ax.plot(hist_timestamps, historical_data.values, 'b-', linewidth=2, label='Historical Data', alpha=0.8)
        
        # Parse forecast timestamps
        forecast_times = [datetime.datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in forecast_timestamps]
        
        # Plot forecast
        ax.plot(forecast_times, forecast, 'r--', linewidth=2, label='Forecast', alpha=0.9)
        
        # Plot confidence intervals if available
        if confidence_interval:
            lower_bound = confidence_interval.get('lower_bound', [])
            upper_bound = confidence_interval.get('upper_bound', [])
            if lower_bound and upper_bound:
                ax.fill_between(forecast_times, lower_bound, upper_bound, 
                              alpha=0.2, color='red', label='95% Confidence Interval')
        
        # Add vertical line at current time
        ax.axvline(x=current_time, color='green', linestyle=':', linewidth=2, label='Current Time')
        
        # Formatting
        ax.set_xlabel('Time')
        ax.set_ylabel('CPU Usage (%)')
        ax.set_title('Traffic Forecast - CPU Usage Prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return plot_base64
        
    except Exception as e:
        logger.error(f"Error generating plot: {e}")
        return None