import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from arima_model import ARIMAModeler

def load_prometheus_data(file_path):
    """Load data from Prometheus JSON file and convert to pandas Series"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract the first time series (can be modified to handle multiple series)
    result = data['raw_data']['data']['result'][0]
    metric_name = list(result['metric'].values())[0]
    
    # Convert to pandas Series
    timestamps = [point[0] for point in result['values']]
    values = [float(point[1]) for point in result['values']]
    
    # Create datetime index
    date_index = pd.to_datetime([datetime.fromtimestamp(ts) for ts in timestamps])
    
    return pd.Series(values, index=date_index, name=metric_name)

def main():
    # Initialize the ARIMA modeler
    arima_modeler = ARIMAModeler()
    
    # Paths to generated data
    data_dir = Path("examples/prometheus")
    files = {
        "traffic": data_dir / "traffic-spike.json",
        "cpu": data_dir / "cpu-spike.json",
        "memory": data_dir / "memory-spike.json"
    }
    
    # Select which dataset to analyze (traffic, cpu, or memory)
    dataset = "traffic"
    
    # Load the data
    ts_data = load_prometheus_data(files[dataset])
    print(f"Loaded {dataset} data with {len(ts_data)} data points")
    
    # Split into training and testing sets (80/20)
    split_point = int(len(ts_data) * 0.8)
    train_data = ts_data[:split_point]
    test_data = ts_data[split_point:]
    
    # Train ARIMA model
    model = arima_modeler.train_model(train_data)
    
    # Generate forecast
    forecast_steps = len(test_data)
    forecast, conf_int = arima_modeler.forecast(model, steps=forecast_steps)
    
    # Evaluate model
    metrics = arima_modeler.evaluate_model(model, test_data)
    print(f"Model evaluation metrics: {metrics}")
    
    # Plot forecast vs actual
    arima_modeler.plot_forecast(
        history=train_data, 
        forecast=forecast, 
        conf_int=conf_int,
        title=f"ARIMA Forecast for {dataset.capitalize()} Spike",
        save_path=f"examples/{dataset}_forecast.png"
    )
    
    # Also plot against the actual test data
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data.values, label='Training Data')
    plt.plot(test_data.index, test_data.values, 'g-', label='Actual Test Data')
    plt.plot(test_data.index, forecast.values, 'r--', label='Forecast')
    
    if conf_int is not None:
        plt.fill_between(
            test_data.index,
            conf_int.iloc[:, 0],
            conf_int.iloc[:, 1],
            color='pink', alpha=0.3
        )
    
    plt.title(f"ARIMA Forecast vs Actual {dataset.capitalize()} Data")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"examples/{dataset}_forecast_vs_actual.png")
    plt.show()

if __name__ == "__main__":
    main()