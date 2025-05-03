import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

def load_settings(settings_file="traffic_settings.json"):
    """
    Load traffic generator settings from a JSON file
    
    Args:
        settings_file: Path to the settings JSON file
        
    Returns:
        Dictionary containing the settings
    """
    try:
        with open(settings_file, 'r') as f:
            settings = json.load(f)
        print(f"Successfully loaded settings from {settings_file}")
        return settings
    except FileNotFoundError:
        print(f"Settings file {settings_file} not found. Using default values.")
        return {}
    except json.JSONDecodeError:
        print(f"Error parsing {settings_file}. Using default values.")
        return {}

def generate_traffic_spike_data(base_timestamp=None, duration_minutes=15, interval_seconds=60, 
                               services=None, base_traffic=None, spike_factor=3.5, noise_level=0.05,
                               early_flat_percentage=30, rapid_growth_percentage=55, plateau_percentage=15):
    """
    Generate synthetic HTTP request rate data showing a traffic spike
    
    Args:
        base_timestamp: Starting timestamp (Unix timestamp)
        duration_minutes: Duration of the data in minutes
        interval_seconds: Interval between data points in seconds
        services: List of service names to generate data for
        base_traffic: Dictionary of base traffic rates per service
        spike_factor: How much the traffic increases (multiplier)
        noise_level: Amount of random noise to add
        early_flat_percentage: Percentage of time where traffic is relatively stable at the beginning
        rapid_growth_percentage: Percentage of time with rapid traffic growth
        plateau_percentage: Percentage of time where traffic levels off at the end
        
    Returns:
        Dictionary in Prometheus response format
    """
    if base_timestamp is None:
        # Use current time if no timestamp provided
        base_timestamp = datetime.now().timestamp()
        
    if services is None:
        # Updated to use your service names
        services = ["gw-nginx", "s0", "s1", "s2"]
        
    # Base request rates per service
    if base_traffic is None:
        base_traffic = {
            "gw-nginx": 120.0,  # High traffic gateway
            "s0": 45.0,         # Medium traffic service
            "s1": 87.0,         # Higher traffic service
            "s2": 65.0          # Medium-high traffic service
        }
    
    # Calculate number of data points
    num_points = int((duration_minutes * 60) / interval_seconds)
    timestamps = [base_timestamp + (i * interval_seconds) for i in range(num_points)]
    
    # Normalize the percentages
    total_percentage = early_flat_percentage + rapid_growth_percentage + plateau_percentage
    early_flat = early_flat_percentage / total_percentage
    rapid_growth = rapid_growth_percentage / total_percentage
    
    # Create a sigmoid-like function to model the traffic spike
    def traffic_pattern(x, start_level=1.0, end_level=spike_factor):
        # Convert to range 0-1
        normalized_x = x / (num_points - 1)
        # Use sigmoid function to create an S-curve
        if normalized_x < early_flat:
            # First portion is relatively flat
            return start_level + (end_level - start_level) * 0.1 * normalized_x / early_flat
        elif normalized_x > (early_flat + rapid_growth):
            # Last portion levels off
            normalized_pos = (normalized_x - (early_flat + rapid_growth)) / (1 - (early_flat + rapid_growth))
            return start_level + (end_level - start_level) * (0.9 + 0.1 * normalized_pos)
        else:
            # Middle part is the rapid growth
            normalized_pos = (normalized_x - early_flat) / rapid_growth
            return start_level + (end_level - start_level) * (0.1 + 0.8 * normalized_pos)
    
    # Generate results
    result = []
    for service in services:
        base_rate = base_traffic.get(service, 100.0)
        values = []
        
        for i, ts in enumerate(timestamps):
            # Calculate the multiplier for this point in time
            multiplier = traffic_pattern(i)
            
            # Apply multiplier to base rate and add some noise
            rate = base_rate * multiplier * (1 + noise_level * (np.random.random() - 0.5))
            
            values.append([ts, str(round(rate, 1))])
        
        result.append({
            "metric": {"service": service},
            "values": values
        })
    
    # Construct the full response
    response = {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": result
        }
    }
    
    return response

def generate_cpu_spike_data(base_timestamp=None, duration_minutes=15, interval_seconds=60,
                          pods=None, base_cpu_usage=None, spike_factor=6.5, noise_level=0.05):
    """
    Generate synthetic CPU usage data corresponding to a traffic spike
    
    Args:
        base_timestamp: Starting timestamp (Unix timestamp)
        duration_minutes: Duration of the data in minutes
        interval_seconds: Interval between data points in seconds
        pods: List of pod names to generate data for
        base_cpu_usage: Dictionary of base CPU usage per pod
        spike_factor: How much the CPU usage increases (multiplier)
        noise_level: Amount of random noise to add
        
    Returns:
        Dictionary in Prometheus response format
    """
    if base_timestamp is None:
        # Use current time if no timestamp provided
        base_timestamp = datetime.now().timestamp()
        
    if pods is None:
        # Updated to use your service names for pod names
        pods = ["gw-nginx-54b7ff4756-jm2ql", "s0-67df8d74c6-l8r9c", 
                "s1-84499d876f-bdfhj", "s2-94499d876f-abcde"]
        
    if base_cpu_usage is None:
        base_cpu_usage = {
            "gw-nginx-54b7ff4756-jm2ql": 0.0125,
            "s0-67df8d74c6-l8r9c": 0.0076,
            "s1-84499d876f-bdfhj": 0.0104,
            "s2-94499d876f-abcde": 0.0095
        }
    
    # Calculate number of data points
    num_points = int((duration_minutes * 60) / interval_seconds)
    timestamps = [base_timestamp + (i * interval_seconds) for i in range(num_points)]
    
    # Create a sigmoid-like function with a lag compared to traffic
    def cpu_pattern(x, start_level=1.0, end_level=spike_factor):
        # Add a slight lag to CPU response (1 data point)
        x = max(0, x - 1)
        # Convert to range 0-1
        normalized_x = x / (num_points - 1)
        # Use sigmoid function to create an S-curve with more dramatic rise
        if normalized_x < 0.35:
            # First 35% is relatively flat
            return start_level + (end_level - start_level) * 0.08 * normalized_x / 0.35
        elif normalized_x > 0.8:
            # Last 20% levels off
            normalized_pos = (normalized_x - 0.8) / 0.2
            return start_level + (end_level - start_level) * (0.9 + 0.1 * normalized_pos)
        else:
            # Middle part is the rapid growth
            normalized_pos = (normalized_x - 0.35) / 0.45
            # Steeper curve for CPU as it tends to spike more dramatically
            curve = normalized_pos ** 0.8  # makes the curve steeper in the middle
            return start_level + (end_level - start_level) * (0.08 + 0.82 * curve)
    
    # Generate results
    result = []
    for pod in pods:
        base_usage = base_cpu_usage.get(pod, 0.01)
        values = []
        
        for i, ts in enumerate(timestamps):
            # Calculate the multiplier for this point in time
            multiplier = cpu_pattern(i)
            
            # Apply multiplier to base usage and add some noise
            usage = base_usage * multiplier * (1 + noise_level * (np.random.random() - 0.5))
            
            values.append([ts, str(round(usage, 6))])
        
        result.append({
            "metric": {"pod": pod},
            "values": values
        })
    
    # Construct the full response
    response = {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": result
        }
    }
    
    return response

def generate_memory_spike_data(base_timestamp=None, duration_minutes=15, interval_seconds=60,
                             pods=None, base_memory_usage=None, spike_factor=1.6, noise_level=0.02):
    """
    Generate synthetic memory usage data corresponding to a traffic spike
    
    Args:
        base_timestamp: Starting timestamp (Unix timestamp)
        duration_minutes: Duration of the data in minutes
        interval_seconds: Interval between data points in seconds
        pods: List of pod names to generate data for
        base_memory_usage: Dictionary of base memory usage per pod (in bytes)
        spike_factor: How much the memory increases (multiplier)
        noise_level: Amount of random noise to add
        
    Returns:
        Dictionary in Prometheus response format
    """
    if base_timestamp is None:
        # Use current time if no timestamp provided
        base_timestamp = datetime.now().timestamp()
        
    if pods is None:
        # Updated to use your service names for pod names
        pods = ["gw-nginx-54b7ff4756-jm2ql", "s0-67df8d74c6-l8r9c", 
                "s1-84499d876f-bdfhj", "s2-94499d876f-abcde"]
        
    if base_memory_usage is None:
        # Memory usage in bytes
        base_memory_usage = {
            "gw-nginx-54b7ff4756-jm2ql": 132 * 1024 * 1024,  # ~132MB
            "s0-67df8d74c6-l8r9c": 110 * 1024 * 1024,        # ~110MB
            "s1-84499d876f-bdfhj": 125 * 1024 * 1024,        # ~125MB
            "s2-94499d876f-abcde": 118 * 1024 * 1024         # ~118MB
        }
    
    # Calculate number of data points
    num_points = int((duration_minutes * 60) / interval_seconds)
    timestamps = [base_timestamp + (i * interval_seconds) for i in range(num_points)]
    
    # Memory increases more gradually and doesn't drop quickly
    def memory_pattern(x, start_level=1.0, end_level=spike_factor):
        # Convert to range 0-1
        normalized_x = x / (num_points - 1)
        
        # Memory usage increases more gradually than CPU
        if normalized_x < 0.25:
            # First quarter is relatively stable
            return start_level + (end_level - start_level) * 0.05 * normalized_x / 0.25
        else:
            # Gradually increases without leveling off much
            curve = (normalized_x - 0.25) / 0.75
            # Square root makes the curve rise faster initially then slow down
            return start_level + (end_level - start_level) * (0.05 + 0.95 * np.sqrt(curve))
    
    # Generate results
    result = []
    for pod in pods:
        base_usage = base_memory_usage.get(pod, 100 * 1024 * 1024)  # Default to 100MB
        values = []
        
        for i, ts in enumerate(timestamps):
            # Calculate the multiplier for this point in time
            multiplier = memory_pattern(i)
            
            # Apply multiplier to base usage and add some noise
            usage = int(base_usage * multiplier * (1 + noise_level * (np.random.random() - 0.5)))
            
            values.append([ts, str(usage)])
        
        result.append({
            "metric": {"pod": pod},
            "values": values
        })
    
    # Construct the full response
    response = {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": result
        }
    }
    
    return response

def save_to_json(data, query_str, output_file):
    """
    Save the generated data to a JSON file in Prometheus format
    
    Args:
        data: The generated data
        query_str: The query string to include
        output_file: Path to save the file
    """
    result = {
        "query": query_str,
        "raw_data": data
    }
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

def visualize_generated_data(traffic_data, cpu_data, memory_data, save_dir=None):
    """
    Visualize the generated data to verify it shows a spike pattern
    
    Args:
        traffic_data: Generated traffic data
        cpu_data: Generated CPU usage data
        memory_data: Generated memory data
        save_dir: Directory to save plots (optional)
    """
    # Extract timestamps and values
    service_names = []
    traffic_series = {}
    
    for result in traffic_data["data"]["result"]:
        service = result["metric"]["service"]
        service_names.append(service)
        timestamps = [point[0] for point in result["values"]]
        values = [float(point[1]) for point in result["values"]]
        traffic_series[service] = (timestamps, values)
    
    pod_names = []
    cpu_series = {}
    
    for result in cpu_data["data"]["result"]:
        pod = result["metric"]["pod"]
        pod_names.append(pod)
        timestamps = [point[0] for point in result["values"]]
        values = [float(point[1]) for point in result["values"]]
        cpu_series[pod] = (timestamps, values)
    
    memory_series = {}
    
    for result in memory_data["data"]["result"]:
        pod = result["metric"]["pod"]
        timestamps = [point[0] for point in result["values"]]
        # Convert bytes to MB
        values = [float(point[1]) / (1024 * 1024) for point in result["values"]]
        memory_series[pod] = (timestamps, values)
    
    # Convert timestamps to datetime for better x-axis labeling
    base_time = datetime.fromtimestamp(timestamps[0])
    time_labels = [(base_time + timedelta(seconds=t - timestamps[0])).strftime('%H:%M:%S') 
                  for t in timestamps]
    
    # Create the figure with three subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot traffic data
    for service, (ts, values) in traffic_series.items():
        axs[0].plot(range(len(ts)), values, label=service)
    axs[0].set_title('HTTP Request Rate')
    axs[0].set_ylabel('Requests/second')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot CPU data
    for pod, (ts, values) in cpu_series.items():
        # Convert to percentage
        axs[1].plot(range(len(ts)), [v * 100 for v in values], label=pod.split('-')[0])
    axs[1].set_title('CPU Usage')
    axs[1].set_ylabel('CPU Usage (%)')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot memory data
    for pod, (ts, values) in memory_series.items():
        axs[2].plot(range(len(ts)), values, label=pod.split('-')[0])
    axs[2].set_title('Memory Usage')
    axs[2].set_ylabel('Memory Usage (MB)')
    axs[2].set_xlabel('Time')
    axs[2].legend()
    axs[2].grid(True)
    
    # Set x-tick labels
    num_ticks = min(10, len(time_labels))
    tick_indices = np.linspace(0, len(time_labels)-1, num_ticks, dtype=int)
    axs[2].set_xticks(tick_indices)
    axs[2].set_xticklabels([time_labels[i] for i in tick_indices], rotation=45)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "traffic_spike_visualization.png"))
    
    plt.show()

def main():
    # Load settings from JSON file
    settings = load_settings("traffic_settings.json")
    
    # Create base timestamp - use current time
    base_timestamp = datetime.now().timestamp()
    
    # Extract settings from the nested structure
    services = settings.get("services", ["gw-nginx", "s0", "s1", "s2"])
    pods = settings.get("pods", ["gw-nginx-54b7ff4756-jm2ql", "s0-67df8d74c6-l8r9c", 
                                "s1-84499d876f-bdfhj", "s2-94499d876f-abcde"])
    
    # Get base traffic, CPU and memory values
    base_traffic = settings.get("base_traffic", None)
    base_cpu_usage = settings.get("base_cpu_usage", None)
    
    # Convert memory values from MB to bytes if they exist
    base_memory_usage = None
    if "base_memory_usage" in settings:
        base_memory_usage = {pod: value * 1024 * 1024 for pod, value in settings["base_memory_usage"].items()}
    
    # Get spike settings
    spike_settings = settings.get("spike_settings", {})
    traffic_spike = spike_settings.get("traffic", {})
    cpu_spike = spike_settings.get("cpu", {})
    memory_spike = spike_settings.get("memory", {})
    
    # Output paths
    output_paths = settings.get("output_paths", {})
    
    # Generate data for traffic spike
    traffic_data = generate_traffic_spike_data(
        base_timestamp=base_timestamp,
        duration_minutes=traffic_spike.get("duration_minutes", 15),
        interval_seconds=traffic_spike.get("interval_seconds", 60),
        services=services,
        base_traffic=base_traffic,
        spike_factor=traffic_spike.get("spike_factor", 4.0),
        noise_level=traffic_spike.get("noise_level", 0.05),
        early_flat_percentage=traffic_spike.get("early_flat_percentage", 30),
        rapid_growth_percentage=traffic_spike.get("rapid_growth_percentage", 55),
        plateau_percentage=traffic_spike.get("plateau_percentage", 15)
    )
    
    cpu_data = generate_cpu_spike_data(
        base_timestamp=base_timestamp,
        duration_minutes=traffic_spike.get("duration_minutes", 15),
        interval_seconds=traffic_spike.get("interval_seconds", 60),
        pods=pods,
        base_cpu_usage=base_cpu_usage,
        spike_factor=cpu_spike.get("spike_factor", 6.5),
        noise_level=cpu_spike.get("noise_level", 0.05)
    )
    
    memory_data = generate_memory_spike_data(
        base_timestamp=base_timestamp,
        duration_minutes=traffic_spike.get("duration_minutes", 15),
        interval_seconds=traffic_spike.get("interval_seconds", 60),
        pods=pods,
        base_memory_usage=base_memory_usage,
        spike_factor=memory_spike.get("spike_factor", 1.6),
        noise_level=memory_spike.get("noise_level", 0.02)
    )
    
    # Ensure examples/prometheus directory exists
    output_dir = Path(os.path.dirname(output_paths.get("traffic", "examples/prometheus/traffic-spike.json")))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON files
    save_to_json(
        traffic_data, 
        "sum(rate(http_requests_total{namespace=\"default\", service=~\".*\"}[1m])) by (service)",
        output_paths.get("traffic", "examples/prometheus/traffic-spike.json")
    )
    
    save_to_json(
        cpu_data,
        "sum(rate(container_cpu_usage_seconds_total{namespace=\"default\", pod=~\"gw-nginx.*|s0.*|s1.*|s2.*\"}[5m])) by (pod)",
        output_paths.get("cpu", "examples/prometheus/cpu-spike.json")
    )
    
    save_to_json(
        memory_data,
        "sum(container_memory_usage_bytes{namespace=\"default\", pod=~\"gw-nginx.*|s0.*|s1.*|s2.*\"}) by (pod)",
        output_paths.get("memory", "examples/prometheus/memory-spike.json")
    )
    
    # Visualize the data
    save_path = os.path.dirname(output_paths.get("visualization", "examples/prometheus/traffic_spike_visualization.png"))
    visualize_generated_data(traffic_data, cpu_data, memory_data, save_path)
    
    print("Generated traffic spike data complete!")

if __name__ == "__main__":
    main()