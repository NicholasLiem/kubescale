import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

import requests

today = datetime.now()

start_time = today.replace(hour=9, minute=55, second=0, microsecond=0)
end_time = today.replace(hour=10, minute=20, second=0, microsecond=0)

start_timestamp = int(start_time.timestamp())
end_timestamp = int(end_time.timestamp())

# Define queries for both namespaces
queries = {
    "default": 'sum(rate(container_cpu_usage_seconds_total{namespace="default", pod=~"s[0-2].*|gw-nginx.*"}[1m])) by (pod)',
    "warm_pool": 'sum(rate(container_cpu_usage_seconds_total{namespace="warm-pool", pod=~"s[0-2].*|gw-nginx.*"}[1m])) by (pod)'
}

# Fetch data for both namespaces
for namespace, query in queries.items():
    payload = {
        "query": query,
        "start_time": start_timestamp,
        "end_time": end_timestamp,
        "step": "15s"
    }

    response = requests.post("http://localhost:5000/query-prometheus", json=payload)
    data = response.json()
    
    print(f"Raw response data: {data}")
    print(f"Data type: {type(data['data'])}")
    
    metrics_data = json.loads(data['data'])
    print(f"Parsed metrics_data keys: {metrics_data.keys()}")

    time_series = metrics_data['total']

    # Convert to DataFrame
    df = pd.DataFrame(list(time_series.items()), columns=['timestamp', 'value'])

    # Convert timestamp from milliseconds to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Save to CSV with namespace suffix
    csv_filename = f'training_data_{namespace}.csv'
    df.to_csv(csv_filename, index=False)
    
    print(f"Saved {namespace} data to {csv_filename}")
    print(f"{namespace} data preview:")
    print(df.head())
    print(f"Data points: {len(df)}")
    print("-" * 50)