import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

import requests

today = datetime.now()

start_time = today.replace(hour=20, minute=24, second=0, microsecond=0)
end_time = today.replace(hour=21, minute=18, second=0, microsecond=0)

start_timestamp = int(start_time.timestamp())
end_timestamp = int(end_time.timestamp())

payload = {
    "query": 'sum(rate(container_cpu_usage_seconds_total{namespace="default", pod=~"s[0-2].*|gw-nginx.*"}[1m])) by (pod)',
    "start_time": start_timestamp,
    "end_time": end_timestamp,
    "step": "15s"
}

response = requests.post("http://localhost:5000/query-prometheus", json=payload)
data = response.json()

metrics_data = json.loads(data['data'])

time_series = metrics_data['total']

# Convert to DataFrame
df = pd.DataFrame(list(time_series.items()), columns=['timestamp', 'value'])

# Convert timestamp from milliseconds to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Set timestamp as index
df = df.set_index('timestamp')

# Save to CSV if needed
df.to_csv('training_data_v2.csv')

print(df.head())

# Plot the dataframe
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['value'], linewidth=1.5, color='blue')
plt.title('Container CPU Usage Over Time')
plt.xlabel('Timestamp')
plt.ylabel('CPU Usage Rate')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()

# Optionally save the plot
plt.savefig('cpu_usage_plot.png', dpi=300, bbox_inches='tight')