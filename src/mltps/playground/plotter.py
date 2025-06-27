import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def group_consecutive_peaks(peak_indices, max_gap=2):
    """Group consecutive peak indices together."""
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

# Load both patterns
df_pattern1 = pd.read_csv('training_data_warm_pool.csv', parse_dates=['timestamp'])
df_pattern2 = pd.read_csv('training_data_default.csv', parse_dates=['timestamp'])

plt.figure(figsize=(12, 8))

# Plot Pattern 1 (Warm Pool) - thinner lines
plt.plot(df_pattern1['timestamp'], df_pattern1['value'],
         'b-', label='Warm Pool (CPU Usage %)', linewidth=0.8, alpha=0.8)

# Plot Pattern 2 (Main Pool) - thinner lines
plt.plot(df_pattern2['timestamp'], df_pattern2['value'],
         'r-', label='Main Pool (CPU Usage %)', linewidth=0.8, alpha=0.8)

# Draw median line for Main Pool
p2_median = df_pattern2['value'].median()
small_spike_threshold = p2_median * 1.1
big_spike_threshold = p2_median * 1.8


estimated_pod_count = 3  # You can adjust this based on your typical pod count
hpa_threshold_per_pod = 70.0  # 70% from HPA config
hpa_threshold_aggregated = hpa_threshold_per_pod * estimated_pod_count

plt.axhline(y=p2_median, color='green', linestyle='-', linewidth=1,
            alpha=0.7, label=f'Main Pool Baseline: {p2_median:.2f}%')
plt.axhline(y=small_spike_threshold, color='orange', linestyle='-', linewidth=1,
            alpha=0.8, label=f'Small Spike Threshold: {small_spike_threshold:.2f}%')
plt.axhline(y=big_spike_threshold, color='purple', linestyle='-', linewidth=1,
            alpha=0.8, label=f'Big Spike Threshold: {big_spike_threshold:.2f}%')
plt.axhline(y=hpa_threshold_aggregated, color='red', linestyle='--', linewidth=1,
            alpha=0.9, label=f'HPA Trigger Threshold: {hpa_threshold_aggregated:.0f}% (70% × {estimated_pod_count} pods)')

# Simple threshold for peaks based on mean + std
threshold = df_pattern2['value'].mean() + df_pattern2['value'].std()
peak_indices = np.where(df_pattern2['value'] > threshold)[0]
peak_groups = group_consecutive_peaks(peak_indices, max_gap=2)

# Calculate averages and trends for peak groups
group_averages = []
group_times = []

# Annotate each peak group in Main Pool
for idx, group in enumerate(peak_groups):
    max_idx = group[np.argmax(df_pattern2['value'].iloc[group])]
    peak_time = df_pattern2['timestamp'].iloc[max_idx]
    peak_value = df_pattern2['value'].iloc[max_idx]
    
    # Calculate average for this group
    group_avg = df_pattern2['value'].iloc[group].mean()
    group_averages.append(group_avg)
    group_times.append(peak_time)
    
    plt.scatter(peak_time, peak_value, color='darkred', marker='^', s=120,
                edgecolors='black', linewidth=1.5)
    plt.annotate(f"Spike\nVal: {peak_value:.2f}%",
                 xy=(peak_time, peak_value), xytext=(45, 0), textcoords='offset points',
                 ha='left', fontsize=9, color='darkred',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9,
                           edgecolor='darkred'),
                 arrowprops=dict(arrowstyle='->', color='darkred', lw=1))

# Plot trend line for group averages
if len(group_averages) > 1:
    plt.plot(group_times, group_averages, 'go-', linewidth=1, markersize=8, 
             label='Spike Trend', alpha=0.8)

# Calculate and print differences between consecutive groups
print("Peak Group Analysis:")
for i in range(len(group_averages)):
    print(f"Group {i+1}: Average = {group_averages[i]:.2f}")
    if i > 0:
        diff = group_averages[i] - group_averages[i-1]
        print(f"  Difference from Group {i}: {diff:+.2f}")

# # Define phases and shade them, plus vertical lines
# phase_intervals = [
#     (pd.to_datetime("2025-06-27 04:30:00"), pd.to_datetime("2025-06-27 04:36:00")),
#     (pd.to_datetime("2025-06-27 04:36:00"), pd.to_datetime("2025-06-27 04:39:00")),
#     (pd.to_datetime("2025-06-27 04:39:00"), pd.to_datetime("2025-06-27 04:42:00")),
# ]

# phase_colors = ['lightblue', 'lightgreen', 'lightcoral']
# phase_labels = ['Forecasting Phase', 'Spike Handling Phase', 'Recovery Phase']
# for idx, (start_time, end_time) in enumerate(phase_intervals):
#     plt.axvspan(start_time, end_time, color=phase_colors[idx % len(phase_colors)], alpha=0.2,
#                 label=phase_labels[idx])
#     plt.axvline(start_time, color='gray', linestyle='--', alpha=0.7, linewidth=0.8)
#     plt.axvline(end_time, color='gray', linestyle='--', alpha=0.7, linewidth=0.8)

# Add grid background
plt.grid(True, alpha=0.3, linewidth=0.5)

# Format x-axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(rotation=45)

# Place legend on the right side
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Label the axes
plt.xlabel('Timestamp')
plt.ylabel('CPU Usage Rate (%)')

plt.title("CPU Usage: Warm Pool vs Main Pool Load Distribution")
plt.tight_layout()
plt.savefig('cpu_usage_comparison.png', dpi=300, bbox_inches='tight')
plt.show()