import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

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

def count_spike_events(df, threshold=210):
    """Count spike events using grouping algorithm and calculate durations."""
    peak_indices = np.where(df['value'] > threshold)[0]
    peak_groups = group_consecutive_peaks(peak_indices, max_gap=2)
    
    # Calculate duration for each group
    spike_info = []
    for group in peak_groups:
        start_idx = group[0]
        end_idx = group[-1]
        max_idx = group[np.argmax(df['value'].iloc[group])]
        
        start_time = df['timestamp'].iloc[start_idx]
        end_time = df['timestamp'].iloc[end_idx]
        peak_time = df['timestamp'].iloc[max_idx]
        peak_value = df['value'].iloc[max_idx]
        
        # Calculate duration in seconds
        duration = (end_time - start_time).total_seconds()
        # Add one interval (15 seconds) since it's inclusive
        duration += 15
        
        spike_info.append({
            'start_time': start_time,
            'end_time': end_time,
            'peak_time': peak_time,
            'peak_value': peak_value,
            'duration_seconds': duration,
            'duration_minutes': duration / 60,
            'indices': group
        })
    
    return len(peak_groups), peak_groups, spike_info

def analyze_spike_data(file1, file2, threshold=210):
    """
    Analyze and visualize spike data comparing with and without KubeScale
    Using grouped spike events with duration analysis
    """
    # Read the CSV files
    df_with = pd.read_csv(file1)
    df_without = pd.read_csv(file2)
    
    # Convert timestamp to datetime
    df_with['timestamp'] = pd.to_datetime(df_with['timestamp'])
    df_without['timestamp'] = pd.to_datetime(df_without['timestamp'])
    
    # Match time periods - use the shorter dataset as reference
    min_length = min(len(df_with), len(df_without))
    df_with_matched = df_with.head(min_length).copy()
    df_without_matched = df_without.head(min_length).copy()
    
    print(f"=== TIME PERIOD MATCHING ===")
    print(f"Original lengths - With KubeScale: {len(df_with)}, Without KubeScale: {len(df_without)}")
    print(f"Matched length: {min_length} data points")
    print(f"Time range: {df_with_matched['timestamp'].min()} to {df_with_matched['timestamp'].max()}")
    
    # Count spike events using grouping algorithm with duration info
    spike_events_with, spike_groups_with, spike_info_with = count_spike_events(df_with_matched, threshold)
    spike_events_without, spike_groups_without, spike_info_without = count_spike_events(df_without_matched, threshold)
    
    # Calculate total spike duration
    total_duration_with = sum([info['duration_seconds'] for info in spike_info_with])
    total_duration_without = sum([info['duration_seconds'] for info in spike_info_without])
    
    # Also get individual points above threshold for comparison
    individual_spikes_with = len(df_with_matched[df_with_matched['value'] > threshold])
    individual_spikes_without = len(df_without_matched[df_without_matched['value'] > threshold])
    
    print(f"\n=== SPIKE ANALYSIS RESULTS (Matched Time Periods) ===")
    print(f"Threshold: {threshold}")
    print(f"\nWith KubeScale:")
    print(f"  - Data points: {len(df_with_matched)}")
    print(f"  - Spike EVENTS (grouped): {spike_events_with}")
    print(f"  - Total spike duration: {total_duration_with:.0f} seconds ({total_duration_with/60:.1f} minutes)")
    print(f"  - Average spike duration: {total_duration_with/spike_events_with:.1f} seconds" if spike_events_with > 0 else "  - Average spike duration: 0 seconds")
    print(f"  - Individual points above threshold: {individual_spikes_with}")
    print(f"  - Points above threshold %: {(individual_spikes_with/len(df_with_matched)*100):.2f}%")
    print(f"  - Max value: {df_with_matched['value'].max():.2f}")
    
    print(f"\nWithout KubeScale:")
    print(f"  - Data points: {len(df_without_matched)}")
    print(f"  - Spike EVENTS (grouped): {spike_events_without}")
    print(f"  - Total spike duration: {total_duration_without:.0f} seconds ({total_duration_without/60:.1f} minutes)")
    print(f"  - Average spike duration: {total_duration_without/spike_events_without:.1f} seconds" if spike_events_without > 0 else "  - Average spike duration: 0 seconds")
    print(f"  - Individual points above threshold: {individual_spikes_without}")
    print(f"  - Points above threshold %: {(individual_spikes_without/len(df_without_matched)*100):.2f}%")
    print(f"  - Max value: {df_without_matched['value'].max():.2f}")
    
    # Calculate improvements
    spike_event_reduction = ((spike_events_without - spike_events_with) / spike_events_without * 100) if spike_events_without > 0 else 0
    duration_reduction = ((total_duration_without - total_duration_with) / total_duration_without * 100) if total_duration_without > 0 else 0
    threshold_breach_reduction = ((individual_spikes_without - individual_spikes_with) / individual_spikes_without * 100) if individual_spikes_without > 0 else 0
    peak_reduction = ((df_without_matched['value'].max() - df_with_matched['value'].max()) / df_without_matched['value'].max() * 100)
    
    print(f"\n=== IMPROVEMENT METRICS ===")
    print(f"Spike event reduction: {spike_event_reduction:.1f}%")
    print(f"Total spike duration reduction: {duration_reduction:.1f}%")
    print(f"Threshold breach reduction: {threshold_breach_reduction:.1f}%")
    print(f"Peak value reduction: {peak_reduction:.1f}%")
    
    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Time series with KubeScale - mark spike events with duration labels
    ax1.plot(df_with_matched['timestamp'], df_with_matched['value'], 'b-', linewidth=1, label='CPU Usage')
    ax1.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    
    # Mark spike events with duration labels
    for i, (group, info) in enumerate(zip(spike_groups_with, spike_info_with)):
        spike_times = df_with_matched['timestamp'].iloc[group]
        spike_values = df_with_matched['value'].iloc[group]
        ax1.fill_between(spike_times, threshold, spike_values, 
                        color='red', alpha=0.3)
        # Mark peak with duration label
        ax1.scatter(info['peak_time'], info['peak_value'], 
                   color='darkred', marker='^', s=80, zorder=5)
        ax1.annotate(f"{info['duration_seconds']:.0f}s", 
                    (info['peak_time'], info['peak_value']), 
                    xytext=(5, 10), textcoords='offset points',
                    fontsize=8, ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax1.set_title(f'With KubeScale System\n({spike_events_with} spike events, {total_duration_with:.0f}s total duration)')
    ax1.set_ylabel('CPU Usage (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time series without KubeScale - mark spike events with duration labels
    ax2.plot(df_without_matched['timestamp'], df_without_matched['value'], 'r-', linewidth=1, label='CPU Usage')
    ax2.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    
    # Mark spike events with duration labels
    for i, (group, info) in enumerate(zip(spike_groups_without, spike_info_without)):
        spike_times = df_without_matched['timestamp'].iloc[group]
        spike_values = df_without_matched['value'].iloc[group]
        ax2.fill_between(spike_times, threshold, spike_values, 
                        color='red', alpha=0.3)
        # Mark peak with duration label
        ax2.scatter(info['peak_time'], info['peak_value'], 
                   color='darkred', marker='^', s=80, zorder=5)
        ax2.annotate(f"{info['duration_seconds']:.0f}s", 
                    (info['peak_time'], info['peak_value']), 
                    xytext=(5, 10), textcoords='offset points',
                    fontsize=8, ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax2.set_title(f'Without KubeScale System\n({spike_events_without} spike events, {total_duration_without:.0f}s total duration)')
    ax2.set_ylabel('CPU Usage (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Comparison bar chart including duration metrics
    categories = ['Spike Events', 'Total Duration (min)', 'Max Value', 'Threshold Breaches']
    with_values = [spike_events_with, total_duration_with/60, df_with_matched['value'].max(), individual_spikes_with]
    without_values = [spike_events_without, total_duration_without/60, df_without_matched['value'].max(), individual_spikes_without]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax3.bar(x - width/2, with_values, width, label='With KubeScale', color='green', alpha=0.7)
    ax3.bar(x + width/2, without_values, width, label='Without KubeScale', color='red', alpha=0.7)
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Values')
    ax3.set_title('System Performance Comparison (Matched Time Periods)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (with_val, without_val) in enumerate(zip(with_values, without_values)):
        ax3.text(i - width/2, with_val + max(max(with_values), max(without_values))*0.01, f'{with_val:.1f}', 
                ha='center', va='bottom', fontsize=8)
        ax3.text(i + width/2, without_val + max(max(with_values), max(without_values))*0.01, f'{without_val:.1f}', 
                ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Histogram comparison (CPU Usage Distribution)
    bins = np.linspace(0, max(df_with_matched['value'].max(), df_without_matched['value'].max()), 30)
    ax4.hist(df_with_matched['value'], bins=bins, alpha=0.7, label='With KubeScale', color='green', density=True)
    ax4.hist(df_without_matched['value'], bins=bins, alpha=0.7, label='Without KubeScale', color='red', density=True)
    ax4.axvline(x=threshold, color='black', linestyle='--', label=f'Threshold ({threshold})')
    ax4.set_xlabel('CPU Usage (%)')
    ax4.set_ylabel('Density')
    ax4.set_title('CPU Usage Distribution (Matched Periods)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spike_analysis_comparison.png', dpi=300)
    plt.show()
    
    # Detailed spike event analysis with duration
    print(f"\n=== DETAILED SPIKE EVENT ANALYSIS ===")
    if spike_info_with:
        print(f"With KubeScale - Spike Events Details:")
        for i, info in enumerate(spike_info_with):
            print(f"  Event {i+1}: Peak {info['peak_value']:.2f}% at {info['peak_time']}")
            print(f"    Duration: {info['duration_seconds']:.0f} seconds ({info['duration_minutes']:.1f} minutes)")
            print(f"    Time span: {info['start_time']} to {info['end_time']}")
    
    if spike_info_without:
        print(f"\nWithout KubeScale - Spike Events Details:")
        for i, info in enumerate(spike_info_without):
            print(f"  Event {i+1}: Peak {info['peak_value']:.2f}% at {info['peak_time']}")
            print(f"    Duration: {info['duration_seconds']:.0f} seconds ({info['duration_minutes']:.1f} minutes)")
            print(f"    Time span: {info['start_time']} to {info['end_time']}")
    
    # Statistics table with duration metrics
    avg_duration_with = total_duration_with / spike_events_with if spike_events_with > 0 else 0
    avg_duration_without = total_duration_without / spike_events_without if spike_events_without > 0 else 0
    
    stats_df = pd.DataFrame({
        'Metric': ['Data Points', 'Spike Events', 'Total Duration (min)', 'Avg Duration (sec)', 
                   'Points Above Threshold', 'Threshold Breach %', 'Max Value (%)', 'Mean Value (%)', 'Std Deviation'],
        'With KubeScale': [
            len(df_with_matched), spike_events_with, f"{total_duration_with/60:.1f}",
            f"{avg_duration_with:.1f}", individual_spikes_with, 
            f"{individual_spikes_with/len(df_with_matched)*100:.2f}",
            f"{df_with_matched['value'].max():.2f}", f"{df_with_matched['value'].mean():.2f}", 
            f"{df_with_matched['value'].std():.2f}"
        ],
        'Without KubeScale': [
            len(df_without_matched), spike_events_without, f"{total_duration_without/60:.1f}",
            f"{avg_duration_without:.1f}", individual_spikes_without,
            f"{individual_spikes_without/len(df_without_matched)*100:.2f}",
            f"{df_without_matched['value'].max():.2f}", f"{df_without_matched['value'].mean():.2f}", 
            f"{df_without_matched['value'].std():.2f}"
        ]
    })
    
    print(f"\n=== DETAILED STATISTICS ===")
    print(stats_df.to_string(index=False))
    
    return {
        'spike_event_reduction_percent': spike_event_reduction,
        'duration_reduction_percent': duration_reduction,
        'threshold_breach_reduction_percent': threshold_breach_reduction,
        'peak_reduction_percent': peak_reduction,
        'spike_events_with': spike_events_with,
        'spike_events_without': spike_events_without,
        'total_duration_with': total_duration_with,
        'total_duration_without': total_duration_without,
        'avg_duration_with': avg_duration_with,
        'avg_duration_without': avg_duration_without,
        'individual_spikes_with': individual_spikes_with,
        'individual_spikes_without': individual_spikes_without,
        'max_with': df_with_matched['value'].max(),
        'max_without': df_without_matched['value'].max(),
    }

# Usage
if __name__ == "__main__":
    # Update these paths to your actual file locations
    file_with_kubescale = "training_data_default_cyclical_4.csv"
    file_without_kubescale = "training_data_default_without.csv"
    
    results = analyze_spike_data(file_with_kubescale, file_without_kubescale, threshold=210)
    
    print(f"\n=== SUMMARY FOR THESIS ===")
    print(f"KubeScale system achieved (matched time periods):")
    print(f"• {results['spike_event_reduction_percent']:.1f}% reduction in spike events")
    print(f"• {results['duration_reduction_percent']:.1f}% reduction in total spike duration")
    print(f"• {results['threshold_breach_reduction_percent']:.1f}% reduction in threshold breaches")
    print(f"• {results['peak_reduction_percent']:.1f}% reduction in peak CPU usage")
    print(f"• Spike events: {results['spike_events_without']} → {results['spike_events_with']}")
    print(f"• Total spike duration: {results['total_duration_without']:.0f}s → {results['total_duration_with']:.0f}s")
    print(f"• Average spike duration: {results['avg_duration_without']:.1f}s → {results['avg_duration_with']:.1f}s")
    print(f"• Threshold breaches: {results['individual_spikes_without']} → {results['individual_spikes_with']}")
    print(f"• Improved system stability and resource utilization")