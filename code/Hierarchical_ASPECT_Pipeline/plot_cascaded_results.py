
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11

def load_event_counts(base_dir):
    three_class_path = os.path.join(base_dir, "result_three_class", "event_counts.csv")
    binary_class_path = os.path.join(base_dir, "result_binary_class", "event_counts.csv")
    
    three_class = pd.read_csv(three_class_path)
    binary_class = pd.read_csv(binary_class_path)
    
    return three_class, binary_class

def plot_event_counts_side_by_side(three_class, binary_class, output_dir):
    events = sorted(three_class['event'].unique())
    n_events = len(events)
    
   
    metrics_data = []
    for event in events:
        three_row = three_class[three_class['event'] == event].iloc[0]
        binary_row = binary_class[binary_class['event'] == event].iloc[0]
        
        metrics_data.append({
            'event': event,
            'actual_count': three_row['actual_count'],
            'predicted_3c': three_row['predicted_count'],
            'predicted_bin': binary_row['predicted_count'],
            'tp_3c': three_row['true_predicted_count'],
            'tp_bin': binary_row['true_predicted_count'],
        })
    

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    x = np.arange(n_events)
    width = 0.15  
    
    colors = {
        'actual_count': '#e74c3c',           # Red
        'predicted_3c': '#f39c12',           # Orange
        'tp_3c': '#d68910',                  # Dark Orange/Brown
        'predicted_bin': '#9b59b6',          # Purple
        'tp_bin': '#7d3c98',                 # Dark Purple
    }
    
    actual_vals = [d['actual_count'] for d in metrics_data]
    bars_actual = ax.bar(x - width*2, actual_vals, width, label='Actual Count', 
                        color=colors['actual_count'], alpha=0.9, edgecolor='black', linewidth=1.5)
    
    pred_3c_vals = [d['predicted_3c'] for d in metrics_data]
    bars_pred_3c = ax.bar(x - width*1, pred_3c_vals, width, label='Predicted (3-Class)', 
                         color=colors['predicted_3c'], alpha=0.85, edgecolor='black', linewidth=1.5)
    
    tp_3c_vals = [d['tp_3c'] for d in metrics_data]
    bars_tp_3c = ax.bar(x, tp_3c_vals, width, label='True Positive (3-Class)', 
                       color=colors['tp_3c'], alpha=0.85, edgecolor='black', linewidth=1.5)
    
    pred_bin_vals = [d['predicted_bin'] for d in metrics_data]
    bars_pred_bin = ax.bar(x + width*1, pred_bin_vals, width, label='Predicted (Binary)', 
                         color=colors['predicted_bin'], alpha=0.85, edgecolor='black', linewidth=1.5)
    
    tp_bin_vals = [d['tp_bin'] for d in metrics_data]
    bars_tp_bin = ax.bar(x + width*2, tp_bin_vals, width, label='True Positive (Binary)', 
                       color=colors['tp_bin'], alpha=0.85, edgecolor='black', linewidth=1.5)
    

    max_val = max(max(actual_vals), max(pred_3c_vals), max(pred_bin_vals),
                 max(tp_3c_vals), max(tp_bin_vals))
    
    all_bars = [bars_actual, bars_pred_3c, bars_tp_3c, bars_pred_bin, bars_tp_bin]
    all_vals = [actual_vals, pred_3c_vals, tp_3c_vals, pred_bin_vals, tp_bin_vals]
    all_positions = [x - width*2, x - width*1, x, x + width*1, x + width*2]
    
    for bars, vals, pos in zip(all_bars, all_vals, all_positions):
        for i, (bar, val) in enumerate(zip(bars, vals)):
            height = bar.get_height()
            ax.text(pos[i], height + max_val * 0.02,
                   f'{int(val)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for i, event_data in enumerate(metrics_data):
        act = event_data['actual_count']
        
       
        acc_3c = (event_data['tp_3c'] / act * 100) if act > 0 else 0
        acc_bin = (event_data['tp_bin'] / act * 100) if act > 0 else 0
        
       
        ax.text(i, event_data['tp_3c'] + max_val * 0.08, f'{acc_3c:.1f}%',
               ha='center', va='bottom', fontsize=8, fontweight='bold', 
               color=colors['tp_3c'], style='italic',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=colors['tp_3c']))
        ax.text(i + width*2, event_data['tp_bin'] + max_val * 0.08, f'{acc_bin:.1f}%',
               ha='center', va='bottom', fontsize=8, fontweight='bold', 
               color=colors['tp_bin'], style='italic',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=colors['tp_bin']))
    
   
    ax.set_xlabel('Event', fontweight='bold', fontsize=14)
    ax.set_ylabel('Count', fontweight='bold', fontsize=14)
    ax.set_title('Event-wise Comparison of Hierarchical ASPECT Pipeline Metrics', 
                fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(events, fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9, ncol=2)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, max_val * 1.15])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'event_counts_side_by_side.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Side-by-side event counts comparison plot saved to: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot three-class pipeline results')
    parser.add_argument('--input-dir', type=str, default='./test_result_test',
                        help='Base directory containing result_three_class and result_binary_class')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: same as input-dir)')
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    print(f"Loading event counts from: {args.input_dir}")
    three_class, binary_class = load_event_counts(args.input_dir)
    
    
    plot_event_counts_side_by_side(three_class, binary_class, args.output_dir)
    
    print(f"\nPlot saved to: {os.path.join(args.output_dir, 'event_counts_side_by_side.png')}")

if __name__ == "__main__":
    main()
