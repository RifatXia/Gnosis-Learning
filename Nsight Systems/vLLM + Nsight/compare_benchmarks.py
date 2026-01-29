import json
import glob
import sys
import os
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

def load_benchmarks(pattern="benchmark_*.json"):
    """load all benchmark json files"""
    files = glob.glob(pattern)
    benchmarks = []
    
    for file in files:
        # skip benchmark_schema.json (template file)
        if 'benchmark_schema.json' in file:
            continue
        with open(file, 'r') as f:
            data = json.load(f)
            # skip if all metrics are zero (template data)
            if data['inference_metrics'].get('total_inference_time_seconds', 0) == 0:
                continue
            benchmarks.append(data)
    
    return benchmarks

def compare_benchmarks(benchmarks):
    """create comparison table from benchmarks"""
    if not benchmarks:
        print("no benchmark files found")
        return
    
    # prepare comparison data
    comparison = []
    
    for b in benchmarks:
        # format model name with parameter count
        model_name = b["model_name"]
        params = b["model_properties"].get("total_parameters", "N/A")
        if params != "N/A" and isinstance(params, (int, float)):
            # format parameter count (e.g., 774030080 -> 774m)
            if params >= 1e9:
                param_str = f"{params/1e9:.1f}b"
            elif params >= 1e6:
                param_str = f"{int(params/1e6)}m"
            else:
                param_str = f"{int(params/1e3)}k"
            # add parameter count to model name if not already there
            if param_str not in model_name.lower():
                model_name = f"{model_name}-{param_str}"
        
        row = {
            "Model": model_name,
            "Params": params,
            "Layers": b["model_properties"].get("num_layers", "N/A"),
            "Model Size (MB)": f"{b['model_properties'].get('model_size_mb', 0):.2f}",
            "Load Time (s)": f"{b['inference_metrics'].get('model_load_time_seconds', 0):.4f}",
            "Inference Time (s)": f"{b['inference_metrics'].get('total_inference_time_seconds', 0):.4f}",
            "Tokens/s": f"{b['inference_metrics'].get('tokens_per_second', 0):.2f}",
            "GPU Mem (MB)": f"{b['gpu_metrics'].get('gpu_memory_allocated_mb', 0):.2f}",
            "GPU Util (%)": f"{b['gpu_metrics'].get('gpu_memory_utilization_percent', 0):.2f}"
        }
        comparison.append(row)
    
    # sort by model name
    comparison.sort(key=lambda x: x["Model"])
    
    # print comparison table
    print("\n" + "="*120)
    print("MODEL COMPARISON")
    print("="*120)
    print(tabulate(comparison, headers="keys", tablefmt="grid"))
    print("="*120)
    
    return comparison

def generate_comparison_png(comparison):
    """generate png visualization of comparison table"""
    if not comparison:
        return
    
    # create figure with table
    fig, ax = plt.subplots(figsize=(16, len(comparison) * 0.8 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # prepare table data
    headers = list(comparison[0].keys())
    rows = [[row[h] for h in headers] for row in comparison]
    
    # create table
    table = ax.table(cellText=rows, colLabels=headers, cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Model Benchmark Comparison', fontsize=16, weight='bold', pad=20)
    
    # save to plots directory
    os.makedirs('plots', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'plots/comparison_table_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nComparison table saved to: {filename}")

def generate_performance_graphs(benchmarks):
    """generate performance comparison graphs"""
    if not benchmarks:
        return
    
    # extract data
    models = [b['model_name'] for b in benchmarks]
    tokens_per_sec = [b['inference_metrics'].get('tokens_per_second', 0) for b in benchmarks]
    inference_time = [b['inference_metrics'].get('total_inference_time_seconds', 0) for b in benchmarks]
    load_time = [b['inference_metrics'].get('model_load_time_seconds', 0) for b in benchmarks]
    gpu_mem_util = [b['gpu_metrics'].get('gpu_memory_utilization_percent', 0) for b in benchmarks]
    gpu_mem_mb = [b['gpu_metrics'].get('gpu_memory_allocated_mb', 0) for b in benchmarks]
    
    # create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=18, weight='bold', y=0.995)
    
    # 1. tokens per second (higher is better)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(models)), tokens_per_sec, color='#4CAF50', alpha=0.8)
    ax1.set_xlabel('Model', fontsize=12, weight='bold')
    ax1.set_ylabel('Tokens/Second', fontsize=12, weight='bold')
    ax1.set_title('Inference Speed (Higher is Better)', fontsize=14, weight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    # add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{tokens_per_sec[i]:.1f}',
                ha='center', va='bottom', fontsize=10)
    
    # 2. inference time (lower is better)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(models)), inference_time, color='#2196F3', alpha=0.8)
    ax2.set_xlabel('Model', fontsize=12, weight='bold')
    ax2.set_ylabel('Time (seconds)', fontsize=12, weight='bold')
    ax2.set_title('Inference Time (Lower is Better)', fontsize=14, weight='bold')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{inference_time[i]:.2f}s',
                ha='center', va='bottom', fontsize=10)
    
    # 3. gpu memory utilization
    ax3 = axes[1, 0]
    bars3 = ax3.bar(range(len(models)), gpu_mem_util, color='#FF9800', alpha=0.8)
    ax3.set_xlabel('Model', fontsize=12, weight='bold')
    ax3.set_ylabel('GPU Memory Utilization (%)', fontsize=12, weight='bold')
    ax3.set_title('GPU Memory Utilization', fontsize=14, weight='bold')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.set_ylim(0, 100)
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='80% threshold')
    ax3.legend()
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{gpu_mem_util[i]:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    # 4. model load time vs inference time
    ax4 = axes[1, 1]
    x = range(len(models))
    width = 0.35
    bars4a = ax4.bar([i - width/2 for i in x], load_time, width, label='Load Time', color='#9C27B0', alpha=0.8)
    bars4b = ax4.bar([i + width/2 for i in x], inference_time, width, label='Inference Time', color='#2196F3', alpha=0.8)
    ax4.set_xlabel('Model', fontsize=12, weight='bold')
    ax4.set_ylabel('Time (seconds)', fontsize=12, weight='bold')
    ax4.set_title('Load Time vs Inference Time', fontsize=14, weight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # save to plots directory
    os.makedirs('plots', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'plots/performance_graphs_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Performance graphs saved to: {filename}")

if __name__ == "__main__":
    pattern = "benchmark_*.json" if len(sys.argv) < 2 else sys.argv[1]
    benchmarks = load_benchmarks(pattern)
    
    if benchmarks:
        print(f"\nfound {len(benchmarks)} benchmark file(s)")
        comparison = compare_benchmarks(benchmarks)
        generate_comparison_png(comparison)
        generate_performance_graphs(benchmarks)
    else:
        print("no benchmark files found. run vllm_demo.py first to generate benchmarks.")
