"""
Plot results from all configurations of Experiment 1.

Discovers result files matching results_phase{1,2,3a,3b}_a*_b*.json and produces:
  experiment1_all_configs_phase1.png  — Phase 1 (vLLM HTTP API) across all configs
  experiment1_all_configs_phase2.png  — Phase 2 (vLLM Python API) across all configs
  experiment1_all_configs_phase3a.png — Phase 3a (llama.cpp HTTP API) across all configs
  experiment1_all_configs_phase3b.png — Phase 3b (llama.cpp Python API) across all configs
"""

import csv
import glob
import json
import re
import statistics
import sys
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def _load_phase3b_csv(path: str) -> dict:
    """Load a phase3b results CSV and build the same data structure as JSON phases.

    The CSV has columns: trial, condition, warmup_time_ms, ttft_ms, total_time_ms,
    throughput_tok_s, prompt_tokens, completion_tokens, finish_reason.
    Rows are split by condition (hit/miss) and stats are computed.
    """
    hit_rows = []
    miss_rows = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {
                "warmup_time_ms": float(row["warmup_time_ms"]),
                "ttft_ms": float(row["ttft_ms"]),
                "total_time_ms": float(row["total_time_ms"]),
                "throughput": float(row["throughput_tok_s"]),
                "prompt_tokens": int(row["prompt_tokens"]),
                "completion_tokens": int(row["completion_tokens"]),
                "finish_reason": row["finish_reason"],
            }
            if row["condition"] == "hit":
                hit_rows.append(parsed)
            else:
                miss_rows.append(parsed)

    def _stats(trials):
        ttft = [t["ttft_ms"] for t in trials]
        total = [t["total_time_ms"] for t in trials]
        tput = [t["throughput"] for t in trials]
        return {
            "ttft_mean": statistics.mean(ttft),
            "ttft_median": statistics.median(ttft),
            "ttft_stdev": statistics.stdev(ttft) if len(ttft) > 1 else 0.0,
            "total_time_mean": statistics.mean(total),
            "total_time_stdev": statistics.stdev(total) if len(total) > 1 else 0.0,
            "throughput_mean": statistics.mean(tput),
            "throughput_stdev": statistics.stdev(tput) if len(tput) > 1 else 0.0,
            "prompt_tokens": trials[0]["prompt_tokens"],
        }

    hit_stats = _stats(hit_rows)
    miss_stats = _stats(miss_rows)

    # Warmup stats from all rows combined (hit + miss)
    all_warmup = [r["warmup_time_ms"] for r in hit_rows + miss_rows]
    warmup_mean = statistics.mean(all_warmup)
    warmup_stdev = statistics.stdev(all_warmup) if len(all_warmup) > 1 else 0.0
    warmup_median = statistics.median(all_warmup)

    for s in (hit_stats, miss_stats):
        s["warmup_time_mean"] = warmup_mean
        s["warmup_time_median"] = warmup_median
        s["warmup_time_stdev"] = warmup_stdev

    ttft_degradation = (miss_stats["ttft_mean"] / hit_stats["ttft_mean"] - 1) * 100

    # Extract A,B sizes from filename: results_phase3b_a{A}_b{B}.csv
    m = re.search(r"a(\d+)_b(\d+)", path)
    a_size = int(m.group(1)) if m else 0
    b_size = int(m.group(2)) if m else 0

    return {
        "phase": "phase3b_python_api",
        "config": {
            "context_size_a": a_size,
            "context_size_b": b_size,
        },
        "cache_hit": {"trials": hit_rows, "stats": hit_stats},
        "cache_miss": {"trials": miss_rows, "stats": miss_stats},
        "ttft_degradation_pct": ttft_degradation,
    }


def discover_results():
    """Find all result files and group by config (A,B sizes).

    Loads JSON for phases 1, 2, 3a and CSV for phase 3b.
    """
    # --- JSON files (phases 1, 2, 3a only — skip phase3b) ---
    json_files = sorted(glob.glob("results/results_phase*_a*_b*.json"))
    # Also check for old-style names (no size suffix)
    for old in ["results/results_phase1.json", "results/results_phase2.json"]:
        try:
            with open(old) as f:
                json.load(f)
            if old not in json_files:
                json_files.append(old)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    configs = {}  # key: (a_size, b_size) -> {"phase1": ..., "phase2": ..., "phase3a": ..., "phase3b": ..., "phase3b_trivia": ...}

    for path in json_files:
        # Skip any phase3b JSON files (we use CSV now)
        if "phase3b" in path:
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue

        a = data["config"]["context_size_a"]
        b = data["config"]["context_size_b"]
        key = (a, b)
        if key not in configs:
            configs[key] = {"phase1": None, "phase2": None, "phase3a": None, "phase3b": None, "phase3b_trivia": None}

        phase_str = data["phase"]
        if "phase3a" in phase_str:
            phase = "phase3a"
        elif "phase2" in phase_str:
            phase = "phase2"
        else:
            phase = "phase1"
        configs[key][phase] = data
        print(f"Loaded: {path}  (A={a}, B={b}, {phase})")

    # --- CSV files (phase 3b) ---
    csv_files = sorted(glob.glob("results/results_phase3b_a*_b*.csv"))
    for path in csv_files:
        data = _load_phase3b_csv(path)
        a = data["config"]["context_size_a"]
        b = data["config"]["context_size_b"]
        key = (a, b)
        if key not in configs:
            configs[key] = {"phase1": None, "phase2": None, "phase3a": None, "phase3b": None, "phase3b_trivia": None}
        configs[key]["phase3b"] = data
        print(f"Loaded: {path}  (A={a}, B={b}, phase3b)")

    # --- CSV files (phase 3b trivia) ---
    trivia_csv_files = sorted(glob.glob("results/results_phase3b_trivia_a*_b*.csv"))
    for path in trivia_csv_files:
        data = _load_phase3b_csv(path)
        data["phase"] = "phase3b_trivia"
        a = data["config"]["context_size_a"]
        b = data["config"]["context_size_b"]
        key = (a, b)
        if key not in configs:
            configs[key] = {"phase1": None, "phase2": None, "phase3a": None, "phase3b": None, "phase3b_trivia": None}
        configs[key]["phase3b_trivia"] = data
        print(f"Loaded: {path}  (A={a}, B={b}, phase3b_trivia)")

    return configs


# ---------------------------------------------------------------------------
# Grouped bar plot for a single phase across all configs
# ---------------------------------------------------------------------------

def plot_phase(configs: dict, phase_key: str, phase_label: str, outfile: str):
    """Grouped bar chart: TTFT + throughput + warmup for one phase across all configs."""
    labels = []
    ttft_hit = []
    ttft_miss = []
    ttft_hit_err = []
    ttft_miss_err = []
    # Non-streaming TTFT (computation only, no HTTP overhead)
    ttft_ns_hit = []
    ttft_ns_miss = []
    overhead_hit = []
    overhead_miss = []
    tput_hit = []
    tput_miss = []
    tput_hit_err = []
    tput_miss_err = []
    warmup_hit = []
    warmup_miss = []
    warmup_hit_err = []
    warmup_miss_err = []
    degrad_pcts = []
    has_warmup = False
    has_overhead = False

    for (a, b), phases in sorted(configs.items()):
        data = phases[phase_key]
        if data is None:
            continue

        hs = data["cache_hit"]["stats"]
        ms = data["cache_miss"]["stats"]

        labels.append(f"A={a}\nB={b}")
        ttft_hit.append(hs["ttft_mean"])
        ttft_miss.append(ms["ttft_mean"])
        ttft_hit_err.append(hs["ttft_stdev"])
        ttft_miss_err.append(ms["ttft_stdev"])
        tput_hit.append(hs["throughput_mean"])
        tput_miss.append(ms["throughput_mean"])
        tput_hit_err.append(hs["throughput_stdev"])
        tput_miss_err.append(ms["throughput_stdev"])
        degrad_pcts.append(data["ttft_degradation_pct"])

        if "warmup_time_mean" in hs:
            has_warmup = True
            warmup_hit.append(hs["warmup_time_mean"])
            warmup_miss.append(ms["warmup_time_mean"])
            warmup_hit_err.append(hs["warmup_time_stdev"])
            warmup_miss_err.append(ms["warmup_time_stdev"])

        if "overhead_mean" in hs:
            has_overhead = True
            ttft_ns_hit.append(hs["ttft_non_streaming_mean"])
            ttft_ns_miss.append(ms["ttft_non_streaming_mean"])
            overhead_hit.append(hs["overhead_mean"])
            overhead_miss.append(ms["overhead_mean"])

    if not labels:
        print(f"No data for {phase_label} — skipping.")
        return

    n = len(labels)
    x = np.arange(n)
    width = 0.35

    ncols = 3 if has_warmup else 2
    fig, axes = plt.subplots(1, ncols, figsize=(max(5 * ncols, 5 * n), 6))
    if ncols == 2:
        ax1, ax2 = axes
    else:
        ax1, ax2, ax3 = axes
    fig.suptitle(f"Prefix Cache Invalidation — All Configurations ({phase_label})",
                 fontsize=14, fontweight="bold")

    # TTFT — stacked bars if overhead data available
    if has_overhead:
        # Bottom: computation (non-streaming TTFT)
        b1_comp = ax1.bar(x - width / 2, ttft_ns_hit, width,
                          label="Cache Hit (computation)",
                          color="#4ecdc4", alpha=0.85, edgecolor="black")
        b2_comp = ax1.bar(x + width / 2, ttft_ns_miss, width,
                          label="Cache Miss (computation)",
                          color="#ff6b6b", alpha=0.85, edgecolor="black")
        # Top: overhead
        b1_oh = ax1.bar(x - width / 2, overhead_hit, width,
                        bottom=ttft_ns_hit,
                        label="Cache Hit (HTTP overhead)",
                        color="#4ecdc4", alpha=0.35, edgecolor="black",
                        hatch="//")
        b2_oh = ax1.bar(x + width / 2, overhead_miss, width,
                        bottom=ttft_ns_miss,
                        label="Cache Miss (HTTP overhead)",
                        color="#ff6b6b", alpha=0.35, edgecolor="black",
                        hatch="//")

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=9)
        ax1.set_xlabel("Context Size Configuration (tokens)", fontsize=10)
        ax1.set_ylabel("TTFT (ms)", fontsize=10)
        ax1.set_title("Time to First Token (stacked: computation + overhead)",
                       fontsize=10, fontweight="bold")
        ax1.legend(fontsize=7, loc="upper left")
        ax1.grid(axis="y", alpha=0.3, linestyle="--")

        all_ttft = ttft_hit + ttft_miss
        # Label total (top of stacked bar) and computation (bottom segment)
        for i in range(n):
            # Hit bar — total at top, computation inside
            ax1.text(x[i] - width / 2, ttft_hit[i] + max(all_ttft) * 0.02,
                     f"{ttft_hit[i]:.1f}", ha="center", va="bottom",
                     fontsize=7, fontweight="bold")
            ax1.text(x[i] - width / 2, ttft_ns_hit[i] / 2,
                     f"{ttft_ns_hit[i]:.1f}", ha="center", va="center",
                     fontsize=7, color="black", fontweight="bold")
            # Miss bar
            ax1.text(x[i] + width / 2, ttft_miss[i] + max(all_ttft) * 0.02,
                     f"{ttft_miss[i]:.1f}", ha="center", va="bottom",
                     fontsize=7, fontweight="bold")
            ax1.text(x[i] + width / 2, ttft_ns_miss[i] / 2,
                     f"{ttft_ns_miss[i]:.1f}", ha="center", va="center",
                     fontsize=7, color="black", fontweight="bold")

        for i, d in enumerate(degrad_pcts):
            sign = "+" if d >= 0 else ""
            ax1.annotate(f"{sign}{d:.1f}%",
                         xy=(x[i], max(ttft_hit[i], ttft_miss[i]) + max(all_ttft) * 0.10),
                         ha="center", fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.6))
    else:
        # Standard non-stacked TTFT bars
        b1 = ax1.bar(x - width / 2, ttft_hit, width, label="Cache Hit",
                     color="#4ecdc4", alpha=0.85, edgecolor="black",
                     yerr=ttft_hit_err, capsize=5)
        b2 = ax1.bar(x + width / 2, ttft_miss, width, label="Cache Miss",
                     color="#ff6b6b", alpha=0.85, edgecolor="black",
                     yerr=ttft_miss_err, capsize=5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=9)
        ax1.set_xlabel("Context Size Configuration (tokens)", fontsize=10)
        ax1.set_ylabel("TTFT (ms)", fontsize=10)
        ax1.set_title("Time to First Token", fontsize=11, fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.grid(axis="y", alpha=0.3, linestyle="--")

        all_ttft = ttft_hit + ttft_miss
        for bars in [b1, b2]:
            for bar in bars:
                h = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2, h + max(all_ttft) * 0.02,
                         f"{h:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        for i, d in enumerate(degrad_pcts):
            sign = "+" if d >= 0 else ""
            ax1.annotate(f"{sign}{d:.1f}%",
                         xy=(x[i], max(ttft_hit[i], ttft_miss[i]) + max(all_ttft) * 0.10),
                         ha="center", fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.6))

    # Throughput
    b3 = ax2.bar(x - width / 2, tput_hit, width, label="Cache Hit",
                 color="#4ecdc4", alpha=0.85, edgecolor="black",
                 yerr=tput_hit_err, capsize=5)
    b4 = ax2.bar(x + width / 2, tput_miss, width, label="Cache Miss",
                 color="#ff6b6b", alpha=0.85, edgecolor="black",
                 yerr=tput_miss_err, capsize=5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_xlabel("Context Size Configuration (tokens)", fontsize=10)
    ax2.set_ylabel("Throughput (tok/s)", fontsize=10)
    ax2.set_title("Generation Throughput", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    all_tput = tput_hit + tput_miss
    for bars in [b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + max(all_tput) * 0.02,
                     f"{h:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Warmup time (if available)
    if has_warmup:
        b5 = ax3.bar(x - width / 2, warmup_hit, width, label="Cache Hit",
                     color="#4ecdc4", alpha=0.85, edgecolor="black",
                     yerr=warmup_hit_err, capsize=5)
        b6 = ax3.bar(x + width / 2, warmup_miss, width, label="Cache Miss",
                     color="#ff6b6b", alpha=0.85, edgecolor="black",
                     yerr=warmup_miss_err, capsize=5)
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, fontsize=9)
        ax3.set_xlabel("Context Size Configuration (tokens)", fontsize=10)
        ax3.set_ylabel("Warmup Time (ms)", fontsize=10)
        ax3.set_title("Cache Warmup [A,B] Load Time", fontsize=11, fontweight="bold")
        ax3.legend(fontsize=9)
        ax3.grid(axis="y", alpha=0.3, linestyle="--")

        all_warmup = warmup_hit + warmup_miss
        for bars in [b5, b6]:
            for bar in bars:
                h = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width() / 2, h + max(all_warmup) * 0.02,
                         f"{h:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved {outfile}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    configs = discover_results()
    if not configs:
        print("Error: no result files found. Run the experiments first.")
        sys.exit(1)

    print(f"\nFound {len(configs)} configuration(s).\n")
    plot_phase(configs, "phase1", "Phase 1 — vLLM HTTP API",
               "plots/experiment1_all_configs_phase1.png")
    plot_phase(configs, "phase2", "Phase 2 — vLLM Python API",
               "plots/experiment1_all_configs_phase2.png")
    plot_phase(configs, "phase3a", "Phase 3a — llama.cpp HTTP",
               "plots/experiment1_all_configs_phase3a.png")
    plot_phase(configs, "phase3b", "Phase 3b — llama.cpp Python",
               "plots/experiment1_all_configs_phase3b.png")
    plot_phase(configs, "phase3b_trivia", "Phase 3b Trivia — llama.cpp Python",
               "plots/experiment1_all_configs_phase3b_trivia.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
