#!/usr/bin/env python3
"""Full analysis script for SDS2 eye-tracking data.

Usage:
    python scripts/run_analysis.py              # Sampled data (100k rows/file)
    python scripts/run_analysis.py --full       # Full data
    python scripts/run_analysis.py --nrows 50000  # Custom sample size
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from tobii_pipeline.analysis import (
    compute_cohens_d,
    compute_effect_size_r,
    compute_recording_summary,
    load_all_recordings,
    mannwhitneyu_groups,
    plot_recording_summary,
)

# =============================================================================
# Configuration
# =============================================================================

DATA_DIRS = [
    Path("Data/data_G/Tobii"),
    Path("Data/data_L/Tobii"),
]

OUTPUT_DIR = Path("figures")

METRICS = [
    ("validity_rate", "Validity Rate"),
    ("gaze_dispersion", "Gaze Dispersion (pixels)"),
    ("pupil_variability", "Pupil Variability (CV)"),
    ("fixation_mean_duration", "Mean Fixation Duration (ms)"),
    ("fixation_rate", "Fixation Rate (per sec)"),
    ("saccade_rate", "Saccade Rate (per sec)"),
]

LONGITUDINAL_METRICS = ["gaze_dispersion", "pupil_variability", "fixation_mean_duration"]


# =============================================================================
# Summary DataFrame Builder
# =============================================================================


def build_summary_df(recordings, progress=True):
    """Build summary DataFrame from loaded recordings using library metrics."""
    rows = []
    iterator = tqdm(recordings, desc="Computing metrics") if progress else recordings

    for df, metadata in iterator:
        try:
            summary = compute_recording_summary(df)
            summary.update(
                {
                    "recording_id": metadata.get("id", ""),
                    "participant": metadata.get("participant", ""),
                    "group": metadata.get("group", ""),
                    "month": metadata.get("month", 0),
                    "visit": metadata.get("visit", 0),
                }
            )
            rows.append(summary)
        except Exception as e:
            if progress:
                tqdm.write(f"Error processing {metadata.get('id', 'unknown')}: {e}")
            continue

    return pd.DataFrame(rows)


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_group_box(summary_df, metric, label, ax):
    """Box plot comparing Patient vs Control."""
    patient = summary_df[summary_df["group"] == "Patient"][metric].dropna()
    control = summary_df[summary_df["group"] == "Control"][metric].dropna()

    bp = ax.boxplot(
        [patient, control],
        tick_labels=[f"Patient (n={len(patient)})", f"Control (n={len(control)})"],
        patch_artist=True,
    )

    colors = ["#ff6b6b", "#4dabf7"]
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(label)
    ax.set_title(f"{label}: Patient vs Control")
    ax.grid(True, alpha=0.3, axis="y")


def plot_longitudinal(summary_df, metric, label, ax):
    """Plot metric trend over M0, M12, M24, M36."""
    months = [0, 12, 24, 36]

    for group, color in [("Patient", "#ff6b6b"), ("Control", "#4dabf7")]:
        data = summary_df[summary_df["group"] == group]
        means = [data[data["month"] == m][metric].mean() for m in months]
        stds = [data[data["month"] == m][metric].std() for m in months]
        means = np.array(means)
        stds = np.array(stds)

        ax.plot(months, means, "o-", color=color, label=group, linewidth=2, markersize=8)
        ax.fill_between(months, means - stds, means + stds, alpha=0.2, color=color)

    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(f"{label} Over Time", fontsize=14)
    ax.set_xticks(months)
    ax.set_xticklabels(["M0", "M12", "M24", "M36"])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)


# =============================================================================
# Statistics
# =============================================================================


def compute_stats(summary_df, metric):
    """Compute group comparison statistics."""
    patient = summary_df[summary_df["group"] == "Patient"][metric].dropna().values
    control = summary_df[summary_df["group"] == "Control"][metric].dropna().values

    if len(patient) < 2 or len(control) < 2:
        return None

    result = mannwhitneyu_groups(patient, control)

    return {
        "metric": metric,
        "patient_n": len(patient),
        "patient_mean": np.mean(patient),
        "patient_std": np.std(patient),
        "control_n": len(control),
        "control_mean": np.mean(control),
        "control_std": np.std(control),
        "U_statistic": result["U_statistic"],
        "p_value": result["p_value"],
        "significant": result["significant"],
        "effect_r": compute_effect_size_r(result["U_statistic"], len(patient), len(control)),
        "effect_d": compute_cohens_d(patient, control),
    }


# =============================================================================
# Main Analysis Pipeline
# =============================================================================


def run_analysis(nrows=None):
    """Run the full analysis pipeline."""
    # Setup output directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "group_comparisons").mkdir(exist_ok=True)
    (OUTPUT_DIR / "longitudinal").mkdir(exist_ok=True)
    (OUTPUT_DIR / "samples").mkdir(exist_ok=True)

    # Load all recordings once
    print(f"\nLoading recordings (nrows={nrows or 'all'})...")
    recordings = load_all_recordings(DATA_DIRS, nrows=nrows)
    print(f"Loaded {len(recordings)} recordings")

    if len(recordings) == 0:
        print("No recordings loaded. Check data directories.")
        return

    # Build summary DataFrame
    summary_df = build_summary_df(recordings)

    if len(summary_df) == 0:
        print("No recordings processed successfully.")
        return

    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Total recordings: {len(summary_df)}")
    print(f"Patients: {len(summary_df[summary_df['group'] == 'Patient'])}")
    print(f"Controls: {len(summary_df[summary_df['group'] == 'Control'])}")
    print("\nRecordings by month:")
    for month in [0, 12, 24, 36]:
        count = len(summary_df[summary_df["month"] == month])
        print(f"  M{month}: {count}")

    # Save summary report
    summary_path = OUTPUT_DIR / "summary_report.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary report saved to: {summary_path}")

    # Group comparison plots and statistics
    print("\n" + "=" * 60)
    print("GROUP COMPARISONS (Patient vs Control)")
    print("=" * 60)

    stats_results = []
    for metric, label in METRICS:
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_group_box(summary_df, metric, label, ax)
        fig.savefig(
            OUTPUT_DIR / "group_comparisons" / f"{metric}_comparison.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Compute statistics
        stats = compute_stats(summary_df, metric)
        if stats:
            stats_results.append(stats)
            sig = "*" if stats["significant"] else ""
            print(
                f"\n{label}:"
                f"\n  Patient: {stats['patient_mean']:.3f} +/- {stats['patient_std']:.3f} (n={stats['patient_n']})"
                f"\n  Control: {stats['control_mean']:.3f} +/- {stats['control_std']:.3f} (n={stats['control_n']})"
                f"\n  Mann-Whitney U: p={stats['p_value']:.4f}{sig}"
                f"\n  Effect size (d): {stats['effect_d']:.3f}"
            )

    # Save statistics
    if stats_results:
        stats_df = pd.DataFrame(stats_results)
        stats_path = OUTPUT_DIR / "group_statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"\nGroup statistics saved to: {stats_path}")

    # Longitudinal plots
    print("\n" + "=" * 60)
    print("LONGITUDINAL ANALYSIS")
    print("=" * 60)

    for metric in LONGITUDINAL_METRICS:
        label = next((lbl for m, lbl in METRICS if m == metric), metric)
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_longitudinal(summary_df, metric, label, ax)
        fig.savefig(
            OUTPUT_DIR / "longitudinal" / f"{metric}_trend.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
        print(f"Created: {metric}_trend.png")

    # Sample visualizations (first 3 recordings)
    print("\nCreating sample visualizations...")
    for df, meta in recordings[:3]:
        recording_id = meta.get("id", "unknown")
        print(f"  - {recording_id}")
        try:
            fig = plot_recording_summary(df, figsize=(16, 12))
            fig.suptitle(
                f"Recording: {recording_id} ({meta.get('group', '?')} - M{meta.get('month', '?')})",
                fontsize=14,
                y=1.02,
            )
            fig.savefig(
                OUTPUT_DIR / "samples" / f"{recording_id}_summary.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
        except Exception as e:
            print(f"    Error: {e}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print("  - summary_report.csv")
    print("  - group_statistics.csv")
    print("  - group_comparisons/")
    print("  - longitudinal/")
    print("  - samples/")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Run full analysis on SDS2 eye-tracking data")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Process full data (no row limit)",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=100000,
        help="Number of rows per recording (default: 100000)",
    )

    args = parser.parse_args()

    nrows = None if args.full else args.nrows

    print("=" * 60)
    print("SDS2 Eye-Tracking Analysis")
    print("=" * 60)
    if nrows:
        print(f"Mode: Sampled ({nrows:,} rows per file)")
    else:
        print("Mode: Full data")

    run_analysis(nrows=nrows)


if __name__ == "__main__":
    main()
