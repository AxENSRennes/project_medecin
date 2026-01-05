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
    compute_fixation_stats,
    compute_gaze_dispersion,
    compute_pupil_stats,
    compute_pupil_variability,
    compute_saccade_stats,
    compute_validity_rate,
    interpret_effect_size,
    mannwhitneyu_groups,
    plot_recording_summary,
)
from tobii_pipeline.analysis.stats import compute_cohens_d, compute_effect_size_r
from tobii_pipeline.cleaner import clean_recording, filter_eye_tracker
from tobii_pipeline.loader import get_recording_files, load_recording
from tobii_pipeline.parser import parse_filename

# =============================================================================
# Configuration
# =============================================================================

DATA_DIRS = [
    Path("Data/data_G/Tobii"),
    Path("Data/data_L/Tobii"),
]

OUTPUT_DIR = Path("figures")


# =============================================================================
# Metric Definitions
# =============================================================================


def get_metric_configs():
    """Define metrics to analyze with their computation functions."""
    return [
        {
            "name": "validity_rate",
            "label": "Validity Rate",
            "func": lambda df: compute_validity_rate(df),
        },
        {
            "name": "gaze_dispersion",
            "label": "Gaze Dispersion (pixels)",
            "func": lambda df: compute_gaze_dispersion(df),
        },
        {
            "name": "pupil_variability",
            "label": "Pupil Variability (CV)",
            "func": lambda df: compute_pupil_variability(df),
        },
        {
            "name": "fixation_mean_duration",
            "label": "Mean Fixation Duration (ms)",
            "func": lambda df: compute_fixation_stats(df)["mean_duration"],
        },
        {
            "name": "fixation_rate",
            "label": "Fixation Rate (per sec)",
            "func": lambda df: compute_fixation_stats(df)["fixation_rate"],
        },
        {
            "name": "saccade_rate",
            "label": "Saccade Rate (per sec)",
            "func": lambda df: compute_saccade_stats(df)["saccade_rate"],
        },
    ]


# =============================================================================
# Data Loading and Metric Computation
# =============================================================================


def load_and_compute_metrics(data_dirs, nrows=None, progress=True, keep_samples=3):
    """Load recordings one at a time and compute metrics (memory-efficient).

    Args:
        data_dirs: List of data directories
        nrows: Optional row limit per recording
        progress: Show progress bar
        keep_samples: Number of sample recordings to keep for visualization

    Returns:
        DataFrame with one row per recording containing all metrics,
        and a list of sample recordings for visualization
    """
    print(f"\nLoading recordings (nrows={nrows or 'all'})...")

    # Gather all files first
    all_files = []
    for data_dir in data_dirs:
        all_files.extend(get_recording_files(data_dir))

    print(f"Found {len(all_files)} recordings")

    rows = []
    sample_recordings = []  # Only keep a few for sample visualizations
    metric_configs = get_metric_configs()

    iterator = tqdm(all_files, desc="Processing recordings") if progress else all_files

    for filepath in iterator:
        try:
            # Load single recording
            df = load_recording(filepath, nrows=nrows)
            metadata = parse_filename(filepath.name)
            metadata["filepath"] = filepath

            # Clean and filter
            df = clean_recording(df)
            df = filter_eye_tracker(df)

            # Compute metrics
            row = {
                "recording_id": metadata.get("id", ""),
                "participant": metadata.get("participant", ""),
                "group": metadata.get("group", ""),
                "month": metadata.get("month", 0),
                "visit": metadata.get("visit", 0),
                "n_samples": len(df),
            }

            for config in metric_configs:
                try:
                    row[config["name"]] = config["func"](df)
                except Exception:
                    row[config["name"]] = np.nan

            # Additional pupil stats
            try:
                pupil = compute_pupil_stats(df)
                row["pupil_mean"] = pupil["mean"]
                row["pupil_left_mean"] = pupil["left_mean"]
                row["pupil_right_mean"] = pupil["right_mean"]
            except Exception:
                row["pupil_mean"] = np.nan
                row["pupil_left_mean"] = np.nan
                row["pupil_right_mean"] = np.nan

            # Additional fixation/saccade stats
            try:
                fix = compute_fixation_stats(df)
                row["fixation_count"] = fix["count"]
            except Exception:
                row["fixation_count"] = np.nan

            try:
                sac = compute_saccade_stats(df)
                row["saccade_count"] = sac["count"]
            except Exception:
                row["saccade_count"] = np.nan

            rows.append(row)

            # Keep a few sample recordings for visualization
            if len(sample_recordings) < keep_samples:
                sample_recordings.append((df.copy(), metadata))
            # else: df goes out of scope and is garbage collected

        except Exception as e:
            if progress:
                tqdm.write(f"Error processing {filepath.name}: {e}")

    print(f"Processed {len(rows)} recordings")
    return pd.DataFrame(rows), sample_recordings


# =============================================================================
# Group Comparison Plots
# =============================================================================


def plot_group_comparison_from_df(
    summary_df, metric_name, metric_label, ax=None
):
    """Create box plot comparing Patient vs Control for a metric."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    patient = summary_df[summary_df["group"] == "Patient"][metric_name].dropna()
    control = summary_df[summary_df["group"] == "Control"][metric_name].dropna()

    data = [patient, control]
    labels = [f"Patient (n={len(patient)})", f"Control (n={len(control)})"]

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)

    # Color boxes
    colors = ["#ff6b6b", "#4dabf7"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label}: Patient vs Control")
    ax.grid(True, alpha=0.3, axis="y")

    return ax


def compute_group_stats(summary_df, metric_name):
    """Compute statistical comparison between groups."""
    patient = summary_df[summary_df["group"] == "Patient"][metric_name].dropna().values
    control = summary_df[summary_df["group"] == "Control"][metric_name].dropna().values

    if len(patient) < 2 or len(control) < 2:
        return None

    # Mann-Whitney U test
    test_result = mannwhitneyu_groups(patient, control)

    # Effect sizes
    effect_r = compute_effect_size_r(
        test_result["U_statistic"], len(patient), len(control)
    )
    effect_d = compute_cohens_d(patient, control)

    return {
        "metric": metric_name,
        "patient_n": len(patient),
        "patient_mean": np.mean(patient),
        "patient_std": np.std(patient),
        "control_n": len(control),
        "control_mean": np.mean(control),
        "control_std": np.std(control),
        "U_statistic": test_result["U_statistic"],
        "p_value": test_result["p_value"],
        "significant": test_result["significant"],
        "effect_r": effect_r,
        "effect_d": effect_d,
        "effect_interpretation": interpret_effect_size(effect_d, "d"),
    }


# =============================================================================
# Longitudinal Plots
# =============================================================================


def plot_longitudinal_from_df(
    summary_df, metric_name, metric_label, ax=None, by_group=True
):
    """Plot metric trend across M0, M12, M24, M36."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    months = [0, 12, 24, 36]

    if by_group:
        for group, color in [("Patient", "#ff6b6b"), ("Control", "#4dabf7")]:
            group_data = summary_df[summary_df["group"] == group]

            means = []
            stds = []
            counts = []

            for month in months:
                month_data = group_data[group_data["month"] == month][metric_name]
                means.append(month_data.mean())
                stds.append(month_data.std())
                counts.append(len(month_data))

            means = np.array(means)
            stds = np.array(stds)

            ax.plot(months, means, "o-", color=color, label=group, linewidth=2, markersize=8)
            ax.fill_between(
                months,
                means - stds,
                means + stds,
                alpha=0.2,
                color=color,
            )
    else:
        means = []
        stds = []

        for month in months:
            month_data = summary_df[summary_df["month"] == month][metric_name]
            means.append(month_data.mean())
            stds.append(month_data.std())

        means = np.array(means)
        stds = np.array(stds)

        ax.plot(months, means, "o-", color="black", linewidth=2, markersize=8)
        ax.fill_between(months, means - stds, means + stds, alpha=0.2, color="gray")

    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f"{metric_label} Over Time", fontsize=14)
    ax.set_xticks(months)
    ax.set_xticklabels(["M0", "M12", "M24", "M36"])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    return ax


# =============================================================================
# Sample Visualizations
# =============================================================================


def create_sample_visualizations(recordings, output_dir, max_samples=3):
    """Create detailed visualizations for a subset of recordings."""
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    # Select samples: first Patient and first Control from each month if available
    selected = []
    for df, metadata in recordings[:max_samples]:
        selected.append((df, metadata))

    print(f"\nCreating sample visualizations for {len(selected)} recordings...")

    for df, metadata in selected:
        recording_id = metadata.get("id", "unknown")
        print(f"  - {recording_id}")

        try:
            fig = plot_recording_summary(df, figsize=(16, 12))
            fig.suptitle(
                f"Recording: {recording_id} ({metadata.get('group', '?')} - M{metadata.get('month', '?')})",
                fontsize=14,
                y=1.02,
            )
            fig.savefig(
                samples_dir / f"{recording_id}_summary.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
        except Exception as e:
            print(f"    Error creating summary for {recording_id}: {e}")


# =============================================================================
# Main Analysis Pipeline
# =============================================================================


def run_analysis(nrows=None):
    """Run the full analysis pipeline."""
    # Setup output directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "group_comparisons").mkdir(exist_ok=True)
    (OUTPUT_DIR / "longitudinal").mkdir(exist_ok=True)

    # Load data and compute metrics
    summary_df, recordings = load_and_compute_metrics(DATA_DIRS, nrows=nrows)

    if len(summary_df) == 0:
        print("No recordings loaded. Check data directories.")
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

    metric_configs = get_metric_configs()
    stats_results = []

    for config in metric_configs:
        metric_name = config["name"]
        metric_label = config["label"]

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_group_comparison_from_df(summary_df, metric_name, metric_label, ax)
        fig.savefig(
            OUTPUT_DIR / "group_comparisons" / f"{metric_name}_comparison.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Compute statistics
        stats = compute_group_stats(summary_df, metric_name)
        if stats:
            stats_results.append(stats)
            sig = "*" if stats["significant"] else ""
            print(
                f"\n{metric_label}:"
                f"\n  Patient: {stats['patient_mean']:.3f} +/- {stats['patient_std']:.3f} (n={stats['patient_n']})"
                f"\n  Control: {stats['control_mean']:.3f} +/- {stats['control_std']:.3f} (n={stats['control_n']})"
                f"\n  Mann-Whitney U: p={stats['p_value']:.4f}{sig}"
                f"\n  Effect size (d): {stats['effect_d']:.3f} ({stats['effect_interpretation']})"
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

    longitudinal_metrics = ["gaze_dispersion", "pupil_variability", "fixation_mean_duration"]

    for metric_name in longitudinal_metrics:
        config = next((c for c in metric_configs if c["name"] == metric_name), None)
        if config:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_longitudinal_from_df(
                summary_df, metric_name, config["label"], ax, by_group=True
            )
            fig.savefig(
                OUTPUT_DIR / "longitudinal" / f"{metric_name}_trend.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
            print(f"Created: {metric_name}_trend.png")

    # Sample visualizations
    create_sample_visualizations(recordings, OUTPUT_DIR, max_samples=3)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print(f"  - summary_report.csv")
    print(f"  - group_statistics.csv")
    print(f"  - group_comparisons/")
    print(f"  - longitudinal/")
    print(f"  - samples/")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run full analysis on SDS2 eye-tracking data"
    )
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
