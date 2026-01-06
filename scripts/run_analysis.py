#!/usr/bin/env python3
"""Publication-quality analysis script for SDS2 eye-tracking data.

Generates three main figures:
    - Figure 1: Group comparison (heatmaps, metric distributions)
    - Figure 2: Longitudinal trends with 95% CI
    - Figure 3: Behavioral analysis (BORIS integration)

Usage:
    python scripts/run_analysis.py              # Sampled data (100k rows/file)
    python scripts/run_analysis.py --full       # Full data
    python scripts/run_analysis.py --nrows 50000  # Custom sample size
    python scripts/run_analysis.py --no-boris   # Skip BORIS integration
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from tobii_pipeline.analysis import (
    apply_publication_style,
    compute_cohens_d,
    compute_effect_size_r,
    # Core metrics
    compute_recording_summary,
    # Group visualizations
    create_group_comparison_figure,
    create_longitudinal_figure,
    load_all_recordings,
    mannwhitneyu_groups,
    plot_recording_summary,
    save_figure,
)

# =============================================================================
# Configuration
# =============================================================================

DATA_DIRS = [
    Path("Data/data_G/Tobii"),
    Path("Data/data_L/Tobii"),
]

BORIS_DIRS = [
    Path("Data/data_G/Boris"),
    Path("Data/data_L/Boris"),
]

OUTPUT_DIR = Path("figures")

# Metrics for analysis
METRICS = [
    ("validity_rate", "Validity Rate"),
    ("gaze_dispersion", "Gaze Dispersion (px)"),
    ("pupil_variability", "Pupil Variability (CV)"),
    ("fixation_mean_duration", "Fixation Duration (ms)"),
    ("fixation_rate", "Fixation Rate (/s)"),
    ("saccade_rate", "Saccade Rate (/s)"),
]

LONGITUDINAL_METRICS = [
    ("gaze_dispersion", "Gaze Dispersion (px)"),
    ("pupil_variability", "Pupil Variability (CV)"),
    ("fixation_mean_duration", "Fixation Duration (ms)"),
    ("fixation_rate", "Fixation Rate (/s)"),
]


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

            # Flatten nested dict structure
            row = {
                "recording_id": metadata.get("id", ""),
                "participant": metadata.get("participant", ""),
                "group": metadata.get("group", ""),
                "month": metadata.get("month", 0),
                "visit": metadata.get("visit", 0),
                # Quality metrics
                "validity_rate": summary.get("quality", {}).get("validity_rate", np.nan),
                "tracking_ratio": summary.get("quality", {}).get("tracking_ratio", np.nan),
                # Gaze metrics
                "gaze_dispersion": summary.get("gaze", {}).get("dispersion", np.nan),
                # Pupil metrics
                "pupil_variability": summary.get("pupil", {}).get("variability", np.nan),
                "pupil_mean": summary.get("pupil", {}).get("stats", {}).get("mean", np.nan),
                "pupil_left_mean": summary.get("pupil", {})
                .get("stats", {})
                .get("left_mean", np.nan),
                "pupil_right_mean": summary.get("pupil", {})
                .get("stats", {})
                .get("right_mean", np.nan),
                # Fixation metrics
                "fixation_count": summary.get("fixation", {}).get("count", 0),
                "fixation_mean_duration": summary.get("fixation", {}).get(
                    "duration_mean_ms", np.nan
                ),
                "fixation_rate": summary.get("fixation", {}).get("rate_per_sec", np.nan),
                # Saccade metrics
                "saccade_count": summary.get("saccade", {}).get("count", 0),
                "saccade_rate": summary.get("saccade", {}).get("rate_per_sec", np.nan),
            }

            # Compute fixation rate if not present but we have count and duration
            if np.isnan(row.get("fixation_rate", np.nan)) and row["fixation_count"] > 0:
                total_duration = len(df) / 100.0  # Assume 100Hz sampling
                if total_duration > 0:
                    row["fixation_rate"] = row["fixation_count"] / total_duration

            # Compute saccade rate similarly
            if np.isnan(row.get("saccade_rate", np.nan)) and row["saccade_count"] > 0:
                total_duration = len(df) / 100.0
                if total_duration > 0:
                    row["saccade_rate"] = row["saccade_count"] / total_duration

            rows.append(row)
        except Exception as e:
            if progress:
                tqdm.write(f"Error processing {metadata.get('id', 'unknown')}: {e}")
            continue

    return pd.DataFrame(rows)


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
# BORIS Integration
# =============================================================================


def load_boris_pairs(recordings, boris_dirs):
    """Load matching BORIS files for each Tobii recording."""
    from boris_pipeline.loader import load_boris_file
    from boris_pipeline.parser import parse_boris_filename

    pairs = []

    for df, metadata in recordings:
        participant = metadata.get("participant", "")

        # Try to find matching BORIS file
        boris_df = None
        for boris_dir in boris_dirs:
            if not boris_dir.exists():
                continue

            # Look for aggregated file matching this recording
            for boris_file in boris_dir.glob("*_agregated.xlsx"):
                try:
                    parsed = parse_boris_filename(boris_file.name)
                    if parsed and parsed.get("participant") == participant:
                        boris_df = load_boris_file(boris_file)
                        break
                except Exception:
                    continue

            if boris_df is not None:
                break

        if boris_df is not None:
            pairs.append((df, boris_df, metadata))

    return pairs


# =============================================================================
# Figure Generation Functions
# =============================================================================


def create_figure1(summary_df, recordings, output_dir):
    """Create Figure 1: Group Comparison.

    Multi-panel figure with:
    - Row 1: Aggregate gaze heatmaps (Patient | Control | Difference)
    - Rows 2-3: Key metric violin plots with significance
    """
    print("\nCreating Figure 1: Group Comparison...")

    with apply_publication_style():
        fig = create_group_comparison_figure(
            summary_df,
            recordings,
            metrics=[
                ("gaze_dispersion", "Gaze Dispersion (px)"),
                ("pupil_variability", "Pupil Variability (CV)"),
                ("fixation_mean_duration", "Fixation Duration (ms)"),
                ("fixation_rate", "Fixation Rate (/s)"),
                ("saccade_rate", "Saccade Rate (/s)"),
                ("validity_rate", "Validity Rate"),
            ],
            figsize=(12, 12),
        )

        fig.suptitle("Patient vs Control: Eye-Tracking Metrics", fontsize=14, y=1.01)

        paths = save_figure(fig, output_dir / "figure1_group_comparison")
        print(f"  Saved: {[p.name for p in paths]}")


def create_figure2(summary_df, output_dir):
    """Create Figure 2: Longitudinal Trends.

    2x2 grid of longitudinal plots with 95% confidence intervals.
    """
    print("\nCreating Figure 2: Longitudinal Trends...")

    with apply_publication_style():
        fig = create_longitudinal_figure(
            summary_df,
            metrics=LONGITUDINAL_METRICS,
            figsize=(10, 8),
        )

        fig.suptitle("Longitudinal Trends (M0 → M36)", fontsize=14, y=1.01)

        paths = save_figure(fig, output_dir / "figure2_longitudinal")
        print(f"  Saved: {[p.name for p in paths]}")


def create_figure3(tobii_boris_pairs, output_dir):
    """Create Figure 3: Behavioral Analysis (BORIS Integration).

    Eye-tracking metrics segmented by behavioral states.
    """
    from tobii_pipeline.analysis import create_behavioral_figure

    print("\nCreating Figure 3: Behavioral Analysis...")

    if not tobii_boris_pairs:
        print("  Skipped: No BORIS data available")
        return

    with apply_publication_style():
        fig = create_behavioral_figure(
            tobii_boris_pairs,
            behaviors=None,  # Auto-detect
            figsize=(12, 8),
        )

        fig.suptitle("Eye-Tracking Metrics by Behavioral State", fontsize=14, y=1.01)

        paths = save_figure(fig, output_dir / "figure3_behavioral")
        print(f"  Saved: {[p.name for p in paths]}")


def create_sample_visualizations(recordings, output_dir, n_samples=3):
    """Create sample visualizations for individual recordings."""
    print(f"\nCreating sample visualizations (n={n_samples})...")

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    for df, meta in recordings[:n_samples]:
        recording_id = meta.get("id", "unknown")
        print(f"  - {recording_id}")

        try:
            with apply_publication_style():
                fig = plot_recording_summary(df, figsize=(16, 12))
                fig.suptitle(
                    f"Recording: {recording_id} ({meta.get('group', '?')} - M{meta.get('month', '?')})",
                    fontsize=14,
                    y=1.02,
                )
                save_figure(fig, samples_dir / f"{recording_id}_summary", formats=["png"])
        except Exception as e:
            print(f"    Error: {e}")


# =============================================================================
# Main Analysis Pipeline
# =============================================================================


def run_analysis(nrows=None, include_boris=True):
    """Run the full analysis pipeline."""
    # Setup output directories
    OUTPUT_DIR.mkdir(exist_ok=True)

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

    # Compute and save group statistics
    print("\n" + "=" * 60)
    print("GROUP STATISTICS")
    print("=" * 60)

    stats_results = []
    for metric, label in METRICS:
        stats = compute_stats(summary_df, metric)
        if stats:
            stats_results.append(stats)
            sig = "*" if stats["significant"] else ""
            print(
                f"\n{label}:"
                f"\n  Patient: {stats['patient_mean']:.3f} ± {stats['patient_std']:.3f} (n={stats['patient_n']})"
                f"\n  Control: {stats['control_mean']:.3f} ± {stats['control_std']:.3f} (n={stats['control_n']})"
                f"\n  Mann-Whitney U: p={stats['p_value']:.4f}{sig}"
                f"\n  Effect size (d): {stats['effect_d']:.3f}"
            )

    if stats_results:
        stats_df = pd.DataFrame(stats_results)
        stats_path = OUTPUT_DIR / "group_statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"\nGroup statistics saved to: {stats_path}")

    # Generate publication figures
    print("\n" + "=" * 60)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 60)

    create_figure1(summary_df, recordings, OUTPUT_DIR)
    create_figure2(summary_df, OUTPUT_DIR)

    # BORIS integration (Figure 3)
    if include_boris:
        print("\nLoading BORIS behavioral data...")
        tobii_boris_pairs = load_boris_pairs(recordings, BORIS_DIRS)
        print(f"Found {len(tobii_boris_pairs)} Tobii-BORIS pairs")

        if tobii_boris_pairs:
            create_figure3(tobii_boris_pairs, OUTPUT_DIR)
        else:
            print("  No BORIS pairs found - skipping Figure 3")
    else:
        print("\nSkipping BORIS integration (--no-boris flag)")

    # Sample visualizations
    create_sample_visualizations(recordings, OUTPUT_DIR, n_samples=3)

    # Final summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print("  - summary_report.csv")
    print("  - group_statistics.csv")
    print("  - figure1_group_comparison.{png,pdf}")
    print("  - figure2_longitudinal.{png,pdf}")
    if include_boris:
        print("  - figure3_behavioral.{png,pdf}")
    print("  - samples/")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures from SDS2 eye-tracking data"
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
    parser.add_argument(
        "--no-boris",
        action="store_true",
        help="Skip BORIS behavioral data integration",
    )

    args = parser.parse_args()

    nrows = None if args.full else args.nrows
    include_boris = not args.no_boris

    print("=" * 60)
    print("SDS2 Eye-Tracking Analysis")
    print("=" * 60)
    if nrows:
        print(f"Mode: Sampled ({nrows:,} rows per file)")
    else:
        print("Mode: Full data")
    print(f"BORIS integration: {'enabled' if include_boris else 'disabled'}")

    run_analysis(nrows=nrows, include_boris=include_boris)


if __name__ == "__main__":
    main()
