#!/usr/bin/env python3
"""Publication-quality analysis script for SDS2 eye-tracking data.

Generates three main figures:
    - Figure 1: Group comparison (heatmaps, metric distributions)
    - Figure 2: Longitudinal trends with 95% CI
    - Figure 3: Behavioral analysis (BORIS integration)

Usage:
    python scripts/run_analysis.py              # Full data (default)
    python scripts/run_analysis.py --nrows 100000  # Sampled data (100k rows/file)
    python scripts/run_analysis.py --nrows 50000  # Custom sample size
    python scripts/run_analysis.py --no-boris   # Skip BORIS integration
"""

import argparse
import gc
from collections.abc import Iterator
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
    create_group_comparison_figure_streaming,
    create_longitudinal_figure,
    mannwhitneyu_groups,
    plot_recording_summary,
    save_figure,
)
from tobii_pipeline.cleaner import clean_recording, filter_eye_tracker
from tobii_pipeline.loader import load_recording
from tobii_pipeline.parser import parse_filename

# =============================================================================
# Streaming Helper Functions
# =============================================================================


def stream_recordings(
    data_dirs: str | Path | list[str | Path],
    nrows: int | None = None,
    clean: bool = True,
    filter_to_eye_tracker: bool = True,
    progress: bool = True,
) -> Iterator[tuple[pd.DataFrame, dict, Path]]:
    """Yield (df, metadata, filepath) one recording at a time.

    Memory-efficient generator that loads and yields recordings one by one,
    avoiding the need to hold all recordings in memory simultaneously.

    Args:
        data_dirs: Directory or list of directories to search
        nrows: Optional row limit per recording
        clean: Whether to apply clean_recording()
        filter_to_eye_tracker: Whether to filter to eye tracker data only
        progress: Whether to show progress bar

    Yields:
        Tuple of (DataFrame, metadata dict, filepath)
    """
    if isinstance(data_dirs, str | Path):
        data_dirs = [data_dirs]

    # Collect all file paths first
    all_files = []
    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        if data_dir.exists():
            all_files.extend(sorted(data_dir.glob("*.tsv")))

    iterator = tqdm(all_files, desc="Processing recordings") if progress else all_files

    for filepath in iterator:
        try:
            # Load recording
            df = load_recording(filepath, nrows=nrows)

            # Parse metadata from filename
            metadata = parse_filename(filepath.name)
            if metadata is None:
                metadata = {}
            metadata["filepath"] = filepath

            # Apply cleaning if requested
            if clean:
                df = clean_recording(df)

            # Filter to eye tracker data if requested
            if filter_to_eye_tracker:
                df = filter_eye_tracker(df)

            yield df, metadata, filepath

        except Exception as e:
            if progress:
                tqdm.write(f"Error loading {filepath.name}: {e}")
            continue


def extract_gaze_data(
    df: pd.DataFrame,
    metadata: dict,
    width: int = 1920,
    height: int = 1080,
) -> dict | None:
    """Extract only gaze X/Y columns for heatmaps.

    Returns lightweight dict with numpy arrays (~96% memory savings vs full DataFrame).

    Args:
        df: Full recording DataFrame
        metadata: Recording metadata
        width: Screen width for bounds filtering
        height: Screen height for bounds filtering

    Returns:
        Dict with group, participant, recording_id, gaze_x, gaze_y arrays,
        or None if no valid gaze data
    """
    gaze_x = df.get("Gaze point X")
    gaze_y = df.get("Gaze point Y")

    if gaze_x is None or gaze_y is None:
        return None

    # Filter valid values within screen bounds
    valid_mask = (
        gaze_x.notna()
        & gaze_y.notna()
        & (gaze_x >= 0)
        & (gaze_x <= width)
        & (gaze_y >= 0)
        & (gaze_y <= height)
    )

    gaze_x_valid = gaze_x[valid_mask].values.astype(np.float32)
    gaze_y_valid = gaze_y[valid_mask].values.astype(np.float32)

    if len(gaze_x_valid) == 0:
        return None

    return {
        "group": metadata.get("group", "Unknown"),
        "participant": metadata.get("participant", "Unknown"),
        "recording_id": metadata.get("id", "Unknown"),
        "gaze_x": gaze_x_valid,
        "gaze_y": gaze_y_valid,
    }


def save_gaze_cache(gaze_data_list: list[dict], cache_path: Path) -> None:
    """Save gaze data list to compressed .npz file.

    Args:
        gaze_data_list: List of gaze data dicts from extract_gaze_data()
        cache_path: Path to save the .npz file
    """
    # Flatten into arrays with metadata
    groups = []
    participants = []
    recording_ids = []
    gaze_x_list = []
    gaze_y_list = []
    offsets = [0]  # Track where each recording's data starts

    for gaze_data in gaze_data_list:
        if gaze_data is None:
            continue
        groups.append(gaze_data["group"])
        participants.append(gaze_data["participant"])
        recording_ids.append(gaze_data["recording_id"])
        gaze_x_list.append(gaze_data["gaze_x"])
        gaze_y_list.append(gaze_data["gaze_y"])
        offsets.append(offsets[-1] + len(gaze_data["gaze_x"]))

    # Concatenate all gaze data
    all_gaze_x = np.concatenate(gaze_x_list) if gaze_x_list else np.array([], dtype=np.float32)
    all_gaze_y = np.concatenate(gaze_y_list) if gaze_y_list else np.array([], dtype=np.float32)

    np.savez_compressed(
        cache_path,
        groups=np.array(groups),
        participants=np.array(participants),
        recording_ids=np.array(recording_ids),
        gaze_x=all_gaze_x,
        gaze_y=all_gaze_y,
        offsets=np.array(offsets),
    )


def load_gaze_cache(cache_path: Path) -> list[dict]:
    """Load gaze data list from .npz file.

    Args:
        cache_path: Path to the .npz file

    Returns:
        List of gaze data dicts (same format as extract_gaze_data output)
    """
    data = np.load(cache_path, allow_pickle=True)

    groups = data["groups"]
    participants = data["participants"]
    recording_ids = data["recording_ids"]
    all_gaze_x = data["gaze_x"]
    all_gaze_y = data["gaze_y"]
    offsets = data["offsets"]

    gaze_data_list = []
    for i in range(len(groups)):
        start = offsets[i]
        end = offsets[i + 1]
        gaze_data_list.append(
            {
                "group": str(groups[i]),
                "participant": str(participants[i]),
                "recording_id": str(recording_ids[i]),
                "gaze_x": all_gaze_x[start:end],
                "gaze_y": all_gaze_y[start:end],
            }
        )

    return gaze_data_list


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

        paths = save_figure(fig, output_dir / "figure1_group_comparison", formats=["png"])
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

        paths = save_figure(fig, output_dir / "figure2_longitudinal", formats=["png"])
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

        paths = save_figure(fig, output_dir / "figure3_behavioral", formats=["png"])
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
# Streaming Figure Generation Functions
# =============================================================================


def create_figure1_streaming(summary_df, gaze_data_list, output_dir):
    """Create Figure 1: Group Comparison (streaming version).

    Uses cached gaze data instead of full recordings for memory efficiency.
    """
    print("\nCreating Figure 1: Group Comparison...")

    with apply_publication_style():
        fig = create_group_comparison_figure_streaming(
            summary_df,
            gaze_data_list,
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

        paths = save_figure(fig, output_dir / "figure1_group_comparison", formats=["png"])
        print(f"  Saved: {[p.name for p in paths]}")


def create_figure3_streaming(data_dirs, boris_dirs, output_dir, nrows=None):
    """Create Figure 3: Behavioral Analysis (streaming version).

    Streams through recordings, matches BORIS files, computes metrics,
    and creates figure without holding all data in memory.
    """
    from boris_pipeline.loader import load_boris_file
    from boris_pipeline.parser import parse_boris_filename
    from tobii_pipeline.analysis import create_behavioral_figure

    print("\nCreating Figure 3: Behavioral Analysis (streaming)...")

    # Stream through recordings and collect pairs one at a time
    tobii_boris_pairs = []

    for df, metadata, _filepath in stream_recordings(data_dirs, nrows=nrows, progress=True):
        participant = metadata.get("participant", "")

        # Try to find matching BORIS file
        boris_df = None
        for boris_dir in boris_dirs:
            if not Path(boris_dir).exists():
                continue

            for boris_file in Path(boris_dir).glob("*_agregated.xlsx"):
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
            tobii_boris_pairs.append((df, boris_df, metadata))

        # Memory cleanup after each recording
        del df
        gc.collect()

    print(f"Found {len(tobii_boris_pairs)} Tobii-BORIS pairs")

    if not tobii_boris_pairs:
        print("  Skipped: No BORIS data available")
        return

    with apply_publication_style():
        fig = create_behavioral_figure(
            tobii_boris_pairs,
            behaviors=None,
            figsize=(12, 8),
        )

        fig.suptitle("Eye-Tracking Metrics by Behavioral State", fontsize=14, y=1.01)

        paths = save_figure(fig, output_dir / "figure3_behavioral", formats=["png"])
        print(f"  Saved: {[p.name for p in paths]}")

    # Clean up pairs
    del tobii_boris_pairs
    gc.collect()


def create_sample_visualizations_streaming(sample_filepaths, output_dir, nrows=None):
    """Create sample visualizations (streaming version).

    Reloads sample recordings one at a time instead of keeping all in memory.
    """
    print(f"\nCreating sample visualizations (n={len(sample_filepaths)})...")

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    for filepath, metadata in sample_filepaths:
        recording_id = metadata.get("id", "unknown")
        print(f"  - {recording_id}")

        try:
            # Reload just this recording
            df = load_recording(filepath, nrows=nrows)
            df = clean_recording(df)
            df = filter_eye_tracker(df)

            with apply_publication_style():
                fig = plot_recording_summary(df, figsize=(16, 12))
                fig.suptitle(
                    f"Recording: {recording_id} ({metadata.get('group', '?')} - M{metadata.get('month', '?')})",
                    fontsize=14,
                    y=1.02,
                )
                save_figure(fig, samples_dir / f"{recording_id}_summary", formats=["png"])

            # Clean up
            del df
            gc.collect()

        except Exception as e:
            print(f"    Error: {e}")


# =============================================================================
# Main Analysis Pipeline
# =============================================================================


def run_analysis(nrows=None, include_boris=True):
    """Run the full analysis pipeline using streaming architecture.

    Memory-efficient processing that streams recordings one at a time,
    extracting only the data needed for each stage.
    """
    # Setup output directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    gaze_cache_path = OUTPUT_DIR / ".gaze_cache.npz"

    # ==========================================================================
    # PASS 1: Stream through recordings, compute metrics, extract gaze data
    # ==========================================================================
    print(f"\nStreaming recordings (nrows={nrows or 'all'})...")

    summary_rows = []
    gaze_data_list = []
    sample_filepaths = []  # Save first 3 filepaths for sample visualizations
    n_recordings = 0

    for df, metadata, filepath in stream_recordings(DATA_DIRS, nrows=nrows):
        try:
            # Compute metrics for this recording
            summary = compute_recording_summary(df)

            # Flatten nested dict structure into row
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

            # Compute fixation rate if not present
            if np.isnan(row.get("fixation_rate", np.nan)) and row["fixation_count"] > 0:
                total_duration = len(df) / 100.0
                if total_duration > 0:
                    row["fixation_rate"] = row["fixation_count"] / total_duration

            # Compute saccade rate similarly
            if np.isnan(row.get("saccade_rate", np.nan)) and row["saccade_count"] > 0:
                total_duration = len(df) / 100.0
                if total_duration > 0:
                    row["saccade_rate"] = row["saccade_count"] / total_duration

            summary_rows.append(row)

            # Extract gaze data (X, Y only) for heatmaps
            gaze_data = extract_gaze_data(df, metadata)
            gaze_data_list.append(gaze_data)

            # Save first 3 filepaths for sample visualizations
            if len(sample_filepaths) < 3:
                sample_filepaths.append((filepath, metadata.copy()))

            n_recordings += 1

        except Exception as e:
            tqdm.write(f"Error processing {metadata.get('id', 'unknown')}: {e}")

        # CRITICAL: Delete DataFrame to free memory
        del df
        gc.collect()

    print(f"Processed {n_recordings} recordings")

    if n_recordings == 0:
        print("No recordings processed. Check data directories.")
        return

    # Build summary DataFrame and save gaze cache
    summary_df = pd.DataFrame(summary_rows)
    save_gaze_cache(gaze_data_list, gaze_cache_path)
    print(f"Gaze cache saved to: {gaze_cache_path}")

    # Free memory from summary_rows (we have summary_df now)
    del summary_rows
    gc.collect()

    if len(summary_df) == 0:
        print("No recordings processed successfully.")
        return

    # ==========================================================================
    # Print summary statistics
    # ==========================================================================
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

    # ==========================================================================
    # Compute and save group statistics
    # ==========================================================================
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

    # ==========================================================================
    # Generate publication figures (using cached gaze data)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 60)

    # Figure 1: Group comparison (uses gaze cache for heatmaps)
    create_figure1_streaming(summary_df, gaze_data_list, OUTPUT_DIR)

    # Free gaze_data_list after Figure 1 is complete
    del gaze_data_list
    gc.collect()

    # Figure 2: Longitudinal trends (uses summary_df only)
    create_figure2(summary_df, OUTPUT_DIR)

    # BORIS integration (Figure 3) - streams recordings again
    if include_boris:
        create_figure3_streaming(DATA_DIRS, BORIS_DIRS, OUTPUT_DIR, nrows=nrows)
    else:
        print("\nSkipping BORIS integration (--no-boris flag)")

    # Sample visualizations - reloads individual recordings
    create_sample_visualizations_streaming(sample_filepaths, OUTPUT_DIR, nrows=nrows)

    # Clean up gaze cache file
    if gaze_cache_path.exists():
        gaze_cache_path.unlink()

    # ==========================================================================
    # Final summary
    # ==========================================================================
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print("  - summary_report.csv")
    print("  - group_statistics.csv")
    print("  - figure1_group_comparison.png")
    print("  - figure2_longitudinal.png")
    if include_boris:
        print("  - figure3_behavioral.png")
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
        help="Process full data (no row limit) - this is now the default behavior",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Number of rows per recording (default: None, processes all rows)",
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
    if nrows is None:
        print("Mode: Full data")
    else:
        print(f"Mode: Sampled ({nrows:,} rows per file)")
    print(f"BORIS integration: {'enabled' if include_boris else 'disabled'}")

    run_analysis(nrows=nrows, include_boris=include_boris)


if __name__ == "__main__":
    main()
