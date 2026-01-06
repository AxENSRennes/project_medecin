#!/usr/bin/env python3
"""Per-patient full Tobii-BORIS analysis script.

Runs a complete analysis for a single participant across all their timepoints,
outputting both metrics (CSV/JSON) and visualizations.

Usage:
    python scripts/run_patient_analysis.py FAUJea           # Analyze participant FAUJea
    python scripts/run_patient_analysis.py FAUJea -o results/  # Custom output directory
    python scripts/run_patient_analysis.py FAUJea --no-plots   # Metrics only
    python scripts/run_patient_analysis.py FAUJea --nrows 50000  # Sample data
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# BORIS pipeline
from boris_pipeline import load_boris_file
from boris_pipeline.analysis.metrics import compute_recording_summary as compute_boris_summary
from boris_pipeline.analysis.plots import plot_recording_summary as plot_boris_summary
from boris_pipeline.loader import is_aggregated_file
from boris_pipeline.parser import parse_boris_filename

# Integration
from integration.alignment import validate_alignment
from integration.cross_modal import (
    compute_gaze_per_behavior,
    compute_gaze_shift_at_behavior,
    compute_pupil_change_at_behavior,
    compute_pupil_per_behavior,
    plot_cross_modal_summary,
    plot_pupil_behavior_timeline,
)
from integration.epochs import (
    align_to_behavior_onset,
    plot_event_locked_response,
)

# Tobii pipeline
from tobii_pipeline import load_recording
from tobii_pipeline.analysis import apply_publication_style, save_figure
from tobii_pipeline.analysis.metrics import compute_recording_summary as compute_tobii_summary
from tobii_pipeline.analysis.plots import plot_recording_summary as plot_tobii_summary
from tobii_pipeline.cleaner import clean_recording, filter_eye_tracker
from tobii_pipeline.parser import parse_filename

# =============================================================================
# Configuration
# =============================================================================

DATA_DIRS = {
    "tobii": [Path("Data/data_G/Tobii"), Path("Data/data_L/Tobii")],
    "boris": [Path("Data/data_G/Boris"), Path("Data/data_L/Boris")],
}

DEFAULT_OUTPUT_DIR = Path("output")


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class AnalysisError:
    """Container for analysis errors."""

    recording_id: str
    stage: str
    message: str


@dataclass
class RecordingMatch:
    """Matched Tobii-BORIS file pair."""

    tobii_path: Path
    boris_path: Path | None
    metadata: dict
    visit_key: str  # e.g., "M0_V1"


@dataclass
class RecordingResult:
    """Results from analyzing a single recording."""

    recording_id: str
    visit_key: str
    metadata: dict
    tobii_metrics: dict = field(default_factory=dict)
    boris_metrics: dict = field(default_factory=dict)
    cross_modal_metrics: dict = field(default_factory=dict)
    event_locked_metrics: dict = field(default_factory=dict)
    alignment_info: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


# =============================================================================
# File Finding and Matching
# =============================================================================


def find_participant_tobii_files(participant_code: str, data_dirs: list[Path]) -> list[Path]:
    """Find all Tobii files for a participant."""
    files = []
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        for f in data_dir.glob(f"*_{participant_code}_*.tsv"):
            files.append(f)
    return sorted(files)


def find_participant_boris_files(participant_code: str, data_dirs: list[Path]) -> list[Path]:
    """Find all BORIS aggregated files for a participant."""
    files = []
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        for f in data_dir.glob(f"*_{participant_code}_*.xlsx"):
            if is_aggregated_file(f):
                files.append(f)
    return sorted(files)


def match_tobii_boris_files(
    tobii_files: list[Path], boris_files: list[Path]
) -> list[RecordingMatch]:
    """Match Tobii recordings with corresponding BORIS files by ID and visit."""
    # Build lookup from BORIS files by (id, visit)
    boris_lookup = {}
    for boris_path in boris_files:
        meta = parse_boris_filename(boris_path.name)
        if meta.get("id") and meta.get("visit"):
            key = (meta["id"], meta["visit"])
            boris_lookup[key] = boris_path

    # Match Tobii files
    matches = []
    for tobii_path in tobii_files:
        meta = parse_filename(tobii_path.name)
        if not meta.get("id"):
            continue

        key = (meta["id"], meta.get("visit"))
        boris_path = boris_lookup.get(key)
        visit_key = f"M{meta.get('month', '?')}_V{meta.get('visit', '?')}"

        matches.append(
            RecordingMatch(
                tobii_path=tobii_path,
                boris_path=boris_path,
                metadata=meta,
                visit_key=visit_key,
            )
        )

    return matches


# =============================================================================
# Single Recording Analysis
# =============================================================================


def analyze_single_recording(
    match: RecordingMatch,
    behaviors: list[str] | None = None,
    nrows: int | None = None,
    generate_plots: bool = True,
    output_dir: Path | None = None,
    verbose: bool = False,
    pbar: tqdm | None = None,
) -> RecordingResult:
    """Analyze a single matched Tobii-BORIS recording pair."""
    result = RecordingResult(
        recording_id=match.metadata.get("id", "unknown"),
        visit_key=match.visit_key,
        metadata=match.metadata,
    )

    # --- Stage 1: Load and clean Tobii data ---
    if pbar is not None:
        pbar.set_description("Loading Tobii")
    try:
        if verbose:
            print(f"  Loading Tobii: {match.tobii_path.name}")
        tobii_df = load_recording(match.tobii_path, nrows=nrows)
        tobii_df = clean_recording(tobii_df)
        tobii_df = filter_eye_tracker(tobii_df)
    except Exception as e:
        result.errors.append(AnalysisError(result.recording_id, "load_tobii", str(e)))
        if pbar is not None:
            pbar.update(pbar.total - pbar.n)  # Complete remaining steps
        return result
    if pbar is not None:
        pbar.update(1)

    # --- Stage 2: Compute Tobii metrics ---
    if pbar is not None:
        pbar.set_description("Tobii metrics")
    try:
        result.tobii_metrics = compute_tobii_summary(tobii_df)
    except Exception as e:
        result.errors.append(AnalysisError(result.recording_id, "tobii_metrics", str(e)))
    if pbar is not None:
        pbar.update(1)

    # --- Stage 3: Load BORIS data (if available) ---
    if pbar is not None:
        pbar.set_description("Loading BORIS")
    boris_df = None
    if match.boris_path is not None:
        try:
            if verbose:
                print(f"  Loading BORIS: {match.boris_path.name}")
            boris_df = load_boris_file(match.boris_path, file_type="aggregated")
        except Exception as e:
            result.warnings.append(f"Failed to load BORIS: {e}")
    else:
        result.warnings.append("No matching BORIS file found")
    if pbar is not None:
        pbar.update(1)

    # --- Stage 4: Compute BORIS metrics ---
    if pbar is not None:
        pbar.set_description("BORIS metrics")
    if boris_df is not None:
        try:
            result.boris_metrics = compute_boris_summary(boris_df)
        except Exception as e:
            result.errors.append(AnalysisError(result.recording_id, "boris_metrics", str(e)))
    if pbar is not None:
        pbar.update(1)

    # --- Stage 5: Validate alignment ---
    if pbar is not None:
        pbar.set_description("Alignment")
    if boris_df is not None:
        try:
            result.alignment_info = validate_alignment(boris_df, tobii_df)
            if result.alignment_info.get("warnings"):
                result.warnings.extend(result.alignment_info["warnings"])
        except Exception as e:
            result.warnings.append(f"Alignment validation failed: {e}")
    if pbar is not None:
        pbar.update(1)

    # --- Stage 6: Cross-modal analysis ---
    if pbar is not None:
        pbar.set_description("Cross-modal")
    if boris_df is not None:
        try:
            if verbose:
                print("  Computing cross-modal metrics...")

            # Get behaviors to analyze
            if behaviors is None:
                behaviors_to_analyze = boris_df["Behavior"].unique().tolist()
            else:
                behaviors_to_analyze = behaviors

            # Compute gaze and pupil per behavior
            gaze_per_beh = compute_gaze_per_behavior(
                tobii_df, boris_df, behaviors=behaviors_to_analyze
            )
            pupil_per_beh = compute_pupil_per_behavior(
                tobii_df, boris_df, behaviors=behaviors_to_analyze
            )

            result.cross_modal_metrics = {
                "gaze_per_behavior": gaze_per_beh,
                "pupil_per_behavior": pupil_per_beh,
            }

            # Event-locked analysis for each behavior
            event_locked = {}
            for behavior in behaviors_to_analyze:
                try:
                    pupil_change = compute_pupil_change_at_behavior(tobii_df, boris_df, behavior)
                    gaze_shift = compute_gaze_shift_at_behavior(tobii_df, boris_df, behavior)
                    event_locked[behavior] = {
                        "pupil_change": {
                            k: v for k, v in pupil_change.items() if k != "per_event_changes"
                        },
                        "gaze_shift": gaze_shift,
                    }
                except Exception:
                    continue

            result.event_locked_metrics = event_locked

        except Exception as e:
            result.errors.append(AnalysisError(result.recording_id, "cross_modal", str(e)))
    if pbar is not None:
        pbar.update(1)

    # --- Stage 7: Generate plots ---
    if pbar is not None:
        pbar.set_description("Plotting")
    if generate_plots and output_dir is not None:
        try:
            plot_dir = output_dir / "plots" / match.visit_key
            plot_dir.mkdir(parents=True, exist_ok=True)

            if verbose:
                print(f"  Generating plots in {plot_dir}")

            # Tobii summary plot
            try:
                with apply_publication_style():
                    fig = plot_tobii_summary(tobii_df)
                    fig.suptitle(f"Tobii: {result.recording_id} ({match.visit_key})", y=1.02)
                    save_figure(fig, plot_dir / "tobii_summary", formats=["png"])
            except Exception as e:
                result.warnings.append(f"Tobii plot failed: {e}")

            # BORIS summary plot (if data available)
            if boris_df is not None:
                try:
                    with apply_publication_style():
                        fig = plot_boris_summary(boris_df)
                        fig.suptitle(f"BORIS: {result.recording_id} ({match.visit_key})", y=1.02)
                        save_figure(fig, plot_dir / "boris_summary", formats=["png"])
                except Exception as e:
                    result.warnings.append(f"BORIS plot failed: {e}")

                # Cross-modal summary
                try:
                    with apply_publication_style():
                        fig = plot_cross_modal_summary(tobii_df, boris_df)
                        fig.suptitle(
                            f"Cross-Modal: {result.recording_id} ({match.visit_key})", y=1.02
                        )
                        save_figure(fig, plot_dir / "cross_modal_summary", formats=["png"])
                except Exception as e:
                    result.warnings.append(f"Cross-modal plot failed: {e}")

                # Pupil-behavior timeline
                try:
                    with apply_publication_style():
                        fig, ax = plt.subplots(figsize=(14, 6))
                        plot_pupil_behavior_timeline(tobii_df, boris_df, ax=ax)
                        save_figure(fig, plot_dir / "pupil_behavior_timeline", formats=["png"])
                except Exception as e:
                    result.warnings.append(f"Timeline plot failed: {e}")

                # Event-locked plots for top behaviors
                if behaviors is None:
                    top_behaviors = boris_df["Behavior"].value_counts().head(3).index
                else:
                    top_behaviors = behaviors[:3]

                for behavior in top_behaviors:
                    try:
                        event_locked_df = align_to_behavior_onset(
                            tobii_df,
                            boris_df,
                            behavior,
                            window_before_s=1.0,
                            window_after_s=3.0,
                        )
                        if len(event_locked_df) > 0:
                            with apply_publication_style():
                                fig, ax = plt.subplots(figsize=(10, 6))
                                plot_event_locked_response(
                                    event_locked_df,
                                    "Pupil diameter left",
                                    ax=ax,
                                    show_individual=True,
                                )
                                ax.set_title(f"Pupil Response: {behavior}")
                                safe_name = behavior.replace(" ", "_").replace("/", "_")
                                save_figure(
                                    fig,
                                    plot_dir / f"event_locked_{safe_name}",
                                    formats=["png"],
                                )
                    except Exception:
                        continue

        except Exception as e:
            result.errors.append(AnalysisError(result.recording_id, "plots", str(e)))
    if pbar is not None:
        pbar.update(1)

    return result


# =============================================================================
# Aggregation and Export
# =============================================================================


def _flatten_tobii_metrics(metrics: dict) -> dict:
    """Flatten nested tobii_metrics dict for CSV export and plotting.

    The tobii_metrics dict from compute_recording_summary has nested structure:
    {
        "quality": {"validity_rate": ..., ...},
        "gaze": {"dispersion": ..., "center": (x, y), ...},
        "pupil": {"variability": ..., "stats": {...}},
        "fixation": {"count": ..., "duration_mean_ms": ...},
        "saccade": {...},
    }

    This function flattens it to:
    {
        "validity_rate": ...,
        "gaze_dispersion": ...,
        "pupil_variability": ...,
        "fixation_mean_duration": ...,
        ...
    }
    """
    flat = {}

    # Quality metrics
    if "quality" in metrics:
        q = metrics["quality"]
        flat["validity_rate"] = q.get("validity_rate")
        flat["validity_rate_either"] = q.get("validity_rate_either")
        flat["tracking_ratio"] = q.get("tracking_ratio")

    # Gaze metrics
    if "gaze" in metrics:
        g = metrics["gaze"]
        flat["gaze_dispersion"] = g.get("dispersion")
        center = g.get("center", (None, None))
        if isinstance(center, tuple):
            flat["gaze_center_x"] = center[0]
            flat["gaze_center_y"] = center[1]

    # Pupil metrics
    if "pupil" in metrics:
        p = metrics["pupil"]
        flat["pupil_variability"] = p.get("variability")
        stats = p.get("stats", {})
        if isinstance(stats, dict):
            flat["pupil_mean"] = stats.get("mean")
            flat["pupil_left_mean"] = stats.get("left_mean")
            flat["pupil_right_mean"] = stats.get("right_mean")

    # Fixation metrics
    if "fixation" in metrics:
        f = metrics["fixation"]
        flat["fixation_count"] = f.get("count")
        flat["fixation_mean_duration"] = f.get("duration_mean_ms")
        flat["fixation_std_duration"] = f.get("duration_std_ms")

    # Saccade metrics
    if "saccade" in metrics:
        s = metrics["saccade"]
        flat["saccade_count"] = s.get("count")
        flat["saccade_mean_duration"] = s.get("duration_mean_ms")
        flat["saccade_mean_amplitude"] = s.get("amplitude_mean_deg")

    return flat


def aggregate_results(results: list[RecordingResult]) -> dict:
    """Aggregate results across all recordings."""
    tobii_rows = []
    boris_rows = []
    cross_modal_rows = []
    event_locked_rows = []

    for result in results:
        base_info = {
            "recording_id": result.recording_id,
            "visit_key": result.visit_key,
            "participant": result.metadata.get("participant", ""),
            "group": result.metadata.get("group", ""),
            "month": result.metadata.get("month", 0),
            "visit": result.metadata.get("visit", 0),
        }

        # Tobii metrics (flatten nested dict for CSV/plotting)
        if result.tobii_metrics:
            flat_metrics = _flatten_tobii_metrics(result.tobii_metrics)
            tobii_rows.append({**base_info, **flat_metrics})

        # BORIS metrics
        if result.boris_metrics:
            boris_rows.append({**base_info, **result.boris_metrics})

        # Cross-modal metrics (flatten per behavior)
        if result.cross_modal_metrics:
            gaze_per_beh = result.cross_modal_metrics.get("gaze_per_behavior", {})
            pupil_per_beh = result.cross_modal_metrics.get("pupil_per_behavior", {})

            for behavior in set(gaze_per_beh.keys()) | set(pupil_per_beh.keys()):
                row = {**base_info, "behavior": behavior}
                gaze = gaze_per_beh.get(behavior, {})
                pupil = pupil_per_beh.get(behavior, {})

                # Flatten gaze metrics
                for key, value in gaze.items():
                    if isinstance(value, tuple):
                        row[f"gaze_{key}_x"] = value[0]
                        row[f"gaze_{key}_y"] = value[1]
                    else:
                        row[f"gaze_{key}"] = value

                # Flatten pupil metrics
                for key, value in pupil.items():
                    row[f"pupil_{key}"] = value

                cross_modal_rows.append(row)

        # Event-locked metrics
        if result.event_locked_metrics:
            for behavior, metrics in result.event_locked_metrics.items():
                row = {**base_info, "behavior": behavior}
                pupil_change = metrics.get("pupil_change", {})
                gaze_shift = metrics.get("gaze_shift", {})

                for key, value in pupil_change.items():
                    row[f"pupil_change_{key}"] = value
                for key, value in gaze_shift.items():
                    row[f"gaze_shift_{key}"] = value

                event_locked_rows.append(row)

    return {
        "tobii_metrics": pd.DataFrame(tobii_rows) if tobii_rows else pd.DataFrame(),
        "boris_metrics": pd.DataFrame(boris_rows) if boris_rows else pd.DataFrame(),
        "cross_modal_metrics": (
            pd.DataFrame(cross_modal_rows) if cross_modal_rows else pd.DataFrame()
        ),
        "event_locked_metrics": (
            pd.DataFrame(event_locked_rows) if event_locked_rows else pd.DataFrame()
        ),
    }


def export_results(
    participant_code: str,
    results: list[RecordingResult],
    aggregated: dict,
    output_dir: Path,
):
    """Export all results to files."""
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Export CSVs
    for name, df in aggregated.items():
        if len(df) > 0:
            df.to_csv(metrics_dir / f"{name}.csv", index=False)
            print(f"  Saved: {name}.csv ({len(df)} rows)")

    # Export summary JSON
    summary = {
        "participant": participant_code,
        "n_recordings": len(results),
        "recordings": [],
    }

    for result in results:
        rec_summary = {
            "recording_id": result.recording_id,
            "visit_key": result.visit_key,
            "metadata": {
                k: str(v)
                if hasattr(v, "__str__") and not isinstance(v, int | float | str | type(None))
                else v
                for k, v in result.metadata.items()
            },
            "has_tobii_metrics": bool(result.tobii_metrics),
            "has_boris_metrics": bool(result.boris_metrics),
            "has_cross_modal": bool(result.cross_modal_metrics),
            "n_errors": len(result.errors),
            "n_warnings": len(result.warnings),
            "errors": [asdict(e) for e in result.errors] if result.errors else [],
            "warnings": result.warnings,
        }
        if result.alignment_info:
            rec_summary["alignment"] = {
                k: v for k, v in result.alignment_info.items() if k not in ("warnings",)
            }
        summary["recordings"].append(rec_summary)

    with open(output_dir / "analysis_report.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("  Saved: analysis_report.json")


def create_longitudinal_plots(
    aggregated: dict,
    participant_code: str,
    output_dir: Path,
):
    """Create longitudinal trend plots."""
    plot_dir = output_dir / "plots" / "longitudinal"
    plot_dir.mkdir(parents=True, exist_ok=True)

    tobii_df = aggregated.get("tobii_metrics")
    if tobii_df is None or len(tobii_df) == 0:
        return

    # Sort by month
    tobii_df = tobii_df.sort_values("month")

    # Metrics to plot over time
    metrics_to_plot = [
        ("validity_rate", "Validity Rate"),
        ("gaze_dispersion", "Gaze Dispersion (pixels)"),
        ("pupil_variability", "Pupil Variability (CV)"),
        ("fixation_mean_duration", "Mean Fixation Duration (ms)"),
    ]

    with apply_publication_style():
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, (metric, label) in enumerate(metrics_to_plot):
            ax = axes[idx]
            if metric in tobii_df.columns:
                months = tobii_df["month"].values
                values = tobii_df[metric].values

                ax.plot(months, values, "o-", markersize=10, linewidth=2)
                ax.set_xlabel("Month")
                ax.set_ylabel(label)
                ax.set_title(f"{label} Over Time")
                ax.set_xticks(months)
                ax.set_xticklabels([f"M{m}" for m in months])
                ax.grid(True, alpha=0.3)
            else:
                ax.set_title(f"{label} (not available)")
                ax.axis("off")

        fig.suptitle(f"Longitudinal Metrics: {participant_code}", fontsize=14)
        fig.tight_layout()
        save_figure(fig, plot_dir / "metrics_over_time", formats=["png"])
    print("  Saved: longitudinal/metrics_over_time.png")


# =============================================================================
# Main Analysis Pipeline
# =============================================================================


def run_patient_analysis(
    participant_code: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    behaviors: list[str] | None = None,
    generate_plots: bool = True,
    nrows: int | None = None,
    verbose: bool = False,
):
    """Run full analysis for a single participant."""
    print("=" * 60)
    print(f"Patient Analysis: {participant_code}")
    print("=" * 60)

    # Create output directory
    patient_output_dir = output_dir / participant_code
    patient_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {patient_output_dir}")

    # Find all files for participant
    print("\nFinding files...")
    tobii_files = find_participant_tobii_files(participant_code, DATA_DIRS["tobii"])
    boris_files = find_participant_boris_files(participant_code, DATA_DIRS["boris"])

    print(f"  Tobii files: {len(tobii_files)}")
    print(f"  BORIS files: {len(boris_files)}")

    if not tobii_files:
        print(f"\nERROR: No Tobii files found for participant '{participant_code}'")
        return

    # Match files
    matches = match_tobii_boris_files(tobii_files, boris_files)
    print(f"  Matched pairs: {len(matches)}")

    # Analyze each recording
    print("\nAnalyzing recordings...")
    results = []
    n_stages = 7  # Number of stages per recording

    with tqdm(total=len(matches), desc="Recordings", position=0) as pbar_recordings:
        for match in matches:
            if verbose:
                print(f"\n{match.visit_key}: {match.metadata.get('id', 'unknown')}")

            pbar_recordings.set_postfix_str(match.visit_key)

            with tqdm(
                total=n_stages,
                desc="Starting",
                position=1,
                leave=False,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
            ) as pbar_stages:
                result = analyze_single_recording(
                    match,
                    behaviors=behaviors,
                    nrows=nrows,
                    generate_plots=generate_plots,
                    output_dir=patient_output_dir,
                    verbose=verbose,
                    pbar=pbar_stages,
                )
            results.append(result)
            pbar_recordings.update(1)

            if result.errors:
                for error in result.errors:
                    tqdm.write(f"  ERROR [{error.stage}]: {error.message}")

    # Aggregate results
    print("\nAggregating results...")
    aggregated = aggregate_results(results)

    # Export results
    print("\nExporting results...")
    export_results(participant_code, results, aggregated, patient_output_dir)

    # Create longitudinal plots
    if generate_plots:
        print("\nCreating longitudinal plots...")
        create_longitudinal_plots(aggregated, participant_code, patient_output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Participant: {participant_code}")
    print(f"Recordings analyzed: {len(results)}")

    n_errors = sum(len(r.errors) for r in results)
    n_warnings = sum(len(r.warnings) for r in results)
    print(f"Total errors: {n_errors}")
    print(f"Total warnings: {n_warnings}")

    print(f"\nOutputs saved to: {patient_output_dir}/")
    print("  - metrics/*.csv")
    if generate_plots:
        print("  - plots/*/")
    print("  - analysis_report.json")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run full Tobii-BORIS analysis for a single participant"
    )
    parser.add_argument(
        "participant_code",
        help="6-character participant code (e.g., 'FAUJea')",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--behaviors",
        nargs="+",
        help="Specific behaviors to analyze (default: all)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation (metrics only)",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        help="Limit rows per Tobii file (for testing)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    run_patient_analysis(
        participant_code=args.participant_code,
        output_dir=args.output_dir,
        behaviors=args.behaviors,
        generate_plots=not args.no_plots,
        nrows=args.nrows,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
