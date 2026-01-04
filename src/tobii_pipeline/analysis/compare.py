"""Group and longitudinal comparison utilities."""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from tqdm import tqdm

from ..cleaner import clean_recording, filter_eye_tracker
from ..loader import get_recording_files, load_recording
from ..parser import parse_filename
from .stats import compute_cohens_d, compute_effect_size_r, mannwhitneyu_groups, ttest_groups

# =============================================================================
# Data Loading Helpers
# =============================================================================


def _normalize_data_dirs(data_dirs: str | Path | list[str | Path]) -> list[Path]:
    """Normalize data directories to list of Paths."""
    if isinstance(data_dirs, str | Path):
        return [Path(data_dirs)]
    return [Path(d) for d in data_dirs]


def load_all_recordings(
    data_dirs: str | Path | list[str | Path],
    nrows: int | None = None,
    clean: bool = True,
    filter_to_eye_tracker: bool = True,
    progress: bool = True,
) -> list[tuple[pd.DataFrame, dict]]:
    """Load all recordings with metadata.

    Args:
        data_dirs: Directory or list of directories containing recordings
        nrows: Optional limit on rows per recording (for testing)
        clean: Apply clean_recording
        filter_to_eye_tracker: Filter to eye tracker data only
        progress: Show progress bar

    Returns:
        List of tuples (DataFrame, metadata dict)
    """
    dirs = _normalize_data_dirs(data_dirs)

    all_files = []
    for data_dir in dirs:
        all_files.extend(get_recording_files(data_dir))

    results = []
    iterator = tqdm(all_files, desc="Loading recordings") if progress else all_files

    for filepath in iterator:
        try:
            df = load_recording(filepath, nrows=nrows)
            metadata = parse_filename(filepath.name)
            metadata["filepath"] = filepath

            if clean:
                df = clean_recording(df)

            if filter_to_eye_tracker:
                df = filter_eye_tracker(df)

            results.append((df, metadata))
        except Exception as e:
            if progress:
                tqdm.write(f"Error loading {filepath.name}: {e}")

    return results


def load_recordings_by_group(
    data_dirs: str | Path | list[str | Path],
    clean: bool = True,
    filter_to_eye_tracker: bool = True,
    progress: bool = True,
) -> dict[str, list[tuple[pd.DataFrame, dict]]]:
    """Load recordings grouped by Patient/Control.

    Args:
        data_dirs: Directory or list of directories
        clean: Apply clean_recording
        filter_to_eye_tracker: Filter to eye tracker data only
        progress: Show progress bar

    Returns:
        Dict with keys "Patient" and "Control", values are lists of (DataFrame, metadata)
    """
    all_recordings = load_all_recordings(
        data_dirs,
        clean=clean,
        filter_to_eye_tracker=filter_to_eye_tracker,
        progress=progress,
    )

    result = {"Patient": [], "Control": []}

    for df, metadata in all_recordings:
        group = metadata.get("group", "Unknown")
        if group in result:
            result[group].append((df, metadata))

    return result


def load_recordings_by_month(
    data_dirs: str | Path | list[str | Path],
    clean: bool = True,
    filter_to_eye_tracker: bool = True,
    progress: bool = True,
) -> dict[int, list[tuple[pd.DataFrame, dict]]]:
    """Load recordings grouped by timepoint.

    Args:
        data_dirs: Directory or list of directories
        clean: Apply clean_recording
        filter_to_eye_tracker: Filter to eye tracker data only
        progress: Show progress bar

    Returns:
        Dict with keys 0, 12, 24, 36 (months), values are lists of (DataFrame, metadata)
    """
    all_recordings = load_all_recordings(
        data_dirs,
        clean=clean,
        filter_to_eye_tracker=filter_to_eye_tracker,
        progress=progress,
    )

    result = {0: [], 12: [], 24: [], 36: []}

    for df, metadata in all_recordings:
        month = metadata.get("month")
        if month in result:
            result[month].append((df, metadata))

    return result


# =============================================================================
# Group Comparisons
# =============================================================================


def compare_patient_vs_control(
    data_dirs: str | Path | list[str | Path],
    metric_func: Callable[[pd.DataFrame], float],
    clean: bool = True,
    filter_to_eye_tracker: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """Apply metric to all recordings and compare groups.

    Args:
        data_dirs: Directory or list of directories
        metric_func: Function that takes DataFrame and returns a numeric value
        clean: Apply clean_recording before metric
        filter_to_eye_tracker: Filter to eye tracker data before metric
        progress: Show progress bar

    Returns:
        DataFrame with columns: recording_id, participant, group, month, metric_value
    """
    all_recordings = load_all_recordings(
        data_dirs,
        clean=clean,
        filter_to_eye_tracker=filter_to_eye_tracker,
        progress=progress,
    )

    rows = []
    for df, metadata in all_recordings:
        try:
            metric_value = metric_func(df)
            rows.append(
                {
                    "recording_id": metadata.get("id", ""),
                    "participant": metadata.get("participant", ""),
                    "group": metadata.get("group", ""),
                    "month": metadata.get("month", 0),
                    "visit": metadata.get("visit", 0),
                    "metric_value": metric_value,
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


def plot_group_comparison(
    data_dirs: str | Path | list[str | Path],
    metric_func: Callable[[pd.DataFrame], float],
    metric_name: str,
    ax: Axes | None = None,
    progress: bool = True,
) -> Axes:
    """Box plot comparing metric between Patient and Control groups.

    Args:
        data_dirs: Directory or list of directories
        metric_func: Function that computes the metric
        metric_name: Name for axis label
        ax: Matplotlib axes
        progress: Show progress bar

    Returns:
        Matplotlib Axes object
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    comparison_df = compare_patient_vs_control(
        data_dirs,
        metric_func,
        progress=progress,
    )

    if len(comparison_df) == 0:
        ax.set_title("No data available")
        return ax

    # Separate by group
    patient_values = comparison_df[comparison_df["group"] == "Patient"]["metric_value"].dropna()
    control_values = comparison_df[comparison_df["group"] == "Control"]["metric_value"].dropna()

    # Box plot
    data = [patient_values, control_values]
    labels = [f"Patient (n={len(patient_values)})", f"Control (n={len(control_values)})"]

    ax.boxplot(data, labels=labels)
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name}: Patient vs Control")
    ax.grid(True, alpha=0.3, axis="y")

    return ax


def test_group_difference(
    data_dirs: str | Path | list[str | Path],
    metric_func: Callable[[pd.DataFrame], float],
    test: str = "mannwhitneyu",
    progress: bool = True,
) -> dict:
    """Statistical test of group difference.

    Args:
        data_dirs: Directory or list of directories
        metric_func: Function that computes the metric
        test: Statistical test ("ttest", "mannwhitneyu")
        progress: Show progress bar

    Returns:
        Dict with test results and effect size
    """
    comparison_df = compare_patient_vs_control(
        data_dirs,
        metric_func,
        progress=progress,
    )

    patient_values = (
        comparison_df[comparison_df["group"] == "Patient"]["metric_value"].dropna().values
    )
    control_values = (
        comparison_df[comparison_df["group"] == "Control"]["metric_value"].dropna().values
    )

    if test == "ttest":
        test_result = ttest_groups(patient_values, control_values)
        effect_size = compute_cohens_d(patient_values, control_values)
        effect_type = "cohens_d"
    else:
        test_result = mannwhitneyu_groups(patient_values, control_values)
        effect_size = compute_effect_size_r(
            test_result["U_statistic"],
            len(patient_values),
            len(control_values),
        )
        effect_type = "r"

    return {
        "test": test,
        "test_result": test_result,
        "effect_size": effect_size,
        "effect_type": effect_type,
        "patient_mean": np.mean(patient_values) if len(patient_values) > 0 else np.nan,
        "patient_std": np.std(patient_values) if len(patient_values) > 0 else np.nan,
        "control_mean": np.mean(control_values) if len(control_values) > 0 else np.nan,
        "control_std": np.std(control_values) if len(control_values) > 0 else np.nan,
    }


# =============================================================================
# Longitudinal Analysis
# =============================================================================


def compute_longitudinal_metrics(
    data_dirs: str | Path | list[str | Path],
    metric_func: Callable[[pd.DataFrame], float],
    progress: bool = True,
) -> pd.DataFrame:
    """Compute metric for each recording organized by timepoint.

    Args:
        data_dirs: Directory or list of directories
        metric_func: Function that computes the metric
        progress: Show progress bar

    Returns:
        DataFrame with participant-level longitudinal data
    """
    comparison_df = compare_patient_vs_control(
        data_dirs,
        metric_func,
        progress=progress,
    )

    if len(comparison_df) == 0:
        return pd.DataFrame()

    # Pivot to get participant x month structure
    pivot_df = comparison_df.pivot_table(
        index=["participant", "group"],
        columns="month",
        values="metric_value",
        aggfunc="mean",
    ).reset_index()

    # Rename columns
    pivot_df.columns = [col if not isinstance(col, int) else f"M{col}" for col in pivot_df.columns]

    return pivot_df


def plot_longitudinal_trend(
    data_dirs: str | Path | list[str | Path],
    metric_func: Callable[[pd.DataFrame], float],
    metric_name: str,
    ax: Axes | None = None,
    by_group: bool = True,
    progress: bool = True,
) -> Axes:
    """Line plot of metric across M0, M12, M24, M36.

    Args:
        data_dirs: Directory or list of directories
        metric_func: Function that computes the metric
        metric_name: Name for axis label
        ax: Matplotlib axes
        by_group: Separate lines for Patient/Control
        progress: Show progress bar

    Returns:
        Matplotlib Axes object
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    comparison_df = compare_patient_vs_control(
        data_dirs,
        metric_func,
        progress=progress,
    )

    if len(comparison_df) == 0:
        ax.set_title("No data available")
        return ax

    months = [0, 12, 24, 36]

    if by_group:
        for group, color in [("Patient", "red"), ("Control", "blue")]:
            group_data = comparison_df[comparison_df["group"] == group]

            means = []
            stds = []
            counts = []

            for month in months:
                month_data = group_data[group_data["month"] == month]["metric_value"]
                means.append(month_data.mean())
                stds.append(month_data.std())
                counts.append(len(month_data))

            means = np.array(means)
            stds = np.array(stds)

            ax.plot(months, means, "o-", color=color, label=f"{group}")
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
            month_data = comparison_df[comparison_df["month"] == month]["metric_value"]
            means.append(month_data.mean())
            stds.append(month_data.std())

        means = np.array(means)
        stds = np.array(stds)

        ax.plot(months, means, "o-", color="black")
        ax.fill_between(months, means - stds, means + stds, alpha=0.2, color="gray")

    ax.set_xlabel("Month")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} Over Time")
    ax.set_xticks(months)
    ax.set_xticklabels(["M0", "M12", "M24", "M36"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


# =============================================================================
# Reports
# =============================================================================


def generate_summary_report(
    data_dirs: str | Path | list[str | Path],
    progress: bool = True,
) -> pd.DataFrame:
    """Generate comprehensive summary of all recordings.

    Computes standard metrics for each recording.

    Args:
        data_dirs: Directory or list of directories
        progress: Show progress bar

    Returns:
        DataFrame with one row per recording and columns for all metrics
    """
    from .metrics import (
        compute_fixation_stats,
        compute_gaze_dispersion,
        compute_pupil_stats,
        compute_pupil_variability,
        compute_saccade_stats,
        compute_validity_rate,
    )

    all_recordings = load_all_recordings(
        data_dirs,
        clean=True,
        filter_to_eye_tracker=True,
        progress=progress,
    )

    rows = []
    for df, metadata in all_recordings:
        try:
            pupil_stats = compute_pupil_stats(df)
            fixation_stats = compute_fixation_stats(df)
            saccade_stats = compute_saccade_stats(df)

            row = {
                "recording_id": metadata.get("id", ""),
                "participant": metadata.get("participant", ""),
                "group": metadata.get("group", ""),
                "month": metadata.get("month", 0),
                "visit": metadata.get("visit", 0),
                "n_samples": len(df),
                "validity_rate": compute_validity_rate(df),
                "validity_rate_either": compute_validity_rate(df, both_eyes=False),
                "gaze_dispersion": compute_gaze_dispersion(df),
                "pupil_mean": pupil_stats["mean"],
                "pupil_left_mean": pupil_stats["left_mean"],
                "pupil_right_mean": pupil_stats["right_mean"],
                "pupil_variability": compute_pupil_variability(df),
                "fixation_count": fixation_stats["count"],
                "fixation_mean_duration": fixation_stats["mean_duration"],
                "fixation_rate": fixation_stats["fixation_rate"],
                "saccade_count": saccade_stats["count"],
                "saccade_mean_duration": saccade_stats["mean_duration"],
                "saccade_rate": saccade_stats["saccade_rate"],
            }
            rows.append(row)
        except Exception:
            continue

    return pd.DataFrame(rows)
