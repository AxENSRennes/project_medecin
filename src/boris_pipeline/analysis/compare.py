"""Group and longitudinal comparison utilities for BORIS behavioral data."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from tqdm import tqdm

from ..loader import get_recording_files, load_boris_file
from ..parser import parse_filename
from .metrics import (
    compute_behavior_durations,
    compute_behavior_frequency,
    compute_sequence_entropy,
    compute_time_budget,
)
from .stats import (
    chi_square_behavior_frequency,
    compare_behavior_duration,
    compute_cohens_d,
    compute_effect_size_r,
    mannwhitneyu_groups,
    ttest_groups,
)

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
    file_type: Literal["time_budget", "aggregated", "all"] = "aggregated",
    add_metadata: bool = True,
    progress: bool = True,
) -> list[tuple[pd.DataFrame, dict]]:
    """Load all BORIS recordings with metadata.

    Args:
        data_dirs: Directory or list of directories containing recordings.
        file_type: Type of BORIS files to load.
        add_metadata: Whether to parse and include metadata from filenames.
        progress: Show progress bar.

    Returns:
        List of tuples (DataFrame, metadata dict).
    """
    dirs = _normalize_data_dirs(data_dirs)

    all_files = []
    for data_dir in dirs:
        all_files.extend(get_recording_files(data_dir, file_type=file_type))

    results = []
    iterator = tqdm(all_files, desc="Loading recordings") if progress else all_files

    for filepath in iterator:
        try:
            df = load_boris_file(filepath)

            if add_metadata:
                metadata = parse_filename(filepath.name)
                metadata["filepath"] = filepath
            else:
                metadata = {"filepath": filepath}

            results.append((df, metadata))
        except Exception as e:
            if progress:
                tqdm.write(f"Error loading {filepath.name}: {e}")

    return results


def load_recordings_by_group(
    data_dirs: str | Path | list[str | Path],
    file_type: Literal["time_budget", "aggregated"] = "aggregated",
    progress: bool = True,
) -> dict[str, list[tuple[pd.DataFrame, dict]]]:
    """Load recordings grouped by Patient/Control.

    Args:
        data_dirs: Directory or list of directories.
        file_type: Type of BORIS files to load.
        progress: Show progress bar.

    Returns:
        Dict with keys "Patient" and "Control", values are lists of (DataFrame, metadata).
    """
    all_recordings = load_all_recordings(
        data_dirs,
        file_type=file_type,
        progress=progress,
    )

    result = {"Patient": [], "Control": []}

    for df, metadata in all_recordings:
        group = metadata.get("group", "Unknown")
        if group == "P":
            result["Patient"].append((df, metadata))
        elif group == "C":
            result["Control"].append((df, metadata))

    return result


def load_recordings_by_month(
    data_dirs: str | Path | list[str | Path],
    file_type: Literal["time_budget", "aggregated"] = "aggregated",
    progress: bool = True,
) -> dict[int, list[tuple[pd.DataFrame, dict]]]:
    """Load recordings grouped by timepoint.

    Args:
        data_dirs: Directory or list of directories.
        file_type: Type of BORIS files to load.
        progress: Show progress bar.

    Returns:
        Dict with keys 0, 12, 24, 36 (months), values are lists of (DataFrame, metadata).
    """
    all_recordings = load_all_recordings(
        data_dirs,
        file_type=file_type,
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
    metric_func: Callable[[pd.DataFrame], float | dict],
    file_type: Literal["time_budget", "aggregated"] = "aggregated",
    progress: bool = True,
) -> pd.DataFrame:
    """Apply metric to all recordings and compare groups.

    Args:
        data_dirs: Directory or list of directories.
        metric_func: Function that takes DataFrame and returns a numeric value
            or dict of values.
        file_type: Type of BORIS files to load.
        progress: Show progress bar.

    Returns:
        DataFrame with columns: recording_id, participant, group, month,
                               visit, and metric column(s).
    """
    all_recordings = load_all_recordings(
        data_dirs,
        file_type=file_type,
        progress=progress,
    )

    rows = []
    for df, metadata in all_recordings:
        try:
            metric_value = metric_func(df)

            row = {
                "recording_id": metadata.get("id", ""),
                "participant": metadata.get("participant", ""),
                "group": "Patient" if metadata.get("group") == "P" else "Control",
                "month": metadata.get("month", 0),
                "visit": metadata.get("visit", 0),
            }

            # Handle dict or single value
            if isinstance(metric_value, dict):
                row.update(metric_value)
            else:
                row["metric_value"] = metric_value

            rows.append(row)
        except Exception:
            continue

    return pd.DataFrame(rows)


def compare_behavior_profiles(
    data_dirs: str | Path | list[str | Path],
    file_type: Literal["time_budget", "aggregated"] = "aggregated",
    behavior_col: str = "Behavior",
    progress: bool = True,
) -> dict:
    """Compare behavior frequency profiles between Patient and Control.

    Args:
        data_dirs: Directory or list of directories.
        file_type: Type of BORIS files to load.
        behavior_col: Column name for behavior labels.
        progress: Show progress bar.

    Returns:
        Dict with: patient_profile, control_profile, chi_square_result,
                   per_behavior_comparisons.
    """
    by_group = load_recordings_by_group(data_dirs, file_type=file_type, progress=progress)

    # Combine all recordings per group
    patient_dfs = [df for df, _ in by_group["Patient"]]
    control_dfs = [df for df, _ in by_group["Control"]]

    if not patient_dfs or not control_dfs:
        return {
            "patient_profile": {},
            "control_profile": {},
            "chi_square_result": None,
            "per_behavior_comparisons": {},
        }

    patient_combined = pd.concat(patient_dfs, ignore_index=True)
    control_combined = pd.concat(control_dfs, ignore_index=True)

    # Compute profiles
    patient_profile = compute_behavior_frequency(patient_combined, behavior_col=behavior_col)
    control_profile = compute_behavior_frequency(control_combined, behavior_col=behavior_col)

    # Chi-square test
    chi_result = chi_square_behavior_frequency(
        patient_combined, control_combined, behavior_col=behavior_col
    )

    # Per-behavior duration comparisons
    all_behaviors = set(patient_profile.keys()) | set(control_profile.keys())
    per_behavior = {}

    for behavior in all_behaviors:
        comparison = compare_behavior_duration(
            patient_combined,
            control_combined,
            behavior,
            behavior_col=behavior_col,
        )
        per_behavior[behavior] = comparison

    return {
        "patient_profile": patient_profile,
        "control_profile": control_profile,
        "chi_square_result": chi_result,
        "per_behavior_comparisons": per_behavior,
    }


def plot_group_comparison(
    data_dirs: str | Path | list[str | Path],
    metric_func: Callable[[pd.DataFrame], float],
    metric_name: str,
    ax: Axes | None = None,
    file_type: Literal["time_budget", "aggregated"] = "aggregated",
    progress: bool = True,
) -> Axes:
    """Box plot comparing metric between Patient and Control groups.

    Args:
        data_dirs: Directory or list of directories.
        metric_func: Function that computes the metric.
        metric_name: Name for axis label.
        ax: Matplotlib axes.
        file_type: Type of BORIS files to load.
        progress: Show progress bar.

    Returns:
        Matplotlib Axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    comparison_df = compare_patient_vs_control(
        data_dirs,
        metric_func,
        file_type=file_type,
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
    file_type: Literal["time_budget", "aggregated"] = "aggregated",
    progress: bool = True,
) -> dict:
    """Statistical test of group difference.

    Args:
        data_dirs: Directory or list of directories.
        metric_func: Function that computes the metric.
        test: Statistical test ("ttest", "mannwhitneyu").
        file_type: Type of BORIS files to load.
        progress: Show progress bar.

    Returns:
        Dict with test results and effect size.
    """
    comparison_df = compare_patient_vs_control(
        data_dirs,
        metric_func,
        file_type=file_type,
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
        "patient_n": len(patient_values),
        "control_n": len(control_values),
    }


# =============================================================================
# Behavior-Specific Comparisons
# =============================================================================


def compare_behavior_duration_by_group(
    data_dirs: str | Path | list[str | Path],
    behavior: str,
    file_type: Literal["time_budget", "aggregated"] = "aggregated",
    behavior_col: str = "Behavior",
    progress: bool = True,
) -> dict:
    """Compare duration of specific behavior between groups.

    Args:
        data_dirs: Directory or list of directories.
        behavior: Behavior to compare.
        file_type: Type of BORIS files to load.
        behavior_col: Column name for behavior labels.
        progress: Show progress bar.

    Returns:
        Dict with: patient_durations, control_durations, test_result, effect_size.
    """
    by_group = load_recordings_by_group(data_dirs, file_type=file_type, progress=progress)

    # Combine all recordings per group
    patient_dfs = [df for df, _ in by_group["Patient"]]
    control_dfs = [df for df, _ in by_group["Control"]]

    if not patient_dfs or not control_dfs:
        return {
            "behavior": behavior,
            "patient_durations": [],
            "control_durations": [],
            "test_result": None,
            "effect_size": np.nan,
        }

    patient_combined = pd.concat(patient_dfs, ignore_index=True)
    control_combined = pd.concat(control_dfs, ignore_index=True)

    return compare_behavior_duration(
        patient_combined,
        control_combined,
        behavior,
        behavior_col=behavior_col,
    )


def compare_behavior_frequency_by_group(
    data_dirs: str | Path | list[str | Path],
    behavior: str,
    file_type: Literal["time_budget", "aggregated"] = "aggregated",
    behavior_col: str = "Behavior",
    progress: bool = True,
) -> dict:
    """Compare frequency of specific behavior between groups.

    Args:
        data_dirs: Directory or list of directories.
        behavior: Behavior to compare.
        file_type: Type of BORIS files to load.
        behavior_col: Column name for behavior labels.
        progress: Show progress bar.

    Returns:
        Dict with comparison results per recording and summary statistics.
    """

    def count_behavior(df: pd.DataFrame) -> float:
        if behavior_col not in df.columns:
            return 0.0
        return (df[behavior_col] == behavior).sum()

    comparison_df = compare_patient_vs_control(
        data_dirs,
        count_behavior,
        file_type=file_type,
        progress=progress,
    )

    patient_counts = comparison_df[comparison_df["group"] == "Patient"]["metric_value"]
    control_counts = comparison_df[comparison_df["group"] == "Control"]["metric_value"]

    test_result = mannwhitneyu_groups(patient_counts, control_counts)

    return {
        "behavior": behavior,
        "patient_mean": patient_counts.mean(),
        "patient_std": patient_counts.std(),
        "control_mean": control_counts.mean(),
        "control_std": control_counts.std(),
        "test_result": test_result,
        "patient_n": len(patient_counts),
        "control_n": len(control_counts),
    }


# =============================================================================
# Longitudinal Analysis
# =============================================================================


def compute_longitudinal_metrics(
    data_dirs: str | Path | list[str | Path],
    metric_func: Callable[[pd.DataFrame], float],
    file_type: Literal["time_budget", "aggregated"] = "aggregated",
    progress: bool = True,
) -> pd.DataFrame:
    """Compute metric for each recording organized by timepoint.

    Args:
        data_dirs: Directory or list of directories.
        metric_func: Function that computes the metric.
        file_type: Type of BORIS files to load.
        progress: Show progress bar.

    Returns:
        DataFrame with participant-level longitudinal data, pivoted by month.
    """
    comparison_df = compare_patient_vs_control(
        data_dirs,
        metric_func,
        file_type=file_type,
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
    file_type: Literal["time_budget", "aggregated"] = "aggregated",
    progress: bool = True,
) -> Axes:
    """Line plot of metric across M0, M12, M24, M36.

    Args:
        data_dirs: Directory or list of directories.
        metric_func: Function that computes the metric.
        metric_name: Name for axis label.
        ax: Matplotlib axes.
        by_group: Separate lines for Patient/Control.
        file_type: Type of BORIS files to load.
        progress: Show progress bar.

    Returns:
        Matplotlib Axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    comparison_df = compare_patient_vs_control(
        data_dirs,
        metric_func,
        file_type=file_type,
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

            for month in months:
                month_data = group_data[group_data["month"] == month]["metric_value"]
                means.append(month_data.mean())
                stds.append(month_data.std())

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


def test_longitudinal_change(
    data_dirs: str | Path | list[str | Path],
    metric_func: Callable[[pd.DataFrame], float],
    baseline_month: int = 0,
    target_month: int = 36,
    file_type: Literal["time_budget", "aggregated"] = "aggregated",
    progress: bool = True,
) -> dict:
    """Test within-subject change from baseline to target month.

    Uses Wilcoxon signed-rank test for paired samples.

    Args:
        data_dirs: Directory or list of directories.
        metric_func: Function that computes the metric.
        baseline_month: Baseline timepoint (default 0).
        target_month: Target timepoint (default 36).
        file_type: Type of BORIS files to load.
        progress: Show progress bar.

    Returns:
        Dict with test results for Patient and Control groups separately.
    """
    from .stats import wilcoxon_paired

    comparison_df = compare_patient_vs_control(
        data_dirs,
        metric_func,
        file_type=file_type,
        progress=progress,
    )

    results = {}

    for group in ["Patient", "Control"]:
        group_data = comparison_df[comparison_df["group"] == group]

        # Pivot to get participant values at each timepoint
        pivot = group_data.pivot_table(
            index="participant", columns="month", values="metric_value", aggfunc="mean"
        )

        if baseline_month not in pivot.columns or target_month not in pivot.columns:
            results[group] = {
                "test_result": None,
                "n_pairs": 0,
                "baseline_mean": np.nan,
                "target_mean": np.nan,
                "change_mean": np.nan,
            }
            continue

        # Get paired values (participants with both timepoints)
        paired = pivot[[baseline_month, target_month]].dropna()

        if len(paired) < 2:
            results[group] = {
                "test_result": None,
                "n_pairs": len(paired),
                "baseline_mean": np.nan,
                "target_mean": np.nan,
                "change_mean": np.nan,
            }
            continue

        baseline_values = paired[baseline_month].values
        target_values = paired[target_month].values

        test_result = wilcoxon_paired(baseline_values, target_values)

        results[group] = {
            "test_result": test_result,
            "n_pairs": len(paired),
            "baseline_mean": np.mean(baseline_values),
            "target_mean": np.mean(target_values),
            "change_mean": np.mean(target_values - baseline_values),
            "change_std": np.std(target_values - baseline_values),
        }

    return results


# =============================================================================
# Reports
# =============================================================================


def generate_summary_report(
    data_dirs: str | Path | list[str | Path],
    file_type: Literal["time_budget", "aggregated"] = "aggregated",
    progress: bool = True,
) -> pd.DataFrame:
    """Generate comprehensive summary of all recordings.

    Computes standard behavioral metrics for each recording.

    Args:
        data_dirs: Directory or list of directories.
        file_type: Type of BORIS files to load.
        progress: Show progress bar.

    Returns:
        DataFrame with one row per recording and columns for all metrics.
    """
    all_recordings = load_all_recordings(
        data_dirs,
        file_type=file_type,
        progress=progress,
    )

    rows = []
    for df, metadata in all_recordings:
        try:
            frequencies = compute_behavior_frequency(df)
            durations = compute_behavior_durations(df)
            time_budget = compute_time_budget(df)
            entropy = compute_sequence_entropy(df)

            total_events = sum(frequencies.values())
            total_behaviors = len(frequencies)
            total_duration = sum(d.get("total_duration", 0) for d in durations.values())

            row = {
                "recording_id": metadata.get("id", ""),
                "participant": metadata.get("participant", ""),
                "group": "Patient" if metadata.get("group") == "P" else "Control",
                "month": metadata.get("month", 0),
                "visit": metadata.get("visit", 0),
                "total_events": total_events,
                "total_behaviors": total_behaviors,
                "total_duration": total_duration,
                "sequence_entropy": entropy,
            }

            # Add per-behavior metrics (top 5 by frequency)
            sorted_behaviors = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:5]
            for behavior, count in sorted_behaviors:
                safe_name = behavior.replace(" ", "_").lower()
                row[f"{safe_name}_count"] = count
                row[f"{safe_name}_pct"] = time_budget.get(behavior, 0) * 100
                if behavior in durations:
                    row[f"{safe_name}_mean_duration"] = durations[behavior].get("mean", np.nan)

            rows.append(row)
        except Exception:
            continue

    return pd.DataFrame(rows)
