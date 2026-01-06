"""Cross-modal metrics and visualizations for Boris-Tobii integration.

Provides functions to compute gaze and pupil metrics during behavioral states,
and visualizations combining both data streams.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .alignment import tobii_to_seconds, validate_alignment
from .epochs import (
    align_to_behavior_onset,
    extract_tobii_epochs,
)

# =============================================================================
# Cross-Modal Metrics
# =============================================================================


def compute_gaze_per_behavior(
    tobii_df: pd.DataFrame,
    boris_df: pd.DataFrame,
    behaviors: list[str] | None = None,
    behavior_col: str = "Behavior",
    boris_start_col: str = "Start (s)",
    boris_stop_col: str = "Stop (s)",
) -> dict[str, dict]:
    """Compute gaze metrics during each behavioral state.

    Args:
        tobii_df: Tobii recording DataFrame.
        boris_df: Boris aggregated events DataFrame.
        behaviors: List of behaviors to analyze. If None, all behaviors.
        behavior_col: Column name for Boris behavior labels.
        boris_start_col: Column name for Boris event start times.
        boris_stop_col: Column name for Boris event stop times.

    Returns:
        Dict per behavior with: gaze_center, gaze_dispersion, validity_rate,
                                n_samples, n_events.
    """
    from tobii_pipeline.analysis.metrics import (
        compute_gaze_center,
        compute_gaze_dispersion,
        compute_validity_rate,
    )

    if behaviors is None:
        behaviors = boris_df[behavior_col].unique().tolist()

    results = {}

    for behavior in behaviors:
        epochs = extract_tobii_epochs(
            tobii_df,
            boris_df,
            behavior=behavior,
            behavior_col=behavior_col,
            boris_start_col=boris_start_col,
            boris_stop_col=boris_stop_col,
        )

        if not epochs:
            results[behavior] = {
                "gaze_center": (np.nan, np.nan),
                "gaze_dispersion": np.nan,
                "validity_rate": np.nan,
                "n_samples": 0,
                "n_events": 0,
            }
            continue

        # Combine all epochs
        combined = pd.concat(epochs, ignore_index=True)

        results[behavior] = {
            "gaze_center": compute_gaze_center(combined),
            "gaze_dispersion": compute_gaze_dispersion(combined),
            "validity_rate": compute_validity_rate(combined),
            "n_samples": len(combined),
            "n_events": len(epochs),
        }

    return results


def compute_pupil_per_behavior(
    tobii_df: pd.DataFrame,
    boris_df: pd.DataFrame,
    behaviors: list[str] | None = None,
    behavior_col: str = "Behavior",
    boris_start_col: str = "Start (s)",
    boris_stop_col: str = "Stop (s)",
) -> dict[str, dict]:
    """Compute pupil metrics during each behavioral state.

    Args:
        tobii_df: Tobii recording DataFrame.
        boris_df: Boris aggregated events DataFrame.
        behaviors: List of behaviors to analyze. If None, all behaviors.
        behavior_col: Column name for Boris behavior labels.
        boris_start_col: Column name for Boris event start times.
        boris_stop_col: Column name for Boris event stop times.

    Returns:
        Dict per behavior with: pupil_mean, pupil_std, pupil_variability,
                                pupil_left_mean, pupil_right_mean, n_samples, n_events.
    """
    from tobii_pipeline.analysis.metrics import (
        compute_pupil_stats,
        compute_pupil_variability,
    )

    if behaviors is None:
        behaviors = boris_df[behavior_col].unique().tolist()

    results = {}

    for behavior in behaviors:
        epochs = extract_tobii_epochs(
            tobii_df,
            boris_df,
            behavior=behavior,
            behavior_col=behavior_col,
            boris_start_col=boris_start_col,
            boris_stop_col=boris_stop_col,
        )

        if not epochs:
            results[behavior] = {
                "pupil_mean": np.nan,
                "pupil_std": np.nan,
                "pupil_variability": np.nan,
                "pupil_left_mean": np.nan,
                "pupil_right_mean": np.nan,
                "n_samples": 0,
                "n_events": 0,
            }
            continue

        combined = pd.concat(epochs, ignore_index=True)
        pupil_stats = compute_pupil_stats(combined)

        results[behavior] = {
            "pupil_mean": pupil_stats.get("mean", np.nan),
            "pupil_std": pupil_stats.get("left_std", np.nan),  # Use left as representative
            "pupil_variability": compute_pupil_variability(combined),
            "pupil_left_mean": pupil_stats.get("left_mean", np.nan),
            "pupil_right_mean": pupil_stats.get("right_mean", np.nan),
            "n_samples": len(combined),
            "n_events": len(epochs),
        }

    return results


def compute_behavior_gaze_correlations(
    tobii_df: pd.DataFrame,
    boris_df: pd.DataFrame,
    behavior_col: str = "Behavior",
    duration_col: str = "Duration (s)",
    boris_start_col: str = "Start (s)",
    boris_stop_col: str = "Stop (s)",
) -> pd.DataFrame:
    """Correlate behavior durations with gaze metrics.

    For each behavioral event, computes gaze metrics and correlates
    with event duration.

    Args:
        tobii_df: Tobii recording DataFrame.
        boris_df: Boris aggregated events DataFrame.
        behavior_col: Column name for Boris behavior labels.
        duration_col: Column name for Boris event durations.
        boris_start_col: Column name for Boris event start times.
        boris_stop_col: Column name for Boris event stop times.

    Returns:
        DataFrame with behavior-metric correlation coefficients and p-values.
    """
    from scipy import stats as scipy_stats

    from tobii_pipeline.analysis.metrics import (
        compute_gaze_dispersion,
        compute_pupil_variability,
        compute_validity_rate,
    )

    # Compute metrics per event
    event_metrics = []

    for idx, event in boris_df.iterrows():
        epoch = extract_tobii_epochs(
            tobii_df,
            boris_df.loc[[idx]],
            behavior_col=behavior_col,
            boris_start_col=boris_start_col,
            boris_stop_col=boris_stop_col,
        )

        if not epoch:
            continue

        epoch_df = epoch[0]

        event_metrics.append(
            {
                "behavior": event[behavior_col],
                "duration": event.get(duration_col, np.nan),
                "gaze_dispersion": compute_gaze_dispersion(epoch_df),
                "validity_rate": compute_validity_rate(epoch_df),
                "pupil_variability": compute_pupil_variability(epoch_df),
            }
        )

    if not event_metrics:
        return pd.DataFrame()

    metrics_df = pd.DataFrame(event_metrics)

    # Compute correlations per behavior
    results = []
    gaze_metrics = ["gaze_dispersion", "validity_rate", "pupil_variability"]

    for behavior in metrics_df["behavior"].unique():
        beh_data = metrics_df[metrics_df["behavior"] == behavior]

        if len(beh_data) < 3:
            continue

        for metric in gaze_metrics:
            valid = beh_data[["duration", metric]].dropna()
            if len(valid) < 3:
                continue

            corr, p_value = scipy_stats.pearsonr(valid["duration"], valid[metric])

            results.append(
                {
                    "behavior": behavior,
                    "metric": metric,
                    "correlation": corr,
                    "p_value": p_value,
                    "n": len(valid),
                    "significant": p_value < 0.05,
                }
            )

    return pd.DataFrame(results)


# =============================================================================
# Behavior-Triggered Analysis
# =============================================================================


def compute_pupil_change_at_behavior(
    tobii_df: pd.DataFrame,
    boris_df: pd.DataFrame,
    behavior: str,
    baseline_duration_s: float = 0.5,
    response_duration_s: float = 2.0,
    behavior_col: str = "Behavior",
    boris_start_col: str = "Start (s)",
) -> dict:
    """Compute pupil dilation/constriction triggered by behavior onset.

    Args:
        tobii_df: Tobii recording DataFrame.
        boris_df: Boris aggregated events DataFrame.
        behavior: Behavior to analyze.
        baseline_duration_s: Duration before onset for baseline.
        response_duration_s: Duration after onset for response measurement.
        behavior_col: Column name for Boris behavior labels.
        boris_start_col: Column name for Boris event start times.

    Returns:
        Dict with: baseline_mean, response_mean, change, change_pct,
                   per_event_changes.
    """
    from tobii_pipeline.analysis.metrics import compute_pupil_stats

    # Get event-locked data
    event_locked = align_to_behavior_onset(
        tobii_df,
        boris_df,
        behavior,
        window_before_s=baseline_duration_s,
        window_after_s=response_duration_s,
        behavior_col=behavior_col,
        boris_start_col=boris_start_col,
    )

    if len(event_locked) == 0:
        return {
            "baseline_mean": np.nan,
            "response_mean": np.nan,
            "change": np.nan,
            "change_pct": np.nan,
            "per_event_changes": [],
            "n_events": 0,
        }

    per_event_changes = []

    for event_id in event_locked["event_id"].unique():
        event_data = event_locked[event_locked["event_id"] == event_id]

        # Baseline: before onset (negative relative time)
        baseline_data = event_data[event_data["relative_time_s"] < 0]
        # Response: after onset (positive relative time)
        response_data = event_data[event_data["relative_time_s"] >= 0]

        if len(baseline_data) == 0 or len(response_data) == 0:
            continue

        baseline_stats = compute_pupil_stats(baseline_data)
        response_stats = compute_pupil_stats(response_data)

        baseline_mean = baseline_stats.get("mean", np.nan)
        response_mean = response_stats.get("mean", np.nan)

        if not np.isnan(baseline_mean) and not np.isnan(response_mean):
            change = response_mean - baseline_mean
            change_pct = (change / baseline_mean * 100) if baseline_mean != 0 else np.nan
            per_event_changes.append(
                {
                    "event_id": event_id,
                    "baseline_mean": baseline_mean,
                    "response_mean": response_mean,
                    "change": change,
                    "change_pct": change_pct,
                }
            )

    if not per_event_changes:
        return {
            "baseline_mean": np.nan,
            "response_mean": np.nan,
            "change": np.nan,
            "change_pct": np.nan,
            "per_event_changes": [],
            "n_events": 0,
        }

    # Aggregate across events
    changes_df = pd.DataFrame(per_event_changes)

    return {
        "baseline_mean": changes_df["baseline_mean"].mean(),
        "response_mean": changes_df["response_mean"].mean(),
        "change": changes_df["change"].mean(),
        "change_pct": changes_df["change_pct"].mean(),
        "change_std": changes_df["change"].std(),
        "per_event_changes": per_event_changes,
        "n_events": len(per_event_changes),
    }


def compute_gaze_shift_at_behavior(
    tobii_df: pd.DataFrame,
    boris_df: pd.DataFrame,
    behavior: str,
    window_before_s: float = 0.5,
    window_after_s: float = 0.5,
    behavior_col: str = "Behavior",
    boris_start_col: str = "Start (s)",
) -> dict:
    """Measure gaze position shift at behavior onset.

    Args:
        tobii_df: Tobii recording DataFrame.
        boris_df: Boris aggregated events DataFrame.
        behavior: Behavior to analyze.
        window_before_s: Time window before onset.
        window_after_s: Time window after onset.
        behavior_col: Column name for Boris behavior labels.
        boris_start_col: Column name for Boris event start times.

    Returns:
        Dict with: shift_distance_mean, shift_distance_std, n_events.
    """
    from tobii_pipeline.analysis.metrics import compute_gaze_center

    event_locked = align_to_behavior_onset(
        tobii_df,
        boris_df,
        behavior,
        window_before_s=window_before_s,
        window_after_s=window_after_s,
        behavior_col=behavior_col,
        boris_start_col=boris_start_col,
    )

    if len(event_locked) == 0:
        return {
            "shift_distance_mean": np.nan,
            "shift_distance_std": np.nan,
            "n_events": 0,
        }

    shifts = []

    for event_id in event_locked["event_id"].unique():
        event_data = event_locked[event_locked["event_id"] == event_id]

        before_data = event_data[event_data["relative_time_s"] < 0]
        after_data = event_data[event_data["relative_time_s"] >= 0]

        if len(before_data) == 0 or len(after_data) == 0:
            continue

        before_center = compute_gaze_center(before_data)
        after_center = compute_gaze_center(after_data)

        if any(np.isnan(v) for v in before_center + after_center):
            continue

        # Euclidean distance
        distance = np.sqrt(
            (after_center[0] - before_center[0]) ** 2 + (after_center[1] - before_center[1]) ** 2
        )
        shifts.append(distance)

    if not shifts:
        return {
            "shift_distance_mean": np.nan,
            "shift_distance_std": np.nan,
            "n_events": 0,
        }

    return {
        "shift_distance_mean": np.mean(shifts),
        "shift_distance_std": np.std(shifts),
        "n_events": len(shifts),
    }


# =============================================================================
# Comparison Utilities
# =============================================================================


def compare_gaze_between_behaviors(
    tobii_df: pd.DataFrame,
    boris_df: pd.DataFrame,
    behavior1: str,
    behavior2: str,
    behavior_col: str = "Behavior",
) -> dict:
    """Statistical comparison of gaze metrics between two behaviors.

    Args:
        tobii_df: Tobii recording DataFrame.
        boris_df: Boris aggregated events DataFrame.
        behavior1: First behavior.
        behavior2: Second behavior.
        behavior_col: Column name for Boris behavior labels.

    Returns:
        Dict with test results for: gaze_dispersion, pupil_mean.
    """

    gaze_per_beh = compute_gaze_per_behavior(
        tobii_df, boris_df, behaviors=[behavior1, behavior2], behavior_col=behavior_col
    )
    pupil_per_beh = compute_pupil_per_behavior(
        tobii_df, boris_df, behaviors=[behavior1, behavior2], behavior_col=behavior_col
    )

    results = {
        "behavior1": behavior1,
        "behavior2": behavior2,
    }

    # Compare gaze dispersion
    beh1_gaze = gaze_per_beh.get(behavior1, {})
    beh2_gaze = gaze_per_beh.get(behavior2, {})

    results["gaze_dispersion"] = {
        "behavior1_value": beh1_gaze.get("gaze_dispersion", np.nan),
        "behavior2_value": beh2_gaze.get("gaze_dispersion", np.nan),
        "behavior1_n": beh1_gaze.get("n_events", 0),
        "behavior2_n": beh2_gaze.get("n_events", 0),
    }

    # Compare pupil
    beh1_pupil = pupil_per_beh.get(behavior1, {})
    beh2_pupil = pupil_per_beh.get(behavior2, {})

    results["pupil_mean"] = {
        "behavior1_value": beh1_pupil.get("pupil_mean", np.nan),
        "behavior2_value": beh2_pupil.get("pupil_mean", np.nan),
        "behavior1_n": beh1_pupil.get("n_events", 0),
        "behavior2_n": beh2_pupil.get("n_events", 0),
    }

    return results


# =============================================================================
# Visualization
# =============================================================================


def plot_gaze_by_behavior(
    tobii_df: pd.DataFrame,
    boris_df: pd.DataFrame,
    ax: Axes | None = None,
    metric: str = "dispersion",
    behavior_col: str = "Behavior",
    figsize: tuple[float, float] = (10, 6),
) -> Axes:
    """Bar plot of gaze metric per behavioral state.

    Args:
        tobii_df: Tobii recording DataFrame.
        boris_df: Boris aggregated events DataFrame.
        ax: Matplotlib axes.
        metric: "dispersion", "validity", or "pupil".
        behavior_col: Column name for Boris behavior labels.
        figsize: Figure size.

    Returns:
        Matplotlib Axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if metric in ("dispersion", "validity"):
        metrics = compute_gaze_per_behavior(tobii_df, boris_df, behavior_col=behavior_col)
        if metric == "dispersion":
            values = {b: m["gaze_dispersion"] for b, m in metrics.items()}
            ylabel = "Gaze Dispersion (pixels)"
        else:
            values = {b: m["validity_rate"] * 100 for b, m in metrics.items()}
            ylabel = "Validity Rate (%)"
    else:  # pupil
        metrics = compute_pupil_per_behavior(tobii_df, boris_df, behavior_col=behavior_col)
        values = {b: m["pupil_mean"] for b, m in metrics.items()}
        ylabel = "Mean Pupil Diameter (mm)"

    # Sort by value
    sorted_items = sorted(
        values.items(), key=lambda x: x[1] if not np.isnan(x[1]) else 0, reverse=True
    )
    behaviors = [item[0] for item in sorted_items]
    vals = [item[1] for item in sorted_items]

    ax.bar(behaviors, vals)
    ax.set_xlabel("Behavior")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by Behavior")

    if len(behaviors) > 5:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.grid(True, alpha=0.3, axis="y")

    return ax


def plot_pupil_behavior_timeline(
    tobii_df: pd.DataFrame,
    boris_df: pd.DataFrame,
    ax: Axes | None = None,
    tobii_time_col: str = "Recording timestamp",
    behavior_col: str = "Behavior",
    boris_start_col: str = "Start (s)",
    boris_stop_col: str = "Stop (s)",
    figsize: tuple[float, float] = (14, 6),
) -> Axes:
    """Combined plot: pupil timeseries with behavior annotations overlaid.

    Args:
        tobii_df: Tobii recording DataFrame.
        boris_df: Boris aggregated events DataFrame.
        ax: Matplotlib axes.
        tobii_time_col: Column name for Tobii timestamps.
        behavior_col: Column name for Boris behavior labels.
        boris_start_col: Column name for Boris event start times.
        boris_stop_col: Column name for Boris event stop times.
        figsize: Figure size.

    Returns:
        Matplotlib Axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Plot pupil timeseries
    time_s = tobii_to_seconds(tobii_df, tobii_time_col)

    if "Pupil diameter left" in tobii_df.columns:
        ax.plot(
            time_s, tobii_df["Pupil diameter left"], label="Pupil (L)", alpha=0.7, linewidth=0.5
        )
    if "Pupil diameter right" in tobii_df.columns:
        ax.plot(
            time_s, tobii_df["Pupil diameter right"], label="Pupil (R)", alpha=0.7, linewidth=0.5
        )

    # Add behavior annotations as colored spans
    behaviors = boris_df[behavior_col].unique()
    cmap = plt.get_cmap("Set3")
    colors = [cmap(i) for i in np.linspace(0, 1, len(behaviors))]
    color_map = dict(zip(behaviors, colors, strict=False))

    for _, event in boris_df.iterrows():
        behavior = event[behavior_col]
        start = event[boris_start_col]
        stop = event[boris_stop_col]

        ax.axvspan(start, stop, alpha=0.3, color=color_map[behavior], label=behavior)

    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pupil Diameter (mm)")
    ax.set_title("Pupil Diameter with Behavioral Annotations")
    ax.grid(True, alpha=0.3)

    return ax


def plot_cross_modal_summary(
    tobii_df: pd.DataFrame,
    boris_df: pd.DataFrame,
    behavior_col: str = "Behavior",
    figsize: tuple[float, float] = (14, 10),
) -> Figure:
    """Multi-panel summary of cross-modal relationships.

    Panels: behavior timeline with pupil overlay, gaze by behavior bars,
            pupil by behavior bars, validation info.

    Args:
        tobii_df: Tobii recording DataFrame.
        boris_df: Boris aggregated events DataFrame.
        behavior_col: Column name for Boris behavior labels.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # Timeline with pupil overlay (top, full width)
    ax1 = fig.add_subplot(gs[0, :])
    plot_pupil_behavior_timeline(tobii_df, boris_df, ax=ax1, behavior_col=behavior_col)

    # Gaze dispersion by behavior (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    plot_gaze_by_behavior(
        tobii_df, boris_df, ax=ax2, metric="dispersion", behavior_col=behavior_col
    )

    # Pupil by behavior (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    plot_gaze_by_behavior(tobii_df, boris_df, ax=ax3, metric="pupil", behavior_col=behavior_col)

    # Validity by behavior (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    plot_gaze_by_behavior(tobii_df, boris_df, ax=ax4, metric="validity", behavior_col=behavior_col)

    # Summary text (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")

    # Validation and summary info
    validation = validate_alignment(tobii_df, boris_df)

    gaze_metrics = compute_gaze_per_behavior(tobii_df, boris_df, behavior_col=behavior_col)

    summary_text = f"""Cross-Modal Summary

Alignment:
  Boris duration: {validation["boris_duration"]:.1f}s
  Tobii duration: {validation["tobii_duration"]:.1f}s
  Overlap: {validation["overlap_pct"]:.1f}%

Behaviors analyzed: {len(gaze_metrics)}
"""

    if validation["warnings"]:
        summary_text += "\nWarnings:\n"
        for warning in validation["warnings"][:3]:
            summary_text += f"  - {warning[:50]}...\n"

    ax5.text(
        0.1,
        0.9,
        summary_text,
        transform=ax5.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    return fig
