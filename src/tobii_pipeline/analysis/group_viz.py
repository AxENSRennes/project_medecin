"""Group-level visualization functions for eye-tracking analysis.

Provides aggregate visualizations for comparing Patient vs Control groups,
including heatmaps, event distributions, and longitudinal trends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .pub_plots import GROUP_COLORS, add_panel_label, add_significance_bar, despine
from .stats import compute_confidence_interval, mannwhitneyu_groups

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# =============================================================================
# Aggregate Heatmap Functions
# =============================================================================


def compute_group_heatmap(
    recordings: list[tuple[pd.DataFrame, dict]],
    group: str,
    width: int = 1920,
    height: int = 1080,
    n_bins: int = 100,
    sigma: float = 5.0,
) -> np.ndarray:
    """Compute aggregate gaze heatmap for a group.

    Aggregates gaze data from all participants in the specified group,
    normalizing each participant's contribution to avoid bias from
    recording length differences.

    Args:
        recordings: List of (DataFrame, metadata) tuples
        group: Group to filter ("Patient" or "Control")
        width: Screen width in pixels
        height: Screen height in pixels
        n_bins: Number of bins in each dimension
        sigma: Gaussian smoothing sigma (in bins)

    Returns:
        2D numpy array (n_bins x n_bins) with normalized density values
    """
    bins_x = np.linspace(0, width, n_bins + 1)
    bins_y = np.linspace(0, height, n_bins + 1)

    combined_heatmap = np.zeros((n_bins, n_bins))
    n_participants = 0

    for df, metadata in recordings:
        if metadata.get("group") != group:
            continue

        # Extract valid gaze points
        gaze_x = df.get("Gaze point X")
        gaze_y = df.get("Gaze point Y")

        if gaze_x is None or gaze_y is None:
            continue

        # Filter valid values within screen bounds
        valid_mask = (
            gaze_x.notna()
            & gaze_y.notna()
            & (gaze_x >= 0)
            & (gaze_x <= width)
            & (gaze_y >= 0)
            & (gaze_y <= height)
        )

        gaze_x = gaze_x[valid_mask].values
        gaze_y = gaze_y[valid_mask].values

        if len(gaze_x) == 0:
            continue

        # Create 2D histogram for this participant
        hist, _, _ = np.histogram2d(gaze_x, gaze_y, bins=[bins_x, bins_y])

        # Normalize participant contribution (max = 1)
        if hist.max() > 0:
            hist = hist / hist.max()

        # Transpose to match image coordinates (y increases downward)
        combined_heatmap += hist.T
        n_participants += 1

    # Normalize by number of participants
    if n_participants > 0:
        combined_heatmap /= n_participants

    # Apply Gaussian smoothing
    if sigma > 0:
        combined_heatmap = gaussian_filter(combined_heatmap, sigma=sigma)

    return combined_heatmap


def plot_group_heatmap_comparison(
    recordings: list[tuple[pd.DataFrame, dict]],
    width: int = 1920,
    height: int = 1080,
    figsize: tuple[float, float] = (10, 4),
    cmap_groups: str = "hot",
    cmap_diff: str = "RdBu_r",
) -> Figure:
    """Create side-by-side heatmap comparison: Patient | Control | Difference.

    Args:
        recordings: List of (DataFrame, metadata) tuples
        width: Screen width in pixels
        height: Screen height in pixels
        figsize: Figure size
        cmap_groups: Colormap for group heatmaps
        cmap_diff: Colormap for difference heatmap (diverging)

    Returns:
        Matplotlib Figure object
    """
    # Compute heatmaps
    patient_heatmap = compute_group_heatmap(recordings, "Patient", width, height)
    control_heatmap = compute_group_heatmap(recordings, "Control", width, height)
    diff_heatmap = patient_heatmap - control_heatmap

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Common extent for all plots
    extent = [0, width, height, 0]

    # Patient heatmap
    vmax_groups = max(patient_heatmap.max(), control_heatmap.max())
    im1 = axes[0].imshow(
        patient_heatmap,
        extent=extent,
        cmap=cmap_groups,
        vmin=0,
        vmax=vmax_groups,
        aspect="auto",
    )
    axes[0].set_title("Patient", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("X (pixels)")
    axes[0].set_ylabel("Y (pixels)")

    # Control heatmap
    im2 = axes[1].imshow(
        control_heatmap,
        extent=extent,
        cmap=cmap_groups,
        vmin=0,
        vmax=vmax_groups,
        aspect="auto",
    )
    axes[1].set_title("Control", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("X (pixels)")
    axes[1].set_ylabel("")

    # Difference heatmap (symmetric around 0)
    vmax_diff = max(abs(diff_heatmap.min()), abs(diff_heatmap.max()))
    if vmax_diff == 0:
        vmax_diff = 1  # Avoid zero range

    im3 = axes[2].imshow(
        diff_heatmap,
        extent=extent,
        cmap=cmap_diff,
        vmin=-vmax_diff,
        vmax=vmax_diff,
        aspect="auto",
    )
    axes[2].set_title("Difference (P - C)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("X (pixels)")
    axes[2].set_ylabel("")

    # Add colorbars
    fig.colorbar(im1, ax=axes[0], shrink=0.8, label="Density")
    fig.colorbar(im2, ax=axes[1], shrink=0.8, label="Density")
    fig.colorbar(im3, ax=axes[2], shrink=0.8, label="Diff")

    # Add panel labels
    for ax, label in zip(axes, ["A", "B", "C"], strict=True):
        add_panel_label(ax, label)

    fig.tight_layout()
    return fig


# =============================================================================
# Longitudinal Visualization with Confidence Intervals
# =============================================================================


def plot_longitudinal_ci(
    summary_df: pd.DataFrame,
    metric: str,
    label: str,
    confidence: float = 0.95,
    ax: Axes | None = None,
    months: list[int] | None = None,
) -> Axes:
    """Plot longitudinal trends with confidence intervals.

    Args:
        summary_df: DataFrame with columns: group, month, and metric
        metric: Column name for the metric to plot
        label: Y-axis label for the metric
        confidence: Confidence level (default 0.95 for 95% CI)
        ax: Matplotlib axes (created if None)
        months: List of month values (default: [0, 12, 24, 36])

    Returns:
        Matplotlib Axes object
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    if months is None:
        months = [0, 12, 24, 36]

    for group, color in GROUP_COLORS.items():
        group_data = summary_df[summary_df["group"] == group]

        means = []
        ci_lower = []
        ci_upper = []
        ns = []

        for month in months:
            month_data = group_data[group_data["month"] == month][metric].dropna()

            if len(month_data) >= 2:
                mean = month_data.mean()
                ci_low, ci_high = compute_confidence_interval(month_data, confidence)
                means.append(mean)
                ci_lower.append(ci_low)
                ci_upper.append(ci_high)
                ns.append(len(month_data))
            else:
                means.append(np.nan)
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)
                ns.append(0)

        means = np.array(means)
        ci_lower = np.array(ci_lower)
        ci_upper = np.array(ci_upper)

        # Plot line with markers
        ax.plot(
            months,
            means,
            "o-",
            color=color,
            label=group,
            linewidth=2,
            markersize=8,
        )

        # Fill confidence interval
        ax.fill_between(
            months,
            ci_lower,
            ci_upper,
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel("Timepoint", fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_xticks(months)
    ax.set_xticklabels([f"M{m}" for m in months])
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    despine(ax)

    return ax


def create_longitudinal_figure(
    summary_df: pd.DataFrame,
    metrics: list[tuple[str, str]],
    figsize: tuple[float, float] = (10, 8),
) -> Figure:
    """Create multi-panel longitudinal figure with CI.

    Args:
        summary_df: DataFrame with metric columns
        metrics: List of (metric_column, label) tuples
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    n_metrics = len(metrics)
    ncols = 2
    nrows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, (metric, label) in enumerate(metrics):
        plot_longitudinal_ci(summary_df, metric, label, ax=axes[i])
        add_panel_label(axes[i], chr(65 + i))  # A, B, C, ...

    # Hide unused axes
    for j in range(len(metrics), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    return fig


# =============================================================================
# Event Distribution Visualizations
# =============================================================================


def plot_metric_violin(
    summary_df: pd.DataFrame,
    metric: str,
    label: str,
    ax: Axes | None = None,
    show_points: bool = True,
    show_stats: bool = True,
) -> Axes:
    """Create violin plot comparing metric between groups.

    Args:
        summary_df: DataFrame with group and metric columns
        metric: Column name for the metric
        label: Y-axis label
        ax: Matplotlib axes (created if None)
        show_points: Overlay individual data points
        show_stats: Show significance annotation

    Returns:
        Matplotlib Axes object
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 5))

    patient_data = summary_df[summary_df["group"] == "Patient"][metric].dropna()
    control_data = summary_df[summary_df["group"] == "Control"][metric].dropna()

    data = [patient_data.values, control_data.values]

    # Create violin plot
    parts = ax.violinplot(data, positions=[0, 1], showmeans=True, showmedians=False)

    # Color the violins
    colors = [GROUP_COLORS["Patient"], GROUP_COLORS["Control"]]
    for pc, color in zip(parts["bodies"], colors, strict=True):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor("black")
        pc.set_linewidth(1)

    # Style other elements
    for partname in ["cmeans", "cmins", "cmaxs", "cbars"]:
        if partname in parts:
            parts[partname].set_color("black")
            parts[partname].set_linewidth(1)

    # Add individual points with jitter
    if show_points:
        for i, (d, color) in enumerate(zip(data, colors, strict=True)):
            jitter = np.random.normal(0, 0.05, len(d))
            ax.scatter(
                np.full(len(d), i) + jitter,
                d,
                alpha=0.4,
                s=20,
                color=color,
                edgecolor="none",
            )

    # Add significance annotation
    if show_stats and len(patient_data) >= 2 and len(control_data) >= 2:
        result = mannwhitneyu_groups(patient_data, control_data)
        p_value = result["p_value"]

        # Position significance bar
        y_max = max(patient_data.max(), control_data.max())
        y_range = y_max - min(patient_data.min(), control_data.min())
        add_significance_bar(ax, 0, 1, y_max + 0.05 * y_range, p_value)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Patient\n(n={len(patient_data)})", f"Control\n(n={len(control_data)})"])
    ax.set_ylabel(label, fontsize=11)
    despine(ax)

    return ax


def plot_multi_metric_comparison(
    summary_df: pd.DataFrame,
    metrics: list[tuple[str, str]],
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Create multi-panel violin plot figure.

    Args:
        summary_df: DataFrame with metric columns
        metrics: List of (metric_column, label) tuples
        figsize: Figure size (auto-calculated if None)

    Returns:
        Matplotlib Figure object
    """
    n_metrics = len(metrics)
    if figsize is None:
        figsize = (3 * n_metrics, 5)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    for i, (metric, label) in enumerate(metrics):
        plot_metric_violin(summary_df, metric, label, ax=axes[i])
        add_panel_label(axes[i], chr(65 + i))

    fig.tight_layout()
    return fig


# =============================================================================
# Combined Group Comparison Figure
# =============================================================================


def create_group_comparison_figure(
    summary_df: pd.DataFrame,
    recordings: list[tuple[pd.DataFrame, dict]],
    metrics: list[tuple[str, str]] | None = None,
    figsize: tuple[float, float] = (12, 10),
) -> Figure:
    """Create comprehensive group comparison figure.

    Layout:
    - Row 1: Aggregate heatmaps (Patient | Control | Difference)
    - Row 2-3: Key metric violin plots

    Args:
        summary_df: DataFrame with metric columns
        recordings: List of (DataFrame, metadata) tuples for heatmaps
        metrics: List of (metric_column, label) tuples for violin plots
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    if metrics is None:
        metrics = [
            ("gaze_dispersion", "Gaze Dispersion (px)"),
            ("pupil_variability", "Pupil Variability (CV)"),
            ("fixation_mean_duration", "Fixation Duration (ms)"),
            ("fixation_rate", "Fixation Rate (/s)"),
            ("saccade_rate", "Saccade Rate (/s)"),
            ("validity_rate", "Validity Rate"),
        ]

    # Create figure with gridspec
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1, 1], hspace=0.35, wspace=0.3)

    # Row 1: Heatmaps (spanning all columns)
    ax_heat = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # Compute and plot heatmaps
    patient_heatmap = compute_group_heatmap(recordings, "Patient")
    control_heatmap = compute_group_heatmap(recordings, "Control")
    diff_heatmap = patient_heatmap - control_heatmap

    extent = [0, 1920, 1080, 0]
    vmax_groups = max(patient_heatmap.max(), control_heatmap.max())

    ax_heat[0].imshow(
        patient_heatmap, extent=extent, cmap="hot", vmin=0, vmax=vmax_groups, aspect="auto"
    )
    ax_heat[0].set_title("Patient", fontsize=12, fontweight="bold")

    ax_heat[1].imshow(
        control_heatmap, extent=extent, cmap="hot", vmin=0, vmax=vmax_groups, aspect="auto"
    )
    ax_heat[1].set_title("Control", fontsize=12, fontweight="bold")

    vmax_diff = max(abs(diff_heatmap.min()), abs(diff_heatmap.max()), 0.01)
    ax_heat[2].imshow(
        diff_heatmap, extent=extent, cmap="RdBu_r", vmin=-vmax_diff, vmax=vmax_diff, aspect="auto"
    )
    ax_heat[2].set_title("Difference", fontsize=12, fontweight="bold")

    for ax in ax_heat:
        ax.set_xlabel("X (px)")
    ax_heat[0].set_ylabel("Y (px)")

    # Add panel labels to heatmaps
    add_panel_label(ax_heat[0], "A")
    add_panel_label(ax_heat[1], "B")
    add_panel_label(ax_heat[2], "C")

    # Rows 2-3: Violin plots
    n_metrics = min(len(metrics), 6)  # Max 6 metrics
    for i, (metric, label) in enumerate(metrics[:n_metrics]):
        row = 1 + i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        plot_metric_violin(summary_df, metric, label, ax=ax, show_points=True)
        add_panel_label(ax, chr(68 + i))  # D, E, F, ...

    return fig


# =============================================================================
# BORIS Behavior-Segmented Visualizations
# =============================================================================


def compute_group_metrics_by_behavior(
    tobii_boris_pairs: list[tuple[pd.DataFrame, pd.DataFrame, dict]],
    behaviors: list[str] | None = None,
    metrics_to_compute: list[str] | None = None,
) -> pd.DataFrame:
    """Compute eye-tracking metrics per behavior, aggregated by group.

    Args:
        tobii_boris_pairs: List of (tobii_df, boris_df, metadata) tuples
        behaviors: List of behaviors to analyze (None = all)
        metrics_to_compute: List of metrics ("pupil_mean", "gaze_dispersion", etc.)

    Returns:
        DataFrame with columns: behavior, group, metric, mean, ci_lower, ci_upper, n
    """
    from integration.cross_modal import compute_gaze_per_behavior, compute_pupil_per_behavior

    if metrics_to_compute is None:
        metrics_to_compute = ["pupil_mean", "gaze_dispersion", "validity_rate"]

    # Collect per-participant metrics
    participant_metrics = []

    for tobii_df, boris_df, metadata in tobii_boris_pairs:
        group = metadata.get("group", "Unknown")
        participant = metadata.get("participant", "Unknown")

        # Get available behaviors if not specified
        if behaviors is None:
            behavior_col = "Behavior"
            if behavior_col in boris_df.columns:
                current_behaviors = boris_df[behavior_col].unique().tolist()
            else:
                continue
        else:
            current_behaviors = behaviors

        # Compute metrics per behavior
        gaze_metrics = compute_gaze_per_behavior(tobii_df, boris_df, behaviors=current_behaviors)
        pupil_metrics = compute_pupil_per_behavior(tobii_df, boris_df, behaviors=current_behaviors)

        for behavior in current_behaviors:
            gaze = gaze_metrics.get(behavior, {})
            pupil = pupil_metrics.get(behavior, {})

            row = {
                "participant": participant,
                "group": group,
                "behavior": behavior,
                "pupil_mean": pupil.get("pupil_mean", np.nan),
                "gaze_dispersion": gaze.get("gaze_dispersion", np.nan),
                "validity_rate": gaze.get("validity_rate", np.nan),
            }
            participant_metrics.append(row)

    if not participant_metrics:
        return pd.DataFrame()

    metrics_df = pd.DataFrame(participant_metrics)

    # Aggregate by (behavior, group)
    results = []

    for behavior in metrics_df["behavior"].unique():
        beh_data = metrics_df[metrics_df["behavior"] == behavior]

        for group in ["Patient", "Control"]:
            group_data = beh_data[beh_data["group"] == group]

            for metric in metrics_to_compute:
                values = group_data[metric].dropna()
                stats = _compute_metric_stats(values, behavior, group, metric)
                results.append(stats)

    return pd.DataFrame(results)


def _compute_metric_stats(values: pd.Series, behavior: str, group: str, metric: str) -> dict:
    """Compute statistics for a single metric."""
    if len(values) >= 2:
        mean = values.mean()
        ci_low, ci_high = compute_confidence_interval(values)
        n = len(values)
    else:
        mean = values.mean() if len(values) > 0 else np.nan
        ci_low = ci_high = np.nan
        n = len(values)

    return {
        "behavior": behavior,
        "group": group,
        "metric": metric,
        "mean": mean,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "n": n,
    }


def plot_behavior_group_comparison(
    metrics_df: pd.DataFrame,
    metric: str,
    label: str,
    ax: Axes | None = None,
    max_behaviors: int = 8,
) -> Axes:
    """Create grouped bar plot comparing behaviors between Patient and Control.

    Args:
        metrics_df: DataFrame from compute_group_metrics_by_behavior
        metric: Metric to plot
        label: Y-axis label
        ax: Matplotlib axes (created if None)
        max_behaviors: Maximum number of behaviors to show

    Returns:
        Matplotlib Axes object
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    # Filter to specific metric
    data = metrics_df[metrics_df["metric"] == metric].copy()

    if len(data) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return ax

    # Get behaviors sorted by overall mean
    behavior_means = data.groupby("behavior")["mean"].mean().sort_values(ascending=False)
    behaviors = behavior_means.head(max_behaviors).index.tolist()

    # Bar positions
    x = np.arange(len(behaviors))
    width = 0.35

    for i, group in enumerate(["Patient", "Control"]):
        group_data = data[data["group"] == group]

        means = []
        errors_lower = []
        errors_upper = []

        for behavior in behaviors:
            beh_data = group_data[group_data["behavior"] == behavior]
            if len(beh_data) > 0:
                mean = beh_data["mean"].values[0]
                ci_low = beh_data["ci_lower"].values[0]
                ci_high = beh_data["ci_upper"].values[0]
                means.append(mean)
                errors_lower.append(mean - ci_low if not np.isnan(ci_low) else 0)
                errors_upper.append(ci_high - mean if not np.isnan(ci_high) else 0)
            else:
                means.append(0)
                errors_lower.append(0)
                errors_upper.append(0)

        offset = width / 2 if i == 0 else -width / 2
        ax.bar(
            x - offset,
            means,
            width,
            label=group,
            color=GROUP_COLORS[group],
            alpha=0.8,
            yerr=[errors_lower, errors_upper],
            capsize=3,
        )

    ax.set_xlabel("Behavior", fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_xticks(x)

    # Abbreviate long behavior names
    abbreviated = [b[:15] + "..." if len(b) > 15 else b for b in behaviors]
    ax.set_xticklabels(abbreviated, rotation=45, ha="right", fontsize=9)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    despine(ax)

    return ax


def create_behavioral_figure(
    tobii_boris_pairs: list[tuple[pd.DataFrame, pd.DataFrame, dict]],
    behaviors: list[str] | None = None,
    figsize: tuple[float, float] = (12, 8),
) -> Figure:
    """Create comprehensive behavioral analysis figure.

    Layout:
    - Row 1: Pupil metrics by behavior (Patient vs Control)
    - Row 2: Gaze dispersion and validity by behavior

    Args:
        tobii_boris_pairs: List of (tobii_df, boris_df, metadata) tuples
        behaviors: List of behaviors to analyze
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    # Compute metrics
    metrics_df = compute_group_metrics_by_behavior(
        tobii_boris_pairs,
        behaviors=behaviors,
        metrics_to_compute=["pupil_mean", "gaze_dispersion", "validity_rate"],
    )

    if len(metrics_df) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "No BORIS data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Pupil mean by behavior
    plot_behavior_group_comparison(metrics_df, "pupil_mean", "Pupil Diameter (mm)", ax=axes[0, 0])
    add_panel_label(axes[0, 0], "A")

    # Gaze dispersion by behavior
    plot_behavior_group_comparison(
        metrics_df, "gaze_dispersion", "Gaze Dispersion (px)", ax=axes[0, 1]
    )
    add_panel_label(axes[0, 1], "B")

    # Validity rate by behavior
    plot_behavior_group_comparison(metrics_df, "validity_rate", "Validity Rate", ax=axes[1, 0])
    add_panel_label(axes[1, 0], "C")

    # Summary statistics panel
    axes[1, 1].axis("off")

    # Count participants and behaviors
    n_patients = metrics_df[metrics_df["group"] == "Patient"]["n"].sum()
    n_controls = metrics_df[metrics_df["group"] == "Control"]["n"].sum()
    n_behaviors = metrics_df["behavior"].nunique()

    summary_text = f"""Behavioral Analysis Summary

Behaviors analyzed: {n_behaviors}
Patient observations: {n_patients}
Control observations: {n_controls}

Error bars show 95% CI
"""

    axes[1, 1].text(
        0.1,
        0.9,
        summary_text,
        transform=axes[1, 1].transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )
    add_panel_label(axes[1, 1], "D")

    fig.tight_layout()
    return fig
