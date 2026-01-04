"""Visualization functions for eye-tracking data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .metrics import (
    compute_fixation_durations,
    compute_fixation_stats,
    compute_pupil_stats,
    compute_validity_rate,
)

# Default screen size
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080


# =============================================================================
# Helper Functions
# =============================================================================


def _get_or_create_axes(ax: Axes | None, figsize: tuple[float, float] = (10, 6)) -> Axes:
    """Get existing axes or create new figure with axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _add_screen_bounds(
    ax: Axes,
    screen_width: int = SCREEN_WIDTH,
    screen_height: int = SCREEN_HEIGHT,
) -> None:
    """Add screen boundary rectangle to axes."""
    rect = Rectangle(
        (0, 0),
        screen_width,
        screen_height,
        fill=False,
        edgecolor="gray",
        linestyle="--",
        linewidth=1,
    )
    ax.add_patch(rect)


# =============================================================================
# Gaze Visualizations
# =============================================================================


def plot_gaze_scatter(
    df: pd.DataFrame,
    ax: Axes | None = None,
    alpha: float = 0.3,
    color: str = "blue",
    screen_size: tuple[int, int] = (SCREEN_WIDTH, SCREEN_HEIGHT),
    show_screen_bounds: bool = True,
    s: float = 1,
) -> Axes:
    """Scatter plot of gaze points on screen coordinates.

    Args:
        df: Input DataFrame with gaze columns
        ax: Matplotlib axes. If None, creates new figure.
        alpha: Point transparency (0.0 to 1.0)
        color: Point color
        screen_size: (width, height) for screen bounds
        show_screen_bounds: Draw rectangle showing screen edges
        s: Point size

    Returns:
        Matplotlib Axes object
    """
    ax = _get_or_create_axes(ax)

    if "Gaze point X" in df.columns and "Gaze point Y" in df.columns:
        x = df["Gaze point X"].dropna()
        y = df["Gaze point Y"].dropna()

        # Align indices
        valid_idx = x.index.intersection(y.index)
        x = x.loc[valid_idx]
        y = y.loc[valid_idx]

        ax.scatter(x, y, alpha=alpha, c=color, s=s)

    if show_screen_bounds:
        _add_screen_bounds(ax, screen_size[0], screen_size[1])

    ax.set_xlim(-50, screen_size[0] + 50)
    ax.set_ylim(screen_size[1] + 50, -50)  # Invert Y axis (screen coordinates)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title("Gaze Points")
    ax.set_aspect("equal")

    return ax


def plot_gaze_heatmap(
    df: pd.DataFrame,
    bins: int = 50,
    cmap: str = "hot",
    ax: Axes | None = None,
    screen_size: tuple[int, int] = (SCREEN_WIDTH, SCREEN_HEIGHT),
) -> Axes:
    """2D histogram heatmap of gaze density.

    Args:
        df: Input DataFrame with gaze columns
        bins: Number of bins for histogram
        cmap: Colormap name
        ax: Matplotlib axes
        screen_size: Screen dimensions

    Returns:
        Matplotlib Axes object
    """
    ax = _get_or_create_axes(ax)

    if "Gaze point X" in df.columns and "Gaze point Y" in df.columns:
        x = df["Gaze point X"].dropna()
        y = df["Gaze point Y"].dropna()

        # Align indices
        valid_idx = x.index.intersection(y.index)
        x = x.loc[valid_idx]
        y = y.loc[valid_idx]

        if len(x) > 0:
            h, _, _ = np.histogram2d(
                x,
                y,
                bins=bins,
                range=[[0, screen_size[0]], [0, screen_size[1]]],
            )

            # Plot heatmap
            im = ax.imshow(
                h.T,
                origin="upper",
                extent=[0, screen_size[0], screen_size[1], 0],
                cmap=cmap,
                aspect="auto",
            )
            plt.colorbar(im, ax=ax, label="Count")

    ax.set_xlim(0, screen_size[0])
    ax.set_ylim(screen_size[1], 0)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title("Gaze Heatmap")

    return ax


def plot_gaze_trajectory(
    df: pd.DataFrame,
    max_samples: int = 1000,
    ax: Axes | None = None,
    color: str = "blue",
    linewidth: float = 0.5,
    screen_size: tuple[int, int] = (SCREEN_WIDTH, SCREEN_HEIGHT),
    show_screen_bounds: bool = True,
) -> Axes:
    """Line plot showing gaze path over time.

    Args:
        df: Input DataFrame with gaze columns
        max_samples: Maximum samples to plot (for performance)
        ax: Matplotlib axes
        color: Line color
        linewidth: Line width
        screen_size: Screen dimensions
        show_screen_bounds: Draw rectangle showing screen edges

    Returns:
        Matplotlib Axes object
    """
    ax = _get_or_create_axes(ax)

    if "Gaze point X" in df.columns and "Gaze point Y" in df.columns:
        # Sample if too many points
        if len(df) > max_samples:
            df = df.iloc[:: len(df) // max_samples]

        x = df["Gaze point X"].values
        y = df["Gaze point Y"].values

        # Remove NaN segments
        valid = ~(np.isnan(x) | np.isnan(y))

        # Create line segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Only keep segments where both points are valid
        valid_segments = valid[:-1] & valid[1:]
        segments = segments[valid_segments]

        if len(segments) > 0:
            lc = LineCollection(segments, colors=color, linewidths=linewidth, alpha=0.5)
            ax.add_collection(lc)

    if show_screen_bounds:
        _add_screen_bounds(ax, screen_size[0], screen_size[1])

    ax.set_xlim(-50, screen_size[0] + 50)
    ax.set_ylim(screen_size[1] + 50, -50)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title("Gaze Trajectory")
    ax.set_aspect("equal")

    return ax


def plot_scanpath(
    df: pd.DataFrame,
    ax: Axes | None = None,
    fixation_scale: float = 1.0,
    saccade_color: str = "gray",
    fixation_color: str = "blue",
    screen_size: tuple[int, int] = (SCREEN_WIDTH, SCREEN_HEIGHT),
    show_screen_bounds: bool = True,
    max_fixations: int = 100,
) -> Axes:
    """Scanpath visualization with fixation circles and saccade lines.

    Fixation circles sized by duration, connected by saccade lines.

    Args:
        df: Input DataFrame with eye movement data
        ax: Matplotlib axes
        fixation_scale: Scale factor for fixation circle sizes
        saccade_color: Color for saccade lines
        fixation_color: Color for fixation circles
        screen_size: Screen dimensions
        show_screen_bounds: Draw rectangle showing screen edges
        max_fixations: Maximum fixations to display

    Returns:
        Matplotlib Axes object
    """
    ax = _get_or_create_axes(ax)

    if "Eye movement type" not in df.columns or "Eye movement type index" not in df.columns:
        ax.set_title("Scanpath (no fixation data)")
        return ax

    # Get fixation data
    fixations = df[df["Eye movement type"] == "Fixation"].copy()

    if len(fixations) == 0:
        ax.set_title("Scanpath (no fixations)")
        return ax

    # Get unique fixations with their positions and durations
    if "Fixation point X" in df.columns and "Fixation point Y" in df.columns:
        x_col, y_col = "Fixation point X", "Fixation point Y"
    else:
        x_col, y_col = "Gaze point X", "Gaze point Y"

    fixation_data = (
        fixations.groupby("Eye movement type index")
        .agg(
            x=(x_col, "mean"),
            y=(y_col, "mean"),
            duration=("Gaze event duration", "first"),
        )
        .dropna()
        .head(max_fixations)
    )

    if len(fixation_data) == 0:
        ax.set_title("Scanpath (no valid fixations)")
        return ax

    # Plot saccade lines
    x_vals = fixation_data["x"].values
    y_vals = fixation_data["y"].values
    ax.plot(x_vals, y_vals, color=saccade_color, linewidth=0.5, alpha=0.5, zorder=1)

    # Plot fixation circles (sized by duration)
    sizes = fixation_data["duration"].values * fixation_scale
    sizes = np.clip(sizes, 10, 500)  # Limit size range

    ax.scatter(
        x_vals,
        y_vals,
        s=sizes,
        c=fixation_color,
        alpha=0.6,
        zorder=2,
        edgecolors="white",
        linewidths=0.5,
    )

    if show_screen_bounds:
        _add_screen_bounds(ax, screen_size[0], screen_size[1])

    ax.set_xlim(-50, screen_size[0] + 50)
    ax.set_ylim(screen_size[1] + 50, -50)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title(f"Scanpath ({len(fixation_data)} fixations)")
    ax.set_aspect("equal")

    return ax


# =============================================================================
# Pupil Visualizations
# =============================================================================


def plot_pupil_timeseries(
    df: pd.DataFrame,
    eye: str = "both",
    ax: Axes | None = None,
    timestamp_col: str = "Recording timestamp",
) -> Axes:
    """Plot pupil diameter over recording time.

    Args:
        df: Input DataFrame with pupil columns
        eye: "left", "right", or "both"
        ax: Matplotlib axes
        timestamp_col: Timestamp column name

    Returns:
        Matplotlib Axes object
    """
    ax = _get_or_create_axes(ax)

    if timestamp_col not in df.columns:
        ax.set_title("Pupil Timeseries (no timestamp)")
        return ax

    # Convert timestamp to seconds
    time_s = (df[timestamp_col] - df[timestamp_col].iloc[0]) / 1_000_000

    if eye in ("left", "both") and "Pupil diameter left" in df.columns:
        ax.plot(time_s, df["Pupil diameter left"], label="Left", alpha=0.7, linewidth=0.5)

    if eye in ("right", "both") and "Pupil diameter right" in df.columns:
        ax.plot(time_s, df["Pupil diameter right"], label="Right", alpha=0.7, linewidth=0.5)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Pupil Diameter (mm)")
    ax.set_title("Pupil Diameter Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_pupil_distribution(
    df: pd.DataFrame,
    ax: Axes | None = None,
    bins: int = 50,
) -> Axes:
    """Histogram of pupil diameter values.

    Args:
        df: Input DataFrame with pupil columns
        ax: Matplotlib axes
        bins: Number of histogram bins

    Returns:
        Matplotlib Axes object
    """
    ax = _get_or_create_axes(ax)

    if "Pupil diameter left" in df.columns:
        left = df["Pupil diameter left"].dropna()
        if len(left) > 0:
            ax.hist(left, bins=bins, alpha=0.5, label="Left", density=True)

    if "Pupil diameter right" in df.columns:
        right = df["Pupil diameter right"].dropna()
        if len(right) > 0:
            ax.hist(right, bins=bins, alpha=0.5, label="Right", density=True)

    ax.set_xlabel("Pupil Diameter (mm)")
    ax.set_ylabel("Density")
    ax.set_title("Pupil Diameter Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_pupil_comparison(
    df_list: list[pd.DataFrame],
    labels: list[str],
    ax: Axes | None = None,
) -> Axes:
    """Box plots comparing pupil diameter across recordings.

    Args:
        df_list: List of DataFrames to compare
        labels: Labels for each DataFrame
        ax: Matplotlib axes

    Returns:
        Matplotlib Axes object
    """
    ax = _get_or_create_axes(ax)

    data = []
    for df in df_list:
        values = []
        if "Pupil diameter left" in df.columns:
            values.extend(df["Pupil diameter left"].dropna().tolist())
        if "Pupil diameter right" in df.columns:
            values.extend(df["Pupil diameter right"].dropna().tolist())
        data.append(values)

    ax.boxplot(data, labels=labels)
    ax.set_ylabel("Pupil Diameter (mm)")
    ax.set_title("Pupil Diameter Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    return ax


# =============================================================================
# Eye Movement Visualizations
# =============================================================================


def plot_fixation_durations(
    df: pd.DataFrame,
    bins: int = 30,
    ax: Axes | None = None,
) -> Axes:
    """Histogram of fixation duration distribution.

    Args:
        df: Input DataFrame with eye movement data
        bins: Number of histogram bins
        ax: Matplotlib axes

    Returns:
        Matplotlib Axes object
    """
    ax = _get_or_create_axes(ax)

    durations = compute_fixation_durations(df)

    if len(durations) > 0:
        ax.hist(durations, bins=bins, edgecolor="black", alpha=0.7)

        # Add statistics
        stats = compute_fixation_stats(df)
        ax.axvline(
            stats["mean_duration"],
            color="red",
            linestyle="--",
            label=f"Mean: {stats['mean_duration']:.0f}ms",
        )

    ax.set_xlabel("Duration (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Fixation Duration Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_fixation_heatmap(
    df: pd.DataFrame,
    bins: int = 50,
    ax: Axes | None = None,
    weight_by_duration: bool = True,
    screen_size: tuple[int, int] = (SCREEN_WIDTH, SCREEN_HEIGHT),
    cmap: str = "YlOrRd",
) -> Axes:
    """Heatmap of fixation locations, optionally weighted by duration.

    Args:
        df: Input DataFrame with fixation data
        bins: Number of bins for histogram
        ax: Matplotlib axes
        weight_by_duration: Weight by fixation duration
        screen_size: Screen dimensions
        cmap: Colormap name

    Returns:
        Matplotlib Axes object
    """
    ax = _get_or_create_axes(ax)

    if "Eye movement type" not in df.columns:
        ax.set_title("Fixation Heatmap (no data)")
        return ax

    fixations = df[df["Eye movement type"] == "Fixation"]

    if "Fixation point X" in df.columns and "Fixation point Y" in df.columns:
        x_col, y_col = "Fixation point X", "Fixation point Y"
    else:
        x_col, y_col = "Gaze point X", "Gaze point Y"

    if x_col not in fixations.columns or y_col not in fixations.columns:
        ax.set_title("Fixation Heatmap (no position data)")
        return ax

    x = fixations[x_col].dropna()
    y = fixations[y_col].dropna()

    # Align indices
    valid_idx = x.index.intersection(y.index)
    x = x.loc[valid_idx]
    y = y.loc[valid_idx]

    if len(x) == 0:
        ax.set_title("Fixation Heatmap (no valid data)")
        return ax

    weights = None
    if weight_by_duration and "Gaze event duration" in fixations.columns:
        weights = fixations.loc[valid_idx, "Gaze event duration"].values

    h, _, _ = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[[0, screen_size[0]], [0, screen_size[1]]],
        weights=weights,
    )

    im = ax.imshow(
        h.T,
        origin="upper",
        extent=[0, screen_size[0], screen_size[1], 0],
        cmap=cmap,
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, label="Duration (ms)" if weight_by_duration else "Count")

    ax.set_xlim(0, screen_size[0])
    ax.set_ylim(screen_size[1], 0)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title("Fixation Heatmap" + (" (duration-weighted)" if weight_by_duration else ""))

    return ax


def plot_eye_movement_timeline(
    df: pd.DataFrame,
    ax: Axes | None = None,
    timestamp_col: str = "Recording timestamp",
) -> Axes:
    """Timeline colored by eye movement type.

    Shows Fixation, Saccade, EyesNotFound, Unclassified as colored segments.

    Args:
        df: Input DataFrame with eye movement columns
        ax: Matplotlib axes
        timestamp_col: Timestamp column name

    Returns:
        Matplotlib Axes object
    """
    ax = _get_or_create_axes(ax, figsize=(12, 3))

    if "Eye movement type" not in df.columns or timestamp_col not in df.columns:
        ax.set_title("Eye Movement Timeline (no data)")
        return ax

    # Convert timestamp to seconds
    time_s = (df[timestamp_col] - df[timestamp_col].iloc[0]) / 1_000_000

    # Color mapping
    color_map = {
        "Fixation": "blue",
        "Saccade": "red",
        "EyesNotFound": "gray",
        "Unclassified": "orange",
    }

    # Plot each eye movement type
    for movement_type, color in color_map.items():
        mask = df["Eye movement type"] == movement_type
        if mask.any():
            ax.scatter(
                time_s[mask],
                [movement_type] * mask.sum(),
                c=color,
                s=1,
                alpha=0.5,
                label=movement_type,
            )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Movement Type")
    ax.set_title("Eye Movement Timeline")
    ax.legend(loc="upper right", markerscale=5)
    ax.grid(True, alpha=0.3, axis="x")

    return ax


# =============================================================================
# Summary Visualization
# =============================================================================


def plot_recording_summary(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (14, 10),
) -> Figure:
    """Create multi-panel summary figure for a recording.

    Creates a figure with:
    - Gaze heatmap
    - Scanpath
    - Pupil timeseries
    - Fixation duration histogram
    - Eye movement timeline
    - Key metrics text

    Args:
        df: Input DataFrame (cleaned eye tracker data)
        figsize: Figure size in inches

    Returns:
        Matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)

    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Gaze heatmap (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_gaze_heatmap(df, ax=ax1)

    # Scanpath (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_scanpath(df, ax=ax2)

    # Fixation heatmap (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    plot_fixation_heatmap(df, ax=ax3)

    # Pupil timeseries (middle, full width)
    ax4 = fig.add_subplot(gs[1, :2])
    plot_pupil_timeseries(df, ax=ax4)

    # Fixation duration histogram (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    plot_fixation_durations(df, ax=ax5)

    # Eye movement timeline (bottom, 2/3 width)
    ax6 = fig.add_subplot(gs[2, :2])
    plot_eye_movement_timeline(df, ax=ax6)

    # Metrics text (bottom right)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis("off")

    # Compute and display metrics
    validity = compute_validity_rate(df)
    pupil_stats = compute_pupil_stats(df)
    fixation_stats = compute_fixation_stats(df)

    metrics_text = f"""Recording Summary

Validity Rate: {validity:.1%}

Pupil Diameter:
  Mean: {pupil_stats["mean"]:.2f} mm
  Left: {pupil_stats["left_mean"]:.2f} mm
  Right: {pupil_stats["right_mean"]:.2f} mm

Fixations:
  Count: {fixation_stats["count"]}
  Mean Duration: {fixation_stats["mean_duration"]:.0f} ms
  Total Time: {fixation_stats["total_fixation_time"]:.0f} ms

Samples: {len(df):,}
"""

    ax7.text(
        0.1,
        0.9,
        metrics_text,
        transform=ax7.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    return fig
