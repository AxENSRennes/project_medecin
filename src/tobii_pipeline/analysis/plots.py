"""Visualization functions for eye-tracking data.

Heatmap visualizations use MNE-Python for smoother, publication-quality plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from tobii_pipeline.adapters.mne_adapter import (
    plot_gaze_heatmap as mne_plot_gaze_heatmap,
)
from tobii_pipeline.adapters.mne_adapter import (
    plot_gaze_on_stimulus,
)

from .metrics import (
    compute_events,
    compute_pupil_stats,
    compute_validity_rate,
    get_fixation_stats,
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
    sigma: float = 50.0,
    cmap: str = "hot",
    ax: Axes | None = None,
    screen_size: tuple[int, int] = (SCREEN_WIDTH, SCREEN_HEIGHT),
    vlim: tuple[float | None, float | None] | None = None,
) -> tuple[Figure, Axes]:
    """Gaze density heatmap using MNE-style visualization with Gaussian smoothing.

    Args:
        df: Input DataFrame with gaze columns
        sigma: Gaussian smoothing parameter (higher = smoother)
        cmap: Colormap name
        ax: Matplotlib axes
        screen_size: Screen dimensions (width, height)
        vlim: Value limits (min, max) for color scaling.
            Use (min_val, None) to make low values transparent.

    Returns:
        Tuple of (Figure, Axes) with the heatmap
    """
    return mne_plot_gaze_heatmap(
        df,
        width=screen_size[0],
        height=screen_size[1],
        sigma=sigma,
        cmap=cmap,
        ax=ax,
        vlim=vlim,
    )


def plot_gaze_on_image(
    df: pd.DataFrame,
    stimulus_path: str | Path,
    sigma: float = 50.0,
    cmap: str = "hot",
    alpha: float = 0.6,
    vlim: tuple[float | None, float | None] = (0.1, None),
) -> tuple[Figure, Axes]:
    """Overlay gaze heatmap on a stimulus image.

    Args:
        df: Input DataFrame with gaze columns
        stimulus_path: Path to the stimulus image file
        sigma: Gaussian smoothing parameter
        cmap: Colormap name
        alpha: Heatmap transparency (0-1)
        vlim: Value limits for color scaling

    Returns:
        Tuple of (Figure, Axes) with the overlay
    """
    return plot_gaze_on_stimulus(
        df,
        stimulus_path=stimulus_path,
        sigma=sigma,
        cmap=cmap,
        alpha=alpha,
        vlim=vlim,
    )


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
    events_df: pd.DataFrame | None = None,
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
    Uses pymovements-detected events if available.

    Args:
        df: Input DataFrame with gaze data
        events_df: Pre-computed events from pymovements. If None, detects events.
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

    # Try to get events from pymovements
    if events_df is None:
        try:
            events_df = compute_events(df)
        except Exception:
            events_df = pd.DataFrame()

    # Filter to fixations
    if len(events_df) > 0 and "name" in events_df.columns:
        fixations = events_df[events_df["name"].str.contains("fixation", case=False)]

        if len(fixations) > 0 and "onset" in fixations.columns:
            # Get fixation positions from gaze data at onset times
            # This is a simplified approach - ideally we'd use fixation centers
            fixation_data = []

            for _, fix in fixations.head(max_fixations).iterrows():
                onset_ms = fix["onset"]
                # Find closest timestamp in df
                if "Recording timestamp" in df.columns:
                    time_ms = df["Recording timestamp"] / 1000
                    idx = (time_ms - onset_ms).abs().idxmin()
                    x = df.loc[idx, "Gaze point X"] if "Gaze point X" in df.columns else np.nan
                    y = df.loc[idx, "Gaze point Y"] if "Gaze point Y" in df.columns else np.nan
                    duration = fix.get("duration", 100)
                    if not np.isnan(x) and not np.isnan(y):
                        fixation_data.append({"x": x, "y": y, "duration": duration})

            if fixation_data:
                fix_df = pd.DataFrame(fixation_data)
                x_vals = fix_df["x"].values
                y_vals = fix_df["y"].values

                # Plot saccade lines
                ax.plot(x_vals, y_vals, color=saccade_color, linewidth=0.5, alpha=0.5, zorder=1)

                # Plot fixation circles (sized by duration)
                sizes = fix_df["duration"].values * fixation_scale
                sizes = np.clip(sizes, 10, 500)

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

                ax.set_title(f"Scanpath ({len(fix_df)} fixations)")
            else:
                ax.set_title("Scanpath (no valid fixation positions)")
        else:
            ax.set_title("Scanpath (no fixations detected)")
    else:
        ax.set_title("Scanpath (event detection failed)")

    if show_screen_bounds:
        _add_screen_bounds(ax, screen_size[0], screen_size[1])

    ax.set_xlim(-50, screen_size[0] + 50)
    ax.set_ylim(screen_size[1] + 50, -50)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
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
    smooth_window: int | None = 50,
) -> Axes:
    """Plot pupil diameter over recording time.

    Args:
        df: Input DataFrame with pupil columns
        eye: "left", "right", or "both"
        ax: Matplotlib axes
        timestamp_col: Timestamp column name
        smooth_window: Rolling average window size (samples).
            Set to None for raw data. Default 50 (~0.5s at 100Hz).

    Returns:
        Matplotlib Axes object
    """
    ax = _get_or_create_axes(ax)

    if timestamp_col not in df.columns:
        ax.set_title("Pupil Timeseries (no timestamp)")
        return ax

    # Convert timestamp to seconds
    time_s = (df[timestamp_col] - df[timestamp_col].iloc[0]) / 1_000_000

    def _smooth(series, window):
        """Apply rolling mean smoothing."""
        if window is None or window <= 1:
            return series
        return series.rolling(window=window, center=True, min_periods=1).mean()

    if eye in ("left", "both") and "Pupil diameter left" in df.columns:
        pupil_left = _smooth(df["Pupil diameter left"], smooth_window)
        ax.plot(time_s, pupil_left, label="Left", alpha=0.7, linewidth=0.8)

    if eye in ("right", "both") and "Pupil diameter right" in df.columns:
        pupil_right = _smooth(df["Pupil diameter right"], smooth_window)
        ax.plot(time_s, pupil_right, label="Right", alpha=0.7, linewidth=0.8)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Pupil Diameter (mm)")
    title = "Pupil Diameter Over Time"
    if smooth_window:
        title += " (smoothed)"
    ax.set_title(title)
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
    events_df: pd.DataFrame | None = None,
    bins: int = 30,
    ax: Axes | None = None,
) -> Axes:
    """Histogram of fixation duration distribution.

    Uses pymovements-detected events.

    Args:
        df: Input DataFrame with gaze data
        events_df: Pre-computed events from pymovements. If None, detects events.
        bins: Number of histogram bins
        ax: Matplotlib axes

    Returns:
        Matplotlib Axes object
    """
    ax = _get_or_create_axes(ax)

    # Get events if not provided
    if events_df is None:
        try:
            events_df = compute_events(df)
        except Exception:
            events_df = pd.DataFrame()

    # Filter to fixations
    if len(events_df) > 0 and "name" in events_df.columns:
        fixations = events_df[events_df["name"].str.contains("fixation", case=False)]

        if len(fixations) > 0 and "duration" in fixations.columns:
            durations = fixations["duration"].dropna()

            if len(durations) > 0:
                ax.hist(durations, bins=bins, edgecolor="black", alpha=0.7)

                # Add mean line
                mean_duration = durations.mean()
                ax.axvline(
                    mean_duration,
                    color="red",
                    linestyle="--",
                    label=f"Mean: {mean_duration:.0f}ms",
                )
                ax.legend()

    ax.set_xlabel("Duration (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Fixation Duration Distribution")
    ax.grid(True, alpha=0.3)

    return ax


def plot_eye_movement_timeline(
    df: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
    ax: Axes | None = None,
    timestamp_col: str = "Recording timestamp",
) -> Axes:
    """Timeline showing detected fixations and saccades.

    Args:
        df: Input DataFrame with timestamp column
        events_df: Pre-computed events from pymovements. If None, detects events.
        ax: Matplotlib axes
        timestamp_col: Timestamp column name

    Returns:
        Matplotlib Axes object
    """
    ax = _get_or_create_axes(ax, figsize=(12, 3))

    if timestamp_col not in df.columns:
        ax.set_title("Eye Movement Timeline (no timestamp)")
        return ax

    # Get events if not provided
    if events_df is None:
        try:
            events_df = compute_events(df)
        except Exception:
            events_df = pd.DataFrame()

    if len(events_df) == 0 or "name" not in events_df.columns:
        ax.set_title("Eye Movement Timeline (no events detected)")
        return ax

    # Color mapping
    color_map = {
        "fixation": "blue",
        "saccade": "red",
    }

    # Plot each event type
    for event_type, color in color_map.items():
        mask = events_df["name"].str.contains(event_type, case=False)
        if mask.any():
            events = events_df[mask]
            for _, event in events.iterrows():
                onset = event.get("onset", 0) / 1000  # Convert to seconds
                offset = event.get("offset", onset) / 1000
                ax.axvspan(onset, offset, alpha=0.5, color=color, label=event_type)

    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Events")
    ax.set_title("Eye Movement Timeline")
    ax.grid(True, alpha=0.3, axis="x")

    return ax


# =============================================================================
# Summary Visualization
# =============================================================================


def plot_recording_summary(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (16, 12),
) -> Figure:
    """Create multi-panel summary figure for a recording.

    Creates a figure with:
    - Gaze heatmap (MNE)
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

    # Create grid layout with increased spacing
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

    # Pre-compute events once for reuse
    try:
        events_df = compute_events(df)
    except Exception:
        events_df = pd.DataFrame()

    # Gaze heatmap (top left) - using MNE
    ax1 = fig.add_subplot(gs[0, 0])
    _, ax1 = plot_gaze_heatmap(df, ax=ax1, sigma=30)

    # Scanpath (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_scanpath(df, events_df=events_df, ax=ax2)

    # Gaze scatter (top right) - alternative to old fixation heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    plot_gaze_scatter(df, ax=ax3, alpha=0.1, s=0.5)

    # Pupil timeseries (middle, full width)
    ax4 = fig.add_subplot(gs[1, :2])
    plot_pupil_timeseries(df, ax=ax4)

    # Fixation duration histogram (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    plot_fixation_durations(df, events_df=events_df, ax=ax5)

    # Eye movement timeline (bottom, 2/3 width)
    ax6 = fig.add_subplot(gs[2, :2])
    plot_eye_movement_timeline(df, events_df=events_df, ax=ax6)

    # Metrics text (bottom right)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis("off")

    # Compute and display metrics
    validity = compute_validity_rate(df)
    pupil_stats = compute_pupil_stats(df)
    fixation_stats = get_fixation_stats(events_df) if len(events_df) > 0 else {"count": 0}

    # Safe formatting
    mean_pupil = pupil_stats.get("mean", float("nan"))
    left_pupil = pupil_stats.get("left_mean", float("nan"))
    right_pupil = pupil_stats.get("right_mean", float("nan"))
    fix_count = fixation_stats.get("count", 0)
    fix_mean = fixation_stats.get("duration_mean_ms")
    fix_mean_str = f"{fix_mean:.0f}" if fix_mean is not None else "N/A"

    metrics_text = f"""Recording Summary

Validity Rate: {validity:.1%}

Pupil Diameter:
  Mean: {mean_pupil:.2f} mm
  Left: {left_pupil:.2f} mm
  Right: {right_pupil:.2f} mm

Fixations (pymovements):
  Count: {fix_count}
  Mean Duration: {fix_mean_str} ms

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
