"""Adapter for MNE-Python library integration.

Provides functions for blink interpolation and gaze visualization using MNE.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Default screen parameters
DEFAULT_SCREEN_WIDTH_PX = 1920
DEFAULT_SCREEN_HEIGHT_PX = 1080
DEFAULT_SCREEN_WIDTH_M = 0.527  # Typical 24" monitor in meters
DEFAULT_SCREEN_HEIGHT_M = 0.296
DEFAULT_DISTANCE_M = 0.60  # Typical viewing distance
DEFAULT_SAMPLING_RATE = 100  # Tobii default


def df_to_mne_raw(
    df: pd.DataFrame,
    sfreq: float = DEFAULT_SAMPLING_RATE,
) -> mne.io.RawArray:
    """Convert a cleaned Tobii DataFrame to MNE Raw object.

    Args:
        df: Cleaned Tobii DataFrame with gaze and pupil data.
        sfreq: Sampling frequency in Hz.

    Returns:
        MNE RawArray object with eye-tracking channels.
    """
    # Prepare channel data
    channels = []
    ch_names = []
    ch_types = []

    # Gaze position channels (in pixels)
    if "Gaze point X" in df.columns:
        channels.append(df["Gaze point X"].fillna(0).values)
        ch_names.append("gaze_x")
        ch_types.append("eyegaze")

    if "Gaze point Y" in df.columns:
        channels.append(df["Gaze point Y"].fillna(0).values)
        ch_names.append("gaze_y")
        ch_types.append("eyegaze")

    # Pupil diameter channels (in mm, convert to m for MNE)
    if "Pupil diameter left" in df.columns:
        channels.append(df["Pupil diameter left"].fillna(0).values / 1000)
        ch_names.append("pupil_left")
        ch_types.append("pupil")

    if "Pupil diameter right" in df.columns:
        channels.append(df["Pupil diameter right"].fillna(0).values / 1000)
        ch_names.append("pupil_right")
        ch_types.append("pupil")

    # Create data array (channels x samples)
    data = np.array(channels)

    # Create MNE info structure
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Create Raw object
    raw = mne.io.RawArray(data, info, verbose=False)

    return raw


def create_blink_annotations(
    df: pd.DataFrame,
    sfreq: float = DEFAULT_SAMPLING_RATE,
) -> mne.Annotations:
    """Create MNE annotations for blink periods from validity columns.

    Args:
        df: Cleaned Tobii DataFrame with validity columns.
        sfreq: Sampling frequency in Hz.

    Returns:
        MNE Annotations object marking blink periods.
    """
    # Identify blinks as periods where both eyes are invalid
    validity_left = df.get("Validity left", pd.Series(["Valid"] * len(df)))
    validity_right = df.get("Validity right", pd.Series(["Valid"] * len(df)))

    both_invalid = (validity_left == "Invalid") & (validity_right == "Invalid")

    # Find blink onset and offset
    blink_starts = []
    blink_durations = []

    in_blink = False
    blink_start_idx = 0

    for i, is_invalid in enumerate(both_invalid):
        if is_invalid and not in_blink:
            in_blink = True
            blink_start_idx = i
        elif not is_invalid and in_blink:
            in_blink = False
            duration_samples = i - blink_start_idx
            # Only count as blink if duration is reasonable (50-500ms)
            duration_ms = duration_samples * 1000 / sfreq
            if 50 <= duration_ms <= 500:
                blink_starts.append(blink_start_idx / sfreq)
                blink_durations.append(duration_samples / sfreq)

    # Handle blink at end of recording
    if in_blink:
        duration_samples = len(both_invalid) - blink_start_idx
        duration_ms = duration_samples * 1000 / sfreq
        if 50 <= duration_ms <= 500:
            blink_starts.append(blink_start_idx / sfreq)
            blink_durations.append(duration_samples / sfreq)

    if not blink_starts:
        return mne.Annotations(onset=[], duration=[], description=[])

    return mne.Annotations(
        onset=blink_starts,
        duration=blink_durations,
        description=["blink"] * len(blink_starts),
    )


def interpolate_blinks_mne(
    df: pd.DataFrame,
    buffer_before: float = 0.05,
    buffer_after: float = 0.2,
    sfreq: float = DEFAULT_SAMPLING_RATE,
) -> pd.DataFrame:
    """Interpolate data during blinks using MNE's algorithm.

    MNE's interpolate_blinks uses cubic spline interpolation with configurable
    buffer periods before and after the detected blink.

    Args:
        df: Cleaned Tobii DataFrame with gaze and pupil data.
        buffer_before: Time in seconds to extend interpolation before blink.
        buffer_after: Time in seconds to extend interpolation after blink.
        sfreq: Sampling frequency in Hz.

    Returns:
        DataFrame with blink periods interpolated.
    """
    # Convert to MNE Raw
    raw = df_to_mne_raw(df, sfreq=sfreq)

    # Create and set blink annotations
    blink_annotations = create_blink_annotations(df, sfreq=sfreq)
    raw.set_annotations(blink_annotations)

    # Interpolate blinks
    mne.preprocessing.eyetracking.interpolate_blinks(
        raw,
        buffer=(buffer_before, buffer_after),
        interpolate_gaze=True,
    )

    # Extract interpolated data back to DataFrame
    data = raw.get_data()
    ch_names = raw.ch_names

    # Update the original DataFrame with interpolated values
    df_interpolated = df.copy()

    for i, ch_name in enumerate(ch_names):
        if ch_name == "gaze_x" and "Gaze point X" in df.columns:
            df_interpolated["Gaze point X"] = data[i]
        elif ch_name == "gaze_y" and "Gaze point Y" in df.columns:
            df_interpolated["Gaze point Y"] = data[i]
        elif ch_name == "pupil_left" and "Pupil diameter left" in df.columns:
            df_interpolated["Pupil diameter left"] = data[i] * 1000  # Back to mm
        elif ch_name == "pupil_right" and "Pupil diameter right" in df.columns:
            df_interpolated["Pupil diameter right"] = data[i] * 1000  # Back to mm

    return df_interpolated


def plot_gaze_heatmap(
    df: pd.DataFrame,
    width: int = DEFAULT_SCREEN_WIDTH_PX,
    height: int = DEFAULT_SCREEN_HEIGHT_PX,
    sigma: float = 50.0,
    cmap: str = "hot",
    ax: Axes | None = None,
    vlim: tuple[float | None, float | None] | None = None,
) -> tuple[Figure, Axes]:
    """Create a gaze heatmap using MNE's visualization.

    Args:
        df: Cleaned Tobii DataFrame with gaze data.
        width: Screen width in pixels.
        height: Screen height in pixels.
        sigma: Gaussian smoothing parameter (higher = smoother).
        cmap: Matplotlib colormap name.
        ax: Existing axes to plot on. If None, creates new figure.
        vlim: Value limits (min, max) for color scaling.
            Use (min_val, None) to make low values transparent.

    Returns:
        Tuple of (Figure, Axes) with the heatmap.
    """
    # Extract valid gaze points
    gaze_x = df["Gaze point X"].dropna().values
    gaze_y = df["Gaze point Y"].dropna().values

    if len(gaze_x) == 0:
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        else:
            fig = ax.get_figure()
        ax.text(0.5, 0.5, "No valid gaze data", ha="center", va="center")
        return fig, ax

    # Create 2D histogram
    bins_x = np.linspace(0, width, 100)
    bins_y = np.linspace(0, height, 100)

    heatmap, _, _ = np.histogram2d(gaze_x, gaze_y, bins=[bins_x, bins_y])
    heatmap = heatmap.T  # Transpose for correct orientation

    # Apply Gaussian smoothing
    from scipy.ndimage import gaussian_filter

    heatmap_smooth = gaussian_filter(heatmap, sigma=sigma / 10)

    # Normalize
    if heatmap_smooth.max() > 0:
        heatmap_smooth = heatmap_smooth / heatmap_smooth.max()

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.get_figure()

    # Get colormap and optionally set transparency for low values
    colormap = plt.get_cmap(cmap)
    if vlim and vlim[0] is not None:
        colormap = colormap.copy()
        colormap.set_under("k", alpha=0)

    # Plot heatmap
    extent = [0, width, height, 0]  # [left, right, bottom, top]
    im = ax.imshow(
        heatmap_smooth,
        extent=extent,
        cmap=colormap,
        aspect="auto",
        vmin=vlim[0] if vlim else None,
        vmax=vlim[1] if vlim else None,
    )

    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title("Gaze Heatmap")

    plt.colorbar(im, ax=ax, label="Normalized density")

    return fig, ax


def plot_gaze_on_stimulus(
    df: pd.DataFrame,
    stimulus_path: str | Path,
    sigma: float = 50.0,
    cmap: str = "hot",
    alpha: float = 0.6,
    vlim: tuple[float | None, float | None] = (0.1, None),
) -> tuple[Figure, Axes]:
    """Overlay gaze heatmap on a stimulus image.

    Args:
        df: Cleaned Tobii DataFrame with gaze data.
        stimulus_path: Path to the stimulus image file.
        sigma: Gaussian smoothing parameter.
        cmap: Matplotlib colormap name.
        alpha: Heatmap transparency (0-1).
        vlim: Value limits for color scaling. Default makes low values transparent.

    Returns:
        Tuple of (Figure, Axes) with the overlay visualization.
    """
    # Load stimulus image
    stimulus = plt.imread(str(stimulus_path))
    height, width = stimulus.shape[:2]

    # Create figure and show stimulus
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(stimulus)

    # Extract valid gaze points
    gaze_x = df["Gaze point X"].dropna().values
    gaze_y = df["Gaze point Y"].dropna().values

    if len(gaze_x) == 0:
        ax.set_title("Gaze on Stimulus (no valid gaze data)")
        return fig, ax

    # Create 2D histogram
    bins_x = np.linspace(0, width, 100)
    bins_y = np.linspace(0, height, 100)

    heatmap, _, _ = np.histogram2d(gaze_x, gaze_y, bins=[bins_x, bins_y])
    heatmap = heatmap.T

    # Apply Gaussian smoothing
    from scipy.ndimage import gaussian_filter

    heatmap_smooth = gaussian_filter(heatmap, sigma=sigma / 10)

    # Normalize
    if heatmap_smooth.max() > 0:
        heatmap_smooth = heatmap_smooth / heatmap_smooth.max()

    # Get colormap with transparency for low values
    colormap = plt.get_cmap(cmap).copy()
    colormap.set_under("k", alpha=0)

    # Overlay heatmap
    extent = [0, width, height, 0]
    ax.imshow(
        heatmap_smooth,
        extent=extent,
        cmap=colormap,
        aspect="auto",
        alpha=alpha,
        vmin=vlim[0] if vlim else None,
        vmax=vlim[1] if vlim else None,
    )

    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title("Gaze on Stimulus")
    ax.axis("off")

    return fig, ax


def get_blink_statistics_mne(
    df: pd.DataFrame,
    sfreq: float = DEFAULT_SAMPLING_RATE,
) -> dict:
    """Compute blink statistics using MNE's blink detection.

    Args:
        df: Cleaned Tobii DataFrame with validity columns.
        sfreq: Sampling frequency in Hz.

    Returns:
        Dictionary with blink statistics.
    """
    annotations = create_blink_annotations(df, sfreq=sfreq)

    if len(annotations) == 0:
        return {
            "count": 0,
            "mean_duration_ms": None,
            "std_duration_ms": None,
            "min_duration_ms": None,
            "max_duration_ms": None,
            "blink_rate_per_min": 0.0,
        }

    durations_ms = np.array(annotations.duration) * 1000
    recording_duration_min = len(df) / sfreq / 60

    return {
        "count": len(annotations),
        "mean_duration_ms": float(np.mean(durations_ms)),
        "std_duration_ms": float(np.std(durations_ms)),
        "min_duration_ms": float(np.min(durations_ms)),
        "max_duration_ms": float(np.max(durations_ms)),
        "blink_rate_per_min": len(annotations) / recording_duration_min,
    }
