"""Core metric calculations for eye-tracking data.

Event metrics (fixations, saccades) are computed using pymovements library.
"""

import numpy as np
import pandas as pd

from tobii_pipeline.adapters.pymovements_adapter import (
    apply_pix2deg,
    apply_pos2vel,
    compute_event_properties,
    detect_events_ivt,
    df_to_gaze_dataframe,
    events_to_df,
    get_fixation_stats,
    get_saccade_stats,
)

# =============================================================================
# Data Quality Metrics
# =============================================================================


def compute_validity_rate(df: pd.DataFrame, both_eyes: bool = True) -> float:
    """Compute percentage of samples with valid gaze data.

    Args:
        df: Input DataFrame with validity columns
        both_eyes: If True, require both eyes valid. If False, at least one.

    Returns:
        Validity rate as float between 0.0 and 1.0
    """
    if "Validity left" not in df.columns or "Validity right" not in df.columns:
        return 0.0

    if len(df) == 0:
        return 0.0

    if both_eyes:
        valid_mask = (df["Validity left"] == "Valid") & (df["Validity right"] == "Valid")
    else:
        valid_mask = (df["Validity left"] == "Valid") | (df["Validity right"] == "Valid")

    return valid_mask.mean()


def compute_tracking_ratio(df: pd.DataFrame) -> float:
    """Compute ratio of eye tracker rows to total rows.

    Args:
        df: Input DataFrame with Sensor column

    Returns:
        Tracking ratio as float between 0.0 and 1.0
    """
    if "Sensor" not in df.columns or len(df) == 0:
        return 0.0

    return (df["Sensor"] == "Eye Tracker").mean()


# =============================================================================
# Gaze Metrics
# =============================================================================


def compute_gaze_center(df: pd.DataFrame) -> tuple[float, float]:
    """Compute mean gaze position (center of attention).

    Args:
        df: Input DataFrame with gaze columns

    Returns:
        Tuple of (mean_x, mean_y) in pixels
    """
    mean_x = df["Gaze point X"].mean() if "Gaze point X" in df.columns else np.nan
    mean_y = df["Gaze point Y"].mean() if "Gaze point Y" in df.columns else np.nan
    return (mean_x, mean_y)


def compute_gaze_dispersion(df: pd.DataFrame) -> float:
    """Compute standard deviation of gaze positions.

    Measures how spread out gaze is across the screen using
    the Euclidean distance from mean position.

    Args:
        df: Input DataFrame with gaze columns

    Returns:
        Dispersion value (RMS distance from center) in pixels
    """
    if "Gaze point X" not in df.columns or "Gaze point Y" not in df.columns:
        return np.nan

    x = df["Gaze point X"].dropna()
    y = df["Gaze point Y"].dropna()

    if len(x) == 0 or len(y) == 0:
        return np.nan

    # Compute standard deviation in both dimensions
    std_x = x.std()
    std_y = y.std()

    # Return Euclidean combination
    return np.sqrt(std_x**2 + std_y**2)


def compute_gaze_quadrant_distribution(
    df: pd.DataFrame,
    screen_width: int = 1920,
    screen_height: int = 1080,
) -> dict[str, float]:
    """Compute percentage of gaze in each screen quadrant.

    Args:
        df: Input DataFrame with gaze columns
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels

    Returns:
        Dict with keys 'top_left', 'top_right', 'bottom_left', 'bottom_right'
        and values as percentages (0.0 to 1.0)
    """
    result = {
        "top_left": 0.0,
        "top_right": 0.0,
        "bottom_left": 0.0,
        "bottom_right": 0.0,
    }

    if "Gaze point X" not in df.columns or "Gaze point Y" not in df.columns:
        return result

    x = df["Gaze point X"]
    y = df["Gaze point Y"]

    valid = x.notna() & y.notna()
    if valid.sum() == 0:
        return result

    mid_x = screen_width / 2
    mid_y = screen_height / 2

    total_valid = valid.sum()

    result["top_left"] = ((x < mid_x) & (y < mid_y) & valid).sum() / total_valid
    result["top_right"] = ((x >= mid_x) & (y < mid_y) & valid).sum() / total_valid
    result["bottom_left"] = ((x < mid_x) & (y >= mid_y) & valid).sum() / total_valid
    result["bottom_right"] = ((x >= mid_x) & (y >= mid_y) & valid).sum() / total_valid

    return result


# =============================================================================
# Pupil Metrics
# =============================================================================


def compute_pupil_stats(df: pd.DataFrame) -> dict:
    """Compute pupil diameter statistics for both eyes.

    Args:
        df: Input DataFrame with pupil columns

    Returns:
        Dict with statistics for left and right eyes
    """
    result = {
        "left_mean": np.nan,
        "left_std": np.nan,
        "left_min": np.nan,
        "left_max": np.nan,
        "right_mean": np.nan,
        "right_std": np.nan,
        "right_min": np.nan,
        "right_max": np.nan,
        "mean": np.nan,
    }

    if "Pupil diameter left" in df.columns:
        left = df["Pupil diameter left"].dropna()
        if len(left) > 0:
            result["left_mean"] = left.mean()
            result["left_std"] = left.std()
            result["left_min"] = left.min()
            result["left_max"] = left.max()

    if "Pupil diameter right" in df.columns:
        right = df["Pupil diameter right"].dropna()
        if len(right) > 0:
            result["right_mean"] = right.mean()
            result["right_std"] = right.std()
            result["right_min"] = right.min()
            result["right_max"] = right.max()

    # Compute overall mean
    means = [v for k, v in result.items() if k.endswith("_mean") and not np.isnan(v)]
    if means:
        result["mean"] = np.mean(means)

    return result


def compute_pupil_variability(df: pd.DataFrame) -> float:
    """Compute coefficient of variation for pupil diameter.

    CV = std / mean, serves as a cognitive load proxy.

    Args:
        df: Input DataFrame with pupil columns

    Returns:
        Coefficient of variation (dimensionless)
    """
    values = []

    if "Pupil diameter left" in df.columns:
        values.extend(df["Pupil diameter left"].dropna().tolist())

    if "Pupil diameter right" in df.columns:
        values.extend(df["Pupil diameter right"].dropna().tolist())

    if not values:
        return np.nan

    values = np.array(values)
    mean = np.mean(values)

    if mean == 0:
        return np.nan

    return np.std(values) / mean


def compute_pupil_over_time(
    df: pd.DataFrame,
    n_bins: int = 10,
    timestamp_col: str = "Recording timestamp",
) -> pd.DataFrame:
    """Compute pupil diameter statistics binned by time periods.

    Args:
        df: Input DataFrame with pupil and timestamp columns
        n_bins: Number of time bins
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with columns: bin, start_time, end_time, mean_pupil, std_pupil
    """
    if timestamp_col not in df.columns:
        return pd.DataFrame()

    if "Pupil diameter left" not in df.columns and "Pupil diameter right" not in df.columns:
        return pd.DataFrame()

    # Compute average pupil diameter
    pupil_cols = []
    if "Pupil diameter left" in df.columns:
        pupil_cols.append("Pupil diameter left")
    if "Pupil diameter right" in df.columns:
        pupil_cols.append("Pupil diameter right")

    df = df.copy()
    df["_pupil_avg"] = df[pupil_cols].mean(axis=1)

    # Create time bins
    df["_time_bin"] = pd.cut(df[timestamp_col], bins=n_bins, labels=range(n_bins))

    # Compute statistics per bin
    result = (
        df.groupby("_time_bin", observed=True)
        .agg(
            start_time=(timestamp_col, "min"),
            end_time=(timestamp_col, "max"),
            mean_pupil=("_pupil_avg", "mean"),
            std_pupil=("_pupil_avg", "std"),
            count=("_pupil_avg", "count"),
        )
        .reset_index()
    )

    result = result.rename(columns={"_time_bin": "bin"})
    return result


# =============================================================================
# Event Metrics (via pymovements)
# =============================================================================


def compute_events(
    df: pd.DataFrame,
    velocity_threshold: float = 30.0,
    minimum_duration: int = 100,
    screen_width_px: int = 1920,
    screen_height_px: int = 1080,
    include_blinks: bool = True,
) -> pd.DataFrame:
    """Detect fixations, saccades, and blinks using pymovements I-VT algorithm.

    Args:
        df: Cleaned Tobii DataFrame with gaze data.
        velocity_threshold: Velocity threshold in deg/s for fixation detection.
        minimum_duration: Minimum event duration in ms.
        screen_width_px: Screen width in pixels.
        screen_height_px: Screen height in pixels.
        include_blinks: Whether to include blink events from validity data.

    Returns:
        DataFrame with detected events including:
        - name: Event type (fixation, saccade, blink)
        - onset: Start time in ms
        - offset: End time in ms
        - duration: Event duration in ms
        - amplitude: Movement amplitude in degrees (fixations/saccades)
        - dispersion: Spatial dispersion in degrees (fixations)
        - peak_velocity: Maximum velocity in deg/s (saccades)
    """
    # Convert to pymovements format
    gaze = df_to_gaze_dataframe(
        df,
        screen_width_px=screen_width_px,
        screen_height_px=screen_height_px,
    )

    # Apply transformations
    gaze = apply_pix2deg(gaze)
    gaze = apply_pos2vel(gaze)

    # Detect events using I-VT
    gaze = detect_events_ivt(
        gaze,
        velocity_threshold=velocity_threshold,
        minimum_duration=minimum_duration,
    )

    # Compute event properties
    gaze = compute_event_properties(gaze)

    # Extract pymovements events
    events_df = events_to_df(gaze)

    # Add blink events from validity data
    if include_blinks:
        from tobii_pipeline.adapters.mne_adapter import get_blink_events_df

        blink_events = get_blink_events_df(df)
        if len(blink_events) > 0:
            events_df = pd.concat([events_df, blink_events], ignore_index=True)
            # Sort by onset time
            events_df = events_df.sort_values("onset").reset_index(drop=True)

    return events_df


def compute_fixation_stats(
    df: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
    velocity_threshold: float = 30.0,
) -> dict:
    """Compute comprehensive fixation statistics using pymovements.

    Args:
        df: Input DataFrame with gaze data (used if events_df is None).
        events_df: Pre-computed events DataFrame from compute_events().
            If None, events are detected from df.
        velocity_threshold: Velocity threshold for event detection.

    Returns:
        Dict with fixation statistics including:
        - count: Number of fixations
        - duration_mean_ms: Mean fixation duration
        - duration_std_ms: Std deviation of fixation duration
        - duration_min_ms: Minimum fixation duration
        - duration_max_ms: Maximum fixation duration
        - dispersion_mean_deg: Mean spatial dispersion (degrees)
        - dispersion_std_deg: Std deviation of dispersion
    """
    if events_df is None:
        try:
            events_df = compute_events(df, velocity_threshold=velocity_threshold)
        except Exception:
            return {
                "count": 0,
                "duration_mean_ms": None,
                "duration_std_ms": None,
                "duration_min_ms": None,
                "duration_max_ms": None,
                "dispersion_mean_deg": None,
                "dispersion_std_deg": None,
            }

    return get_fixation_stats(events_df)


def compute_saccade_stats(
    df: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
    velocity_threshold: float = 30.0,
) -> dict:
    """Compute comprehensive saccade statistics using pymovements.

    Args:
        df: Input DataFrame with gaze data (used if events_df is None).
        events_df: Pre-computed events DataFrame from compute_events().
            If None, events are detected from df.
        velocity_threshold: Velocity threshold for event detection.

    Returns:
        Dict with saccade statistics including:
        - count: Number of saccades
        - duration_mean_ms: Mean saccade duration
        - duration_std_ms: Std deviation of saccade duration
        - amplitude_mean_deg: Mean saccade amplitude (degrees)
        - amplitude_std_deg: Std deviation of amplitude
        - peak_velocity_mean_deg_s: Mean peak velocity (deg/s)
        - peak_velocity_std_deg_s: Std deviation of peak velocity
    """
    if events_df is None:
        try:
            events_df = compute_events(df, velocity_threshold=velocity_threshold)
        except Exception:
            return {
                "count": 0,
                "duration_mean_ms": None,
                "duration_std_ms": None,
                "amplitude_mean_deg": None,
                "amplitude_std_deg": None,
                "peak_velocity_mean_deg_s": None,
                "peak_velocity_std_deg_s": None,
            }

    return get_saccade_stats(events_df)


# =============================================================================
# Summary
# =============================================================================


def compute_recording_summary(
    df: pd.DataFrame,
    detect_events: bool = True,
    velocity_threshold: float = 30.0,
) -> dict:
    """Compute all key metrics for a recording.

    Convenience function that calls all metric functions.

    Args:
        df: Input DataFrame (cleaned and filtered eye tracker data)
        detect_events: Whether to detect fixations/saccades via pymovements.
            Set to False for faster execution if event metrics not needed.
        velocity_threshold: Velocity threshold for event detection.

    Returns:
        Dict with all computed metrics organized by category
    """
    summary = {
        "quality": {
            "validity_rate": compute_validity_rate(df),
            "validity_rate_either": compute_validity_rate(df, both_eyes=False),
            "tracking_ratio": compute_tracking_ratio(df),
        },
        "gaze": {
            "center": compute_gaze_center(df),
            "dispersion": compute_gaze_dispersion(df),
            "quadrant_distribution": compute_gaze_quadrant_distribution(df),
        },
        "pupil": {
            "stats": compute_pupil_stats(df),
            "variability": compute_pupil_variability(df),
        },
    }

    if detect_events:
        # Compute events once and reuse for both fixation and saccade stats
        try:
            events_df = compute_events(df, velocity_threshold=velocity_threshold)
            summary["fixation"] = get_fixation_stats(events_df)
            summary["saccade"] = get_saccade_stats(events_df)
        except Exception:
            summary["fixation"] = compute_fixation_stats(df, events_df=None)
            summary["saccade"] = compute_saccade_stats(df, events_df=None)
    else:
        summary["fixation"] = {
            "count": 0,
            "duration_mean_ms": None,
            "duration_std_ms": None,
            "duration_min_ms": None,
            "duration_max_ms": None,
            "dispersion_mean_deg": None,
            "dispersion_std_deg": None,
        }
        summary["saccade"] = {
            "count": 0,
            "duration_mean_ms": None,
            "duration_std_ms": None,
            "amplitude_mean_deg": None,
            "amplitude_std_deg": None,
            "peak_velocity_mean_deg_s": None,
            "peak_velocity_std_deg_s": None,
        }

    return summary
