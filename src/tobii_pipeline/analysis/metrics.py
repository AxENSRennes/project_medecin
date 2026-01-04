"""Core metric calculations for eye-tracking data."""

import numpy as np
import pandas as pd

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
# Fixation Metrics
# =============================================================================


def compute_fixation_count(df: pd.DataFrame) -> int:
    """Count unique fixations in recording.

    Args:
        df: Input DataFrame with Eye movement type and index columns

    Returns:
        Number of fixations
    """
    if "Eye movement type" not in df.columns:
        return 0

    fixation_df = df[df["Eye movement type"] == "Fixation"]

    if "Eye movement type index" in df.columns:
        return fixation_df["Eye movement type index"].nunique()

    return 0


def compute_fixation_durations(df: pd.DataFrame) -> pd.Series:
    """Get duration of each unique fixation.

    Args:
        df: Input DataFrame with eye movement columns

    Returns:
        Series of fixation durations in milliseconds
    """
    if "Eye movement type" not in df.columns or "Gaze event duration" not in df.columns:
        return pd.Series(dtype=float)

    if "Eye movement type index" not in df.columns:
        return pd.Series(dtype=float)

    fixation_df = df[df["Eye movement type"] == "Fixation"]

    # Get the first duration value for each unique fixation
    durations = fixation_df.groupby("Eye movement type index")["Gaze event duration"].first()

    return durations


def compute_fixation_stats(df: pd.DataFrame) -> dict:
    """Compute comprehensive fixation statistics.

    Args:
        df: Input DataFrame with eye movement columns

    Returns:
        Dict with fixation statistics
    """
    durations = compute_fixation_durations(df)

    result = {
        "count": len(durations),
        "mean_duration": np.nan,
        "std_duration": np.nan,
        "min_duration": np.nan,
        "max_duration": np.nan,
        "total_fixation_time": np.nan,
        "fixation_rate": np.nan,
    }

    if len(durations) == 0:
        return result

    result["mean_duration"] = durations.mean()
    result["std_duration"] = durations.std()
    result["min_duration"] = durations.min()
    result["max_duration"] = durations.max()
    result["total_fixation_time"] = durations.sum()

    # Compute fixation rate (fixations per second)
    if "Recording timestamp" in df.columns and len(df) > 1:
        total_time_s = (
            df["Recording timestamp"].iloc[-1] - df["Recording timestamp"].iloc[0]
        ) / 1_000_000
        if total_time_s > 0:
            result["fixation_rate"] = len(durations) / total_time_s

    return result


# =============================================================================
# Saccade Metrics
# =============================================================================


def compute_saccade_count(df: pd.DataFrame) -> int:
    """Count unique saccades in recording.

    Args:
        df: Input DataFrame with Eye movement type and index columns

    Returns:
        Number of saccades
    """
    if "Eye movement type" not in df.columns:
        return 0

    saccade_df = df[df["Eye movement type"] == "Saccade"]

    if "Eye movement type index" in df.columns:
        return saccade_df["Eye movement type index"].nunique()

    return 0


def compute_saccade_durations(df: pd.DataFrame) -> pd.Series:
    """Get duration of each unique saccade.

    Args:
        df: Input DataFrame with eye movement columns

    Returns:
        Series of saccade durations in milliseconds
    """
    if "Eye movement type" not in df.columns or "Gaze event duration" not in df.columns:
        return pd.Series(dtype=float)

    if "Eye movement type index" not in df.columns:
        return pd.Series(dtype=float)

    saccade_df = df[df["Eye movement type"] == "Saccade"]

    # Get the first duration value for each unique saccade
    durations = saccade_df.groupby("Eye movement type index")["Gaze event duration"].first()

    return durations


def compute_saccade_stats(df: pd.DataFrame) -> dict:
    """Compute comprehensive saccade statistics.

    Args:
        df: Input DataFrame with eye movement columns

    Returns:
        Dict with saccade statistics
    """
    durations = compute_saccade_durations(df)

    result = {
        "count": len(durations),
        "mean_duration": np.nan,
        "std_duration": np.nan,
        "min_duration": np.nan,
        "max_duration": np.nan,
        "total_saccade_time": np.nan,
        "saccade_rate": np.nan,
    }

    if len(durations) == 0:
        return result

    result["mean_duration"] = durations.mean()
    result["std_duration"] = durations.std()
    result["min_duration"] = durations.min()
    result["max_duration"] = durations.max()
    result["total_saccade_time"] = durations.sum()

    # Compute saccade rate (saccades per second)
    if "Recording timestamp" in df.columns and len(df) > 1:
        total_time_s = (
            df["Recording timestamp"].iloc[-1] - df["Recording timestamp"].iloc[0]
        ) / 1_000_000
        if total_time_s > 0:
            result["saccade_rate"] = len(durations) / total_time_s

    return result


# =============================================================================
# Summary
# =============================================================================


def compute_recording_summary(df: pd.DataFrame) -> dict:
    """Compute all key metrics for a recording.

    Convenience function that calls all metric functions.

    Args:
        df: Input DataFrame (cleaned and filtered eye tracker data)

    Returns:
        Dict with all computed metrics organized by category
    """
    return {
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
        "fixation": compute_fixation_stats(df),
        "saccade": compute_saccade_stats(df),
    }
