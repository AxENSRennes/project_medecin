"""Post-processing functions for cleaned Tobii eye-tracking data.

This module provides data quality improvements including:
- Missing data handling (interpolation, fill, removal)
- Outlier detection and removal (pupil, gaze, velocity)
- Blink detection and handling
- Gap detection and splitting
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

# Screen dimensions (Tobii recording setup)
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# Physiological bounds for pupil diameter (mm)
PUPIL_MIN_MM = 2.0
PUPIL_MAX_MM = 8.0

# Blink detection parameters
BLINK_MIN_DURATION_MS = 50
BLINK_MAX_DURATION_MS = 500
BLINK_PUPIL_CHANGE_THRESHOLD = 0.5  # mm change threshold

# Sampling rate
SAMPLING_RATE_HZ = 100
SAMPLE_INTERVAL_MS = 10  # 1000 / 100 Hz

# Velocity thresholds (degrees/second)
MAX_SACCADE_VELOCITY = 1000

# Gap detection
GAP_THRESHOLD_MS = 100

# Type aliases
InterpolationMethod = Literal["linear", "ffill", "bfill", "nearest", "mean", "median", "drop"]
OutlierHandling = Literal["remove", "nan", "interpolate"]
BlinkHandling = Literal["mark", "nan", "remove", "interpolate"]


@dataclass
class BlinkEvent:
    """Represents a detected blink event."""

    start_index: int
    end_index: int
    start_timestamp: float
    end_timestamp: float
    duration_ms: float


@dataclass
class GapInfo:
    """Information about a data gap."""

    start_index: int
    end_index: int
    start_timestamp: float
    end_timestamp: float
    duration_ms: float
    gap_samples: int


# =============================================================================
# Column Helpers
# =============================================================================


def get_interpolatable_columns() -> list[str]:
    """Get list of columns suitable for interpolation.

    Returns columns that contain continuous numeric data (gaze, pupil, 3D positions).
    """
    return [
        "Gaze point X",
        "Gaze point Y",
        "Gaze point 3D X",
        "Gaze point 3D Y",
        "Gaze point 3D Z",
        "Pupil diameter left",
        "Pupil diameter right",
        "Pupil position left X",
        "Pupil position left Y",
        "Pupil position left Z",
        "Pupil position right X",
        "Pupil position right Y",
        "Pupil position right Z",
        "Gaze direction left X",
        "Gaze direction left Y",
        "Gaze direction left Z",
        "Gaze direction right X",
        "Gaze direction right Y",
        "Gaze direction right Z",
    ]


# =============================================================================
# Missing Data Functions
# =============================================================================


def compute_missing_rate(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict[str, float]:
    """Compute missing data rate for each column.

    Args:
        df: Input DataFrame
        columns: Columns to check. If None, checks all numeric columns.

    Returns:
        Dict mapping column name to missing rate (0.0 to 1.0)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    result = {}
    for col in columns:
        if col in df.columns:
            result[col] = df[col].isna().mean()
    return result


def interpolate_missing(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: InterpolationMethod = "linear",
    max_gap: int | None = None,
) -> pd.DataFrame:
    """Interpolate missing values in specified columns.

    Args:
        df: Input DataFrame (should be eye tracker data after cleaning)
        columns: Columns to interpolate. If None, uses interpolatable columns.
        method: Interpolation strategy:
            - "linear": Linear interpolation (default, best for continuous signals)
            - "ffill": Forward fill
            - "bfill": Backward fill
            - "nearest": Nearest valid value
            - "mean": Fill with column mean
            - "median": Fill with column median
            - "drop": Remove rows with missing values
        max_gap: Maximum number of consecutive NaN values to interpolate.
                 Gaps larger than this remain NaN. None = no limit.

    Returns:
        DataFrame with interpolated values
    """
    df = df.copy()

    if columns is None:
        columns = [c for c in get_interpolatable_columns() if c in df.columns]

    if method == "drop":
        return df.dropna(subset=columns)

    for col in columns:
        if col not in df.columns:
            continue

        if method == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif method == "median":
            df[col] = df[col].fillna(df[col].median())
        elif method == "ffill":
            df[col] = df[col].ffill(limit=max_gap)
        elif method == "bfill":
            df[col] = df[col].bfill(limit=max_gap)
        elif method == "nearest":
            df[col] = df[col].interpolate(method="nearest", limit=max_gap)
        else:  # linear
            df[col] = df[col].interpolate(method="linear", limit=max_gap)

    return df


def drop_high_missing_rows(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Remove rows where missing data exceeds threshold.

    Args:
        df: Input DataFrame
        columns: Columns to consider for missing calculation. If None, uses all.
        threshold: Maximum allowed missing rate per row (0.0 to 1.0)

    Returns:
        DataFrame with high-missing rows removed
    """
    df = df.copy()

    if columns is None:
        columns = df.columns.tolist()
    else:
        columns = [c for c in columns if c in df.columns]

    if not columns:
        return df

    missing_rate = df[columns].isna().mean(axis=1)
    return df[missing_rate <= threshold].copy()


# =============================================================================
# Outlier Detection Functions
# =============================================================================


def detect_pupil_outliers(
    df: pd.DataFrame,
    min_diameter: float = PUPIL_MIN_MM,
    max_diameter: float = PUPIL_MAX_MM,
) -> pd.Series:
    """Detect pupil diameter values outside physiological range.

    Args:
        df: Input DataFrame with pupil columns
        min_diameter: Minimum valid pupil diameter in mm (default: 2.0)
        max_diameter: Maximum valid pupil diameter in mm (default: 8.0)

    Returns:
        Boolean Series where True indicates outlier rows
    """
    outliers = pd.Series(False, index=df.index)

    for col in ["Pupil diameter left", "Pupil diameter right"]:
        if col in df.columns:
            col_outliers = (df[col] < min_diameter) | (df[col] > max_diameter)
            # Only flag as outlier if not NaN (NaN is missing, not outlier)
            col_outliers = col_outliers & df[col].notna()
            outliers = outliers | col_outliers

    return outliers


def detect_gaze_outliers(
    df: pd.DataFrame,
    screen_width: int = SCREEN_WIDTH,
    screen_height: int = SCREEN_HEIGHT,
    margin: float = 0.1,
) -> pd.Series:
    """Detect gaze coordinates outside screen bounds.

    Args:
        df: Input DataFrame with gaze columns
        screen_width: Screen width in pixels (default: 1920)
        screen_height: Screen height in pixels (default: 1080)
        margin: Allowed margin outside screen as fraction (default: 0.1 = 10%)

    Returns:
        Boolean Series where True indicates outlier rows
    """
    outliers = pd.Series(False, index=df.index)

    x_min = -margin * screen_width
    x_max = screen_width * (1 + margin)
    y_min = -margin * screen_height
    y_max = screen_height * (1 + margin)

    if "Gaze point X" in df.columns:
        x_outliers = (df["Gaze point X"] < x_min) | (df["Gaze point X"] > x_max)
        x_outliers = x_outliers & df["Gaze point X"].notna()
        outliers = outliers | x_outliers

    if "Gaze point Y" in df.columns:
        y_outliers = (df["Gaze point Y"] < y_min) | (df["Gaze point Y"] > y_max)
        y_outliers = y_outliers & df["Gaze point Y"].notna()
        outliers = outliers | y_outliers

    return outliers


def compute_gaze_velocity(
    df: pd.DataFrame,
    timestamp_col: str = "Recording timestamp",
) -> pd.Series:
    """Compute point-to-point gaze velocity.

    Args:
        df: Input DataFrame with gaze and timestamp columns
        timestamp_col: Name of timestamp column (in microseconds)

    Returns:
        Series with velocity values (pixels/second)
    """
    if "Gaze point X" not in df.columns or "Gaze point Y" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    if timestamp_col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    # Calculate displacement
    dx = df["Gaze point X"].diff()
    dy = df["Gaze point Y"].diff()
    distance = np.sqrt(dx**2 + dy**2)

    # Calculate time difference (convert from microseconds to seconds)
    dt = df[timestamp_col].diff() / 1_000_000

    # Velocity in pixels/second
    velocity = distance / dt
    velocity = velocity.replace([np.inf, -np.inf], np.nan)

    return velocity


def detect_velocity_outliers(
    df: pd.DataFrame,
    max_velocity: float = MAX_SACCADE_VELOCITY,
    timestamp_col: str = "Recording timestamp",
) -> pd.Series:
    """Detect implausible eye movement velocities.

    Calculates point-to-point velocity and flags samples exceeding
    physiological limits for saccades.

    Args:
        df: Input DataFrame with gaze and timestamp columns
        max_velocity: Maximum allowed velocity in pixels/second
        timestamp_col: Name of timestamp column (in microseconds)

    Returns:
        Boolean Series where True indicates velocity outlier
    """
    velocity = compute_gaze_velocity(df, timestamp_col)
    outliers = velocity > max_velocity
    return outliers.fillna(False)


def remove_outliers(
    df: pd.DataFrame,
    outlier_mask: pd.Series,
    method: OutlierHandling = "nan",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Handle outliers based on specified method.

    Args:
        df: Input DataFrame
        outlier_mask: Boolean Series indicating outlier rows
        method: How to handle outliers:
            - "remove": Delete outlier rows
            - "nan": Replace outlier values with NaN
            - "interpolate": Replace with NaN and interpolate
        columns: Columns to set to NaN (for "nan" and "interpolate" methods).
                 If None, uses interpolatable columns.

    Returns:
        DataFrame with outliers handled
    """
    df = df.copy()

    if method == "remove":
        return df[~outlier_mask].copy()

    if columns is None:
        columns = [c for c in get_interpolatable_columns() if c in df.columns]

    for col in columns:
        if col in df.columns:
            df.loc[outlier_mask, col] = np.nan

    if method == "interpolate":
        df = interpolate_missing(df, columns=columns, method="linear")

    return df


def filter_physiological_range(
    df: pd.DataFrame,
    remove_pupil_outliers: bool = True,
    remove_gaze_outliers: bool = True,
    remove_velocity_outliers: bool = False,
    method: OutlierHandling = "nan",
) -> pd.DataFrame:
    """Apply all physiological range filters.

    Convenience function combining multiple outlier detection methods.

    Args:
        df: Input DataFrame
        remove_pupil_outliers: Filter pupil diameter outliers
        remove_gaze_outliers: Filter off-screen gaze
        remove_velocity_outliers: Filter velocity spikes
        method: How to handle outliers ("remove", "nan", "interpolate")

    Returns:
        DataFrame with outliers handled
    """
    df = df.copy()
    combined_outliers = pd.Series(False, index=df.index)

    if remove_pupil_outliers:
        combined_outliers = combined_outliers | detect_pupil_outliers(df)

    if remove_gaze_outliers:
        combined_outliers = combined_outliers | detect_gaze_outliers(df)

    if remove_velocity_outliers:
        combined_outliers = combined_outliers | detect_velocity_outliers(df)

    return remove_outliers(df, combined_outliers, method=method)


# =============================================================================
# Blink Detection Functions
# =============================================================================


def detect_blinks(
    df: pd.DataFrame,
    min_duration_ms: float = BLINK_MIN_DURATION_MS,
    max_duration_ms: float = BLINK_MAX_DURATION_MS,
    timestamp_col: str = "Recording timestamp",
) -> list[BlinkEvent]:
    """Detect blink events from pupil and validity data.

    Blinks are identified by consecutive samples where both eyes are invalid.
    Duration must fall within typical blink range.

    Args:
        df: Input DataFrame (eye tracker data)
        min_duration_ms: Minimum blink duration in milliseconds
        max_duration_ms: Maximum blink duration in milliseconds
        timestamp_col: Name of timestamp column (in microseconds)

    Returns:
        List of BlinkEvent objects
    """
    if "Validity left" not in df.columns or "Validity right" not in df.columns:
        return []

    if timestamp_col not in df.columns:
        return []

    # Find samples where both eyes are invalid (typical of blinks)
    invalid_mask = (df["Validity left"] != "Valid") & (df["Validity right"] != "Valid")

    blinks = []
    in_blink = False
    blink_start_idx = 0
    blink_start_ts = 0.0

    for i, (idx, row) in enumerate(df.iterrows()):
        is_invalid = invalid_mask.iloc[i]

        if is_invalid and not in_blink:
            # Start of potential blink
            in_blink = True
            blink_start_idx = idx
            blink_start_ts = row[timestamp_col]

        elif not is_invalid and in_blink:
            # End of potential blink
            in_blink = False
            blink_end_idx = df.index[i - 1] if i > 0 else idx
            blink_end_ts = df.loc[blink_end_idx, timestamp_col]

            # Calculate duration in ms (timestamp is in microseconds)
            duration_ms = (blink_end_ts - blink_start_ts) / 1000

            # Check if duration is within blink range
            if min_duration_ms <= duration_ms <= max_duration_ms:
                blinks.append(
                    BlinkEvent(
                        start_index=blink_start_idx,
                        end_index=blink_end_idx,
                        start_timestamp=blink_start_ts,
                        end_timestamp=blink_end_ts,
                        duration_ms=duration_ms,
                    )
                )

    # Handle case where recording ends during a blink
    if in_blink:
        blink_end_idx = df.index[-1]
        blink_end_ts = df.loc[blink_end_idx, timestamp_col]
        duration_ms = (blink_end_ts - blink_start_ts) / 1000

        if min_duration_ms <= duration_ms <= max_duration_ms:
            blinks.append(
                BlinkEvent(
                    start_index=blink_start_idx,
                    end_index=blink_end_idx,
                    start_timestamp=blink_start_ts,
                    end_timestamp=blink_end_ts,
                    duration_ms=duration_ms,
                )
            )

    return blinks


def mark_blinks(
    df: pd.DataFrame,
    blinks: list[BlinkEvent] | None = None,
    column_name: str = "is_blink",
) -> pd.DataFrame:
    """Add column marking blink periods.

    Args:
        df: Input DataFrame
        blinks: List of blink events. If None, detects blinks first.
        column_name: Name for the blink marker column

    Returns:
        DataFrame with blink marker column added
    """
    df = df.copy()

    if blinks is None:
        blinks = detect_blinks(df)

    df[column_name] = False

    for blink in blinks:
        mask = (df.index >= blink.start_index) & (df.index <= blink.end_index)
        df.loc[mask, column_name] = True

    return df


def remove_blinks(
    df: pd.DataFrame,
    blinks: list[BlinkEvent] | None = None,
    method: Literal["remove", "nan", "interpolate"] = "nan",
    padding_ms: float = 50.0,
    timestamp_col: str = "Recording timestamp",
) -> pd.DataFrame:
    """Remove or replace blink periods.

    Args:
        df: Input DataFrame
        blinks: List of blink events. If None, detects blinks first.
        method: How to handle blink periods:
            - "remove": Delete blink rows
            - "nan": Replace with NaN
            - "interpolate": Replace and interpolate across blink
        padding_ms: Extra time to remove before/after blink (in ms)
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with blink periods handled
    """
    df = df.copy()

    if blinks is None:
        blinks = detect_blinks(df)

    if not blinks:
        return df

    # Convert padding from ms to microseconds
    padding_us = padding_ms * 1000

    # Create mask for all blink periods (with padding)
    blink_mask = pd.Series(False, index=df.index)

    for blink in blinks:
        start_ts = blink.start_timestamp - padding_us
        end_ts = blink.end_timestamp + padding_us
        mask = (df[timestamp_col] >= start_ts) & (df[timestamp_col] <= end_ts)
        blink_mask = blink_mask | mask

    if method == "remove":
        return df[~blink_mask].copy()

    # Set values to NaN for blink periods
    columns = [c for c in get_interpolatable_columns() if c in df.columns]
    for col in columns:
        df.loc[blink_mask, col] = np.nan

    if method == "interpolate":
        df = interpolate_missing(df, columns=columns, method="linear")

    return df


def get_blink_statistics(blinks: list[BlinkEvent], total_duration_ms: float | None = None) -> dict:
    """Compute statistics for detected blinks.

    Args:
        blinks: List of BlinkEvent objects
        total_duration_ms: Total recording duration in ms (for rate calculation)

    Returns:
        Dict with blink statistics
    """
    if not blinks:
        return {
            "count": 0,
            "mean_duration_ms": 0.0,
            "std_duration_ms": 0.0,
            "min_duration_ms": 0.0,
            "max_duration_ms": 0.0,
            "blink_rate_per_min": 0.0,
        }

    durations = [b.duration_ms for b in blinks]

    stats = {
        "count": len(blinks),
        "mean_duration_ms": np.mean(durations),
        "std_duration_ms": np.std(durations),
        "min_duration_ms": np.min(durations),
        "max_duration_ms": np.max(durations),
    }

    if total_duration_ms and total_duration_ms > 0:
        stats["blink_rate_per_min"] = len(blinks) / (total_duration_ms / 60000)
    else:
        stats["blink_rate_per_min"] = 0.0

    return stats


# =============================================================================
# Gap Detection Functions
# =============================================================================


def detect_gaps(
    df: pd.DataFrame,
    threshold_ms: float = GAP_THRESHOLD_MS,
    timestamp_col: str = "Recording timestamp",
) -> list[GapInfo]:
    """Detect gaps in recording (missing data periods).

    Args:
        df: Input DataFrame with timestamp column
        threshold_ms: Minimum gap duration to report (milliseconds)
        timestamp_col: Name of timestamp column (in microseconds)

    Returns:
        List of GapInfo objects describing each gap
    """
    if timestamp_col not in df.columns or len(df) < 2:
        return []

    # Calculate time differences (convert to ms)
    timestamps = df[timestamp_col].values
    diffs_ms = np.diff(timestamps) / 1000  # microseconds to ms

    # Expected interval at 100Hz is 10ms
    expected_interval_ms = SAMPLE_INTERVAL_MS

    gaps = []
    for i, diff in enumerate(diffs_ms):
        if diff > threshold_ms:
            gap_samples = int(diff / expected_interval_ms) - 1
            gaps.append(
                GapInfo(
                    start_index=df.index[i],
                    end_index=df.index[i + 1],
                    start_timestamp=timestamps[i],
                    end_timestamp=timestamps[i + 1],
                    duration_ms=diff,
                    gap_samples=gap_samples,
                )
            )

    return gaps


def get_gap_statistics(gaps: list[GapInfo], total_duration_ms: float | None = None) -> dict:
    """Compute statistics for detected gaps.

    Args:
        gaps: List of GapInfo objects
        total_duration_ms: Total recording duration in ms (for percentage calculation)

    Returns:
        Dict with gap statistics
    """
    if not gaps:
        return {
            "count": 0,
            "total_duration_ms": 0.0,
            "mean_duration_ms": 0.0,
            "max_duration_ms": 0.0,
            "total_samples_lost": 0,
            "gap_percentage": 0.0,
        }

    durations = [g.duration_ms for g in gaps]
    samples_lost = sum(g.gap_samples for g in gaps)

    stats = {
        "count": len(gaps),
        "total_duration_ms": sum(durations),
        "mean_duration_ms": np.mean(durations),
        "max_duration_ms": np.max(durations),
        "total_samples_lost": samples_lost,
    }

    if total_duration_ms and total_duration_ms > 0:
        stats["gap_percentage"] = (stats["total_duration_ms"] / total_duration_ms) * 100
    else:
        stats["gap_percentage"] = 0.0

    return stats


def mark_gaps(
    df: pd.DataFrame,
    gaps: list[GapInfo] | None = None,
    threshold_ms: float = GAP_THRESHOLD_MS,
    column_name: str = "after_gap",
) -> pd.DataFrame:
    """Add column marking samples after gaps.

    Useful for excluding post-gap samples from analysis.

    Args:
        df: Input DataFrame
        gaps: List of gap events. If None, detects gaps first.
        threshold_ms: Gap threshold if detecting gaps
        column_name: Name for the gap marker column

    Returns:
        DataFrame with gap marker column added
    """
    df = df.copy()

    if gaps is None:
        gaps = detect_gaps(df, threshold_ms=threshold_ms)

    df[column_name] = False

    for gap in gaps:
        df.loc[gap.end_index, column_name] = True

    return df


def split_at_gaps(
    df: pd.DataFrame,
    min_gap_ms: float = 1000.0,
    timestamp_col: str = "Recording timestamp",
) -> list[pd.DataFrame]:
    """Split recording into segments at large gaps.

    Args:
        df: Input DataFrame
        min_gap_ms: Minimum gap duration to split at
        timestamp_col: Name of timestamp column

    Returns:
        List of DataFrame segments
    """
    gaps = detect_gaps(df, threshold_ms=min_gap_ms, timestamp_col=timestamp_col)

    if not gaps:
        return [df.copy()]

    segments = []
    start_idx = 0

    for gap in gaps:
        # Get position in DataFrame
        gap_start_pos = df.index.get_loc(gap.start_index)
        segment = df.iloc[start_idx : gap_start_pos + 1].copy()
        if len(segment) > 0:
            segments.append(segment)
        start_idx = df.index.get_loc(gap.end_index)

    # Add final segment
    final_segment = df.iloc[start_idx:].copy()
    if len(final_segment) > 0:
        segments.append(final_segment)

    return segments


# =============================================================================
# Pipeline Integration Function
# =============================================================================


def postprocess_recording(
    df: pd.DataFrame,
    interpolate: bool = True,
    interpolate_method: InterpolationMethod = "linear",
    interpolate_max_gap: int | None = 5,
    remove_physiological_outliers: bool = True,
    detect_and_handle_blinks: bool = True,
    blink_handling: BlinkHandling = "mark",
    report_gaps: bool = True,
    gap_threshold_ms: float = 100.0,
    timestamp_col: str = "Recording timestamp",
) -> tuple[pd.DataFrame, dict]:
    """Full post-processing pipeline for a recording.

    Convenience function that applies all post-processing steps in order:
    1. Detect and report gaps
    2. Remove physiological outliers
    3. Detect and handle blinks
    4. Interpolate missing values

    Args:
        df: Cleaned DataFrame from clean_recording()
        interpolate: Whether to interpolate missing values
        interpolate_method: Interpolation strategy
        interpolate_max_gap: Maximum consecutive NaNs to interpolate
        remove_physiological_outliers: Filter values outside ranges
        detect_and_handle_blinks: Process blinks
        blink_handling: How to handle blinks ("mark", "nan", "remove", "interpolate")
        report_gaps: Whether to detect and report gaps
        gap_threshold_ms: Gap detection threshold
        timestamp_col: Name of timestamp column

    Returns:
        Tuple of (processed DataFrame, report dict with statistics)
    """
    df = df.copy()
    rows_before = len(df)
    report = {}

    # Calculate total duration for statistics
    if timestamp_col in df.columns and len(df) > 1:
        total_duration_ms = (df[timestamp_col].iloc[-1] - df[timestamp_col].iloc[0]) / 1000
    else:
        total_duration_ms = None

    # Step 1: Detect gaps
    if report_gaps:
        gaps = detect_gaps(df, threshold_ms=gap_threshold_ms, timestamp_col=timestamp_col)
        report["gaps"] = get_gap_statistics(gaps, total_duration_ms)
        report["gaps"]["events"] = gaps

    # Step 2: Compute initial missing rate
    report["missing_before"] = compute_missing_rate(df)

    # Step 3: Remove physiological outliers
    if remove_physiological_outliers:
        pupil_outliers = detect_pupil_outliers(df)
        gaze_outliers = detect_gaze_outliers(df)
        report["outliers"] = {
            "pupil_outlier_count": pupil_outliers.sum(),
            "gaze_outlier_count": gaze_outliers.sum(),
        }
        df = filter_physiological_range(df, method="nan")

    # Step 4: Detect and handle blinks
    if detect_and_handle_blinks:
        blinks = detect_blinks(df, timestamp_col=timestamp_col)
        report["blinks"] = get_blink_statistics(blinks, total_duration_ms)

        if blink_handling == "mark":
            df = mark_blinks(df, blinks)
        elif blink_handling in ("nan", "remove", "interpolate"):
            df = remove_blinks(df, blinks, method=blink_handling, timestamp_col=timestamp_col)

    # Step 5: Interpolate missing values
    if interpolate:
        df = interpolate_missing(df, method=interpolate_method, max_gap=interpolate_max_gap)

    # Step 6: Compute final missing rate
    report["missing_after"] = compute_missing_rate(df)

    # Summary statistics
    report["summary"] = {
        "rows_before": rows_before,
        "rows_after": len(df),
        "total_duration_ms": total_duration_ms,
    }

    return df, report
