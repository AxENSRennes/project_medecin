"""Post-processing functions for cleaned Tobii eye-tracking data.

This module provides data quality improvements including:
- Missing data handling (interpolation, fill, removal)
- Outlier detection and removal (pupil, gaze)
- Blink interpolation (via MNE)
- Event detection (via pymovements)
- Gap detection and splitting
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from tobii_pipeline.adapters.mne_adapter import (
    get_blink_statistics_mne,
    interpolate_blinks_mne,
)
from tobii_pipeline.adapters.pymovements_adapter import (
    apply_pix2deg,
    apply_pos2vel,
    compute_event_properties,
    detect_events_idt,
    detect_events_ivt,
    df_to_gaze_dataframe,
    events_to_df,
)

# Screen dimensions (Tobii recording setup)
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# Physiological bounds for pupil diameter (mm)
PUPIL_MIN_MM = 2.0
PUPIL_MAX_MM = 8.0

# Sampling rate
SAMPLING_RATE_HZ = 100
SAMPLE_INTERVAL_MS = 10  # 1000 / 100 Hz

# Gap detection
GAP_THRESHOLD_MS = 100

# Type aliases
InterpolationMethod = Literal["linear", "ffill", "bfill", "nearest", "mean", "median", "drop"]
OutlierHandling = Literal["remove", "nan", "interpolate"]
EventDetectionMethod = Literal["ivt", "idt"]


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
    method: OutlierHandling = "nan",
) -> pd.DataFrame:
    """Apply all physiological range filters.

    Convenience function combining multiple outlier detection methods.

    Args:
        df: Input DataFrame
        remove_pupil_outliers: Filter pupil diameter outliers
        remove_gaze_outliers: Filter off-screen gaze
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

    return remove_outliers(df, combined_outliers, method=method)


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
# Event Detection (via pymovements)
# =============================================================================


def detect_events(
    df: pd.DataFrame,
    method: EventDetectionMethod = "ivt",
    velocity_threshold: float = 30.0,
    dispersion_threshold: float = 1.0,
    minimum_duration: int = 100,
    screen_width_px: int = SCREEN_WIDTH,
    screen_height_px: int = SCREEN_HEIGHT,
) -> tuple[pd.DataFrame, dict]:
    """Detect fixations and saccades using pymovements algorithms.

    Args:
        df: Cleaned Tobii DataFrame with gaze data.
        method: Detection algorithm:
            - "ivt": Velocity-threshold (faster, recommended)
            - "idt": Dispersion-threshold (more accurate for noisy data)
        velocity_threshold: For I-VT, velocity threshold in deg/s.
        dispersion_threshold: For I-DT, max dispersion in degrees.
        minimum_duration: Minimum event duration in ms.
        screen_width_px: Screen width in pixels.
        screen_height_px: Screen height in pixels.

    Returns:
        Tuple of (events DataFrame, statistics dict)
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

    # Detect events
    if method == "ivt":
        gaze = detect_events_ivt(
            gaze,
            velocity_threshold=velocity_threshold,
            minimum_duration=minimum_duration,
        )
    else:  # idt
        gaze = detect_events_idt(
            gaze,
            dispersion_threshold=dispersion_threshold,
            minimum_duration=minimum_duration,
        )

    # Compute event properties
    gaze = compute_event_properties(gaze)

    # Extract events
    events_df = events_to_df(gaze)

    # Compute statistics
    from tobii_pipeline.adapters.pymovements_adapter import (
        get_fixation_stats,
        get_saccade_stats,
    )

    stats = {
        "fixations": get_fixation_stats(events_df),
        "saccades": get_saccade_stats(events_df),
    }

    return events_df, stats


# =============================================================================
# Pipeline Integration Function
# =============================================================================


def postprocess_recording(
    df: pd.DataFrame,
    interpolate: bool = True,
    interpolate_method: InterpolationMethod = "linear",
    interpolate_max_gap: int | None = 5,
    remove_physiological_outliers: bool = True,
    interpolate_blinks: bool = True,
    blink_buffer_before: float = 0.05,
    blink_buffer_after: float = 0.2,
    detect_eye_events: bool = True,
    event_detection_method: EventDetectionMethod = "ivt",
    report_gaps: bool = True,
    gap_threshold_ms: float = 100.0,
    timestamp_col: str = "Recording timestamp",
) -> tuple[pd.DataFrame, dict]:
    """Full post-processing pipeline for a recording.

    Applies all post-processing steps using library implementations:
    1. Detect and report gaps
    2. Remove physiological outliers
    3. Interpolate blinks (via MNE)
    4. Interpolate remaining missing values
    5. Detect eye events (via pymovements)

    Args:
        df: Cleaned DataFrame from clean_recording()
        interpolate: Whether to interpolate missing values
        interpolate_method: Interpolation strategy
        interpolate_max_gap: Maximum consecutive NaNs to interpolate
        remove_physiological_outliers: Filter values outside ranges
        interpolate_blinks: Use MNE to interpolate blink periods
        blink_buffer_before: Time before blink to include (seconds)
        blink_buffer_after: Time after blink to include (seconds)
        detect_eye_events: Detect fixations/saccades via pymovements
        event_detection_method: Algorithm for event detection ("ivt" or "idt")
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
            "pupil_outlier_count": int(pupil_outliers.sum()),
            "gaze_outlier_count": int(gaze_outliers.sum()),
        }
        df = filter_physiological_range(df, method="nan")

    # Step 4: Interpolate blinks using MNE
    if interpolate_blinks:
        report["blinks"] = get_blink_statistics_mne(df, sfreq=SAMPLING_RATE_HZ)
        df = interpolate_blinks_mne(
            df,
            buffer_before=blink_buffer_before,
            buffer_after=blink_buffer_after,
            sfreq=SAMPLING_RATE_HZ,
        )

    # Step 5: Interpolate remaining missing values
    if interpolate:
        df = interpolate_missing(df, method=interpolate_method, max_gap=interpolate_max_gap)

    # Step 6: Compute final missing rate
    report["missing_after"] = compute_missing_rate(df)

    # Step 7: Detect eye events using pymovements
    if detect_eye_events:
        try:
            events_df, event_stats = detect_events(df, method=event_detection_method)
            report["events"] = events_df
            report["event_stats"] = event_stats
        except Exception as e:
            report["event_detection_error"] = str(e)

    # Summary statistics
    report["summary"] = {
        "rows_before": rows_before,
        "rows_after": len(df),
        "total_duration_ms": total_duration_ms,
    }

    return df, report
