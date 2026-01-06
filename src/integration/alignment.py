"""Timestamp alignment utilities for Boris-Tobii integration.

Provides functions to convert and align timestamps between Boris behavioral
observations (seconds) and Tobii eye-tracking data (microseconds).
"""

from __future__ import annotations

import pandas as pd

# =============================================================================
# Timestamp Conversion
# =============================================================================


def align_boris_to_tobii(
    boris_df: pd.DataFrame,
    tobii_df: pd.DataFrame,
    boris_start_col: str = "Start (s)",
    boris_stop_col: str = "Stop (s)",
    tobii_time_col: str = "Recording timestamp",
    offset_s: float = 0.0,
) -> pd.DataFrame:
    """Convert Boris timestamps to Tobii timestamp units.

    Boris uses seconds from recording start, Tobii uses microseconds.

    Args:
        boris_df: Boris aggregated events DataFrame.
        tobii_df: Tobii recording DataFrame.
        boris_start_col: Column name for Boris event start times.
        boris_stop_col: Column name for Boris event stop times.
        tobii_time_col: Column name for Tobii timestamps.
        offset_s: Manual offset to apply in seconds (positive = Boris starts after Tobii).

    Returns:
        Boris DataFrame with added 'tobii_start_us' and 'tobii_stop_us' columns.
    """
    boris_df = boris_df.copy()

    # Get Tobii recording start time (first timestamp)
    if tobii_time_col not in tobii_df.columns:
        raise ValueError(f"Tobii DataFrame missing column: {tobii_time_col}")

    tobii_start_us = tobii_df[tobii_time_col].min()

    # Convert Boris seconds to Tobii microseconds
    # Boris time (s) -> microseconds + Tobii start + offset
    if boris_start_col in boris_df.columns:
        boris_df["tobii_start_us"] = (
            boris_df[boris_start_col] + offset_s
        ) * 1_000_000 + tobii_start_us

    if boris_stop_col in boris_df.columns:
        boris_df["tobii_stop_us"] = (
            boris_df[boris_stop_col] + offset_s
        ) * 1_000_000 + tobii_start_us

    return boris_df


def tobii_to_seconds(
    tobii_df: pd.DataFrame,
    tobii_time_col: str = "Recording timestamp",
) -> pd.Series:
    """Convert Tobii timestamps to seconds from recording start.

    Args:
        tobii_df: Tobii recording DataFrame.
        tobii_time_col: Column name for Tobii timestamps.

    Returns:
        Series with time in seconds from recording start.
    """
    if tobii_time_col not in tobii_df.columns:
        raise ValueError(f"Missing column: {tobii_time_col}")

    start_us = tobii_df[tobii_time_col].min()
    return (tobii_df[tobii_time_col] - start_us) / 1_000_000


def find_tobii_time_range(
    tobii_df: pd.DataFrame,
    tobii_time_col: str = "Recording timestamp",
) -> tuple[float, float]:
    """Get start and end timestamps from Tobii data in seconds.

    Args:
        tobii_df: Tobii recording DataFrame.
        tobii_time_col: Column name for Tobii timestamps.

    Returns:
        Tuple of (start_s, end_s) in seconds from recording start.
    """
    if tobii_time_col not in tobii_df.columns:
        raise ValueError(f"Missing column: {tobii_time_col}")

    start_us = tobii_df[tobii_time_col].min()
    end_us = tobii_df[tobii_time_col].max()

    # Return as seconds from start (so start is always 0)
    return (0.0, (end_us - start_us) / 1_000_000)


def find_boris_time_range(
    boris_df: pd.DataFrame,
    start_col: str = "Start (s)",
    stop_col: str = "Stop (s)",
) -> tuple[float, float]:
    """Get start and end timestamps from Boris data in seconds.

    Args:
        boris_df: Boris aggregated events DataFrame.
        start_col: Column name for event start times.
        stop_col: Column name for event stop times.

    Returns:
        Tuple of (start_s, end_s) in seconds.
    """
    if start_col not in boris_df.columns or stop_col not in boris_df.columns:
        raise ValueError(f"Missing columns: {start_col} or {stop_col}")

    return (boris_df[start_col].min(), boris_df[stop_col].max())


# =============================================================================
# Offset Estimation
# =============================================================================


def compute_alignment_offset(
    boris_df: pd.DataFrame,
    tobii_df: pd.DataFrame,
    method: str = "start",
    tobii_time_col: str = "Recording timestamp",
    boris_start_col: str = "Start (s)",
    boris_stop_col: str = "Stop (s)",
) -> float:
    """Estimate temporal offset between Boris and Tobii recordings.

    Useful when recordings don't start at exactly the same time.

    Args:
        boris_df: Boris aggregated events DataFrame.
        tobii_df: Tobii recording DataFrame.
        method:
            - "start": Align recording starts (assume both start at t=0).
            - "end": Align recording ends.
            - "center": Align centers of recordings.
        tobii_time_col: Column name for Tobii timestamps.
        boris_start_col: Column name for Boris start times.
        boris_stop_col: Column name for Boris stop times.

    Returns:
        Offset in seconds (positive = Boris starts after Tobii).
    """
    # Get time ranges
    _, tobii_duration = find_tobii_time_range(tobii_df, tobii_time_col)
    boris_start, boris_end = find_boris_time_range(boris_df, boris_start_col, boris_stop_col)

    if method == "start":
        # Assume both recordings start at the same time
        # Offset is how much Boris's first event is offset from t=0
        return -boris_start  # Subtract boris_start to align to 0

    if method == "end":
        # Align ends
        # Boris end should match Tobii end
        return tobii_duration - boris_end

    if method == "center":
        # Align centers
        tobii_center = tobii_duration / 2
        boris_center = (boris_start + boris_end) / 2
        return tobii_center - boris_center

    raise ValueError(f"Unknown method: {method}. Use 'start', 'end', or 'center'.")


# =============================================================================
# Validation
# =============================================================================


def validate_alignment(
    boris_df: pd.DataFrame,
    tobii_df: pd.DataFrame,
    tobii_time_col: str = "Recording timestamp",
    boris_start_col: str = "Start (s)",
    boris_stop_col: str = "Stop (s)",
    duration_tolerance: float = 0.1,
) -> dict:
    """Validate that Boris and Tobii data can be aligned.

    Checks duration compatibility, overlapping time ranges, and potential issues.

    Args:
        boris_df: Boris aggregated events DataFrame.
        tobii_df: Tobii recording DataFrame.
        tobii_time_col: Column name for Tobii timestamps.
        boris_start_col: Column name for Boris start times.
        boris_stop_col: Column name for Boris stop times.
        duration_tolerance: Acceptable difference in duration (as fraction).

    Returns:
        Dict with: is_valid, warnings, boris_duration, tobii_duration,
                   duration_diff_pct, overlap_pct.
    """
    warnings = []
    is_valid = True

    # Get durations
    _, tobii_duration = find_tobii_time_range(tobii_df, tobii_time_col)
    boris_start, boris_end = find_boris_time_range(boris_df, boris_start_col, boris_stop_col)
    boris_duration = boris_end - boris_start

    # Check for empty data
    if len(boris_df) == 0:
        warnings.append("Boris DataFrame is empty")
        is_valid = False

    if len(tobii_df) == 0:
        warnings.append("Tobii DataFrame is empty")
        is_valid = False

    # Duration comparison
    if tobii_duration > 0:
        duration_diff = abs(boris_duration - tobii_duration) / tobii_duration
    else:
        duration_diff = float("inf")

    if duration_diff > duration_tolerance:
        warnings.append(
            f"Duration mismatch: Boris={boris_duration:.1f}s, Tobii={tobii_duration:.1f}s "
            f"(diff={duration_diff * 100:.1f}%)"
        )

    # Check if Boris events fall within Tobii range
    if boris_start < 0:
        warnings.append(f"Boris has negative timestamps (earliest: {boris_start:.2f}s)")

    if boris_end > tobii_duration * 1.1:  # Allow 10% overshoot
        warnings.append(
            f"Boris extends beyond Tobii recording "
            f"(Boris end: {boris_end:.1f}s, Tobii duration: {tobii_duration:.1f}s)"
        )

    # Calculate overlap
    overlap_start = max(0, boris_start)
    overlap_end = min(tobii_duration, boris_end)
    overlap_duration = max(0, overlap_end - overlap_start)

    if boris_duration > 0:
        overlap_pct = overlap_duration / boris_duration
    else:
        overlap_pct = 0.0

    if overlap_pct < 0.5:
        warnings.append(
            f"Low overlap: only {overlap_pct * 100:.1f}% of Boris events in Tobii range"
        )
        is_valid = False

    return {
        "is_valid": is_valid,
        "warnings": warnings,
        "boris_duration": boris_duration,
        "tobii_duration": tobii_duration,
        "duration_diff_pct": duration_diff * 100,
        "overlap_pct": overlap_pct * 100,
        "boris_start": boris_start,
        "boris_end": boris_end,
    }


# =============================================================================
# Timestamp Lookup
# =============================================================================


def find_tobii_index_at_time(
    tobii_df: pd.DataFrame,
    time_s: float,
    tobii_time_col: str = "Recording timestamp",
) -> int:
    """Find the Tobii DataFrame index closest to a given time in seconds.

    Args:
        tobii_df: Tobii recording DataFrame.
        time_s: Time in seconds from recording start.
        tobii_time_col: Column name for Tobii timestamps.

    Returns:
        DataFrame index of closest sample.
    """
    if tobii_time_col not in tobii_df.columns:
        raise ValueError(f"Missing column: {tobii_time_col}")

    start_us = tobii_df[tobii_time_col].min()
    target_us = start_us + time_s * 1_000_000

    # Find closest index
    time_diff = (tobii_df[tobii_time_col] - target_us).abs()
    return time_diff.idxmin()


def find_tobii_indices_in_range(
    tobii_df: pd.DataFrame,
    start_s: float,
    stop_s: float,
    tobii_time_col: str = "Recording timestamp",
) -> pd.Index:
    """Find all Tobii DataFrame indices within a time range.

    Args:
        tobii_df: Tobii recording DataFrame.
        start_s: Start time in seconds from recording start.
        stop_s: Stop time in seconds from recording start.
        tobii_time_col: Column name for Tobii timestamps.

    Returns:
        Index of samples within the range.
    """
    if tobii_time_col not in tobii_df.columns:
        raise ValueError(f"Missing column: {tobii_time_col}")

    tobii_start_us = tobii_df[tobii_time_col].min()
    start_us = tobii_start_us + start_s * 1_000_000
    stop_us = tobii_start_us + stop_s * 1_000_000

    mask = (tobii_df[tobii_time_col] >= start_us) & (tobii_df[tobii_time_col] <= stop_us)
    return tobii_df.index[mask]
