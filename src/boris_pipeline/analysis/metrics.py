"""Core metric calculations for BORIS behavioral observation data.

Provides functions to compute behavioral frequencies, durations,
temporal patterns, and transition analysis from BORIS aggregated events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# =============================================================================
# Behavioral Frequency & Duration Metrics
# =============================================================================


def compute_behavior_frequency(
    df: pd.DataFrame,
    behavior_col: str = "Behavior",
    normalize: bool = False,
) -> dict[str, int | float]:
    """Compute occurrence counts for each behavior.

    Args:
        df: Aggregated events DataFrame with behavior column.
        behavior_col: Column name for behavior labels.
        normalize: If True, return proportions instead of counts.

    Returns:
        Dict mapping behavior names to counts or proportions.
    """
    if behavior_col not in df.columns or len(df) == 0:
        return {}

    counts = df[behavior_col].value_counts()

    if normalize:
        counts = counts / counts.sum()

    return counts.to_dict()


def compute_behavior_durations(
    df: pd.DataFrame,
    behavior_col: str = "Behavior",
    duration_col: str = "Duration (s)",
) -> dict[str, dict]:
    """Compute duration statistics per behavior.

    Args:
        df: Aggregated events DataFrame.
        behavior_col: Column name for behavior labels.
        duration_col: Column name for event durations.

    Returns:
        Dict per behavior with: total_duration, mean, std, min, max, count.
    """
    if behavior_col not in df.columns or duration_col not in df.columns:
        return {}

    if len(df) == 0:
        return {}

    result = {}
    for behavior, group in df.groupby(behavior_col):
        durations = group[duration_col].dropna()
        if len(durations) > 0:
            result[behavior] = {
                "total_duration": durations.sum(),
                "mean": durations.mean(),
                "std": durations.std() if len(durations) > 1 else 0.0,
                "min": durations.min(),
                "max": durations.max(),
                "count": len(durations),
            }

    return result


def compute_behavior_rate(
    df: pd.DataFrame,
    total_duration_s: float | None = None,
    behavior_col: str = "Behavior",
) -> dict[str, float]:
    """Compute behavior rate (occurrences per minute).

    Args:
        df: Aggregated events DataFrame.
        total_duration_s: Total observation duration in seconds.
            If None, inferred from data (max stop - min start).
        behavior_col: Column name for behavior labels.

    Returns:
        Dict mapping behavior to occurrences per minute.
    """
    if behavior_col not in df.columns or len(df) == 0:
        return {}

    if total_duration_s is None:
        if "Start (s)" in df.columns and "Stop (s)" in df.columns:
            total_duration_s = df["Stop (s)"].max() - df["Start (s)"].min()
        else:
            return {}

    if total_duration_s <= 0:
        return {}

    counts = df[behavior_col].value_counts()
    # Convert to per-minute rate
    duration_minutes = total_duration_s / 60.0

    return (counts / duration_minutes).to_dict()


def compute_time_budget(
    df: pd.DataFrame,
    total_duration_s: float | None = None,
    behavior_col: str = "Behavior",
    duration_col: str = "Duration (s)",
) -> dict[str, float]:
    """Compute percentage of time spent in each behavior.

    Args:
        df: Aggregated events DataFrame.
        total_duration_s: Total observation duration. If None, inferred from data.
        behavior_col: Column name for behavior labels.
        duration_col: Column name for event durations.

    Returns:
        Dict mapping behavior to percentage of total time (0.0 to 1.0).
    """
    if behavior_col not in df.columns or duration_col not in df.columns:
        return {}

    if len(df) == 0:
        return {}

    if total_duration_s is None:
        if "Start (s)" in df.columns and "Stop (s)" in df.columns:
            total_duration_s = df["Stop (s)"].max() - df["Start (s)"].min()
        else:
            total_duration_s = df[duration_col].sum()

    if total_duration_s <= 0:
        return {}

    time_per_behavior = df.groupby(behavior_col)[duration_col].sum()

    return (time_per_behavior / total_duration_s).to_dict()


# =============================================================================
# Temporal Pattern Metrics
# =============================================================================


def compute_inter_event_intervals(
    df: pd.DataFrame,
    behavior: str | None = None,
    behavior_col: str = "Behavior",
    start_col: str = "Start (s)",
    stop_col: str = "Stop (s)",
) -> dict[str, dict]:
    """Compute inter-event interval statistics.

    The inter-event interval is the gap between the end of one event
    and the start of the next event of the same behavior.

    Args:
        df: Aggregated events DataFrame (must be sorted by start time).
        behavior: If specified, compute only for this behavior.
        behavior_col: Column name for behavior labels.
        start_col: Column name for event start times.
        stop_col: Column name for event stop times.

    Returns:
        Dict per behavior with: mean, std, min, max, count.
    """
    required_cols = [behavior_col, start_col, stop_col]
    if not all(col in df.columns for col in required_cols):
        return {}

    if len(df) == 0:
        return {}

    # Ensure sorted by start time
    df = df.sort_values(start_col)

    behaviors_to_process = [behavior] if behavior else df[behavior_col].unique()
    result = {}

    for beh in behaviors_to_process:
        beh_df = df[df[behavior_col] == beh].copy()
        if len(beh_df) < 2:
            continue

        # Compute intervals: start of next event - stop of current event
        intervals = beh_df[start_col].iloc[1:].values - beh_df[stop_col].iloc[:-1].values
        intervals = intervals[intervals >= 0]  # Filter out overlaps

        if len(intervals) > 0:
            result[beh] = {
                "mean": float(np.mean(intervals)),
                "std": float(np.std(intervals)) if len(intervals) > 1 else 0.0,
                "min": float(np.min(intervals)),
                "max": float(np.max(intervals)),
                "count": len(intervals),
            }

    return result


def compute_behavior_latency(
    df: pd.DataFrame,
    reference_time: float = 0.0,
    behavior_col: str = "Behavior",
    start_col: str = "Start (s)",
) -> dict[str, float]:
    """Compute latency to first occurrence of each behavior.

    Args:
        df: Aggregated events DataFrame.
        reference_time: Start time reference (usually 0).
        behavior_col: Column name for behavior labels.
        start_col: Column name for event start times.

    Returns:
        Dict mapping behavior to latency in seconds.
    """
    if behavior_col not in df.columns or start_col not in df.columns:
        return {}

    if len(df) == 0:
        return {}

    # Get first occurrence of each behavior
    first_occurrences = df.groupby(behavior_col)[start_col].min()

    return (first_occurrences - reference_time).to_dict()


# =============================================================================
# Transition Analysis
# =============================================================================


def compute_transition_matrix(
    df: pd.DataFrame,
    behavior_col: str = "Behavior",
    normalize: bool = True,
) -> pd.DataFrame:
    """Compute behavior transition probability matrix.

    Args:
        df: Aggregated events DataFrame (must be sorted by Start time).
        behavior_col: Column name for behavior labels.
        normalize: If True, normalize rows to probabilities.

    Returns:
        DataFrame with behaviors as both index and columns.
        Values are transition counts or probabilities.
    """
    if behavior_col not in df.columns or len(df) < 2:
        return pd.DataFrame()

    # Ensure sorted by start time
    if "Start (s)" in df.columns:
        df = df.sort_values("Start (s)")

    behaviors = df[behavior_col].values

    # Count transitions
    transitions = {}
    for i in range(len(behaviors) - 1):
        from_beh = behaviors[i]
        to_beh = behaviors[i + 1]
        if from_beh not in transitions:
            transitions[from_beh] = {}
        transitions[from_beh][to_beh] = transitions[from_beh].get(to_beh, 0) + 1

    # Create matrix
    all_behaviors = sorted(df[behavior_col].unique())
    matrix = pd.DataFrame(0, index=all_behaviors, columns=all_behaviors)

    for from_beh, to_dict in transitions.items():
        for to_beh, count in to_dict.items():
            matrix.loc[from_beh, to_beh] = count

    if normalize:
        row_sums = matrix.sum(axis=1)
        # Avoid division by zero
        row_sums = row_sums.replace(0, 1)
        matrix = matrix.div(row_sums, axis=0)

    return matrix


def compute_transition_counts(
    df: pd.DataFrame,
    behavior_col: str = "Behavior",
) -> pd.DataFrame:
    """Compute raw transition counts between behaviors.

    Convenience wrapper for compute_transition_matrix with normalize=False.

    Args:
        df: Aggregated events DataFrame.
        behavior_col: Column name for behavior labels.

    Returns:
        DataFrame with raw transition counts.
    """
    return compute_transition_matrix(df, behavior_col=behavior_col, normalize=False)


def compute_sequence_entropy(
    df: pd.DataFrame,
    behavior_col: str = "Behavior",
) -> float:
    """Compute entropy of behavior sequence.

    Higher entropy indicates more random/diverse behavior patterns.
    Lower entropy indicates more predictable/repetitive patterns.

    Args:
        df: Aggregated events DataFrame.
        behavior_col: Column name for behavior labels.

    Returns:
        Shannon entropy value (in bits).
    """
    if behavior_col not in df.columns or len(df) == 0:
        return 0.0

    # Compute behavior probabilities
    value_counts = df[behavior_col].value_counts(normalize=True)
    probabilities = value_counts.values

    # Compute Shannon entropy
    # H = -sum(p * log2(p))
    probabilities = probabilities[probabilities > 0]  # Avoid log(0)
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return float(entropy)


def compute_behavior_bout_stats(
    df: pd.DataFrame,
    behavior: str,
    behavior_col: str = "Behavior",
    start_col: str = "Start (s)",
    stop_col: str = "Stop (s)",
    duration_col: str = "Duration (s)",
    max_gap_s: float = 1.0,
) -> dict:
    """Compute bout (consecutive occurrence) statistics for a behavior.

    A bout is a sequence of events of the same behavior separated by
    gaps smaller than max_gap_s.

    Args:
        df: Aggregated events DataFrame.
        behavior: Behavior to analyze.
        behavior_col: Column name for behavior labels.
        start_col: Column name for event start times.
        stop_col: Column name for event stop times.
        duration_col: Column name for event durations.
        max_gap_s: Maximum gap in seconds to consider events as same bout.

    Returns:
        Dict with: bout_count, bout_duration_mean, bout_duration_std,
                   inter_bout_interval_mean.
    """
    required_cols = [behavior_col, start_col, stop_col, duration_col]
    if not all(col in df.columns for col in required_cols):
        return {
            "bout_count": 0,
            "bout_duration_mean": np.nan,
            "bout_duration_std": np.nan,
            "inter_bout_interval_mean": np.nan,
        }

    # Filter for target behavior and sort
    beh_df = df[df[behavior_col] == behavior].sort_values(start_col).copy()

    if len(beh_df) == 0:
        return {
            "bout_count": 0,
            "bout_duration_mean": np.nan,
            "bout_duration_std": np.nan,
            "inter_bout_interval_mean": np.nan,
        }

    # Identify bouts
    bouts = []
    current_bout_start = beh_df[start_col].iloc[0]
    current_bout_end = beh_df[stop_col].iloc[0]

    for i in range(1, len(beh_df)):
        gap = beh_df[start_col].iloc[i] - current_bout_end
        if gap <= max_gap_s:
            # Continue current bout
            current_bout_end = beh_df[stop_col].iloc[i]
        else:
            # End current bout, start new one
            bouts.append((current_bout_start, current_bout_end))
            current_bout_start = beh_df[start_col].iloc[i]
            current_bout_end = beh_df[stop_col].iloc[i]

    # Don't forget the last bout
    bouts.append((current_bout_start, current_bout_end))

    bout_durations = [end - start for start, end in bouts]

    # Compute inter-bout intervals
    inter_bout_intervals = []
    for i in range(1, len(bouts)):
        interval = bouts[i][0] - bouts[i - 1][1]
        inter_bout_intervals.append(interval)

    return {
        "bout_count": len(bouts),
        "bout_duration_mean": float(np.mean(bout_durations)),
        "bout_duration_std": float(np.std(bout_durations)) if len(bout_durations) > 1 else 0.0,
        "inter_bout_interval_mean": float(np.mean(inter_bout_intervals))
        if inter_bout_intervals
        else np.nan,
    }


# =============================================================================
# Summary
# =============================================================================


def compute_recording_summary(
    df: pd.DataFrame,
    compute_transitions: bool = True,
    behavior_col: str = "Behavior",
) -> dict:
    """Compute all behavioral metrics for a recording.

    Args:
        df: Aggregated events DataFrame.
        compute_transitions: Whether to compute transition matrix.
        behavior_col: Column name for behavior labels.

    Returns:
        Dict with: behavior_frequency, behavior_durations, behavior_rate,
                   time_budget, inter_event_intervals, sequence_entropy,
                   transitions (optional).
    """
    summary = {
        "behavior_frequency": compute_behavior_frequency(df, behavior_col=behavior_col),
        "behavior_durations": compute_behavior_durations(df, behavior_col=behavior_col),
        "behavior_rate": compute_behavior_rate(df, behavior_col=behavior_col),
        "time_budget": compute_time_budget(df, behavior_col=behavior_col),
        "inter_event_intervals": compute_inter_event_intervals(df, behavior_col=behavior_col),
        "behavior_latency": compute_behavior_latency(df, behavior_col=behavior_col),
        "sequence_entropy": compute_sequence_entropy(df, behavior_col=behavior_col),
    }

    if compute_transitions:
        summary["transitions"] = compute_transition_matrix(df, behavior_col=behavior_col).to_dict()

    return summary
