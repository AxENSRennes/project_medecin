"""Epoch extraction for Boris-Tobii integration.

Provides functions to extract Tobii eye-tracking data during specific
behavioral events from Boris observations.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from .alignment import find_tobii_indices_in_range, tobii_to_seconds

# =============================================================================
# Epoch Extraction
# =============================================================================


def extract_tobii_epochs(
    tobii_df: pd.DataFrame,
    boris_df: pd.DataFrame,
    behavior: str | None = None,
    time_before_s: float = 0.0,
    time_after_s: float = 0.0,
    behavior_col: str = "Behavior",
    boris_start_col: str = "Start (s)",
    boris_stop_col: str = "Stop (s)",
    tobii_time_col: str = "Recording timestamp",
) -> list[pd.DataFrame]:
    """Extract Tobii data epochs during Boris behavioral events.

    Args:
        tobii_df: Tobii recording DataFrame.
        boris_df: Boris aggregated events DataFrame.
        behavior: If specified, only extract for this behavior. Otherwise, all.
        time_before_s: Include this much time before event onset.
        time_after_s: Include this much time after event offset.
        behavior_col: Column name for Boris behavior labels.
        boris_start_col: Column name for Boris event start times.
        boris_stop_col: Column name for Boris event stop times.
        tobii_time_col: Column name for Tobii timestamps.

    Returns:
        List of DataFrame slices, one per behavioral event.
    """
    # Filter Boris events if behavior specified
    if behavior is not None:
        boris_events = boris_df[boris_df[behavior_col] == behavior].copy()
    else:
        boris_events = boris_df.copy()

    if len(boris_events) == 0:
        return []

    epochs = []

    for _, event in boris_events.iterrows():
        start_s = event[boris_start_col] - time_before_s
        stop_s = event[boris_stop_col] + time_after_s

        # Get indices in this time range
        indices = find_tobii_indices_in_range(
            tobii_df, start_s, stop_s, tobii_time_col=tobii_time_col
        )

        if len(indices) > 0:
            epoch = tobii_df.loc[indices].copy()
            # Add metadata about the event
            epoch["_event_behavior"] = event[behavior_col]
            epoch["_event_start_s"] = event[boris_start_col]
            epoch["_event_stop_s"] = event[boris_stop_col]
            epochs.append(epoch)

    return epochs


def extract_epoch_by_event_index(
    tobii_df: pd.DataFrame,
    boris_df: pd.DataFrame,
    event_index: int,
    time_before_s: float = 0.0,
    time_after_s: float = 0.0,
    boris_start_col: str = "Start (s)",
    boris_stop_col: str = "Stop (s)",
    tobii_time_col: str = "Recording timestamp",
) -> pd.DataFrame:
    """Extract Tobii data for a specific Boris event by index.

    Args:
        tobii_df: Tobii recording DataFrame.
        boris_df: Boris aggregated events DataFrame.
        event_index: Index of the event in boris_df.
        time_before_s: Include this much time before event onset.
        time_after_s: Include this much time after event offset.
        boris_start_col: Column name for Boris event start times.
        boris_stop_col: Column name for Boris event stop times.
        tobii_time_col: Column name for Tobii timestamps.

    Returns:
        DataFrame slice for the specified event.
    """
    if event_index not in boris_df.index:
        return pd.DataFrame()

    event = boris_df.loc[event_index]
    start_s = event[boris_start_col] - time_before_s
    stop_s = event[boris_stop_col] + time_after_s

    indices = find_tobii_indices_in_range(tobii_df, start_s, stop_s, tobii_time_col=tobii_time_col)

    if len(indices) == 0:
        return pd.DataFrame()

    return tobii_df.loc[indices].copy()


def create_epoch_dataset(
    tobii_df: pd.DataFrame,
    boris_df: pd.DataFrame,
    behaviors: list[str] | None = None,
    time_before_s: float = 0.0,
    time_after_s: float = 0.0,
    add_behavior_label: bool = True,
    behavior_col: str = "Behavior",
    boris_start_col: str = "Start (s)",
    boris_stop_col: str = "Stop (s)",
    tobii_time_col: str = "Recording timestamp",
) -> pd.DataFrame:
    """Create combined dataset with behavior labels added to Tobii data.

    Adds 'behavior' and 'event_id' columns to Tobii samples that fall
    within behavioral events. Samples outside events are labeled as "none".

    Args:
        tobii_df: Tobii recording DataFrame.
        boris_df: Boris aggregated events DataFrame.
        behaviors: List of behaviors to include. If None, all behaviors.
        time_before_s: Include this much time before event onset.
        time_after_s: Include this much time after event offset.
        add_behavior_label: Whether to add behavior label column.
        behavior_col: Column name for Boris behavior labels.
        boris_start_col: Column name for Boris event start times.
        boris_stop_col: Column name for Boris event stop times.
        tobii_time_col: Column name for Tobii timestamps.

    Returns:
        Tobii DataFrame with added behavior annotation columns.
    """
    result = tobii_df.copy()

    # Initialize columns
    if add_behavior_label:
        result["behavior"] = "none"
    result["event_id"] = -1

    # Convert Tobii timestamps to seconds for comparison
    tobii_time_s = tobii_to_seconds(result, tobii_time_col)

    # Filter behaviors if specified
    if behaviors is not None:
        boris_filtered = boris_df[boris_df[behavior_col].isin(behaviors)]
    else:
        boris_filtered = boris_df

    # Label each event
    for event_idx, (_, event) in enumerate(boris_filtered.iterrows()):
        start_s = event[boris_start_col] - time_before_s
        stop_s = event[boris_stop_col] + time_after_s

        mask = (tobii_time_s >= start_s) & (tobii_time_s <= stop_s)

        if add_behavior_label:
            result.loc[mask, "behavior"] = event[behavior_col]
        result.loc[mask, "event_id"] = event_idx

    return result


# =============================================================================
# Event-Locked Analysis
# =============================================================================


def align_to_behavior_onset(
    tobii_df: pd.DataFrame,
    boris_df: pd.DataFrame,
    behavior: str,
    window_before_s: float = 1.0,
    window_after_s: float = 3.0,
    behavior_col: str = "Behavior",
    boris_start_col: str = "Start (s)",
    tobii_time_col: str = "Recording timestamp",
) -> pd.DataFrame:
    """Create event-locked Tobii data aligned to behavior onsets.

    Re-centers time to 0 at each behavior onset for averaging across events.

    Args:
        tobii_df: Tobii recording DataFrame.
        boris_df: Boris aggregated events DataFrame.
        behavior: Behavior to align to.
        window_before_s: Time before onset to include.
        window_after_s: Time after onset to include.
        behavior_col: Column name for Boris behavior labels.
        boris_start_col: Column name for Boris event start times.
        tobii_time_col: Column name for Tobii timestamps.

    Returns:
        DataFrame with 'relative_time_s' (0 at onset) and 'event_id' columns.
    """
    boris_events = boris_df[boris_df[behavior_col] == behavior]

    if len(boris_events) == 0:
        return pd.DataFrame()

    # Convert Tobii to seconds from start
    tobii_time_s = tobii_to_seconds(tobii_df, tobii_time_col)

    all_epochs = []

    for event_idx, (_, event) in enumerate(boris_events.iterrows()):
        onset_s = event[boris_start_col]

        # Define window
        start_s = onset_s - window_before_s
        stop_s = onset_s + window_after_s

        # Get samples in window
        mask = (tobii_time_s >= start_s) & (tobii_time_s <= stop_s)

        if mask.sum() > 0:
            epoch = tobii_df.loc[mask].copy()
            # Re-center time relative to onset
            epoch["relative_time_s"] = tobii_time_s.loc[mask] - onset_s
            epoch["event_id"] = event_idx
            all_epochs.append(epoch)

    if not all_epochs:
        return pd.DataFrame()

    return pd.concat(all_epochs, ignore_index=True)


def align_to_behavior_offset(
    tobii_df: pd.DataFrame,
    boris_df: pd.DataFrame,
    behavior: str,
    window_before_s: float = 1.0,
    window_after_s: float = 3.0,
    behavior_col: str = "Behavior",
    boris_stop_col: str = "Stop (s)",
    tobii_time_col: str = "Recording timestamp",
) -> pd.DataFrame:
    """Create event-locked Tobii data aligned to behavior offsets.

    Args:
        tobii_df: Tobii recording DataFrame.
        boris_df: Boris aggregated events DataFrame.
        behavior: Behavior to align to.
        window_before_s: Time before offset to include.
        window_after_s: Time after offset to include.
        behavior_col: Column name for Boris behavior labels.
        boris_stop_col: Column name for Boris event stop times.
        tobii_time_col: Column name for Tobii timestamps.

    Returns:
        DataFrame with 'relative_time_s' (0 at offset) and 'event_id' columns.
    """
    boris_events = boris_df[boris_df[behavior_col] == behavior]

    if len(boris_events) == 0:
        return pd.DataFrame()

    tobii_time_s = tobii_to_seconds(tobii_df, tobii_time_col)

    all_epochs = []

    for event_idx, (_, event) in enumerate(boris_events.iterrows()):
        offset_s = event[boris_stop_col]

        start_s = offset_s - window_before_s
        stop_s = offset_s + window_after_s

        mask = (tobii_time_s >= start_s) & (tobii_time_s <= stop_s)

        if mask.sum() > 0:
            epoch = tobii_df.loc[mask].copy()
            epoch["relative_time_s"] = tobii_time_s.loc[mask] - offset_s
            epoch["event_id"] = event_idx
            all_epochs.append(epoch)

    if not all_epochs:
        return pd.DataFrame()

    return pd.concat(all_epochs, ignore_index=True)


def compute_event_locked_average(
    event_locked_df: pd.DataFrame,
    value_cols: list[str],
    time_col: str = "relative_time_s",
    n_bins: int = 50,
) -> pd.DataFrame:
    """Compute average values across events at each time point.

    Args:
        event_locked_df: DataFrame from align_to_behavior_onset/offset.
        value_cols: Columns to average (e.g., pupil diameter).
        n_bins: Number of time bins for averaging.
        time_col: Column containing relative time.

    Returns:
        DataFrame with time bins and mean/std for each value column.
    """
    if len(event_locked_df) == 0 or time_col not in event_locked_df.columns:
        return pd.DataFrame()

    # Create time bins
    t_min = event_locked_df[time_col].min()
    t_max = event_locked_df[time_col].max()
    bins = np.linspace(t_min, t_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Bin the data
    event_locked_df = event_locked_df.copy()
    event_locked_df["_time_bin"] = pd.cut(
        event_locked_df[time_col], bins=bins, labels=range(n_bins), include_lowest=True
    )

    # Compute statistics per bin
    result_data = {"time_s": bin_centers}

    for col in value_cols:
        if col not in event_locked_df.columns:
            continue

        grouped = event_locked_df.groupby("_time_bin", observed=True)[col]
        result_data[f"{col}_mean"] = grouped.mean().reindex(range(n_bins)).values
        result_data[f"{col}_std"] = grouped.std().reindex(range(n_bins)).values
        result_data[f"{col}_n"] = grouped.count().reindex(range(n_bins)).values

    return pd.DataFrame(result_data)


# =============================================================================
# Visualization
# =============================================================================


def plot_event_locked_response(
    event_locked_df: pd.DataFrame,
    value_col: str,
    ax: Axes | None = None,
    show_individual: bool = False,
    ci: float = 0.95,
    time_col: str = "relative_time_s",
    event_id_col: str = "event_id",
    n_bins: int = 50,
    color: str = "blue",
    figsize: tuple[float, float] = (10, 6),
) -> Axes:
    """Plot average response locked to behavior onset/offset.

    Shows mean with confidence band, optionally individual traces.

    Args:
        event_locked_df: DataFrame from align_to_behavior_onset/offset.
        value_col: Column to plot (e.g., 'Pupil diameter left').
        ax: Matplotlib axes. If None, creates new figure.
        show_individual: Whether to show individual event traces.
        ci: Confidence interval (e.g., 0.95 for 95% CI).
        time_col: Column containing relative time.
        event_id_col: Column containing event identifiers.
        n_bins: Number of time bins for averaging.
        color: Color for mean line and confidence band.
        figsize: Figure size if creating new figure.

    Returns:
        Matplotlib Axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if len(event_locked_df) == 0:
        ax.set_title("Event-Locked Response (no data)")
        return ax

    # Plot individual traces
    if show_individual:
        for event_id in event_locked_df[event_id_col].unique():
            event_data = event_locked_df[event_locked_df[event_id_col] == event_id]
            ax.plot(
                event_data[time_col],
                event_data[value_col],
                alpha=0.2,
                color="gray",
                linewidth=0.5,
            )

    # Compute and plot average
    avg_df = compute_event_locked_average(
        event_locked_df, value_cols=[value_col], time_col=time_col, n_bins=n_bins
    )

    if len(avg_df) > 0:
        mean_col = f"{value_col}_mean"
        std_col = f"{value_col}_std"
        n_col = f"{value_col}_n"

        if mean_col in avg_df.columns:
            mean_vals = avg_df[mean_col].values
            time_vals = avg_df["time_s"].values

            # Plot mean line
            ax.plot(time_vals, mean_vals, color=color, linewidth=2, label="Mean")

            # Plot confidence band
            if std_col in avg_df.columns and n_col in avg_df.columns:
                std_vals = avg_df[std_col].values
                n_vals = avg_df[n_col].values

                # Standard error
                se = std_vals / np.sqrt(np.maximum(n_vals, 1))

                # CI multiplier (approximate for large n)
                from scipy import stats as scipy_stats

                z = scipy_stats.norm.ppf((1 + ci) / 2)

                lower = mean_vals - z * se
                upper = mean_vals + z * se

                ax.fill_between(time_vals, lower, upper, alpha=0.3, color=color)

    # Add vertical line at onset (t=0)
    ax.axvline(0, color="red", linestyle="--", alpha=0.7, label="Event onset")

    ax.set_xlabel("Time relative to event (s)")
    ax.set_ylabel(value_col)
    ax.set_title(f"Event-Locked Response: {value_col}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_epochs_overview(
    epochs: list[pd.DataFrame],
    value_col: str,
    ax: Axes | None = None,
    tobii_time_col: str = "Recording timestamp",
    figsize: tuple[float, float] = (14, 6),
) -> Axes:
    """Plot overview of all extracted epochs.

    Shows each epoch as a separate line segment.

    Args:
        epochs: List of epoch DataFrames from extract_tobii_epochs.
        value_col: Column to plot.
        ax: Matplotlib axes.
        tobii_time_col: Column containing Tobii timestamps.
        figsize: Figure size.

    Returns:
        Matplotlib Axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if not epochs:
        ax.set_title("Epochs Overview (no data)")
        return ax

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in np.linspace(0, 1, min(len(epochs), 10))]

    for i, epoch in enumerate(epochs):
        if value_col not in epoch.columns or tobii_time_col not in epoch.columns:
            continue

        time_s = tobii_to_seconds(epoch, tobii_time_col)
        color = colors[i % len(colors)]

        label = None
        if "_event_behavior" in epoch.columns:
            behavior = epoch["_event_behavior"].iloc[0]
            if i == 0 or epochs[i - 1]["_event_behavior"].iloc[0] != behavior:
                label = behavior

        ax.plot(time_s, epoch[value_col], color=color, alpha=0.7, label=label, linewidth=0.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(value_col)
    ax.set_title(f"Epochs Overview: {value_col}")

    # Only show legend if reasonable number of labels
    handles, _ = ax.get_legend_handles_labels()
    if len(handles) <= 10:
        ax.legend()

    ax.grid(True, alpha=0.3)

    return ax
