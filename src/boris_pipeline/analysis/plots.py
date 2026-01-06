"""Visualization functions for BORIS behavioral observation data.

Provides timeline plots, frequency charts, duration distributions,
and transition visualizations for behavioral data.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .metrics import (
    compute_behavior_durations,
    compute_behavior_frequency,
    compute_inter_event_intervals,
    compute_sequence_entropy,
    compute_time_budget,
    compute_transition_matrix,
)

# Default color palette for behaviors
DEFAULT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


# =============================================================================
# Helper Functions
# =============================================================================


def _get_or_create_axes(
    ax: Axes | None,
    figsize: tuple[float, float] = (10, 6),
) -> Axes:
    """Get existing axes or create new figure with axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _get_color_map(
    behaviors: list[str],
    color_map: dict[str, str] | None = None,
) -> dict[str, str]:
    """Create color mapping for behaviors."""
    if color_map is not None:
        return color_map

    result = {}
    for i, behavior in enumerate(behaviors):
        result[behavior] = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
    return result


def abbreviate_behavior_label(label: str) -> str:
    """Abbreviate common behavior prefixes for readability.

    Replaces verbose prefixes with short codes:
    - "E.D. - " -> "ED:"
    - "E.N. - " -> "EN:"
    - "T. - " -> "T:"
    - "O. - " -> "O:"

    Args:
        label: Full behavior label string.

    Returns:
        Abbreviated label.
    """
    abbreviations = {
        "E.D. - ": "ED:",
        "E.D. -": "ED:",
        "E.N. - ": "EN:",
        "E.N. -": "EN:",
        "T. - ": "T:",
        "T. -": "T:",
        "O. - ": "O:",
        "O. -": "O:",
    }
    for prefix, abbrev in abbreviations.items():
        if label.startswith(prefix):
            return abbrev + label[len(prefix) :]
    return label


def abbreviate_labels(labels: list[str]) -> list[str]:
    """Apply abbreviation to a list of labels.

    Args:
        labels: List of behavior label strings.

    Returns:
        List of abbreviated labels.
    """
    return [abbreviate_behavior_label(label) for label in labels]


# =============================================================================
# Timeline Visualizations
# =============================================================================


def plot_behavior_timeline(
    df: pd.DataFrame,
    ax: Axes | None = None,
    color_map: dict[str, str] | None = None,
    show_labels: bool = True,
    behavior_col: str = "Behavior",
    start_col: str = "Start (s)",
    stop_col: str = "Stop (s)",
    figsize: tuple[float, float] = (14, 4),
) -> Axes:
    """Plot behavioral events as horizontal bars over time.

    Each behavior type gets a row, events shown as colored spans.

    Args:
        df: Aggregated events DataFrame.
        ax: Matplotlib axes. If None, creates new figure.
        color_map: Dict mapping behavior names to colors.
        show_labels: Whether to show behavior labels on y-axis.
        behavior_col: Column name for behavior labels.
        start_col: Column name for event start times.
        stop_col: Column name for event stop times.
        figsize: Figure size if creating new figure.

    Returns:
        Matplotlib Axes object.
    """
    ax = _get_or_create_axes(ax, figsize=figsize)

    required_cols = [behavior_col, start_col, stop_col]
    if not all(col in df.columns for col in required_cols):
        ax.set_title("Behavior Timeline (missing columns)")
        return ax

    if len(df) == 0:
        ax.set_title("Behavior Timeline (no data)")
        return ax

    # Get unique behaviors and assign y positions
    behaviors = sorted(df[behavior_col].unique())
    y_positions = {beh: i for i, beh in enumerate(behaviors)}
    colors = _get_color_map(behaviors, color_map)

    # Plot each event as a horizontal bar
    bar_height = 0.8
    for _, row in df.iterrows():
        behavior = row[behavior_col]
        start = row[start_col]
        stop = row[stop_col]
        duration = stop - start

        y = y_positions[behavior]
        ax.barh(
            y,
            duration,
            left=start,
            height=bar_height,
            color=colors[behavior],
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )

    # Configure axes
    ax.set_yticks(range(len(behaviors)))
    if show_labels:
        abbreviated = abbreviate_labels(behaviors)
        fontsize = 8 if len(behaviors) > 10 else 10
        ax.set_yticklabels(abbreviated, fontsize=fontsize)
    else:
        ax.set_yticklabels([])

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Behavior")
    ax.set_title("Behavioral Timeline")
    ax.set_ylim(-0.5, len(behaviors) - 0.5)
    ax.grid(True, alpha=0.3, axis="x")

    return ax


def plot_ethogram(
    df: pd.DataFrame,
    ax: Axes | None = None,
    behavior_order: list[str] | None = None,
    time_range: tuple[float, float] | None = None,
    behavior_col: str = "Behavior",
    start_col: str = "Start (s)",
    stop_col: str = "Stop (s)",
    figsize: tuple[float, float] = (14, 6),
) -> Axes:
    """Plot ethogram (raster plot) of behavioral events.

    Traditional ethology visualization with behaviors as rows
    and time on x-axis.

    Args:
        df: Aggregated events DataFrame.
        ax: Matplotlib axes.
        behavior_order: Custom order for behaviors (top to bottom).
        time_range: (start, end) time range to display.
        behavior_col: Column name for behavior labels.
        start_col: Column name for event start times.
        stop_col: Column name for event stop times.
        figsize: Figure size.

    Returns:
        Matplotlib Axes object.
    """
    ax = _get_or_create_axes(ax, figsize=figsize)

    required_cols = [behavior_col, start_col, stop_col]
    if not all(col in df.columns for col in required_cols):
        ax.set_title("Ethogram (missing columns)")
        return ax

    if len(df) == 0:
        ax.set_title("Ethogram (no data)")
        return ax

    # Filter time range if specified
    if time_range is not None:
        df = df[(df[start_col] >= time_range[0]) & (df[stop_col] <= time_range[1])]

    # Determine behavior order
    if behavior_order is None:
        behavior_order = sorted(df[behavior_col].unique())

    # Create y position mapping (reversed so first item is at top)
    y_positions = {beh: len(behavior_order) - 1 - i for i, beh in enumerate(behavior_order)}
    colors = _get_color_map(behavior_order)

    # Plot events
    for _, row in df.iterrows():
        behavior = row[behavior_col]
        if behavior not in y_positions:
            continue

        start = row[start_col]
        stop = row[stop_col]
        y = y_positions[behavior]

        ax.plot([start, stop], [y, y], color=colors[behavior], linewidth=6, solid_capstyle="butt")

    # Configure axes
    ax.set_yticks(range(len(behavior_order)))
    ax.set_yticklabels(reversed(behavior_order))
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Behavior")
    ax.set_title("Ethogram")
    ax.set_ylim(-0.5, len(behavior_order) - 0.5)
    ax.grid(True, alpha=0.3, axis="x")

    return ax


# =============================================================================
# Duration & Frequency Visualizations
# =============================================================================


def plot_duration_histogram(
    df: pd.DataFrame,
    behavior: str | None = None,
    ax: Axes | None = None,
    bins: int = 30,
    behavior_col: str = "Behavior",
    duration_col: str = "Duration (s)",
    figsize: tuple[float, float] = (10, 6),
) -> Axes:
    """Histogram of event durations.

    Args:
        df: Aggregated events DataFrame.
        behavior: If specified, plot only this behavior. Otherwise, all.
        ax: Matplotlib axes.
        bins: Number of histogram bins.
        behavior_col: Column name for behavior labels.
        duration_col: Column name for event durations.
        figsize: Figure size.

    Returns:
        Matplotlib Axes object.
    """
    ax = _get_or_create_axes(ax, figsize=figsize)

    if duration_col not in df.columns:
        ax.set_title("Duration Histogram (missing column)")
        return ax

    if behavior is not None:
        df = df[df[behavior_col] == behavior]
        title = f"Duration Distribution: {behavior}"
    else:
        title = "Duration Distribution (All Behaviors)"

    durations = df[duration_col].dropna()

    if len(durations) == 0:
        ax.set_title(f"{title} (no data)")
        return ax

    ax.hist(durations, bins=bins, edgecolor="black", alpha=0.7)

    # Add mean line
    mean_duration = durations.mean()
    ax.axvline(mean_duration, color="red", linestyle="--", label=f"Mean: {mean_duration:.2f}s")
    ax.legend()

    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax


def plot_duration_boxplot(
    df: pd.DataFrame,
    ax: Axes | None = None,
    behaviors: list[str] | None = None,
    behavior_col: str = "Behavior",
    duration_col: str = "Duration (s)",
    figsize: tuple[float, float] = (10, 6),
) -> Axes:
    """Box plot comparing durations across behaviors.

    Args:
        df: Aggregated events DataFrame.
        ax: Matplotlib axes.
        behaviors: List of behaviors to include. If None, all behaviors.
        behavior_col: Column name for behavior labels.
        duration_col: Column name for event durations.
        figsize: Figure size.

    Returns:
        Matplotlib Axes object.
    """
    ax = _get_or_create_axes(ax, figsize=figsize)

    if behavior_col not in df.columns or duration_col not in df.columns:
        ax.set_title("Duration Boxplot (missing columns)")
        return ax

    if behaviors is None:
        behaviors = sorted(df[behavior_col].unique())

    data = []
    labels = []
    for behavior in behaviors:
        durations = df[df[behavior_col] == behavior][duration_col].dropna()
        if len(durations) > 0:
            data.append(durations.values)
            labels.append(behavior)

    if not data:
        ax.set_title("Duration Boxplot (no data)")
        return ax

    abbreviated = abbreviate_labels(labels)
    ax.boxplot(data, labels=abbreviated)
    ax.set_xlabel("Behavior")
    ax.set_ylabel("Duration (seconds)")
    ax.set_title("Duration by Behavior")
    ax.grid(True, alpha=0.3, axis="y")

    # Rotate labels if many behaviors
    if len(labels) > 5:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.gcf().subplots_adjust(bottom=0.25)

    return ax


def plot_frequency_bars(
    df: pd.DataFrame,
    ax: Axes | None = None,
    sort_by: str = "count",
    horizontal: bool = False,
    behavior_col: str = "Behavior",
    figsize: tuple[float, float] = (10, 6),
) -> Axes:
    """Bar chart of behavior occurrence frequencies.

    Args:
        df: Aggregated events DataFrame.
        ax: Matplotlib axes.
        sort_by: "count", "duration", or "alphabetical".
        horizontal: If True, plot horizontal bars.
        behavior_col: Column name for behavior labels.
        figsize: Figure size.

    Returns:
        Matplotlib Axes object.
    """
    ax = _get_or_create_axes(ax, figsize=figsize)

    if behavior_col not in df.columns or len(df) == 0:
        ax.set_title("Frequency Chart (no data)")
        return ax

    frequencies = compute_behavior_frequency(df, behavior_col=behavior_col)

    if not frequencies:
        ax.set_title("Frequency Chart (no data)")
        return ax

    # Sort
    if sort_by == "count":
        sorted_items = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
    elif sort_by == "duration":
        durations = compute_behavior_durations(df, behavior_col=behavior_col)
        sorted_items = sorted(
            frequencies.items(),
            key=lambda x: durations.get(x[0], {}).get("total_duration", 0),
            reverse=True,
        )
    else:  # alphabetical
        sorted_items = sorted(frequencies.items(), key=lambda x: x[0])

    behaviors = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]
    colors = _get_color_map(behaviors)
    bar_colors = [colors[b] for b in behaviors]
    abbreviated = abbreviate_labels(behaviors)

    if horizontal:
        ax.barh(abbreviated, counts, color=bar_colors)
        ax.set_xlabel("Count")
        ax.set_ylabel("Behavior")
        if len(behaviors) > 10:
            plt.setp(ax.get_yticklabels(), fontsize=8)
    else:
        ax.bar(abbreviated, counts, color=bar_colors)
        ax.set_xlabel("Behavior")
        ax.set_ylabel("Count")
        if len(behaviors) > 5:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
            plt.gcf().subplots_adjust(bottom=0.25)

    ax.set_title("Behavior Frequencies")
    ax.grid(True, alpha=0.3, axis="y" if not horizontal else "x")

    return ax


def plot_time_budget_pie(
    df: pd.DataFrame,
    ax: Axes | None = None,
    min_pct: float = 1.0,
    behavior_col: str = "Behavior",
    figsize: tuple[float, float] = (8, 8),
) -> Axes:
    """Pie chart of time budget distribution.

    Args:
        df: Aggregated events DataFrame.
        ax: Matplotlib axes.
        min_pct: Minimum percentage to show as separate slice (others grouped).
        behavior_col: Column name for behavior labels.
        figsize: Figure size.

    Returns:
        Matplotlib Axes object.
    """
    ax = _get_or_create_axes(ax, figsize=figsize)

    time_budget = compute_time_budget(df, behavior_col=behavior_col)

    if not time_budget:
        ax.set_title("Time Budget (no data)")
        return ax

    # Group small slices into "Other"
    main_items = {}
    other_pct = 0.0

    for behavior, pct in time_budget.items():
        if pct * 100 >= min_pct:
            main_items[behavior] = pct
        else:
            other_pct += pct

    if other_pct > 0:
        main_items["Other"] = other_pct

    labels = list(main_items.keys())
    sizes = list(main_items.values())
    colors = _get_color_map(labels)
    pie_colors = [colors[label] for label in labels]
    abbreviated = abbreviate_labels(labels)

    # Dynamic label fontsize based on number of slices
    label_fontsize = 8 if len(labels) > 8 else 10

    ax.pie(
        sizes,
        labels=abbreviated,
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        textprops={"fontsize": label_fontsize},
    )
    ax.set_title("Time Budget")

    return ax


# =============================================================================
# Transition Visualizations
# =============================================================================


def plot_transition_matrix(
    df: pd.DataFrame,
    ax: Axes | None = None,
    cmap: str = "Blues",
    annotate: bool = True,
    behavior_col: str = "Behavior",
    figsize: tuple[float, float] = (10, 8),
) -> Axes:
    """Heatmap of behavior transition probabilities.

    Args:
        df: Aggregated events DataFrame.
        ax: Matplotlib axes.
        cmap: Colormap name.
        annotate: Whether to show values in cells.
        behavior_col: Column name for behavior labels.
        figsize: Figure size.

    Returns:
        Matplotlib Axes object.
    """
    ax = _get_or_create_axes(ax, figsize=figsize)

    matrix = compute_transition_matrix(df, behavior_col=behavior_col, normalize=True)

    if matrix.empty:
        ax.set_title("Transition Matrix (insufficient data)")
        return ax

    # Plot heatmap
    im = ax.imshow(matrix.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Probability")

    n_behaviors = len(matrix)

    # Add annotations (only if matrix is small enough to be readable)
    if annotate and n_behaviors <= 10:
        annotation_fontsize = max(6, 12 - n_behaviors // 2)
        for i in range(len(matrix)):
            for j in range(len(matrix.columns)):
                value = matrix.iloc[i, j]
                if value > 0:
                    text_color = "white" if value > 0.5 else "black"
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=annotation_fontsize,
                    )

    # Configure axes with abbreviated labels
    x_labels = abbreviate_labels(list(matrix.columns))
    y_labels = abbreviate_labels(list(matrix.index))
    label_fontsize = max(6, 10 - n_behaviors // 3)

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=label_fontsize)
    ax.set_yticklabels(y_labels, fontsize=label_fontsize)
    ax.set_xlabel("To Behavior")
    ax.set_ylabel("From Behavior")
    ax.set_title("Behavior Transition Probabilities")

    return ax


def plot_transition_diagram(
    df: pd.DataFrame,
    ax: Axes | None = None,
    min_prob: float = 0.1,
    behavior_col: str = "Behavior",
    figsize: tuple[float, float] = (10, 10),
) -> Axes:
    """Network diagram of behavior transitions.

    Uses circular layout with edge widths proportional to probability.
    Requires networkx library (optional).

    Args:
        df: Aggregated events DataFrame.
        ax: Matplotlib axes.
        min_prob: Minimum probability to draw edge.
        behavior_col: Column name for behavior labels.
        figsize: Figure size.

    Returns:
        Matplotlib Axes object.
    """
    ax = _get_or_create_axes(ax, figsize=figsize)

    matrix = compute_transition_matrix(df, behavior_col=behavior_col, normalize=True)

    if matrix.empty:
        ax.set_title("Transition Diagram (insufficient data)")
        return ax

    try:
        import networkx as nx
    except ImportError:
        ax.text(
            0.5,
            0.5,
            "networkx not installed\nRun: pip install networkx",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Transition Diagram (requires networkx)")
        return ax

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes
    behaviors = list(matrix.index)
    G.add_nodes_from(behaviors)

    # Add edges with weights
    for from_beh in matrix.index:
        for to_beh in matrix.columns:
            prob = matrix.loc[from_beh, to_beh]
            if prob >= min_prob:
                G.add_edge(from_beh, to_beh, weight=prob)

    # Layout
    pos = nx.circular_layout(G)

    # Draw nodes
    colors = _get_color_map(behaviors)
    node_colors = [colors[node] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=2000, alpha=0.8)

    # Draw edges with width proportional to weight
    edges = G.edges(data=True)
    if edges:
        weights = [d["weight"] * 5 for _, _, d in edges]
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            width=weights,
            alpha=0.6,
            edge_color="gray",
            arrows=True,
            arrowsize=20,
            connectionstyle="arc3,rad=0.1",
        )

    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")

    ax.set_title(f"Behavior Transitions (p >= {min_prob})")
    ax.axis("off")

    return ax


# =============================================================================
# Temporal Pattern Visualizations
# =============================================================================


def plot_inter_event_intervals(
    df: pd.DataFrame,
    behavior: str | None = None,
    ax: Axes | None = None,
    bins: int = 30,
    behavior_col: str = "Behavior",
    figsize: tuple[float, float] = (10, 6),
) -> Axes:
    """Histogram of inter-event intervals.

    Args:
        df: Aggregated events DataFrame.
        behavior: Specific behavior to analyze (None for all).
        ax: Matplotlib axes.
        bins: Number of histogram bins.
        behavior_col: Column name for behavior labels.
        figsize: Figure size.

    Returns:
        Matplotlib Axes object.
    """
    ax = _get_or_create_axes(ax, figsize=figsize)

    intervals = compute_inter_event_intervals(df, behavior=behavior, behavior_col=behavior_col)

    if not intervals:
        ax.set_title("Inter-Event Intervals (no data)")
        return ax

    # For simplicity, recompute from data (intervals dict has stats, not raw values)
    if behavior is not None:
        beh_df = df[df[behavior_col] == behavior].sort_values("Start (s)")
    else:
        beh_df = df.sort_values("Start (s)")

    if len(beh_df) < 2:
        ax.set_title("Inter-Event Intervals (insufficient data)")
        return ax

    raw_intervals = beh_df["Start (s)"].iloc[1:].values - beh_df["Stop (s)"].iloc[:-1].values
    raw_intervals = raw_intervals[raw_intervals >= 0]

    if len(raw_intervals) == 0:
        ax.set_title("Inter-Event Intervals (no gaps)")
        return ax

    ax.hist(raw_intervals, bins=bins, edgecolor="black", alpha=0.7)

    mean_interval = np.mean(raw_intervals)
    ax.axvline(mean_interval, color="red", linestyle="--", label=f"Mean: {mean_interval:.2f}s")
    ax.legend()

    title = f"Inter-Event Intervals: {behavior}" if behavior else "Inter-Event Intervals"
    ax.set_xlabel("Interval (seconds)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax


def plot_cumulative_duration(
    df: pd.DataFrame,
    ax: Axes | None = None,
    behaviors: list[str] | None = None,
    behavior_col: str = "Behavior",
    start_col: str = "Start (s)",
    duration_col: str = "Duration (s)",
    figsize: tuple[float, float] = (12, 6),
) -> Axes:
    """Cumulative duration plot over time for each behavior.

    Args:
        df: Aggregated events DataFrame.
        ax: Matplotlib axes.
        behaviors: List of behaviors to plot (None for all).
        behavior_col: Column name for behavior labels.
        start_col: Column name for event start times.
        duration_col: Column name for event durations.
        figsize: Figure size.

    Returns:
        Matplotlib Axes object.
    """
    ax = _get_or_create_axes(ax, figsize=figsize)

    required_cols = [behavior_col, start_col, duration_col]
    if not all(col in df.columns for col in required_cols):
        ax.set_title("Cumulative Duration (missing columns)")
        return ax

    if behaviors is None:
        behaviors = sorted(df[behavior_col].unique())

    colors = _get_color_map(behaviors)

    for behavior in behaviors:
        beh_df = df[df[behavior_col] == behavior].sort_values(start_col)
        if len(beh_df) == 0:
            continue

        times = beh_df[start_col].values
        durations = beh_df[duration_col].values
        cumulative = np.cumsum(durations)

        ax.step(times, cumulative, where="post", label=behavior, color=colors[behavior])

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Cumulative Duration (seconds)")
    ax.set_title("Cumulative Time per Behavior")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    return ax


def plot_behavior_rate_over_time(
    df: pd.DataFrame,
    window_s: float = 60.0,
    ax: Axes | None = None,
    behaviors: list[str] | None = None,
    behavior_col: str = "Behavior",
    start_col: str = "Start (s)",
    figsize: tuple[float, float] = (12, 6),
) -> Axes:
    """Sliding window behavior rate over recording time.

    Args:
        df: Aggregated events DataFrame.
        window_s: Window size in seconds.
        ax: Matplotlib axes.
        behaviors: List of behaviors to plot (None for all).
        behavior_col: Column name for behavior labels.
        start_col: Column name for event start times.
        figsize: Figure size.

    Returns:
        Matplotlib Axes object.
    """
    ax = _get_or_create_axes(ax, figsize=figsize)

    if behavior_col not in df.columns or start_col not in df.columns:
        ax.set_title("Behavior Rate (missing columns)")
        return ax

    if len(df) == 0:
        ax.set_title("Behavior Rate (no data)")
        return ax

    if behaviors is None:
        behaviors = sorted(df[behavior_col].unique())

    colors = _get_color_map(behaviors)

    # Create time bins
    t_min = df[start_col].min()
    t_max = df[start_col].max()
    bin_edges = np.arange(t_min, t_max + window_s, window_s)

    if len(bin_edges) < 2:
        ax.set_title("Behavior Rate (recording too short)")
        return ax

    for behavior in behaviors:
        beh_times = df[df[behavior_col] == behavior][start_col].values

        # Count events per bin
        counts, _ = np.histogram(beh_times, bins=bin_edges)
        rates = counts / (window_s / 60.0)  # events per minute

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.plot(bin_centers, rates, label=behavior, color=colors[behavior])

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Rate (events/minute)")
    ax.set_title(f"Behavior Rate Over Time (window={window_s}s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    return ax


# =============================================================================
# Summary Visualization
# =============================================================================


def plot_recording_summary(
    df: pd.DataFrame,
    behavior_col: str = "Behavior",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Multi-panel summary figure for a recording.

    Panels: timeline, frequency bars, duration boxplot, transition matrix,
            time budget pie, key metrics text.

    Args:
        df: Aggregated events DataFrame.
        behavior_col: Column name for behavior labels.
        figsize: Figure size. If None, adapts to number of behaviors.

    Returns:
        Matplotlib Figure object.
    """
    # Adaptive figure sizing based on number of behaviors
    n_behaviors = len(df[behavior_col].unique()) if behavior_col in df.columns else 1
    if figsize is None:
        height = max(12, 10 + n_behaviors * 0.15)
        figsize = (16, height)

    fig = plt.figure(figsize=figsize)

    # Create grid layout with increased spacing
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    # Timeline (top row, full width)
    ax1 = fig.add_subplot(gs[0, :])
    plot_behavior_timeline(df, ax=ax1, behavior_col=behavior_col)

    # Frequency bars (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    plot_frequency_bars(df, ax=ax2, behavior_col=behavior_col)

    # Duration boxplot (middle center)
    ax3 = fig.add_subplot(gs[1, 1])
    plot_duration_boxplot(df, ax=ax3, behavior_col=behavior_col)

    # Time budget pie (middle right)
    ax4 = fig.add_subplot(gs[1, 2])
    plot_time_budget_pie(df, ax=ax4, behavior_col=behavior_col)

    # Transition matrix (bottom left and center)
    ax5 = fig.add_subplot(gs[2, :2])
    plot_transition_matrix(df, ax=ax5, behavior_col=behavior_col)

    # Metrics text (bottom right)
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis("off")

    # Compute and display metrics
    frequencies = compute_behavior_frequency(df, behavior_col=behavior_col)
    durations = compute_behavior_durations(df, behavior_col=behavior_col)
    time_budget = compute_time_budget(df, behavior_col=behavior_col)
    entropy = compute_sequence_entropy(df, behavior_col=behavior_col)

    total_events = sum(frequencies.values()) if frequencies else 0
    total_behaviors = len(frequencies)
    total_duration = sum(d.get("total_duration", 0) for d in durations.values()) if durations else 0

    metrics_text = f"""Recording Summary

Total Events: {total_events}
Unique Behaviors: {total_behaviors}
Total Duration: {total_duration:.1f}s

Sequence Entropy: {entropy:.2f} bits

Top Behaviors by Frequency:
"""

    # Add top 3 behaviors
    sorted_freq = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:3]
    for beh, count in sorted_freq:
        pct = time_budget.get(beh, 0) * 100
        metrics_text += f"  {beh}: {count} ({pct:.1f}%)\n"

    ax6.text(
        0.1,
        0.9,
        metrics_text,
        transform=ax6.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    return fig
