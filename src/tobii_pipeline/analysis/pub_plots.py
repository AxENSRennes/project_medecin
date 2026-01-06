"""Publication-quality plotting utilities for eye-tracking analysis.

Provides consistent styling, figure sizing, and export utilities for
generating publication-ready figures.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# =============================================================================
# Publication Style Configuration
# =============================================================================

PUB_STYLE: dict = {
    # Font settings
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    # Axes labels and titles
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    # Tick labels
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    # Legend
    "legend.fontsize": 9,
    "legend.framealpha": 0.8,
    "legend.edgecolor": "0.8",
    # Figure
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "figure.autolayout": False,
    # Saving
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    # Axes appearance
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.axisbelow": True,
    # Grid (when enabled)
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    # Lines
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
}

# Group color palette (colorblind-friendly)
GROUP_COLORS: dict[str, str] = {
    "Patient": "#D62728",  # Red
    "Control": "#1F77B4",  # Blue
}

# Extended color palette for behaviors/categories
CATEGORY_COLORS: list[str] = [
    "#1F77B4",  # Blue
    "#FF7F0E",  # Orange
    "#2CA02C",  # Green
    "#D62728",  # Red
    "#9467BD",  # Purple
    "#8C564B",  # Brown
    "#E377C2",  # Pink
    "#7F7F7F",  # Gray
    "#BCBD22",  # Olive
    "#17BECF",  # Cyan
]

# Figure sizes (width, height) in inches - typical journal formats
FIGURE_SIZES: dict[str, tuple[float, float]] = {
    "single_column": (3.5, 3.0),  # Single column width
    "double_column": (7.0, 3.5),  # Double column width
    "half_page": (7.0, 5.0),  # Half page
    "full_page": (7.0, 9.0),  # Full page
    "square": (5.0, 5.0),  # Square format
    "wide": (10.0, 4.0),  # Wide format for timelines
}


# =============================================================================
# Style Context Manager
# =============================================================================


@contextmanager
def apply_publication_style():
    """Context manager to temporarily apply publication styling.

    Usage:
        with apply_publication_style():
            fig, ax = plt.subplots()
            # ... create plot ...
            fig.savefig("figure.png")
    """
    original_params = {key: plt.rcParams.get(key) for key in PUB_STYLE}
    try:
        plt.rcParams.update(PUB_STYLE)
        yield
    finally:
        # Restore original parameters
        for key, value in original_params.items():
            if value is not None:
                plt.rcParams[key] = value


# =============================================================================
# Figure Factory
# =============================================================================


def create_figure(
    layout: str = "single_column",
    nrows: int = 1,
    ncols: int = 1,
    height_ratios: list[float] | None = None,
    width_ratios: list[float] | None = None,
    **kwargs,
) -> tuple[Figure, np.ndarray | Axes]:
    """Create a figure with publication-ready sizing.

    Args:
        layout: One of "single_column", "double_column", "half_page",
                "full_page", "square", "wide"
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        height_ratios: Relative heights of rows
        width_ratios: Relative widths of columns
        **kwargs: Additional arguments passed to plt.subplots

    Returns:
        Tuple of (Figure, Axes or array of Axes)
    """
    figsize = FIGURE_SIZES.get(layout, FIGURE_SIZES["single_column"])

    gridspec_kw = {}
    if height_ratios is not None:
        gridspec_kw["height_ratios"] = height_ratios
    if width_ratios is not None:
        gridspec_kw["width_ratios"] = width_ratios

    if gridspec_kw:
        kwargs["gridspec_kw"] = gridspec_kw

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)

    return fig, axes


# =============================================================================
# Figure Export
# =============================================================================


def save_figure(
    fig: Figure,
    path: str | Path,
    formats: list[str] | None = None,
    close: bool = True,
) -> list[Path]:
    """Save figure in multiple formats for publication.

    Args:
        fig: Matplotlib Figure object
        path: Base path (without extension)
        formats: List of formats to save (default: ["png", "pdf"])
        close: Whether to close the figure after saving

    Returns:
        List of saved file paths
    """
    if formats is None:
        formats = ["png", "pdf"]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for fmt in formats:
        save_path = path.with_suffix(f".{fmt}")
        fig.savefig(save_path, format=fmt, dpi=300, bbox_inches="tight")
        saved_paths.append(save_path)

    if close:
        plt.close(fig)

    return saved_paths


# =============================================================================
# Statistical Annotations
# =============================================================================


def add_significance_bar(
    ax: Axes,
    x1: float,
    x2: float,
    y: float,
    p_value: float,
    height: float = 0.02,
    text_offset: float = 0.01,
) -> None:
    """Add a significance bar with asterisks between two x positions.

    Args:
        ax: Matplotlib Axes object
        x1: Left x position
        x2: Right x position
        y: Y position (in data coordinates)
        p_value: P-value to determine significance level
        height: Height of the bar ends (fraction of y range)
        text_offset: Offset for text above bar (fraction of y range)
    """
    # Determine significance text
    if p_value < 0.001:
        sig_text = "***"
    elif p_value < 0.01:
        sig_text = "**"
    elif p_value < 0.05:
        sig_text = "*"
    else:
        sig_text = "n.s."

    # Get y-axis range for scaling
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    bar_height = height * y_range
    text_y = y + bar_height + text_offset * y_range

    # Draw the bar
    ax.plot([x1, x1, x2, x2], [y, y + bar_height, y + bar_height, y], "k-", linewidth=1)

    # Add text
    ax.text(
        (x1 + x2) / 2,
        text_y,
        sig_text,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def format_p_value(p_value: float) -> str:
    """Format p-value for display.

    Args:
        p_value: The p-value

    Returns:
        Formatted string
    """
    if p_value < 0.001:
        return "p < 0.001"
    if p_value < 0.01:
        return f"p = {p_value:.3f}"
    return f"p = {p_value:.2f}"


# =============================================================================
# Axis Formatting Utilities
# =============================================================================


def format_axis_labels(
    ax: Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
) -> None:
    """Apply consistent formatting to axis labels.

    Args:
        ax: Matplotlib Axes object
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Subplot title
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=11, fontweight="normal")
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=11, fontweight="normal")
    if title is not None:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)


def add_panel_label(
    ax: Axes,
    label: str,
    x: float = -0.1,
    y: float = 1.1,
) -> None:
    """Add a panel label (A, B, C, etc.) to a subplot.

    Args:
        ax: Matplotlib Axes object
        label: Panel label (e.g., "A", "B")
        x: X position in axes coordinates
        y: Y position in axes coordinates
    """
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="right",
    )


def despine(ax: Axes, top: bool = True, right: bool = True) -> None:
    """Remove spines from axes.

    Args:
        ax: Matplotlib Axes object
        top: Remove top spine
        right: Remove right spine
    """
    if top:
        ax.spines["top"].set_visible(False)
    if right:
        ax.spines["right"].set_visible(False)
