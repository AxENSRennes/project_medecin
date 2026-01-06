"""Filename metadata parsing for BORIS behavioral observation files.

BORIS files use the same naming convention as Tobii files:
{ID}_{Name}_{Study}_{Group}_{Month}_{Visit}_{Date}[_agregated].xlsx

This module re-exports parse_filename from tobii_pipeline and provides
a helper to strip the aggregated suffix before parsing.
"""

from tobii_pipeline.parser import parse_filename

from .constants import AGGREGATED_PATTERNS

__all__ = ["parse_filename", "strip_aggregated_suffix"]


def strip_aggregated_suffix(filename: str) -> str:
    """Remove aggregated suffix from BORIS filename for parsing.

    Handles various suffix patterns: _agregated, -agregated, _aggregated, -aggregated

    Args:
        filename: The filename (with or without path and extension)

    Returns:
        Filename with aggregated suffix removed, or original if no suffix found.

    Examples:
        >>> strip_aggregated_suffix("G213_FAUJea_SDS2_P_M36_V4_25062025_agregated.xlsx")
        'G213_FAUJea_SDS2_P_M36_V4_25062025.xlsx'
        >>> strip_aggregated_suffix("G213_FAUJea_SDS2_P_M36_V4_25062025.xlsx")
        'G213_FAUJea_SDS2_P_M36_V4_25062025.xlsx'
    """
    filename_lower = filename.lower()

    for pattern in AGGREGATED_PATTERNS:
        idx = filename_lower.find(pattern)
        if idx != -1:
            # Find extension
            ext_idx = filename_lower.rfind(".")
            if ext_idx > idx:
                # Keep extension
                return filename[:idx] + filename[ext_idx:]
            return filename[:idx]

    return filename


def parse_boris_filename(filename: str) -> dict:
    """Parse BORIS filename to extract metadata, handling aggregated suffix.

    This is a convenience wrapper that strips the aggregated suffix before
    calling parse_filename from tobii_pipeline.

    Args:
        filename: The filename (with or without path and extension)

    Returns:
        dict with keys: id, participant, study, group, month, visit, date, raw_filename
        Plus 'is_aggregated' indicating if this was an aggregated file.
    """
    # Check if aggregated
    filename_lower = filename.lower()
    is_aggregated = any(pattern in filename_lower for pattern in AGGREGATED_PATTERNS)

    # Strip suffix and parse
    clean_filename = strip_aggregated_suffix(filename)
    result = parse_filename(clean_filename)

    # Add aggregated flag
    result["is_aggregated"] = is_aggregated

    return result
