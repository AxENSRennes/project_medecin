"""Adapters for integrating external eye-tracking libraries.

This module provides adapters for:
- pymovements: Event detection (I-VT, I-DT), coordinate transformations, event properties
- MNE-Python: Blink interpolation, gaze heatmap visualization
"""

from tobii_pipeline.adapters.mne_adapter import (
    df_to_mne_raw,
    interpolate_blinks_mne,
    plot_gaze_heatmap,
    plot_gaze_on_stimulus,
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

__all__ = [
    # pymovements
    "df_to_gaze_dataframe",
    "apply_pix2deg",
    "apply_pos2vel",
    "detect_events_ivt",
    "detect_events_idt",
    "compute_event_properties",
    "events_to_df",
    # MNE
    "df_to_mne_raw",
    "interpolate_blinks_mne",
    "plot_gaze_heatmap",
    "plot_gaze_on_stimulus",
]
