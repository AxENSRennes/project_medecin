"""Integration module for Boris behavioral and Tobii eye-tracking data.

Provides utilities for timestamp alignment, epoch extraction, and
cross-modal analysis combining behavioral observations with eye-tracking metrics.

Example:
    from integration import (
        align_boris_to_tobii,
        extract_tobii_epochs,
        compute_pupil_per_behavior,
        plot_cross_modal_summary,
    )

    # Load data
    from tobii_pipeline import load_recording, clean_recording, filter_eye_tracker
    from boris_pipeline import load_boris_file

    tobii_df = filter_eye_tracker(clean_recording(load_recording("tobii_file.tsv")))
    boris_df = load_boris_file("boris_file.xlsx")

    # Align timestamps
    boris_aligned = align_boris_to_tobii(boris_df, tobii_df)

    # Extract epochs during behaviors
    epochs = extract_tobii_epochs(tobii_df, boris_df, behavior="LookingAtFace")

    # Compute cross-modal metrics
    pupil_metrics = compute_pupil_per_behavior(tobii_df, boris_df)

    # Visualize
    fig = plot_cross_modal_summary(tobii_df, boris_df)
"""

from .alignment import (
    align_boris_to_tobii,
    compute_alignment_offset,
    find_boris_time_range,
    find_tobii_index_at_time,
    find_tobii_indices_in_range,
    find_tobii_time_range,
    tobii_to_seconds,
    validate_alignment,
)
from .cross_modal import (
    compare_gaze_between_behaviors,
    compute_behavior_gaze_correlations,
    compute_gaze_per_behavior,
    compute_gaze_shift_at_behavior,
    compute_pupil_change_at_behavior,
    compute_pupil_per_behavior,
    plot_cross_modal_summary,
    plot_gaze_by_behavior,
    plot_pupil_behavior_timeline,
)
from .epochs import (
    align_to_behavior_offset,
    align_to_behavior_onset,
    compute_event_locked_average,
    create_epoch_dataset,
    extract_epoch_by_event_index,
    extract_tobii_epochs,
    plot_epochs_overview,
    plot_event_locked_response,
)

__all__ = [
    # alignment
    "align_boris_to_tobii",
    "tobii_to_seconds",
    "find_tobii_time_range",
    "find_boris_time_range",
    "compute_alignment_offset",
    "validate_alignment",
    "find_tobii_index_at_time",
    "find_tobii_indices_in_range",
    # epochs
    "extract_tobii_epochs",
    "extract_epoch_by_event_index",
    "create_epoch_dataset",
    "align_to_behavior_onset",
    "align_to_behavior_offset",
    "compute_event_locked_average",
    "plot_event_locked_response",
    "plot_epochs_overview",
    # cross_modal
    "compute_gaze_per_behavior",
    "compute_pupil_per_behavior",
    "compute_behavior_gaze_correlations",
    "compute_pupil_change_at_behavior",
    "compute_gaze_shift_at_behavior",
    "compare_gaze_between_behaviors",
    "plot_gaze_by_behavior",
    "plot_pupil_behavior_timeline",
    "plot_cross_modal_summary",
]
