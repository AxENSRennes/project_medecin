"""Tobii eye-tracking data pipeline.

A toolkit for loading, cleaning, processing, and analyzing Tobii eye-tracking data.

Event detection uses pymovements library (I-VT, I-DT algorithms).
Blink interpolation uses MNE-Python.

Example usage:
    from tobii_pipeline import (
        load_recording,
        clean_recording,
        filter_eye_tracker,
        postprocess_recording,
        save_parquet,
    )

    # Load and clean a single recording
    df = load_recording("path/to/recording.tsv")
    df_clean = clean_recording(df)

    # Filter to eye tracker data only
    df_gaze = filter_eye_tracker(df_clean)

    # Post-process: interpolate blinks (MNE), detect events (pymovements)
    df_processed, report = postprocess_recording(df_gaze)

    # Save to Parquet for faster future access
    save_parquet(df_processed, "output/recording_processed.parquet")

    # For analysis:
    from tobii_pipeline.analysis import compute_recording_summary, plot_recording_summary
    summary = compute_recording_summary(df_processed)
    fig = plot_recording_summary(df_processed)
"""

from .cleaner import (
    clean_recording,
    filter_accelerometer,
    filter_by_sensor,
    filter_eye_tracker,
    filter_gyroscope,
    filter_valid_gaze,
    fix_decimal_separator,
    get_gaze_columns,
    get_motion_columns,
    select_gaze_data,
    select_motion_data,
)
from .loader import (
    get_recording_files,
    load_participant,
    load_recording,
    load_recordings_from_dir,
)
from .parser import parse_filename
from .postprocess import (
    GapInfo,
    compute_missing_rate,
    detect_events,
    detect_gaps,
    detect_gaze_outliers,
    detect_pupil_outliers,
    drop_high_missing_rows,
    filter_physiological_range,
    get_gap_statistics,
    get_interpolatable_columns,
    interpolate_missing,
    mark_gaps,
    postprocess_recording,
    remove_outliers,
    split_at_gaps,
)
from .utils import (
    batch_process,
    get_data_summary,
    list_participants,
    list_recordings,
    load_parquet,
    save_parquet,
)

__version__ = "0.1.0"

__all__ = [
    # Parser
    "parse_filename",
    # Loader
    "load_recording",
    "load_recordings_from_dir",
    "load_participant",
    "get_recording_files",
    # Cleaner
    "clean_recording",
    "fix_decimal_separator",
    "filter_by_sensor",
    "filter_eye_tracker",
    "filter_accelerometer",
    "filter_gyroscope",
    "filter_valid_gaze",
    "get_gaze_columns",
    "get_motion_columns",
    "select_gaze_data",
    "select_motion_data",
    # Postprocess
    "interpolate_missing",
    "compute_missing_rate",
    "drop_high_missing_rows",
    "get_interpolatable_columns",
    "detect_pupil_outliers",
    "detect_gaze_outliers",
    "remove_outliers",
    "filter_physiological_range",
    "detect_events",
    "detect_gaps",
    "get_gap_statistics",
    "mark_gaps",
    "split_at_gaps",
    "postprocess_recording",
    "GapInfo",
    # Utils
    "save_parquet",
    "load_parquet",
    "list_recordings",
    "list_participants",
    "get_data_summary",
    "batch_process",
]
