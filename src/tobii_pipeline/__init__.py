"""Tobii eye-tracking data pipeline.

A toolkit for loading, cleaning, and processing Tobii eye-tracking data.

Example usage:
    from tobii_pipeline import load_recording, clean_recording, save_parquet

    # Load and clean a single recording
    df = load_recording("path/to/recording.tsv")
    df_clean = clean_recording(df)

    # Filter to eye tracker data only
    df_gaze = filter_eye_tracker(df_clean)

    # Keep only valid gaze samples
    df_valid = filter_valid_gaze(df_gaze)

    # Save to Parquet for faster future access
    save_parquet(df_valid, "output/recording_clean.parquet")
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
    # Utils
    "save_parquet",
    "load_parquet",
    "list_recordings",
    "list_participants",
    "get_data_summary",
    "batch_process",
]
