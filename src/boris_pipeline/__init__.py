"""BORIS behavioral observation data pipeline.

A toolkit for loading, processing, and analyzing BORIS behavioral observation
data exported as Excel files.

Example usage:
    from boris_pipeline import (
        load_boris_file,
        load_time_budget,
        load_aggregated_events,
        load_recordings_from_dir,
        parse_filename,
        save_parquet,
    )

    # Load a single BORIS file (auto-detects type)
    df = load_boris_file("path/to/recording.xlsx")

    # Or load specific type
    df_budget = load_time_budget("path/to/recording.xlsx")
    df_events = load_aggregated_events("path/to/recording_agregated.xlsx")

    # Load all recordings from a directory
    df_all = load_recordings_from_dir("Data/data_G/Boris/")

    # Save to Parquet
    save_parquet(df, "output/recording.parquet")

For analysis functions, see the analysis subpackage:
    from boris_pipeline.analysis import (
        compute_recording_summary,
        plot_recording_summary,
        compare_patient_vs_control,
    )
"""

from .constants import (
    AGGREGATED_COLUMNS,
    AGGREGATED_PATTERNS,
    AGGREGATED_SHEET,
    TIME_BUDGET_COLUMNS,
    TIME_BUDGET_SHEET,
)
from .loader import (
    get_file_type,
    get_recording_files,
    is_aggregated_file,
    load_aggregated_events,
    load_boris_file,
    load_participant,
    load_recordings_from_dir,
    load_time_budget,
)
from .parser import parse_filename, strip_aggregated_suffix
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
    # Constants
    "TIME_BUDGET_SHEET",
    "AGGREGATED_SHEET",
    "AGGREGATED_PATTERNS",
    "TIME_BUDGET_COLUMNS",
    "AGGREGATED_COLUMNS",
    # Parser
    "parse_filename",
    "strip_aggregated_suffix",
    # Loader
    "is_aggregated_file",
    "get_file_type",
    "load_boris_file",
    "load_time_budget",
    "load_aggregated_events",
    "load_recordings_from_dir",
    "load_participant",
    "get_recording_files",
    # Utils
    "save_parquet",
    "load_parquet",
    "list_recordings",
    "list_participants",
    "get_data_summary",
    "batch_process",
]
