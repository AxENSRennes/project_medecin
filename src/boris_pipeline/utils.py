"""Utility functions for BORIS data pipeline."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pandas as pd
from tqdm import tqdm

from .loader import get_file_type, get_recording_files, load_boris_file
from .parser import parse_boris_filename


def save_parquet(
    df: pd.DataFrame,
    output_path: str | Path,
    compression: str = "snappy",
) -> None:
    """Save DataFrame to Parquet format.

    Args:
        df: DataFrame to save
        output_path: Path to output Parquet file
        compression: Compression codec (default: 'snappy')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, compression=compression, index=False)


def load_parquet(filepath: str | Path) -> pd.DataFrame:
    """Load DataFrame from Parquet file.

    Args:
        filepath: Path to Parquet file

    Returns:
        DataFrame loaded from Parquet
    """
    return pd.read_parquet(filepath)


def list_recordings(
    data_dir: str | Path,
    file_type: Literal["time_budget", "aggregated", "all"] = "all",
) -> list[dict]:
    """List all BORIS recordings with parsed metadata.

    Args:
        data_dir: Directory containing recording files
        file_type: Which files to include

    Returns:
        List of dicts with metadata and file info for each recording
    """
    data_dir = Path(data_dir)
    files = get_recording_files(data_dir, file_type)

    recordings = []
    for filepath in files:
        metadata = parse_boris_filename(filepath.name)
        metadata["filepath"] = filepath
        metadata["file_size_kb"] = filepath.stat().st_size / 1024
        recordings.append(metadata)

    return recordings


def list_participants(
    data_dirs: str | Path | list[str | Path],
    file_type: Literal["time_budget", "aggregated", "all"] = "all",
) -> list[str]:
    """Get unique participant codes from directories.

    Args:
        data_dirs: Directory or list of directories to search
        file_type: Which files to include

    Returns:
        Sorted list of unique participant codes
    """
    if isinstance(data_dirs, str | Path):
        data_dirs = [data_dirs]

    participants = set()

    for data_dir in data_dirs:
        recordings = list_recordings(data_dir, file_type)
        for rec in recordings:
            if rec["participant"]:
                participants.add(rec["participant"])

    return sorted(participants)


def get_data_summary(
    data_dirs: str | Path | list[str | Path],
    file_type: Literal["time_budget", "aggregated", "all"] = "all",
) -> dict:
    """Get summary statistics for all BORIS recordings.

    Args:
        data_dirs: Directory or list of directories to search
        file_type: Which files to include

    Returns:
        dict with summary statistics:
        - total_files: Total number of files
        - time_budget_files: Number of original files
        - aggregated_files: Number of aggregated files
        - participants: List of unique participant codes
        - groups: Dict with counts per group (Patient/Control)
        - months: Dict with counts per month
        - total_size_mb: Total file size in MB
    """
    if isinstance(data_dirs, str | Path):
        data_dirs = [data_dirs]

    all_recordings = []
    for data_dir in data_dirs:
        all_recordings.extend(list_recordings(data_dir, file_type))

    participants = set()
    groups = {"Patient": 0, "Control": 0}
    months = {}
    time_budget_count = 0
    aggregated_count = 0
    total_size_kb = 0

    for rec in all_recordings:
        if rec["participant"]:
            participants.add(rec["participant"])
        if rec["group"]:
            groups[rec["group"]] = groups.get(rec["group"], 0) + 1
        if rec["month"] is not None:
            month_key = f"M{rec['month']}"
            months[month_key] = months.get(month_key, 0) + 1
        if rec["is_aggregated"]:
            aggregated_count += 1
        else:
            time_budget_count += 1
        total_size_kb += rec["file_size_kb"]

    return {
        "total_files": len(all_recordings),
        "time_budget_files": time_budget_count,
        "aggregated_files": aggregated_count,
        "participants": sorted(participants),
        "participant_count": len(participants),
        "groups": groups,
        "months": dict(sorted(months.items())),
        "total_size_mb": total_size_kb / 1024,
    }


def batch_process(
    data_dir: str | Path,
    output_dir: str | Path,
    process_func: Callable[[pd.DataFrame], pd.DataFrame],
    file_type: Literal["time_budget", "aggregated", "all"] = "all",
    output_format: str = "parquet",
    progress: bool = True,
) -> list[Path]:
    """Batch process all BORIS recordings in a directory.

    Args:
        data_dir: Directory containing input Excel files
        output_dir: Directory to save processed files
        process_func: Function that takes DataFrame and returns processed DataFrame
        file_type: Which files to process
        output_format: Output file format ('parquet' or 'csv')
        progress: If True, show progress bar

    Returns:
        List of paths to output files
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = get_recording_files(data_dir, file_type)
    output_files = []

    iterator = tqdm(files, desc="Processing", disable=not progress)

    for filepath in iterator:
        try:
            # Load file
            detected_type = get_file_type(filepath)
            df = load_boris_file(filepath, file_type=detected_type)

            # Apply processing function
            df_processed = process_func(df)

            # Determine output filename
            stem = filepath.stem
            if output_format == "parquet":
                output_path = output_dir / f"{stem}.parquet"
                save_parquet(df_processed, output_path)
            elif output_format == "csv":
                output_path = output_dir / f"{stem}.csv"
                df_processed.to_csv(output_path, index=False)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            output_files.append(output_path)

        except Exception as e:
            print(f"Warning: Failed to process {filepath.name}: {e}")
            continue

    return output_files
