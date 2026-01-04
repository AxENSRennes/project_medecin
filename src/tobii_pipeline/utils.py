"""Utility functions for the Tobii data pipeline."""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .parser import parse_filename


def save_parquet(
    df: pd.DataFrame,
    output_path: str | Path,
    compression: str = "snappy",
) -> None:
    """Save DataFrame to Parquet format.

    Args:
        df: DataFrame to save
        output_path: Output file path (will create parent directories)
        compression: Compression algorithm ('snappy', 'gzip', 'brotli', or None)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, compression=compression, index=False)


def load_parquet(filepath: str | Path) -> pd.DataFrame:
    """Load DataFrame from Parquet file.

    Args:
        filepath: Path to Parquet file

    Returns:
        DataFrame
    """
    return pd.read_parquet(filepath)


def list_recordings(data_dir: str | Path, pattern: str = "*.tsv") -> list[dict]:
    """List all recordings with parsed metadata.

    Args:
        data_dir: Directory containing recording files
        pattern: Glob pattern to match files

    Returns:
        List of dicts with metadata and file info for each recording
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(pattern))

    recordings = []
    for filepath in files:
        metadata = parse_filename(filepath.name)
        metadata["filepath"] = filepath
        metadata["filesize_mb"] = filepath.stat().st_size / (1024 * 1024)
        recordings.append(metadata)

    return recordings


def list_participants(data_dirs: str | Path | list[str | Path]) -> list[str]:
    """Get unique participant codes from directories.

    Args:
        data_dirs: Directory or list of directories to search

    Returns:
        Sorted list of unique participant codes
    """
    if isinstance(data_dirs, str | Path):
        data_dirs = [data_dirs]

    participants = set()
    for data_dir in data_dirs:
        for recording in list_recordings(data_dir):
            if recording["participant"]:
                participants.add(recording["participant"])

    return sorted(participants)


def get_data_summary(data_dirs: str | Path | list[str | Path]) -> dict:
    """Get summary statistics for all recordings.

    Args:
        data_dirs: Directory or list of directories to search

    Returns:
        Dict with summary statistics
    """
    if isinstance(data_dirs, str | Path):
        data_dirs = [data_dirs]

    all_recordings = []
    for data_dir in data_dirs:
        all_recordings.extend(list_recordings(data_dir))

    if not all_recordings:
        return {"total_files": 0}

    total_size = sum(r["filesize_mb"] for r in all_recordings)
    participants = {r["participant"] for r in all_recordings if r["participant"]}
    groups = {}
    months = {}

    for r in all_recordings:
        if r["group"]:
            groups[r["group"]] = groups.get(r["group"], 0) + 1
        if r["month"] is not None:
            months[r["month"]] = months.get(r["month"], 0) + 1

    return {
        "total_files": len(all_recordings),
        "total_size_mb": round(total_size, 1),
        "total_size_gb": round(total_size / 1024, 2),
        "unique_participants": len(participants),
        "participants": sorted(participants),
        "groups": groups,
        "months": dict(sorted(months.items())),
    }


def batch_process(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    data_dir: str | Path,
    output_dir: str | Path,
    process_func,
    pattern: str = "*.tsv",
    output_format: str = "parquet",
    progress: bool = True,
) -> list[Path]:
    """Batch process all recordings in a directory.

    Args:
        data_dir: Directory containing input TSV files
        output_dir: Directory to save processed files
        process_func: Function that takes a DataFrame and returns processed DataFrame
        pattern: Glob pattern to match input files
        output_format: Output format ('parquet' or 'csv')
        progress: If True, show progress bar

    Returns:
        List of output file paths

    Example:
        from tobii_pipeline import batch_process, clean_recording

        output_files = batch_process(
            data_dir="Data/data_G/Tobii",
            output_dir="Data/processed/data_G",
            process_func=clean_recording,
        )
    """
    from .loader import load_recording  # pylint: disable=import-outside-toplevel

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob(pattern))
    output_files = []

    iterator = tqdm(files, desc="Processing") if progress else files

    for filepath in iterator:
        # Load and process
        df = load_recording(filepath)
        df_processed = process_func(df)

        # Generate output filename
        stem = filepath.stem.replace(" Data Export", "")
        if output_format == "parquet":
            output_path = output_dir / f"{stem}.parquet"
            save_parquet(df_processed, output_path)
        else:
            output_path = output_dir / f"{stem}.csv"
            df_processed.to_csv(output_path, index=False)

        output_files.append(output_path)

    return output_files
