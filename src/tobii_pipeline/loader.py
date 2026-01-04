"""Data loading functions for Tobii eye-tracking recordings."""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .parser import parse_filename


def load_recording(
    filepath: str | Path,
    chunksize: int | None = None,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Load a single Tobii TSV recording file.

    Args:
        filepath: Path to the TSV file
        chunksize: If specified, return an iterator of DataFrames with this many rows each.
                   Useful for processing large files without loading into memory.

    Returns:
        DataFrame if chunksize is None, otherwise an Iterator of DataFrames
    """
    filepath = Path(filepath)

    # Tobii exports use tab separator and European decimal format (comma)
    read_kwargs = {
        "sep": "\t",
        "low_memory": False,
    }

    if chunksize:
        return pd.read_csv(filepath, chunksize=chunksize, **read_kwargs)
    return pd.read_csv(filepath, **read_kwargs)


def load_recordings_from_dir(
    data_dir: str | Path,
    pattern: str = "*.tsv",
    add_metadata: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """Load all Tobii recordings from a directory.

    Args:
        data_dir: Directory containing TSV files
        pattern: Glob pattern to match files (default: *.tsv)
        add_metadata: If True, add columns with parsed filename metadata
        progress: If True, show progress bar

    Returns:
        Combined DataFrame with all recordings
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(pattern))

    if not files:
        raise ValueError(f"No files matching '{pattern}' found in {data_dir}")

    dfs = []
    iterator = tqdm(files, desc="Loading recordings") if progress else files

    for filepath in iterator:
        df = load_recording(filepath)

        if add_metadata:
            metadata = parse_filename(filepath.name)
            df["source_file"] = filepath.name
            df["recording_id"] = metadata["id"]
            df["participant_code"] = metadata["participant"]
            df["group"] = metadata["group"]
            df["month"] = metadata["month"]
            df["visit"] = metadata["visit"]

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_participant(
    data_dirs: str | Path | list[str | Path],
    participant_code: str,
    add_metadata: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """Load all recordings for a specific participant.

    Args:
        data_dirs: Directory or list of directories to search for recordings
        participant_code: 6-character participant code (e.g., 'FAUJea')
        add_metadata: If True, add columns with parsed filename metadata
        progress: If True, show progress bar

    Returns:
        Combined DataFrame with all recordings for the participant
    """
    if isinstance(data_dirs, str | Path):
        data_dirs = [data_dirs]

    files = []
    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        # Match files containing the participant code
        files.extend(data_dir.glob(f"*_{participant_code}_*.tsv"))

    if not files:
        raise ValueError(f"No recordings found for participant '{participant_code}'")

    files = sorted(files)
    dfs = []
    iterator = tqdm(files, desc=f"Loading {participant_code}") if progress else files

    for filepath in iterator:
        df = load_recording(filepath)

        if add_metadata:
            metadata = parse_filename(filepath.name)
            df["source_file"] = filepath.name
            df["recording_id"] = metadata["id"]
            df["participant_code"] = metadata["participant"]
            df["group"] = metadata["group"]
            df["month"] = metadata["month"]
            df["visit"] = metadata["visit"]

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def get_recording_files(
    data_dir: str | Path,
    pattern: str = "*.tsv",
) -> list[Path]:
    """Get list of recording files in a directory.

    Args:
        data_dir: Directory to search
        pattern: Glob pattern to match files

    Returns:
        Sorted list of file paths
    """
    data_dir = Path(data_dir)
    return sorted(data_dir.glob(pattern))
