"""Data loading functions for BORIS behavioral observation files."""

from pathlib import Path
from typing import Literal

import pandas as pd
from tqdm import tqdm

from .constants import AGGREGATED_PATTERNS, AGGREGATED_SHEET, TIME_BUDGET_SHEET
from .parser import parse_boris_filename


def is_aggregated_file(filepath: str | Path) -> bool:
    """Check if file is an aggregated BORIS file based on filename.

    Handles various suffix patterns: _agregated, -agregated, _aggregated, -aggregated

    Args:
        filepath: Path to the Excel file

    Returns:
        True if filename contains aggregated suffix pattern

    Examples:
        >>> is_aggregated_file("G213_FAUJea_SDS2_P_M36_V4_25062025_agregated.xlsx")
        True
        >>> is_aggregated_file("G213_FAUJea_SDS2_P_M36_V4_25062025.xlsx")
        False
    """
    filepath = Path(filepath)
    stem_lower = filepath.stem.lower()
    return any(pattern in stem_lower for pattern in AGGREGATED_PATTERNS)


def get_file_type(filepath: str | Path) -> Literal["time_budget", "aggregated"]:
    """Detect BORIS file type based on Excel sheet names.

    Args:
        filepath: Path to the Excel file

    Returns:
        'time_budget' for original files, 'aggregated' for aggregated files

    Raises:
        ValueError: If neither expected sheet is found
    """
    filepath = Path(filepath)
    xlsx = pd.ExcelFile(filepath)
    sheet_names = xlsx.sheet_names

    if TIME_BUDGET_SHEET in sheet_names:
        return "time_budget"
    if AGGREGATED_SHEET in sheet_names:
        return "aggregated"

    raise ValueError(
        f"Could not detect file type for {filepath.name}. "
        f"Expected sheet '{TIME_BUDGET_SHEET}' or '{AGGREGATED_SHEET}', "
        f"found: {sheet_names}"
    )


def load_time_budget(filepath: str | Path) -> pd.DataFrame:
    """Load original BORIS file with Time budget sheet.

    Args:
        filepath: Path to the Excel file

    Returns:
        DataFrame with time budget summary data (behavioral statistics)
    """
    filepath = Path(filepath)
    return pd.read_excel(filepath, sheet_name=TIME_BUDGET_SHEET)


def load_aggregated_events(filepath: str | Path) -> pd.DataFrame:
    """Load aggregated BORIS file with event-level data.

    Args:
        filepath: Path to the Excel file

    Returns:
        DataFrame with individual behavioral events (start/stop times)
    """
    filepath = Path(filepath)
    return pd.read_excel(filepath, sheet_name=AGGREGATED_SHEET)


def load_boris_file(
    filepath: str | Path,
    file_type: Literal["time_budget", "aggregated", "auto"] = "auto",
) -> pd.DataFrame:
    """Load a BORIS Excel file (either type).

    Args:
        filepath: Path to the Excel file
        file_type: Type of file to load. 'auto' detects from sheet names.

    Returns:
        DataFrame with behavioral observation data

    Raises:
        ValueError: If file_type is invalid or sheet not found
    """
    filepath = Path(filepath)

    if file_type == "auto":
        file_type = get_file_type(filepath)

    if file_type == "time_budget":
        return load_time_budget(filepath)
    if file_type == "aggregated":
        return load_aggregated_events(filepath)

    raise ValueError(f"Invalid file_type: {file_type}. Use 'time_budget', 'aggregated', or 'auto'")


def get_recording_files(
    data_dir: str | Path,
    file_type: Literal["time_budget", "aggregated", "all"] = "all",
) -> list[Path]:
    """Get list of BORIS recording files in a directory.

    Args:
        data_dir: Directory to search
        file_type: Which files to include:
            - 'time_budget': Only original files (no aggregated suffix)
            - 'aggregated': Only aggregated files
            - 'all': All Excel files

    Returns:
        Sorted list of file paths
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    all_files = sorted(data_dir.glob("*.xlsx"))

    if file_type == "all":
        return all_files
    if file_type == "time_budget":
        return [f for f in all_files if not is_aggregated_file(f)]
    if file_type == "aggregated":
        return [f for f in all_files if is_aggregated_file(f)]

    raise ValueError(f"Invalid file_type: {file_type}. Use 'time_budget', 'aggregated', or 'all'")


def _add_metadata_columns(df: pd.DataFrame, filepath: Path, file_type: str) -> pd.DataFrame:
    """Add metadata columns to DataFrame based on filename.

    Args:
        df: DataFrame to modify
        filepath: Source file path for metadata extraction
        file_type: 'time_budget' or 'aggregated'

    Returns:
        DataFrame with added metadata columns
    """
    metadata = parse_boris_filename(filepath.name)

    df = df.copy()
    df["source_file"] = filepath.name
    df["recording_id"] = metadata["id"]
    df["participant_code"] = metadata["participant"]
    df["group"] = metadata["group"]
    df["month"] = metadata["month"]
    df["visit"] = metadata["visit"]
    df["file_type"] = file_type

    return df


def load_recordings_from_dir(
    data_dir: str | Path,
    file_type: Literal["time_budget", "aggregated", "all"] = "all",
    add_metadata: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """Load all BORIS recordings from a directory.

    Args:
        data_dir: Directory containing Excel files
        file_type: Which files to load - 'time_budget', 'aggregated', or 'all'
        add_metadata: If True, add columns with parsed filename metadata
        progress: If True, show progress bar

    Returns:
        Combined DataFrame with all recordings

    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If no files found
    """
    data_dir = Path(data_dir)
    files = get_recording_files(data_dir, file_type)

    if not files:
        raise ValueError(f"No BORIS files found in {data_dir} with file_type='{file_type}'")

    dfs = []
    iterator = tqdm(files, desc="Loading BORIS files", disable=not progress)

    for filepath in iterator:
        try:
            # Detect type for each file when loading "all"
            detected_type = get_file_type(filepath)
            df = load_boris_file(filepath, file_type=detected_type)

            if add_metadata:
                df = _add_metadata_columns(df, filepath, detected_type)

            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {filepath.name}: {e}")
            continue

    if not dfs:
        raise ValueError(f"Failed to load any files from {data_dir}")

    return pd.concat(dfs, ignore_index=True)


def load_participant(
    data_dirs: str | Path | list[str | Path],
    participant_code: str,
    file_type: Literal["time_budget", "aggregated", "all"] = "all",
    add_metadata: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """Load all BORIS recordings for a specific participant.

    Args:
        data_dirs: Directory or list of directories to search
        participant_code: 6-character participant code (e.g., 'FAUJea')
        file_type: Which files to load
        add_metadata: If True, add columns with parsed filename metadata
        progress: If True, show progress bar

    Returns:
        Combined DataFrame with all recordings for the participant

    Raises:
        ValueError: If no files found for participant
    """
    if isinstance(data_dirs, str | Path):
        data_dirs = [data_dirs]

    data_dirs = [Path(d) for d in data_dirs]

    # Find all files for participant
    participant_files = []
    for data_dir in data_dirs:
        all_files = get_recording_files(data_dir, file_type)
        for filepath in all_files:
            if f"_{participant_code}_" in filepath.name:
                participant_files.append(filepath)

    if not participant_files:
        dirs_str = ", ".join(str(d) for d in data_dirs)
        raise ValueError(f"No files found for participant '{participant_code}' in: {dirs_str}")

    dfs = []
    iterator = tqdm(
        participant_files,
        desc=f"Loading {participant_code}",
        disable=not progress,
    )

    for filepath in iterator:
        try:
            detected_type = get_file_type(filepath)
            df = load_boris_file(filepath, file_type=detected_type)

            if add_metadata:
                df = _add_metadata_columns(df, filepath, detected_type)

            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {filepath.name}: {e}")
            continue

    if not dfs:
        raise ValueError(f"Failed to load any files for participant '{participant_code}'")

    return pd.concat(dfs, ignore_index=True)
