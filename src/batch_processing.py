"""Batch processing pipeline for Tobii and Boris data.

This module provides functions to process entire datasets with support for:
- Parallel processing (one patient per CPU core)
- Selective processing (single file, multiple files, or entire directories)
- Automatic alignment of Tobii and Boris data
- Preservation of directory structure in processed output
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import pandas as pd
from tqdm import tqdm

from boris_pipeline import load_boris_file
from boris_pipeline.loader import get_file_type
from boris_pipeline.postprocess import BorisPostprocessor
from integration import (
    align_boris_to_tobii,
    compute_alignment_offset,
    create_epoch_dataset,
)
from tobii_pipeline.class_processing_tobii import ProcessingTobii
from tobii_pipeline.loader import load_recording
from tobii_pipeline.parser import parse_filename
from tobii_pipeline.utils import save_parquet

# =============================================================================
# Configuration
# =============================================================================

# Structure des données
DATA_ROOT = Path("Data")
PROCESSED_ROOT = DATA_ROOT / "processed"

# Colonnes à traiter par défaut pour Tobii
DEFAULT_TOBII_COLUMNS = [
    "Gaze point X",
    "Gaze point Y",
    "Gaze point 3D X",
    "Gaze point 3D Y",
    "Gaze point 3D Z",
    "Gaze direction left X",
    "Gaze direction left Y",
    "Gaze direction left Z",
    "Gaze direction right X",
    "Gaze direction right Y",
    "Gaze direction right Z",
    "Pupil position left X",
    "Pupil position left Y",
    "Pupil position left Z",
    "Pupil position right X",
    "Pupil position right Y",
    "Pupil position right Z",
    "Pupil diameter left",
    "Pupil diameter right",
    "Gyro X",
    "Gyro Y",
    "Gyro Z",
    "Accelerometer X",
    "Accelerometer Y",
    "Accelerometer Z",
]

# Paramètres de processing par défaut pour Tobii
DEFAULT_TOBII_PARAMS = {
    "artifact_detection": {
        "method": "both",
        "z_threshold": 4,
        "iqr_factor": 2,
    },
    "resampling": {
        "enabled": False,
        "target_interval_ms": None,
        "force": False,
    },
    "missing_data": {
        "max_gap_ms": 10000,
        "method": "interpolate",
    },
    "smoothing": {
        "method": "median",
        "window_size": 3,
        "poly_order": 2,
    },
    "baseline_correction": {
        "baseline_method": "None",
        "baseline_duration": 30,
    },
}

# =============================================================================
# Helper Functions
# =============================================================================


def get_tobii_files(data_dir: Path) -> list[Path]:
    """Get all Tobii TSV files in a directory.

    Args:
        data_dir: Directory to search

    Returns:
        List of TSV file paths
    """
    return sorted(data_dir.glob("*.tsv"))


def get_boris_files(data_dir: Path) -> list[Path]:
    """Get all Boris Excel files in a directory.

    Args:
        data_dir: Directory to search

    Returns:
        List of Excel file paths
    """
    return sorted(data_dir.glob("*.xlsx"))


def get_filename_prefix(filename: str, n_underscores: int = 6) -> str:
    """Extract filename prefix up to nth underscore.

    Args:
        filename: Filename (with or without path/extension)
        n_underscores: Number of underscores to include (default: 6)

    Returns:
        Prefix string (e.g., "G213_FAUJea_SDS2_P_M36_V4" for n=6)

    Example:
        >>> get_filename_prefix("G213_FAUJea_SDS2_P_M36_V4_25062025.tsv", 6)
        'G213_FAUJea_SDS2_P_M36_V4'
        >>> get_filename_prefix("G213_FAUJea_SDS2_P_M36_V4_25062025 Data Export.tsv", 6)
        'G213_FAUJea_SDS2_P_M36_V4'
    """
    # Remove path and extension
    stem = Path(filename).stem
    # Remove " Data Export" suffix if present
    stem = stem.replace(" Data Export", "")
    # Remove "_processed" or "_agregated" suffixes if present
    stem = stem.replace("_processed", "").replace("_agregated", "").replace("-aggregated", "")

    parts = stem.split("_")
    if len(parts) >= n_underscores + 1:
        return "_".join(parts[: n_underscores + 1])
    return stem


def find_matching_files(
    tobii_file: Path, boris_files: list[Path], n_underscores: int = 6
    ) -> Path | None:
    """Find Boris file matching Tobii file based on filename prefix.

    Args:
        tobii_file: Path to Tobii file
        boris_files: List of Boris file paths to search
        n_underscores: Number of underscores to match (default: 6)

    Returns:
        Matching Boris file path or None
    """
    tobii_prefix = get_filename_prefix(tobii_file.name, n_underscores)

    for boris_file in boris_files:
        boris_prefix = get_filename_prefix(boris_file.name, n_underscores)
        if tobii_prefix == boris_prefix:
            return boris_file

    return None


def find_original_boris_file(processed_boris_file: Path) -> Path | None:
    """Find original Boris Excel file from processed file path.

    Args:
        processed_boris_file: Path to processed Boris parquet file

    Returns:
        Path to original Excel file or None if not found
    """
    # Remove "_processed" suffix and change extension to .xlsx
    name_without_suffix = processed_boris_file.stem.replace("_processed", "")
    
    # Determine original directory (Data/data_G/Boris or Data/data_L/Boris)
    # processed file is in Data/processed/data_G/Boris or Data/processed/data_L/Boris
    processed_path = processed_boris_file
    parts = processed_path.parts
    
    # Determine which data directory (data_G or data_L)
    if "data_G" in parts:
        original_dir = DATA_ROOT / "data_G" / "Boris"
    elif "data_L" in parts:
        original_dir = DATA_ROOT / "data_L" / "Boris"
    else:
        # Fallback: try to find in both directories
        for data_dir in [DATA_ROOT / "data_G" / "Boris", DATA_ROOT / "data_L" / "Boris"]:
            if data_dir.exists():
                # Try with and without aggregated suffix
                for suffix in ["", "_agregated", "-agregated", "_aggregated", "-aggregated"]:
                    original_file = data_dir / f"{name_without_suffix}{suffix}.xlsx"
                    if original_file.exists():
                        return original_file
        return None
    
    if not original_dir.exists():
        return None
    
    # Try to find the original file (with or without aggregated suffix)
    # First try exact match
    original_file = original_dir / f"{name_without_suffix}.xlsx"
    if original_file.exists():
        return original_file
    
    # Try with aggregated suffixes
    for suffix in ["_agregated", "-agregated", "_aggregated", "-aggregated"]:
        original_file = original_dir / f"{name_without_suffix}{suffix}.xlsx"
        if original_file.exists():
            return original_file
    
    return None


def preserve_directory_structure(
    source_path: Path, source_root: Path, target_root: Path
    ) -> Path:
    """Preserve directory structure when moving from source to target.

    Args:
        source_path: Original file path
        source_root: Root of source directory structure
        target_root: Root of target directory structure

    Returns:
        Target path preserving relative structure

    Example:
        >>> preserve_directory_structure(
        ...     Path("Data/data_G/Tobii/file.tsv"),
        ...     Path("Data"),
        ...     Path("Data/processed")
        ... )
        Path("Data/processed/data_G/Tobii/file.tsv")
    """
    try:
        relative_path = source_path.relative_to(source_root)
    except ValueError:
        # If source_path is not under source_root, use just the filename
        relative_path = Path(source_path.name)

    return target_root / relative_path


# =============================================================================
# Processing Functions
# =============================================================================


def process_tobii_file(
    filepath: Path,
    output_path: Path,
    columns_to_process: list[str] | None = None,
    params_dict: dict | None = None,
    verbose: int = 0,
    ) -> dict:
    """Process a single Tobii file and save result.

    Args:
        filepath: Path to input TSV file
        output_path: Path to save processed file
        columns_to_process: Columns to process (default: DEFAULT_TOBII_COLUMNS)
        params_dict: Processing parameters (default: DEFAULT_TOBII_PARAMS)
        verbose: Verbose level for ProcessingTobii (0: aucun print, 1: étapes, 2+: tout)

    Returns:
        Dict with processing status and metadata
    """
    columns_to_process = columns_to_process or DEFAULT_TOBII_COLUMNS
    params_dict = params_dict or DEFAULT_TOBII_PARAMS

    try:
        # Load raw data
        tobii_data = load_recording(filepath)

        # Process
        processor = ProcessingTobii(tobii_data, params_dict, columns_to_process, verbose=verbose)
        processed_df, report = processor.apply_processing_steps()

        # Ensure "Recording timestamp" is preserved for alignment
        if "Recording timestamp" not in processed_df.columns:
            # Try to get from original data
            if "Recording timestamp" in tobii_data.columns:
                # Match by index length
                n_rows = len(processed_df)
                if n_rows <= len(tobii_data):
                    processed_df["Recording timestamp"] = tobii_data["Recording timestamp"].values[
                        :n_rows
                    ]
                else:
                    # If processed has more rows (resampling), interpolate timestamps
                    print(
                        f"Warning: Processed data has more rows than original. "
                        f"Timestamp interpolation may be needed for {filepath.name}"
                    )

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as parquet
        save_parquet(processed_df, output_path)

        # Parse metadata
        metadata = parse_filename(filepath.name)

        return {
            "status": "success",
            "filepath": filepath,
            "output_path": output_path,
            "n_rows": len(processed_df),
            "n_columns": len(processed_df.columns),
            "metadata": metadata,
            "report": report,
        }

    except Exception as e:
        return {
            "status": "error",
            "filepath": filepath,
            "error": str(e),
        }

def process_boris_file(
    filepath: Path,
    output_path: Path,
    file_type: Literal["time_budget", "aggregated", "auto"] = "auto",
    ) -> dict:
    """Process a single Boris file and save result.

    Args:
        filepath: Path to input Excel file
        output_path: Path to save processed file
        file_type: Type of Boris file (default: "auto")

    Returns:
        Dict with processing status and metadata
    """
    try:
        # Detect file type if auto
        if file_type == "auto":
            file_type = get_file_type(filepath)

        # Load Boris data
        boris_data = load_boris_file(filepath, file_type=file_type)

        # Post-process: remove columns, split Behavior, fill NaN
        postprocessor = BorisPostprocessor(file_type)
        boris_data = postprocessor.process(boris_data)

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Try to save as parquet (normal case - works for most files)
        try:
            save_parquet(boris_data, output_path)
        except Exception as parquet_error:
            # Fallback: if Parquet conversion fails (e.g., date column issues),
            # convert problematic columns to string and retry
            error_str = str(parquet_error)
            if "date" in error_str.lower() or "Conversion failed" in error_str:
                # Convert any remaining date-like columns to string
                for col in boris_data.columns:
                    if "date" in col.lower() or "Date" in col:
                        if boris_data[col].dtype == 'object' or pd.api.types.is_integer_dtype(boris_data[col]):
                            try:
                                # Try to convert Excel date integers to datetime then string
                                if pd.api.types.is_integer_dtype(boris_data[col]):
                                    boris_data[col] = pd.to_datetime(
                                        boris_data[col],
                                        origin="1899-12-30",
                                        unit="D",
                                        errors="coerce"
                                    ).astype(str)
                                else:
                                    boris_data[col] = boris_data[col].astype(str)
                            except Exception:
                                # If all else fails, just convert to string
                                boris_data[col] = boris_data[col].astype(str)
                
                # Retry saving after fixing date columns
                save_parquet(boris_data, output_path)
            else:
                # If it's a different error, re-raise it
                raise

        # Parse metadata
        from boris_pipeline.parser import parse_boris_filename

        metadata = parse_boris_filename(filepath.name)

        return {
            "status": "success",
            "filepath": filepath,
            "output_path": output_path,
            "n_rows": len(boris_data),
            "n_columns": len(boris_data.columns),
            "metadata": metadata,
        }

    except Exception as e:
        return {
            "status": "error",
            "filepath": filepath,
            "error": str(e),
        }


def align_tobii_boris_files(
    tobii_file: Path,
    boris_file: Path,
    output_path: Path,
    alignment_method: str = "start",
    time_before_s: float = 0.0,
    time_after_s: float = 0.0,
    boris_original_file: Path | None = None,
    ) -> dict:
    """Align Tobii and Boris data and create epoch dataset.

    Args:
        tobii_file: Path to processed Tobii file
        boris_file: Path to processed Boris file (for reference, not used for alignment)
        output_path: Path to save aligned dataset
        alignment_method: Method for alignment ("start", "end", "center")
        time_before_s: Time before events to include
        time_after_s: Time after events to include
        boris_original_file: Path to original Boris Excel file (required for alignment)

    Returns:
        Dict with alignment status and metadata
    """
    try:
        # Load processed Tobii data
        tobii_df = pd.read_parquet(tobii_file)
        
        # Load original Boris file (not processed) for alignment
        # This is needed because processed files don't have "Start (s)" and "Stop (s)" columns
        if boris_original_file is None:
            raise ValueError(
                "boris_original_file is required for alignment. "
                "Processed Boris files don't contain 'Start (s)' and 'Stop (s)' columns."
            )
        
        # Detect file type and load original Boris data
        boris_file_type = get_file_type(boris_original_file)
        boris_df = load_boris_file(boris_original_file, file_type=boris_file_type)

        # Ensure timestamp column exists
        if "Recording timestamp" not in tobii_df.columns:
            raise ValueError(
                f"Tobii file {tobii_file.name} missing 'Recording timestamp' column"
            )

        # Compute alignment offset
        offset = compute_alignment_offset(
            boris_df, tobii_df, method=alignment_method
        )

        # Align Boris to Tobii
        boris_aligned = align_boris_to_tobii(boris_df, tobii_df, offset_s=offset)

        # Create epoch dataset (annotate Tobii with behavior labels)
        aligned_df = create_epoch_dataset(
            tobii_df,
            boris_aligned,
            time_before_s=time_before_s,
            time_after_s=time_after_s,
        )

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save aligned dataset
        save_parquet(aligned_df, output_path)

        return {
            "status": "success",
            "tobii_file": tobii_file,
            "boris_file": boris_file,
            "output_path": output_path,
            "n_rows": len(aligned_df),
            "offset_s": offset,
        }

    except Exception as e:
        return {
            "status": "error",
            "tobii_file": tobii_file,
            "boris_file": boris_file,
            "error": str(e),
        }


# =============================================================================
# Batch Processing Functions
# =============================================================================


def process_tobii_batch(
    data_dirs: str | Path | list[str | Path] | None = None,
    files: list[Path] | None = None,
    output_root: Path | None = None,
    columns_to_process: list[str] | None = None,
    params_dict: dict | None = None,
    parallel: bool = True,
    n_jobs: int | None = None,
    progress: bool = True,
    file_organization: Literal["flat", "preserve"] = "preserve",
    skip_existing: bool = True,
    verbose: int = 0,
    ) -> list[dict]:
    """Process multiple Tobii files in batch.

    Args:
        data_dirs: Directory or list of directories to process (e.g., ["Data/data_G/Tobii"])
        files: Specific files to process (overrides data_dirs)
        output_root: Root directory for output (default: PROCESSED_ROOT)
        columns_to_process: Columns to process
        params_dict: Processing parameters
        parallel: If True, use parallel processing
        n_jobs: Number of parallel jobs (default: number of CPU cores)
        progress: Show progress bar
        file_organization: How to organize output files:
            - "preserve": Keep same directory structure
            - "flat": Put all files in output_root
        skip_existing: If True, skip files that already have processed output (default: True)
        verbose: Verbose level for ProcessingTobii (0: aucun print, 1: étapes, 2+: tout)

    Returns:
        List of processing result dicts
    """
    output_root = output_root or PROCESSED_ROOT
    columns_to_process = columns_to_process or DEFAULT_TOBII_COLUMNS
    params_dict = params_dict or DEFAULT_TOBII_PARAMS

    # Collect files to process
    if files:
        files_to_process = [Path(f) for f in files]
        # Determine source root from first file
        if files_to_process:
            # Try to find common root (Data/data_G or Data/data_L)
            first_file = files_to_process[0]
            if "data_G" in str(first_file):
                source_root = DATA_ROOT / "data_G"
            elif "data_L" in str(first_file):
                source_root = DATA_ROOT / "data_L"
            else:
                source_root = first_file.parent.parent.parent
        else:
            source_root = DATA_ROOT
    else:
        if data_dirs is None:
            # Default: process all data_G and data_L
            data_dirs = [
                DATA_ROOT / "data_G" / "Tobii",
                DATA_ROOT / "data_L" / "Tobii",
            ]
        elif isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]

        files_to_process = []
        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            if not data_dir.exists():
                print(f"Warning: Directory not found: {data_dir}")
                continue
            files_to_process.extend(get_tobii_files(data_dir))

        # Determine source root for structure preservation
        if files_to_process:
            # Find common root
            source_root = Path(files_to_process[0]).parent.parent.parent
        else:
            source_root = DATA_ROOT

    if not files_to_process:
        print("No Tobii files found to process")
        return []

    # Determine output paths
    tasks = []
    skipped_count = 0
    for filepath in files_to_process:
        if file_organization == "preserve":
            if source_root is None:
                source_root = filepath.parent.parent.parent
            output_path = preserve_directory_structure(
                filepath, source_root, output_root
            )
        else:
            output_path = output_root / filepath.name

        # Add "_processed" suffix
        output_path = output_path.parent / f"{output_path.stem}_processed.parquet"
        # Check if file already exists
        if skip_existing and output_path.exists():
            skipped_count += 1
            if progress:
                print(f"Skipping {filepath.name} (already processed: {output_path.name})")
            continue
        tasks.append((filepath, output_path))

    # Process files
    if parallel and len(tasks) > 1:
        n_jobs = n_jobs or mp.cpu_count()
        print(f"n_jobs: {n_jobs}")
        print(f"max coeur: {mp.cpu_count()}")
        results = []

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit tasks
            future_to_task = {
                executor.submit(
                    process_tobii_file,
                    filepath,
                    output_path,
                    columns_to_process,
                    params_dict,
                    verbose,
                ): (filepath, output_path)
                for filepath, output_path in tasks
            }

            # Collect results with progress bar
            iterator = (
                tqdm(as_completed(future_to_task), total=len(tasks), desc="Processing Tobii")
                if progress
                else as_completed(future_to_task)
            )

            for future in iterator:
                result = future.result()
                results.append(result)

                if result["status"] == "error" and progress:
                    tqdm.write(f"Error processing {result['filepath'].name}: {result['error']}")

    else:
        # Sequential processing
        iterator = tqdm(tasks, desc="Processing Tobii") if progress else tasks
        results = []

        for filepath, output_path in iterator:
            result = process_tobii_file(
                filepath, output_path, columns_to_process, params_dict, verbose
            )
            results.append(result)

            if result["status"] == "error":
                print(f"Error processing {filepath.name}: {result['error']}")

    return results


def process_boris_batch(
    data_dirs: str | Path | list[str | Path] | None = None,
    files: list[Path] | None = None,
    output_root: Path | None = None,
    file_type: Literal["time_budget", "aggregated", "auto"] = "auto",
    parallel: bool = True,
    n_jobs: int | None = None,
    progress: bool = True,
    file_organization: Literal["flat", "preserve"] = "preserve",
    skip_existing: bool = True,
    ) -> list[dict]:
    """Process multiple Boris files in batch.

    Args:
        data_dirs: Directory or list of directories to process
        files: Specific files to process (overrides data_dirs)
        output_root: Root directory for output (default: PROCESSED_ROOT)
        file_type: Type of Boris files to process
        parallel: If True, use parallel processing
        n_jobs: Number of parallel jobs
        progress: Show progress bar
        file_organization: How to organize output files
        skip_existing: If True, skip files that already have processed output (default: True)

    Returns:
        List of processing result dicts
    """
    output_root = output_root or PROCESSED_ROOT

    # Collect files to process
    if files:
        files_to_process = [Path(f) for f in files]
        # Determine source root from first file
        if files_to_process:
            first_file = files_to_process[0]
            if "data_G" in str(first_file):
                source_root = DATA_ROOT / "data_G"
            elif "data_L" in str(first_file):
                source_root = DATA_ROOT / "data_L"
            else:
                source_root = first_file.parent.parent.parent
        else:
            source_root = DATA_ROOT
    else:
        if data_dirs is None:
            # Default: process all data_G and data_L
            data_dirs = [
                DATA_ROOT / "data_G" / "Boris",
                DATA_ROOT / "data_L" / "Boris",
            ]
        elif isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]

        files_to_process = []
        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            if not data_dir.exists():
                print(f"Warning: Directory not found: {data_dir}")
                continue
            files_to_process.extend(get_boris_files(data_dir))

        if files_to_process:
            source_root = Path(files_to_process[0]).parent.parent.parent
        else:
            source_root = DATA_ROOT

    if not files_to_process:
        print("No Boris files found to process")
        return []

    # Determine output paths
    tasks = []
    skipped_count = 0
    for filepath in files_to_process:
        if file_organization == "preserve":
            if source_root is None:
                source_root = filepath.parent.parent.parent
            output_path = preserve_directory_structure(
                filepath, source_root, output_root
            )
        else:
            output_path = output_root / filepath.name

        # Add "_processed" suffix
        output_path = output_path.parent / f"{output_path.stem}_processed.parquet"

        # Check if file already exists
        if skip_existing and output_path.exists():
            skipped_count += 1
            if progress:
                print(f"Skipping {filepath.name} (already processed: {output_path.name})")
            continue

        tasks.append((filepath, output_path))

    if skipped_count > 0:
        print(f"Skipped {skipped_count} already processed Boris files")

    if not tasks:
        print("All Boris files already processed")
        return []

    # Process files
    if parallel and len(tasks) > 1:
        n_jobs = n_jobs or mp.cpu_count()
        results = []

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_task = {
                executor.submit(process_boris_file, filepath, output_path, file_type): (
                    filepath,
                    output_path,
                )
                for filepath, output_path in tasks
            }

            iterator = (
                tqdm(as_completed(future_to_task), total=len(tasks), desc="Processing Boris")
                if progress
                else as_completed(future_to_task)
            )

            for future in iterator:
                result = future.result()
                results.append(result)

                if result["status"] == "error" and progress:
                    tqdm.write(
                        f"Error processing {result['filepath'].name}: {result['error']}"
                    )

    else:
        iterator = tqdm(tasks, desc="Processing Boris") if progress else tasks
        results = []

        for filepath, output_path in iterator:
            result = process_boris_file(filepath, output_path, file_type)
            results.append(result)

            if result["status"] == "error":
                print(f"Error processing {filepath.name}: {result['error']}")

    return results


def align_tobii_boris_batch(
    processed_tobii_dir: Path | None = None,
    processed_boris_dir: Path | None = None,
    output_root: Path | None = None,
    alignment_method: str = "start",
    time_before_s: float = 0.0,
    time_after_s: float = 0.0,
    n_underscores: int = 6,
    parallel: bool = True,
    n_jobs: int | None = None,
    progress: bool = True,
    file_organization: Literal["flat", "preserve"] = "preserve",
    skip_existing: bool = True,
    ) -> list[dict]:
    """Align processed Tobii and Boris files in batch.

    Args:
        processed_tobii_dir: Directory with processed Tobii files
        processed_boris_dir: Directory with processed Boris files
        output_root: Root directory for aligned output
        alignment_method: Method for alignment
        time_before_s: Time before events to include
        time_after_s: Time after events to include
        n_underscores: Number of underscores to match filenames
        parallel: If True, use parallel processing
        n_jobs: Number of parallel jobs
        progress: Show progress bar
        file_organization: How to organize output files
        skip_existing: If True, skip files that already have aligned output (default: True)

    Returns:
        List of alignment result dicts
    """
    output_root = output_root or PROCESSED_ROOT / "aligned"

    # Find processed files
    if processed_tobii_dir is None:
        processed_tobii_dir = PROCESSED_ROOT / "data_G" / "Tobii"
        # Also check data_L
        tobii_dirs = [
            PROCESSED_ROOT / "data_G" / "Tobii",
            PROCESSED_ROOT / "data_L" / "Tobii",
        ]
    else:
        tobii_dirs = [Path(processed_tobii_dir)]

    if processed_boris_dir is None:
        boris_dirs = [
            PROCESSED_ROOT / "data_G" / "Boris",
            PROCESSED_ROOT / "data_L" / "Boris",
        ]
    else:
        boris_dirs = [Path(processed_boris_dir)]

    # Collect all processed files
    tobii_files = []
    for dir_path in tobii_dirs:
        if dir_path.exists():
            tobii_files.extend(dir_path.glob("*_processed.parquet"))

    boris_files = []
    for dir_path in boris_dirs:
        if dir_path.exists():
            boris_files.extend(dir_path.glob("*_processed.parquet"))

    if not tobii_files:
        print("No processed Tobii files found")
        return []

    if not boris_files:
        print("No processed Boris files found")
        return []

    # Find matching pairs and original Boris files
    tasks = []
    skipped_count = 0
    for tobii_file in tobii_files:
        boris_file = find_matching_files(tobii_file, boris_files, n_underscores)

        if boris_file is None:
            if progress:
                tqdm.write(
                    f"No matching Boris file found for {tobii_file.name}"
                )
            continue

        # Find original Boris Excel file (needed for alignment)
        boris_original_file = find_original_boris_file(boris_file)
        if boris_original_file is None:
            if progress:
                tqdm.write(
                    f"Warning: Could not find original Boris file for {boris_file.name}. "
                    f"Skipping alignment."
                )
            continue

        # Determine output path
        if file_organization == "preserve":
            # Preserve structure but put in "aligned" subdirectory
            source_root = tobii_file.parent.parent.parent
            output_path = preserve_directory_structure(
                tobii_file, source_root, output_root
            )
        else:
            output_path = output_root / tobii_file.name

        # Change suffix from "_processed" to "_aligned"
        output_path = output_path.parent / output_path.name.replace(
            "_processed", "_aligned"
        )

        # Check if file already exists
        if skip_existing and output_path.exists():
            skipped_count += 1
            if progress:
                print(f"Skipping alignment for {tobii_file.name} (already aligned: {output_path.name})")
            continue

        tasks.append((tobii_file, boris_file, output_path, boris_original_file))

    if skipped_count > 0:
        print(f"Skipped {skipped_count} already aligned files")

    if not tasks:
        print("No matching Tobii-Boris pairs found or all already aligned")
        return []

    # Align files
    if parallel and len(tasks) > 1:
        n_jobs = n_jobs or mp.cpu_count()
        results = []

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_task = {
                executor.submit(
                    align_tobii_boris_files,
                    tobii_file,
                    boris_file,
                    output_path,
                    alignment_method,
                    time_before_s,
                    time_after_s,
                    boris_original_file,
                ): (tobii_file, boris_file)
                for tobii_file, boris_file, output_path, boris_original_file in tasks
            }

            iterator = (
                tqdm(as_completed(future_to_task), total=len(tasks), desc="Aligning data")
                if progress
                else as_completed(future_to_task)
            )

            for future in iterator:
                result = future.result()
                results.append(result)

                if result["status"] == "error" and progress:
                    tqdm.write(
                        f"Error aligning {result.get('tobii_file', 'unknown')}: {result['error']}"
                    )

    else:
        iterator = tqdm(tasks, desc="Aligning data") if progress else tasks
        results = []

        for tobii_file, boris_file, output_path, boris_original_file in iterator:
            result = align_tobii_boris_files(
                tobii_file,
                boris_file,
                output_path,
                alignment_method,
                time_before_s,
                time_after_s,
                boris_original_file,
            )
            results.append(result)

            if result["status"] == "error":
                print(
                    f"Error aligning {tobii_file.name} and {boris_file.name}: {result['error']}"
                )

    return results


# =============================================================================
# Main Pipeline Function
# =============================================================================


def process_full_dataset(
    data_root: Path | None = None,
    output_root: Path | None = None,
    process_tobii: bool = True,
    process_boris: bool = True,
    align_data: bool = True,
    tobii_columns: list[str] | None = None,
    tobii_params: dict | None = None,
    boris_file_type: Literal["time_budget", "aggregated", "auto"] = "auto",
    alignment_method: str = "start",
    parallel: bool = True,
    n_jobs: int | None = None,
    file_organization: Literal["flat", "preserve"] = "preserve",
    progress: bool = True,
    skip_existing: bool = True,
    verbose: int = 0,
    ) -> tuple[list[dict], list[dict], list[dict]]:
    """Process full dataset: Tobii, Boris, and alignment.

    Args:
        data_root: Root directory of data (default: DATA_ROOT)
        output_root: Root directory for output (default: PROCESSED_ROOT)
        process_tobii: If True, process Tobii files
        process_boris: If True, process Boris files
        align_data: If True, align Tobii and Boris data
        tobii_columns: Columns to process for Tobii
        tobii_params: Processing parameters for Tobii
        boris_file_type: Type of Boris files to process
        alignment_method: Method for alignment
        parallel: If True, use parallel processing
        n_jobs: Number of parallel jobs
        file_organization: How to organize output files
        progress: Show progress bars
        skip_existing: If True, skip files that already have processed output (default: True)
        verbose: Verbose level for ProcessingTobii (0: aucun print, 1: étapes, 2+: tout)

    Returns:
        Dict with processing results
    """
    data_root = data_root or DATA_ROOT
    output_root = output_root or PROCESSED_ROOT

    

    results = {
        "tobii": [],
        "boris": [],
        "aligned": [],
    }

    # Process Tobii files
    if process_tobii:
        print("\n" + "=" * 80)
        print("PROCESSING TOBII FILES")
        print("=" * 80)
        tobii_dirs = [
            data_root / "data_G" / "Tobii",
            data_root / "data_L" / "Tobii",
        ]
        results["tobii"] = process_tobii_batch(
            data_dirs=tobii_dirs,
            output_root=output_root,
            columns_to_process=tobii_columns,
            params_dict=tobii_params,
            parallel=parallel,
            n_jobs=n_jobs,
            progress=progress,
            file_organization=file_organization,
            skip_existing=skip_existing,
            verbose=verbose,
        )

    # Process Boris files
    if process_boris:
        print("\n" + "=" * 80)
        print("PROCESSING BORIS FILES")
        print("=" * 80)
        boris_dirs = [
            data_root / "data_G" / "Boris",
            data_root / "data_L" / "Boris",
        ]
        results["boris"] = process_boris_batch(
            data_dirs=boris_dirs,
            output_root=output_root,
            file_type=boris_file_type,
            parallel=parallel,
            n_jobs=n_jobs,
            progress=progress,
            file_organization=file_organization,
            skip_existing=skip_existing,
        )

    # Align data
    if align_data:
        print("\n" + "=" * 80)
        print("ALIGNING TOBII AND BORIS DATA")
        print("=" * 80)
        results["aligned"] = align_tobii_boris_batch(
            output_root=output_root / "aligned",
            alignment_method=alignment_method,
            parallel=parallel,
            n_jobs=n_jobs,
            progress=progress,
            file_organization=file_organization,
            skip_existing=skip_existing,
        )

    # Summary
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)

    if process_tobii:
        tobii_success = sum(1 for r in results["tobii"] if r["status"] == "success")
        tobii_errors = sum(1 for r in results["tobii"] if r["status"] == "error")
        print(f"Tobii: {tobii_success} successful, {tobii_errors} errors")

    if process_boris:
        boris_success = sum(1 for r in results["boris"] if r["status"] == "success")
        boris_errors = sum(1 for r in results["boris"] if r["status"] == "error")
        print(f"Boris: {boris_success} successful, {boris_errors} errors")

    if align_data:
        aligned_success = sum(1 for r in results["aligned"] if r["status"] == "success")
        aligned_errors = sum(1 for r in results["aligned"] if r["status"] == "error")
        print(f"Aligned: {aligned_success} successful, {aligned_errors} errors")

    return results


# =============================================================================
# Exemples d'utilisation
# =============================================================================

def run_examples():
    """Exemples d'utilisation de toutes les fonctions de batch_processing.
    
    Décommentez l'exemple que vous voulez exécuter.
    """
    
    # ========================================================================
    # EXEMPLE 1: Pipeline complet (Tobii + Boris + Alignement)
    # ========================================================================
    results = process_full_dataset(
        process_tobii=False,      # Traiter les fichiers Tobii
        process_boris=True,       # Traiter les fichiers Boris
        align_data=True,         # Aligner les données
        parallel=True,           # Traitement en parallèle
        n_jobs=None,             # Utilise tous les cœurs (ou mettre un nombre)
        file_organization="preserve",  # Conserve la structure de dossiers
        progress=True,           # Affiche les barres de progression
        skip_existing=False,      # Ignore les fichiers déjà traités (reprise après crash)
        verbose=1,               # 0: aucun print, 1: étapes principales, 2+: tous les prints
    )
    
    # ========================================================================
    # EXEMPLE 2: Seulement Tobii
    # ========================================================================
    # results = process_full_dataset(
    #     process_tobii=True,
    #     process_boris=False,
    #     align_data=False,
    #     parallel=True,
    #     skip_existing=True,  # Ignore les fichiers déjà traités
    #     n_jobs=None, # Utilise tous les cœurs disponibles
    #     verbose=0
    # )
    
    # ========================================================================
    # EXEMPLE 3: Seulement Boris
    # ========================================================================
    # results = process_full_dataset(
    #     process_tobii=False,
    #     process_boris=True,
    #     align_data=False,
    #     parallel=True,
    #     n_jobs=-1,  # Utilise tous les cœurs disponibles
    #     skip_existing=True,  # Ignore les fichiers déjà traités
    # )
    
    # ========================================================================
    # EXEMPLE 4: Tobii et Boris sans alignement
    # ========================================================================
    # results = process_full_dataset(
    #     process_tobii=True,
    #     process_boris=True,
    #     align_data=False,
    #     parallel=True,
    #     n_jobs=-1,  # Utilise tous les cœurs disponibles
    #     skip_existing=True,  # Ignore les fichiers déjà traités
    # )
    
    # ========================================================================
    # EXEMPLE 5: Traiter un seul fichier Tobii
    # ========================================================================
    # from pathlib import Path
    # filepath = Path("Data/data_G/Tobii/G213_FAUJea_SDS2_P_M36_V4_25062025 Data Export.tsv")
    # results = process_tobii_batch(
    #     files=[filepath],
    #     parallel=False,
    #     progress=True,
    # )
    
    # ========================================================================
    # EXEMPLE 6: Traiter un dossier spécifique
    # ========================================================================
    # results = process_tobii_batch(
    #     data_dirs=[Path("Data/data_G/Tobii")],
    #     parallel=True,
    #     progress=True,
    # )
    
    # ========================================================================
    # EXEMPLE 7: Traiter plusieurs fichiers spécifiques
    # ========================================================================
    # files = [
    #     Path("Data/data_G/Tobii/G213_FAUJea_SDS2_P_M36_V4_25062025 Data Export.tsv"),
    #     Path("Data/data_G/Tobii/G215_BENNaw_SDS2_C_M0_V1_30062025 Data Export.tsv"),
    # ]
    # results = process_tobii_batch(
    #     files=files,
    #     parallel=True,
    #     progress=True,
    # )
    
    # ========================================================================
    # EXEMPLE 8: Traiter Boris avec type spécifique
    # ========================================================================
    # results = process_boris_batch(
    #     file_type="aggregated",  # Ou "time_budget" ou "auto"
    #     parallel=True,
    #     progress=True,
    # )
    
    # ========================================================================
    # EXEMPLE 9: Aligner seulement (si déjà traité)
    # ========================================================================
    # results = align_tobii_boris_batch(
    #     alignment_method="start",  # Ou "end" ou "center"
    #     time_before_s=0.0,
    #     time_after_s=0.0,
    #     parallel=True,
    #     progress=True,
    #     n_jobs=-1,  # Utilise tous les cœurs disponibles
    #     skip_existing=True,  # Ignore les fichiers déjà traités
    # )
    
    # ========================================================================
    # EXEMPLE 10: Traitement avec paramètres personnalisés Tobii
    # ========================================================================
    # custom_params = {
    #     "artifact_detection": {
    #         "method": "both",
    #         "z_threshold": 3,
    #         "iqr_factor": 1.5,
    #     },
    #     "resampling": {
    #         "enabled": False,
    #     },
    #     "missing_data": {
    #         "max_gap_ms": 5000,
    #         "method": "interpolate",
    #     },
    #     "smoothing": {
    #         "method": "savgol",
    #         "window_size": 5,
    #         "poly_order": 2,
    #     },
    #     "baseline_correction": {
    #         "baseline_method": "first_n_seconds",
    #         "baseline_duration": 30,
    #     },
    # }
    # 
    # custom_columns = [
    #     "Pupil diameter left",
    #     "Pupil diameter right",
    #     "Gaze point X",
    #     "Gaze point Y",
    # ]
    # 
    # results = process_full_dataset(
    #     process_tobii=True,
    #     process_boris=True,
    #     align_data=True,
    #     tobii_columns=custom_columns,
    #     tobii_params=custom_params,
    #     parallel=True,
    # )
    
    return results


if __name__ == "__main__":
    # Exécuter les exemples
    results = run_examples()
    
    