"""Data cleaning functions for Tobii eye-tracking data."""

import pandas as pd

# Columns that contain numeric data with European decimal format (comma as separator)
NUMERIC_COLUMNS = [
    "Recording timestamp",
    "Computer timestamp",
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
    "Recording media width",
    "Recording media height",
    "Gaze event duration",
    "Eye movement type index",
    "Fixation point X",
    "Fixation point Y",
    "Gyro X",
    "Gyro Y",
    "Gyro Z",
    "Accelerometer X",
    "Accelerometer Y",
    "Accelerometer Z",
]


def fix_decimal_separator(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Convert European decimal format (comma) to standard format (dot).

    Args:
        df: Input DataFrame
        columns: List of columns to convert. If None, uses NUMERIC_COLUMNS.

    Returns:
        DataFrame with corrected numeric columns
    """
    df = df.copy()
    columns = columns or NUMERIC_COLUMNS

    for col in columns:
        if col in df.columns:
            # Only process if column contains strings with commas
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean_recording(df: pd.DataFrame, fix_decimals: bool = True) -> pd.DataFrame:
    """Main cleaning pipeline for Tobii data.

    Performs:
    - Fixes decimal separators (comma -> dot)
    - Converts columns to appropriate dtypes
    - Standardizes column names (optional)

    Args:
        df: Raw DataFrame from load_recording()
        fix_decimals: If True, convert European decimal format

    Returns:
        Cleaned DataFrame
    """
    df = df.copy()

    if fix_decimals:
        df = fix_decimal_separator(df)

    # Convert validity columns to categorical
    for col in ["Validity left", "Validity right"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Convert sensor column to categorical
    if "Sensor" in df.columns:
        df["Sensor"] = df["Sensor"].astype("category")

    # Convert eye movement type to categorical
    if "Eye movement type" in df.columns:
        df["Eye movement type"] = df["Eye movement type"].astype("category")

    return df


def filter_by_sensor(df: pd.DataFrame, sensor: str) -> pd.DataFrame:
    """Filter data by sensor type.

    Args:
        df: Input DataFrame
        sensor: Sensor type - 'Eye Tracker', 'Accelerometer', or 'Gyroscope'

    Returns:
        Filtered DataFrame containing only rows from the specified sensor
    """
    if "Sensor" not in df.columns:
        raise ValueError("DataFrame must contain 'Sensor' column")

    return df[df["Sensor"] == sensor].copy()


def filter_eye_tracker(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to keep only eye tracker data."""
    return filter_by_sensor(df, "Eye Tracker")


def filter_accelerometer(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to keep only accelerometer data."""
    return filter_by_sensor(df, "Accelerometer")


def filter_gyroscope(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to keep only gyroscope data."""
    return filter_by_sensor(df, "Gyroscope")


def filter_valid_gaze(df: pd.DataFrame, both_eyes: bool = True) -> pd.DataFrame:
    """Filter to keep only valid gaze samples.

    Args:
        df: Input DataFrame (should be eye tracker data)
        both_eyes: If True, require both eyes valid. If False, require at least one.

    Returns:
        Filtered DataFrame with only valid gaze samples
    """
    if "Validity left" not in df.columns or "Validity right" not in df.columns:
        raise ValueError("DataFrame must contain 'Validity left' and 'Validity right' columns")

    if both_eyes:
        mask = (df["Validity left"] == "Valid") & (df["Validity right"] == "Valid")
    else:
        mask = (df["Validity left"] == "Valid") | (df["Validity right"] == "Valid")

    return df[mask].copy()


def get_gaze_columns() -> list[str]:
    """Get list of gaze-related column names."""
    return [
        "Recording timestamp",
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
        "Pupil diameter left",
        "Pupil diameter right",
        "Validity left",
        "Validity right",
        "Eye movement type",
        "Gaze event duration",
        "Fixation point X",
        "Fixation point Y",
    ]


def get_motion_columns() -> list[str]:
    """Get list of motion sensor column names."""
    return [
        "Recording timestamp",
        "Gyro X",
        "Gyro Y",
        "Gyro Z",
        "Accelerometer X",
        "Accelerometer Y",
        "Accelerometer Z",
    ]


def select_gaze_data(df: pd.DataFrame) -> pd.DataFrame:
    """Select only gaze-related columns from eye tracker data.

    Args:
        df: Input DataFrame (should be eye tracker data)

    Returns:
        DataFrame with only gaze-related columns that exist in the input
    """
    cols = [c for c in get_gaze_columns() if c in df.columns]
    return df[cols].copy()


def select_motion_data(df: pd.DataFrame) -> pd.DataFrame:
    """Select only motion sensor columns.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with only motion-related columns that exist in the input
    """
    cols = [c for c in get_motion_columns() if c in df.columns]
    return df[cols].copy()
