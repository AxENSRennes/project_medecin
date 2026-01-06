"""Adapter for pymovements library integration.

Provides functions to convert Tobii data to pymovements format and use
pymovements for event detection and metric computation.
"""

from typing import Literal

import pandas as pd
import polars as pl
import pymovements as pm

# Use Gaze class (GazeDataFrame is deprecated in pymovements v0.23.0+)
try:
    GazeClass = pm.Gaze
except AttributeError:
    GazeClass = pm.GazeDataFrame

# Default screen parameters for Tobii recordings
DEFAULT_SCREEN_WIDTH_PX = 1920
DEFAULT_SCREEN_HEIGHT_PX = 1080
DEFAULT_SCREEN_WIDTH_CM = 52.7  # Typical 24" monitor
DEFAULT_SCREEN_HEIGHT_CM = 29.6
DEFAULT_DISTANCE_CM = 60.0  # Typical viewing distance
DEFAULT_SAMPLING_RATE = 100  # Tobii default


def df_to_gaze_dataframe(
    df: pd.DataFrame,
    screen_width_px: int = DEFAULT_SCREEN_WIDTH_PX,
    screen_height_px: int = DEFAULT_SCREEN_HEIGHT_PX,
    screen_width_cm: float = DEFAULT_SCREEN_WIDTH_CM,
    screen_height_cm: float = DEFAULT_SCREEN_HEIGHT_CM,
    distance_cm: float = DEFAULT_DISTANCE_CM,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
) -> pm.GazeDataFrame:
    """Convert a cleaned Tobii DataFrame to pymovements GazeDataFrame.

    Args:
        df: Cleaned Tobii DataFrame with gaze data.
        screen_width_px: Screen width in pixels.
        screen_height_px: Screen height in pixels.
        screen_width_cm: Screen width in centimeters.
        screen_height_cm: Screen height in centimeters.
        distance_cm: Distance from eyes to screen in centimeters.
        sampling_rate: Eye tracker sampling rate in Hz.

    Returns:
        pymovements GazeDataFrame ready for processing.
    """
    # Create experiment definition with screen/tracker metadata
    experiment = pm.Experiment(
        screen_width_px=screen_width_px,
        screen_height_px=screen_height_px,
        screen_width_cm=screen_width_cm,
        screen_height_cm=screen_height_cm,
        distance_cm=distance_cm,
        origin="center",
        sampling_rate=sampling_rate,
    )

    # Prepare data for pymovements
    # pymovements expects time in ms and pixel coordinates
    gaze_data = {
        "time": df["Recording timestamp"].values / 1000,  # Convert to ms
        "x": df["Gaze point X"].values,
        "y": df["Gaze point Y"].values,
    }

    # Add pupil data if available
    if "Pupil diameter left" in df.columns:
        gaze_data["pupil_left"] = df["Pupil diameter left"].values
    if "Pupil diameter right" in df.columns:
        gaze_data["pupil_right"] = df["Pupil diameter right"].values

    # Convert to polars (pymovements native format)
    pl_df = pl.DataFrame(gaze_data)

    # Create Gaze object (using GazeClass for compatibility)
    gaze = GazeClass(
        samples=pl_df,
        experiment=experiment,
        time_column="time",
        time_unit="ms",
        pixel_columns=["x", "y"],
    )

    return gaze


def apply_pix2deg(gaze: pm.GazeDataFrame) -> pm.GazeDataFrame:
    """Convert pixel coordinates to degrees of visual angle.

    Args:
        gaze: pymovements GazeDataFrame with pixel coordinates.

    Returns:
        GazeDataFrame with position in degrees added.
    """
    gaze.pix2deg()
    return gaze


def apply_pos2vel(
    gaze: pm.GazeDataFrame,
    method: Literal["smooth", "neighbors", "preceding"] = "smooth",
) -> pm.GazeDataFrame:
    """Compute velocity from position data.

    Args:
        gaze: pymovements GazeDataFrame with position data.
        method: Velocity computation method.
            - "smooth": Savitzky-Golay filter (recommended)
            - "neighbors": Finite difference using neighboring samples
            - "preceding": Finite difference using preceding sample

    Returns:
        GazeDataFrame with velocity columns added.
    """
    gaze.pos2vel(method=method)
    return gaze


def detect_events_ivt(
    gaze: pm.GazeDataFrame,
    velocity_threshold: float = 30.0,
    minimum_duration: int = 100,
    saccade_minimum_duration: int = 10,
) -> pm.GazeDataFrame:
    """Detect fixations and saccades using I-VT (velocity-threshold) algorithm.

    The I-VT algorithm classifies samples as fixations when velocity is below
    the threshold, and as saccades when velocity exceeds the threshold.

    Args:
        gaze: pymovements GazeDataFrame with velocity computed.
        velocity_threshold: Velocity threshold in degrees/second.
            Samples below this are classified as fixations.
        minimum_duration: Minimum fixation duration in milliseconds.
        saccade_minimum_duration: Minimum saccade duration in milliseconds.
            Default 10ms (1 sample at 100Hz).

    Returns:
        GazeDataFrame with detected events.
    """
    # Ensure velocity is computed (use samples for new API, frame for old)
    samples = gaze.samples if hasattr(gaze, "samples") else gaze.frame
    if "velocity" not in samples.columns:
        gaze.pos2vel()

    # Detect fixations using I-VT
    gaze.detect(
        "ivt",
        velocity_threshold=velocity_threshold,
        minimum_duration=minimum_duration,
        name="fixation",
    )

    # Detect saccades using microsaccades with explicit threshold
    # The threshold is an elliptic (x, y) velocity threshold in deg/s
    # This detects periods where velocity exceeds the threshold
    try:
        gaze.detect(
            "microsaccades",
            threshold=(velocity_threshold, velocity_threshold),
            minimum_duration=saccade_minimum_duration,
            name="saccade",
        )
    except (ValueError, TypeError):
        # If detection fails (e.g., not enough variance), skip saccade detection
        pass

    return gaze


def detect_events_idt(
    gaze: pm.GazeDataFrame,
    dispersion_threshold: float = 1.0,
    minimum_duration: int = 100,
) -> pm.GazeDataFrame:
    """Detect fixations using I-DT (dispersion-threshold) algorithm.

    The I-DT algorithm identifies fixations as periods where gaze points
    remain within a spatial dispersion threshold for a minimum duration.

    Args:
        gaze: pymovements GazeDataFrame with position in degrees.
        dispersion_threshold: Maximum dispersion in degrees for a fixation.
        minimum_duration: Minimum fixation duration in milliseconds.

    Returns:
        GazeDataFrame with detected events.
    """
    # Ensure position in degrees is computed (use samples for new API, frame for old)
    samples = gaze.samples if hasattr(gaze, "samples") else gaze.frame
    if "position" not in samples.columns:
        gaze.pix2deg()

    # Detect fixations using I-DT
    gaze.detect(
        "idt",
        dispersion_threshold=dispersion_threshold,
        minimum_duration=minimum_duration,
        name="fixation",
    )

    return gaze


def compute_event_properties(
    gaze: pm.GazeDataFrame,
    properties: list[str] | None = None,
) -> pm.GazeDataFrame:
    """Compute properties for detected events.

    Args:
        gaze: pymovements GazeDataFrame with detected events.
        properties: List of properties to compute. If None, computes all
            standard properties: amplitude, dispersion, peak_velocity, duration.

    Returns:
        GazeDataFrame with event properties computed.
    """
    if properties is None:
        properties = ["amplitude", "dispersion", "peak_velocity", "duration"]

    # Filter out properties that already exist to avoid errors
    if gaze.events:
        event_obj = gaze.events[0]
        existing_cols = set()

        # Handle different event object types:
        # - New API: events might be polars DataFrames directly
        # - Old API: events might be Event objects with samples/frame attribute
        if hasattr(event_obj, "columns"):
            # Direct polars DataFrame
            existing_cols = set(event_obj.columns)
        elif hasattr(event_obj, "samples"):
            existing_cols = set(event_obj.samples.columns)
        elif hasattr(event_obj, "frame"):
            existing_cols = set(event_obj.frame.columns)

        properties = [p for p in properties if p not in existing_cols]

    if properties:
        gaze.compute_event_properties(properties)

    return gaze


def events_to_df(gaze: pm.GazeDataFrame) -> pd.DataFrame:
    """Extract detected events as a pandas DataFrame.

    Args:
        gaze: pymovements GazeDataFrame with detected events.

    Returns:
        DataFrame with event information including:
        - name: Event type (fixation, saccade)
        - onset: Start time in ms
        - offset: End time in ms
        - duration: Event duration in ms
        - amplitude: Movement amplitude in degrees (for saccades)
        - dispersion: Spatial dispersion in degrees (for fixations)
        - peak_velocity: Maximum velocity in deg/s
    """
    if not gaze.events:
        return pd.DataFrame()

    # Get events from first (and typically only) trial
    event_obj = gaze.events[0]

    # Handle different event object types:
    # - New API: events might be polars DataFrames directly
    # - Old API: events might be Event objects with samples/frame attribute
    if hasattr(event_obj, "columns") and hasattr(event_obj, "to_pandas"):
        # Direct polars DataFrame
        events_pl = event_obj
    elif hasattr(event_obj, "samples"):
        events_pl = event_obj.samples
    elif hasattr(event_obj, "frame"):
        events_pl = event_obj.frame
    else:
        # Fallback
        events_pl = event_obj

    if hasattr(events_pl, "to_pandas"):
        return events_pl.to_pandas()
    return pd.DataFrame(events_pl)


def get_fixation_stats(events_df: pd.DataFrame) -> dict:
    """Compute fixation statistics from events DataFrame.

    Args:
        events_df: DataFrame from events_to_df with fixation events.

    Returns:
        Dictionary with fixation statistics.
    """
    if len(events_df) == 0 or "name" not in events_df.columns:
        return {
            "count": 0,
            "duration_mean_ms": None,
            "duration_std_ms": None,
            "duration_min_ms": None,
            "duration_max_ms": None,
            "dispersion_mean_deg": None,
            "dispersion_std_deg": None,
        }

    fixations = events_df[events_df["name"].str.contains("fixation", case=False)]

    if len(fixations) == 0:
        return {
            "count": 0,
            "duration_mean_ms": None,
            "duration_std_ms": None,
            "duration_min_ms": None,
            "duration_max_ms": None,
            "dispersion_mean_deg": None,
            "dispersion_std_deg": None,
        }

    stats = {
        "count": len(fixations),
        "duration_mean_ms": fixations["duration"].mean(),
        "duration_std_ms": fixations["duration"].std(),
        "duration_min_ms": fixations["duration"].min(),
        "duration_max_ms": fixations["duration"].max(),
    }

    if "dispersion" in fixations.columns:
        stats["dispersion_mean_deg"] = fixations["dispersion"].mean()
        stats["dispersion_std_deg"] = fixations["dispersion"].std()

    return stats


def get_saccade_stats(events_df: pd.DataFrame) -> dict:
    """Compute saccade statistics from events DataFrame.

    Args:
        events_df: DataFrame from events_to_df with saccade events.

    Returns:
        Dictionary with saccade statistics.
    """
    if len(events_df) == 0 or "name" not in events_df.columns:
        return {
            "count": 0,
            "duration_mean_ms": None,
            "duration_std_ms": None,
            "amplitude_mean_deg": None,
            "amplitude_std_deg": None,
            "peak_velocity_mean_deg_s": None,
            "peak_velocity_std_deg_s": None,
        }

    saccades = events_df[events_df["name"].str.contains("saccade", case=False)]

    if len(saccades) == 0:
        return {
            "count": 0,
            "duration_mean_ms": None,
            "duration_std_ms": None,
            "amplitude_mean_deg": None,
            "amplitude_std_deg": None,
            "peak_velocity_mean_deg_s": None,
            "peak_velocity_std_deg_s": None,
        }

    stats = {
        "count": len(saccades),
        "duration_mean_ms": saccades["duration"].mean(),
        "duration_std_ms": saccades["duration"].std(),
    }

    if "amplitude" in saccades.columns:
        stats["amplitude_mean_deg"] = saccades["amplitude"].mean()
        stats["amplitude_std_deg"] = saccades["amplitude"].std()

    if "peak_velocity" in saccades.columns:
        stats["peak_velocity_mean_deg_s"] = saccades["peak_velocity"].mean()
        stats["peak_velocity_std_deg_s"] = saccades["peak_velocity"].std()

    return stats
