"""Tests for the cleaner module."""

import pandas as pd
import pytest

from tobii_pipeline.cleaner import (
    clean_recording,
    filter_by_sensor,
    filter_eye_tracker,
    filter_valid_gaze,
    fix_decimal_separator,
    get_gaze_columns,
    get_motion_columns,
    select_gaze_data,
    select_motion_data,
)


class TestFixDecimalSeparator:
    """Tests for fix_decimal_separator function."""

    def test_fix_decimal_separator(self):
        """Convert European decimal format to standard format."""
        df = pd.DataFrame(
            {
                "Gaze point X": ["1,5", "2,75", "3,0"],
                "Gaze point Y": ["4,25", "5,5", "6,125"],
            }
        )

        result = fix_decimal_separator(df, columns=["Gaze point X", "Gaze point Y"])

        assert result["Gaze point X"].tolist() == [1.5, 2.75, 3.0]
        assert result["Gaze point Y"].tolist() == [4.25, 5.5, 6.125]

    def test_fix_decimal_separator_already_numeric(self):
        """No change for already numeric columns."""
        df = pd.DataFrame(
            {
                "Gaze point X": [1.5, 2.75, 3.0],
                "Other": ["a", "b", "c"],
            }
        )

        result = fix_decimal_separator(df, columns=["Gaze point X"])

        assert result["Gaze point X"].tolist() == [1.5, 2.75, 3.0]

    def test_fix_decimal_separator_missing_column(self):
        """Gracefully handle missing columns."""
        df = pd.DataFrame({"Other": ["1,5", "2,5"]})

        result = fix_decimal_separator(df, columns=["Nonexistent"])

        assert "Other" in result.columns

    def test_fix_decimal_separator_mixed_values(self):
        """Handle mixed valid and invalid values."""
        df = pd.DataFrame(
            {
                "Gaze point X": ["1,5", "invalid", "3,0"],
            }
        )

        result = fix_decimal_separator(df, columns=["Gaze point X"])

        assert result["Gaze point X"].iloc[0] == 1.5
        assert pd.isna(result["Gaze point X"].iloc[1])
        assert result["Gaze point X"].iloc[2] == 3.0


class TestCleanRecording:
    """Tests for clean_recording function."""

    def test_clean_recording(self):
        """Full cleaning pipeline."""
        df = pd.DataFrame(
            {
                "Gaze point X": ["1,5", "2,5"],
                "Validity left": ["Valid", "Invalid"],
                "Validity right": ["Valid", "Valid"],
                "Sensor": ["Eye Tracker", "Eye Tracker"],
                "Eye movement type": ["Fixation", "Saccade"],
            }
        )

        result = clean_recording(df)

        assert result["Gaze point X"].dtype in [float, "float64"]
        assert result["Validity left"].dtype.name == "category"
        assert result["Sensor"].dtype.name == "category"

    def test_clean_recording_no_decimal_fix(self):
        """Cleaning without decimal fix."""
        df = pd.DataFrame(
            {
                "Gaze point X": ["1,5", "2,5"],
                "Sensor": ["Eye Tracker", "Accelerometer"],
            }
        )

        result = clean_recording(df, fix_decimals=False)

        assert result["Gaze point X"].dtype == object
        assert result["Sensor"].dtype.name == "category"


class TestFilterBySensor:
    """Tests for sensor filtering functions."""

    def test_filter_by_sensor(self):
        """Filter by specific sensor type."""
        df = pd.DataFrame(
            {
                "Sensor": ["Eye Tracker", "Accelerometer", "Eye Tracker", "Gyroscope"],
                "value": [1, 2, 3, 4],
            }
        )

        result = filter_by_sensor(df, "Eye Tracker")

        assert len(result) == 2
        assert result["value"].tolist() == [1, 3]

    def test_filter_by_sensor_missing_column(self):
        """Raise ValueError when Sensor column is missing."""
        df = pd.DataFrame({"value": [1, 2, 3]})

        with pytest.raises(ValueError, match="must contain 'Sensor' column"):
            filter_by_sensor(df, "Eye Tracker")

    def test_filter_eye_tracker(self):
        """Filter to eye tracker data only."""
        df = pd.DataFrame(
            {
                "Sensor": ["Eye Tracker", "Accelerometer", "Eye Tracker"],
                "value": [1, 2, 3],
            }
        )

        result = filter_eye_tracker(df)

        assert len(result) == 2
        assert all(result["Sensor"] == "Eye Tracker")


class TestFilterValidGaze:
    """Tests for filter_valid_gaze function."""

    def test_filter_valid_gaze_both_eyes(self):
        """Filter requiring both eyes valid."""
        df = pd.DataFrame(
            {
                "Validity left": ["Valid", "Valid", "Invalid", "Invalid"],
                "Validity right": ["Valid", "Invalid", "Valid", "Invalid"],
                "value": [1, 2, 3, 4],
            }
        )

        result = filter_valid_gaze(df, both_eyes=True)

        assert len(result) == 1
        assert result["value"].iloc[0] == 1

    def test_filter_valid_gaze_one_eye(self):
        """Filter requiring at least one eye valid."""
        df = pd.DataFrame(
            {
                "Validity left": ["Valid", "Valid", "Invalid", "Invalid"],
                "Validity right": ["Valid", "Invalid", "Valid", "Invalid"],
                "value": [1, 2, 3, 4],
            }
        )

        result = filter_valid_gaze(df, both_eyes=False)

        assert len(result) == 3
        assert result["value"].tolist() == [1, 2, 3]

    def test_filter_valid_gaze_missing_columns(self):
        """Raise ValueError when validity columns are missing."""
        df = pd.DataFrame({"value": [1, 2, 3]})

        with pytest.raises(ValueError, match="must contain 'Validity left'"):
            filter_valid_gaze(df)


class TestColumnSelectors:
    """Tests for column selection functions."""

    def test_get_gaze_columns(self):
        """Returns expected gaze column list."""
        cols = get_gaze_columns()

        assert "Recording timestamp" in cols
        assert "Gaze point X" in cols
        assert "Gaze point Y" in cols
        assert "Pupil diameter left" in cols
        assert "Validity left" in cols

    def test_get_motion_columns(self):
        """Returns expected motion column list."""
        cols = get_motion_columns()

        assert "Recording timestamp" in cols
        assert "Gyro X" in cols
        assert "Accelerometer X" in cols

    def test_select_gaze_data(self):
        """Select only gaze columns that exist."""
        df = pd.DataFrame(
            {
                "Recording timestamp": [1, 2, 3],
                "Gaze point X": [0.5, 0.6, 0.7],
                "Unrelated column": ["a", "b", "c"],
            }
        )

        result = select_gaze_data(df)

        assert "Recording timestamp" in result.columns
        assert "Gaze point X" in result.columns
        assert "Unrelated column" not in result.columns

    def test_select_motion_data(self):
        """Select only motion columns that exist."""
        df = pd.DataFrame(
            {
                "Recording timestamp": [1, 2, 3],
                "Gyro X": [0.1, 0.2, 0.3],
                "Accelerometer X": [0.01, 0.02, 0.03],
                "Gaze point X": [0.5, 0.6, 0.7],
            }
        )

        result = select_motion_data(df)

        assert "Recording timestamp" in result.columns
        assert "Gyro X" in result.columns
        assert "Accelerometer X" in result.columns
        assert "Gaze point X" not in result.columns
