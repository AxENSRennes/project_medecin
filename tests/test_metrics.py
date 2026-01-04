"""Tests for the analysis.metrics module."""

import numpy as np
import pandas as pd
import pytest

from tobii_pipeline.analysis.metrics import (
    compute_fixation_count,
    compute_fixation_durations,
    compute_fixation_stats,
    compute_gaze_center,
    compute_gaze_dispersion,
    compute_gaze_quadrant_distribution,
    compute_pupil_over_time,
    compute_pupil_stats,
    compute_pupil_variability,
    compute_recording_summary,
    compute_saccade_count,
    compute_saccade_stats,
    compute_tracking_ratio,
    compute_validity_rate,
)


class TestValidityRate:
    """Tests for compute_validity_rate."""

    def test_both_eyes_required(self):
        """Counts only samples with both eyes valid."""
        df = pd.DataFrame(
            {
                "Validity left": ["Valid", "Valid", "Invalid", "Invalid"],
                "Validity right": ["Valid", "Invalid", "Valid", "Invalid"],
            }
        )
        rate = compute_validity_rate(df, both_eyes=True)
        assert rate == 0.25  # 1/4

    def test_one_eye_sufficient(self):
        """Counts samples with at least one eye valid."""
        df = pd.DataFrame(
            {
                "Validity left": ["Valid", "Valid", "Invalid", "Invalid"],
                "Validity right": ["Valid", "Invalid", "Valid", "Invalid"],
            }
        )
        rate = compute_validity_rate(df, both_eyes=False)
        assert rate == 0.75  # 3/4

    def test_empty_dataframe(self):
        """Returns 0 for empty DataFrame."""
        df = pd.DataFrame({"Validity left": [], "Validity right": []})
        rate = compute_validity_rate(df)
        assert rate == 0.0

    def test_missing_columns(self):
        """Returns 0 when validity columns missing."""
        df = pd.DataFrame({"other": [1, 2, 3]})
        rate = compute_validity_rate(df)
        assert rate == 0.0


class TestTrackingRatio:
    """Tests for compute_tracking_ratio."""

    def test_computes_ratio(self):
        """Computes correct tracking ratio."""
        df = pd.DataFrame(
            {
                "Sensor": ["Eye Tracker", "Eye Tracker", "Accelerometer", "Gyroscope"],
            }
        )
        ratio = compute_tracking_ratio(df)
        assert ratio == 0.5

    def test_missing_sensor_column(self):
        """Returns 0 when Sensor column missing."""
        df = pd.DataFrame({"other": [1, 2, 3]})
        ratio = compute_tracking_ratio(df)
        assert ratio == 0.0


class TestGazeCenter:
    """Tests for compute_gaze_center."""

    def test_computes_mean_position(self):
        """Returns mean X and Y coordinates."""
        df = pd.DataFrame(
            {
                "Gaze point X": [100, 200, 300],
                "Gaze point Y": [50, 100, 150],
            }
        )
        center = compute_gaze_center(df)
        assert center == (200.0, 100.0)

    def test_handles_missing_columns(self):
        """Returns NaN for missing columns."""
        df = pd.DataFrame({"other": [1, 2, 3]})
        center = compute_gaze_center(df)
        assert np.isnan(center[0])
        assert np.isnan(center[1])


class TestGazeDispersion:
    """Tests for compute_gaze_dispersion."""

    def test_computes_dispersion(self):
        """Returns Euclidean combination of std."""
        df = pd.DataFrame(
            {
                "Gaze point X": [0, 100, 200],
                "Gaze point Y": [0, 0, 0],
            }
        )
        dispersion = compute_gaze_dispersion(df)
        assert dispersion > 0

    def test_zero_dispersion(self):
        """Returns 0 when all points same."""
        df = pd.DataFrame(
            {
                "Gaze point X": [100, 100, 100],
                "Gaze point Y": [100, 100, 100],
            }
        )
        dispersion = compute_gaze_dispersion(df)
        assert dispersion == 0.0


class TestGazeQuadrantDistribution:
    """Tests for compute_gaze_quadrant_distribution."""

    def test_computes_distribution(self):
        """Returns correct quadrant percentages."""
        df = pd.DataFrame(
            {
                "Gaze point X": [100, 1000, 100, 1000],  # left, right, left, right
                "Gaze point Y": [100, 100, 600, 600],  # top, top, bottom, bottom
            }
        )
        dist = compute_gaze_quadrant_distribution(df, screen_width=1920, screen_height=1080)
        assert dist["top_left"] == 0.25
        assert dist["top_right"] == 0.25
        assert dist["bottom_left"] == 0.25
        assert dist["bottom_right"] == 0.25


class TestPupilStats:
    """Tests for compute_pupil_stats."""

    def test_computes_stats(self):
        """Computes correct pupil statistics."""
        df = pd.DataFrame(
            {
                "Pupil diameter left": [3.0, 4.0, 5.0],
                "Pupil diameter right": [3.5, 4.5, 5.5],
            }
        )
        stats = compute_pupil_stats(df)
        assert stats["left_mean"] == 4.0
        assert stats["right_mean"] == 4.5
        assert stats["mean"] == pytest.approx(4.25)

    def test_handles_nan(self):
        """Handles NaN values."""
        df = pd.DataFrame(
            {
                "Pupil diameter left": [3.0, np.nan, 5.0],
                "Pupil diameter right": [3.5, 4.5, 5.5],
            }
        )
        stats = compute_pupil_stats(df)
        assert stats["left_mean"] == 4.0  # Mean of 3 and 5


class TestPupilVariability:
    """Tests for compute_pupil_variability."""

    def test_computes_cv(self):
        """Computes coefficient of variation."""
        df = pd.DataFrame(
            {
                "Pupil diameter left": [4.0, 4.0, 4.0],
                "Pupil diameter right": [4.0, 4.0, 4.0],
            }
        )
        cv = compute_pupil_variability(df)
        assert cv == 0.0  # No variation

    def test_positive_cv(self):
        """Returns positive CV for varying data."""
        df = pd.DataFrame(
            {
                "Pupil diameter left": [2.0, 4.0, 6.0],
            }
        )
        cv = compute_pupil_variability(df)
        assert cv > 0


class TestPupilOverTime:
    """Tests for compute_pupil_over_time."""

    def test_returns_binned_data(self):
        """Returns DataFrame with time bins."""
        df = pd.DataFrame(
            {
                "Recording timestamp": list(range(0, 100000, 10000)),
                "Pupil diameter left": [3.0, 3.5, 4.0, 4.5, 5.0, 4.5, 4.0, 3.5, 3.0, 3.5],
            }
        )
        result = compute_pupil_over_time(df, n_bins=5)
        assert len(result) <= 5
        assert "mean_pupil" in result.columns


class TestFixationCount:
    """Tests for compute_fixation_count."""

    def test_counts_unique_fixations(self):
        """Counts each fixation once regardless of samples."""
        df = pd.DataFrame(
            {
                "Eye movement type": ["Fixation", "Fixation", "Saccade", "Fixation", "Fixation"],
                "Eye movement type index": [1, 1, 2, 3, 3],
            }
        )
        count = compute_fixation_count(df)
        assert count == 2  # Indices 1 and 3

    def test_no_fixations(self):
        """Returns 0 when no fixations."""
        df = pd.DataFrame(
            {
                "Eye movement type": ["Saccade", "Saccade"],
                "Eye movement type index": [1, 2],
            }
        )
        count = compute_fixation_count(df)
        assert count == 0


class TestFixationDurations:
    """Tests for compute_fixation_durations."""

    def test_returns_durations(self):
        """Returns duration for each fixation."""
        df = pd.DataFrame(
            {
                "Eye movement type": ["Fixation", "Fixation", "Fixation"],
                "Eye movement type index": [1, 1, 2],
                "Gaze event duration": [200, 200, 300],
            }
        )
        durations = compute_fixation_durations(df)
        assert len(durations) == 2
        assert 200 in durations.values
        assert 300 in durations.values


class TestFixationStats:
    """Tests for compute_fixation_stats."""

    def test_computes_stats(self):
        """Computes comprehensive statistics."""
        df = pd.DataFrame(
            {
                "Eye movement type": ["Fixation", "Fixation", "Fixation"],
                "Eye movement type index": [1, 2, 3],
                "Gaze event duration": [100, 200, 300],
                "Recording timestamp": [0, 100000, 200000],
            }
        )
        stats = compute_fixation_stats(df)
        assert stats["count"] == 3
        assert stats["mean_duration"] == 200.0
        assert stats["min_duration"] == 100.0
        assert stats["max_duration"] == 300.0


class TestSaccadeCount:
    """Tests for compute_saccade_count."""

    def test_counts_unique_saccades(self):
        """Counts each saccade once."""
        df = pd.DataFrame(
            {
                "Eye movement type": ["Saccade", "Saccade", "Fixation", "Saccade"],
                "Eye movement type index": [1, 1, 2, 3],
            }
        )
        count = compute_saccade_count(df)
        assert count == 2  # Indices 1 and 3


class TestSaccadeStats:
    """Tests for compute_saccade_stats."""

    def test_computes_stats(self):
        """Computes comprehensive statistics."""
        df = pd.DataFrame(
            {
                "Eye movement type": ["Saccade", "Saccade", "Saccade"],
                "Eye movement type index": [1, 2, 3],
                "Gaze event duration": [30, 40, 50],
                "Recording timestamp": [0, 100000, 200000],
            }
        )
        stats = compute_saccade_stats(df)
        assert stats["count"] == 3
        assert stats["mean_duration"] == 40.0


class TestRecordingSummary:
    """Tests for compute_recording_summary."""

    def test_returns_all_categories(self):
        """Returns dict with all metric categories."""
        df = pd.DataFrame(
            {
                "Validity left": ["Valid", "Valid"],
                "Validity right": ["Valid", "Valid"],
                "Sensor": ["Eye Tracker", "Eye Tracker"],
                "Gaze point X": [100.0, 200.0],
                "Gaze point Y": [100.0, 200.0],
                "Pupil diameter left": [3.0, 4.0],
                "Pupil diameter right": [3.0, 4.0],
                "Eye movement type": ["Fixation", "Fixation"],
                "Eye movement type index": [1, 1],
                "Gaze event duration": [200, 200],
                "Recording timestamp": [0, 10000],
            }
        )
        summary = compute_recording_summary(df)
        assert "quality" in summary
        assert "gaze" in summary
        assert "pupil" in summary
        assert "fixation" in summary
        assert "saccade" in summary
