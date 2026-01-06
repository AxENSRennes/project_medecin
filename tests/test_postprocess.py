"""Tests for the postprocess module."""

import numpy as np
import pandas as pd
import pytest

from tobii_pipeline.postprocess import (
    compute_missing_rate,
    detect_gaps,
    detect_gaze_outliers,
    detect_pupil_outliers,
    drop_high_missing_rows,
    filter_physiological_range,
    get_gap_statistics,
    get_interpolatable_columns,
    interpolate_missing,
    mark_gaps,
    postprocess_recording,
    remove_outliers,
    split_at_gaps,
)


class TestInterpolateMissing:
    """Tests for interpolate_missing function."""

    def test_linear_interpolation(self):
        """Linear interpolation fills gaps correctly."""
        df = pd.DataFrame(
            {
                "Gaze point X": [1.0, np.nan, 3.0, np.nan, np.nan, 6.0],
            }
        )
        result = interpolate_missing(df, method="linear")
        expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        np.testing.assert_array_almost_equal(result["Gaze point X"].values, expected)

    def test_max_gap_respected(self):
        """Gaps larger than max_gap remain partially unfilled."""
        df = pd.DataFrame(
            {
                "Gaze point X": [1.0, np.nan, np.nan, np.nan, np.nan, 6.0],
            }
        )
        result = interpolate_missing(df, method="linear", max_gap=2)
        # With max_gap=2, only 2 consecutive NaNs are filled from each end
        # The middle values should remain NaN
        assert pd.isna(result["Gaze point X"].iloc[2]) or pd.isna(result["Gaze point X"].iloc[3])

    def test_ffill_method(self):
        """Forward fill propagates last valid value."""
        df = pd.DataFrame(
            {
                "Gaze point X": [1.0, np.nan, np.nan, 4.0],
            }
        )
        result = interpolate_missing(df, method="ffill")
        expected = [1.0, 1.0, 1.0, 4.0]
        np.testing.assert_array_almost_equal(result["Gaze point X"].values, expected)

    def test_bfill_method(self):
        """Backward fill propagates next valid value."""
        df = pd.DataFrame(
            {
                "Gaze point X": [1.0, np.nan, np.nan, 4.0],
            }
        )
        result = interpolate_missing(df, method="bfill")
        expected = [1.0, 4.0, 4.0, 4.0]
        np.testing.assert_array_almost_equal(result["Gaze point X"].values, expected)

    def test_mean_method(self):
        """Mean fill uses column mean."""
        df = pd.DataFrame(
            {
                "Gaze point X": [2.0, np.nan, 4.0],
            }
        )
        result = interpolate_missing(df, method="mean")
        assert result["Gaze point X"].iloc[1] == 3.0

    def test_drop_method(self):
        """Drop method removes NaN rows."""
        df = pd.DataFrame(
            {
                "Gaze point X": [1.0, np.nan, 3.0],
            }
        )
        result = interpolate_missing(df, method="drop")
        assert len(result) == 2
        assert 1.0 in result["Gaze point X"].values
        assert 3.0 in result["Gaze point X"].values


class TestComputeMissingRate:
    """Tests for compute_missing_rate function."""

    def test_no_missing(self):
        """Returns 0 when no missing values."""
        df = pd.DataFrame({"Gaze point X": [1.0, 2.0, 3.0]})
        result = compute_missing_rate(df, columns=["Gaze point X"])
        assert result["Gaze point X"] == 0.0

    def test_all_missing(self):
        """Returns 1 when all values missing."""
        df = pd.DataFrame({"Gaze point X": [np.nan, np.nan, np.nan]})
        result = compute_missing_rate(df, columns=["Gaze point X"])
        assert result["Gaze point X"] == 1.0

    def test_partial_missing(self):
        """Returns correct rate for partial missing."""
        df = pd.DataFrame({"Gaze point X": [1.0, np.nan, 3.0, np.nan]})
        result = compute_missing_rate(df, columns=["Gaze point X"])
        assert result["Gaze point X"] == 0.5


class TestDetectPupilOutliers:
    """Tests for pupil outlier detection."""

    def test_detects_below_minimum(self):
        """Detects pupil values below physiological minimum."""
        df = pd.DataFrame(
            {
                "Pupil diameter left": [3.0, 1.5, 3.5],
                "Pupil diameter right": [3.0, 3.0, 3.0],
            }
        )
        outliers = detect_pupil_outliers(df)
        assert outliers.iloc[1]
        assert not outliers.iloc[0]

    def test_detects_above_maximum(self):
        """Detects pupil values above physiological maximum."""
        df = pd.DataFrame(
            {
                "Pupil diameter left": [3.0, 9.0, 3.5],
                "Pupil diameter right": [3.0, 3.0, 3.0],
            }
        )
        outliers = detect_pupil_outliers(df)
        assert outliers.iloc[1]
        assert not outliers.iloc[0]

    def test_custom_range(self):
        """Respects custom min/max parameters."""
        df = pd.DataFrame(
            {
                "Pupil diameter left": [3.0, 3.5, 4.0],
                "Pupil diameter right": [3.5, 3.5, 3.5],  # All in range
            }
        )
        outliers = detect_pupil_outliers(df, min_diameter=3.2, max_diameter=3.8)
        assert outliers.iloc[0]  # left 3.0 < 3.2
        assert not outliers.iloc[1]  # both in range
        assert outliers.iloc[2]  # left 4.0 > 3.8

    def test_nan_not_flagged_as_outlier(self):
        """NaN values are not flagged as outliers."""
        df = pd.DataFrame(
            {
                "Pupil diameter left": [3.0, np.nan, 3.5],
                "Pupil diameter right": [3.0, 3.0, 3.0],
            }
        )
        outliers = detect_pupil_outliers(df)
        assert not outliers.iloc[1]


class TestDetectGazeOutliers:
    """Tests for gaze outlier detection."""

    def test_detects_offscreen_gaze(self):
        """Detects gaze coordinates outside screen bounds."""
        df = pd.DataFrame(
            {
                "Gaze point X": [960, -50, 2000],
                "Gaze point Y": [540, 540, 540],
            }
        )
        outliers = detect_gaze_outliers(df, screen_width=1920, screen_height=1080, margin=0)
        assert not outliers.iloc[0]
        assert outliers.iloc[1]
        assert outliers.iloc[2]

    def test_margin_allows_overflow(self):
        """Margin parameter allows some overflow."""
        df = pd.DataFrame(
            {
                "Gaze point X": [1950, 2100],
                "Gaze point Y": [540, 540],
            }
        )
        # 10% margin on 1920 = 192 overflow allowed
        outliers = detect_gaze_outliers(df, screen_width=1920, screen_height=1080, margin=0.1)
        assert not outliers.iloc[0]  # 1950 < 1920 * 1.1 = 2112
        assert not outliers.iloc[1]  # 2100 < 2112


class TestDetectGaps:
    """Tests for gap detection."""

    def test_detects_timestamp_gaps(self):
        """Detects gaps in timestamp sequence."""
        # Timestamps in microseconds, 100Hz = 10000us interval
        df = pd.DataFrame(
            {
                "Recording timestamp": [0, 10000, 20000, 200000, 210000],
            }
        )
        gaps = detect_gaps(df, threshold_ms=100)
        assert len(gaps) == 1
        assert gaps[0].duration_ms == pytest.approx(180, rel=0.01)

    def test_no_gaps(self):
        """Returns empty list when no gaps."""
        df = pd.DataFrame(
            {
                "Recording timestamp": [0, 10000, 20000, 30000],
            }
        )
        gaps = detect_gaps(df, threshold_ms=100)
        assert len(gaps) == 0


class TestGapStatistics:
    """Tests for gap statistics."""

    def test_empty_gaps(self):
        """Returns zeros for empty gap list."""
        stats = get_gap_statistics([])
        assert stats["count"] == 0
        assert stats["total_duration_ms"] == 0.0


class TestRemoveOutliers:
    """Tests for remove_outliers function."""

    def test_remove_method(self):
        """Remove method deletes outlier rows."""
        df = pd.DataFrame(
            {
                "Gaze point X": [100, 200, 300],
                "other": [1, 2, 3],
            }
        )
        mask = pd.Series([False, True, False])
        result = remove_outliers(df, mask, method="remove")
        assert len(result) == 2

    def test_nan_method(self):
        """NaN method replaces values with NaN."""
        df = pd.DataFrame(
            {
                "Gaze point X": [100.0, 200.0, 300.0],
            }
        )
        mask = pd.Series([False, True, False])
        result = remove_outliers(df, mask, method="nan", columns=["Gaze point X"])
        assert pd.isna(result["Gaze point X"].iloc[1])
        assert result["Gaze point X"].iloc[0] == 100.0


class TestFilterPhysiologicalRange:
    """Tests for filter_physiological_range function."""

    def test_filters_pupil_outliers(self):
        """Filters pupil diameter outliers."""
        df = pd.DataFrame(
            {
                "Pupil diameter left": [3.0, 1.0, 3.5],
                "Pupil diameter right": [3.0, 3.0, 3.0],
            }
        )
        result = filter_physiological_range(df, remove_pupil_outliers=True, method="nan")
        assert pd.isna(result["Pupil diameter left"].iloc[1])
        assert result["Pupil diameter left"].iloc[0] == 3.0


class TestPostprocessRecording:
    """Tests for postprocess_recording function."""

    def test_returns_dataframe_and_report(self):
        """Returns tuple of DataFrame and report dict."""
        df = pd.DataFrame(
            {
                "Recording timestamp": [0, 10000, 20000, 30000],
                "Gaze point X": [100.0, np.nan, 300.0, 400.0],
                "Gaze point Y": [100.0, 200.0, 300.0, 400.0],
                "Pupil diameter left": [3.0, 3.5, 3.0, 3.5],
                "Pupil diameter right": [3.0, 3.5, 3.0, 3.5],
                "Validity left": ["Valid", "Valid", "Valid", "Valid"],
                "Validity right": ["Valid", "Valid", "Valid", "Valid"],
            }
        )
        # Disable blink interpolation and event detection for basic test
        result_df, report = postprocess_recording(
            df,
            interpolate_blinks=False,
            detect_eye_events=False,
        )

        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(report, dict)
        assert "missing_before" in report
        assert "missing_after" in report

    def test_interpolates_when_enabled(self):
        """Interpolates missing values when enabled."""
        df = pd.DataFrame(
            {
                "Recording timestamp": [0, 10000, 20000],
                "Gaze point X": [100.0, np.nan, 300.0],
                "Validity left": ["Valid", "Valid", "Valid"],
                "Validity right": ["Valid", "Valid", "Valid"],
            }
        )
        result_df, _ = postprocess_recording(
            df,
            interpolate=True,
            remove_physiological_outliers=False,
            interpolate_blinks=False,
            detect_eye_events=False,
        )
        # After interpolation, the NaN should be filled
        assert result_df["Gaze point X"].iloc[1] == 200.0


class TestDropHighMissingRows:
    """Tests for drop_high_missing_rows function."""

    def test_drops_rows_above_threshold(self):
        """Removes rows where missing rate exceeds threshold."""
        df = pd.DataFrame(
            {
                "col1": [1.0, np.nan, 3.0],
                "col2": [1.0, np.nan, 3.0],
                "col3": [1.0, 2.0, 3.0],
            }
        )
        result = drop_high_missing_rows(df, columns=["col1", "col2", "col3"], threshold=0.5)
        assert len(result) == 2  # Middle row has 66% missing


class TestGetInterpolatableColumns:
    """Tests for get_interpolatable_columns function."""

    def test_returns_expected_columns(self):
        """Returns list of continuous numeric columns."""
        cols = get_interpolatable_columns()
        assert "Gaze point X" in cols
        assert "Gaze point Y" in cols
        assert "Pupil diameter left" in cols
        assert "Pupil diameter right" in cols


class TestMarkGaps:
    """Tests for mark_gaps function."""

    def test_adds_gap_column(self):
        """Adds after_gap column to DataFrame."""
        df = pd.DataFrame(
            {
                "Recording timestamp": [0, 10000, 20000],
            }
        )
        result = mark_gaps(df, gaps=[])
        assert "after_gap" in result.columns


class TestSplitAtGaps:
    """Tests for split_at_gaps function."""

    def test_no_gaps_returns_single_segment(self):
        """Returns single segment when no gaps."""
        df = pd.DataFrame(
            {
                "Recording timestamp": [0, 10000, 20000, 30000],
            }
        )
        segments = split_at_gaps(df, min_gap_ms=1000)
        assert len(segments) == 1

    def test_splits_at_large_gap(self):
        """Splits into multiple segments at large gaps."""
        df = pd.DataFrame(
            {
                "Recording timestamp": [0, 10000, 20000, 2000000, 2010000],
            }
        )
        segments = split_at_gaps(df, min_gap_ms=1000)
        assert len(segments) == 2
