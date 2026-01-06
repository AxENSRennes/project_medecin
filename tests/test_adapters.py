"""Tests for the adapters module (pymovements and MNE integration)."""

import numpy as np
import pandas as pd
import pytest


class TestPymovementsAdapter:
    """Tests for pymovements adapter functions."""

    @pytest.fixture
    def sample_gaze_df(self):
        """Create a sample DataFrame with gaze data for testing."""
        n_samples = 100
        timestamps = np.arange(0, n_samples * 10000, 10000)  # 100 samples at 100Hz (microseconds)

        # Create smooth gaze movement pattern
        x_base = np.linspace(400, 1500, n_samples)
        y_base = np.linspace(300, 800, n_samples)

        # Add some noise
        np.random.seed(42)
        x = x_base + np.random.normal(0, 5, n_samples)
        y = y_base + np.random.normal(0, 5, n_samples)

        return pd.DataFrame(
            {
                "Recording timestamp": timestamps,
                "Gaze point X": x,
                "Gaze point Y": y,
                "Pupil diameter left": np.random.uniform(3.0, 5.0, n_samples),
                "Pupil diameter right": np.random.uniform(3.0, 5.0, n_samples),
            }
        )

    def test_df_to_gaze_dataframe_conversion(self, sample_gaze_df):
        """Test conversion from Tobii DataFrame to pymovements GazeDataFrame."""
        from tobii_pipeline.adapters.pymovements_adapter import df_to_gaze_dataframe

        gaze = df_to_gaze_dataframe(sample_gaze_df)

        # Check that Gaze object was created (use samples or frame for compatibility)
        assert gaze is not None
        samples = gaze.samples if hasattr(gaze, "samples") else gaze.frame
        assert len(samples) == len(sample_gaze_df)

    def test_df_to_gaze_dataframe_columns(self, sample_gaze_df):
        """Test that required columns are present after conversion."""
        from tobii_pipeline.adapters.pymovements_adapter import df_to_gaze_dataframe

        gaze = df_to_gaze_dataframe(sample_gaze_df)
        samples = gaze.samples if hasattr(gaze, "samples") else gaze.frame
        columns = samples.columns

        assert "time" in columns
        assert "x" in columns or "pixel" in str(columns)

    def test_apply_pix2deg(self, sample_gaze_df):
        """Test pixel to degree conversion."""
        from tobii_pipeline.adapters.pymovements_adapter import (
            apply_pix2deg,
            df_to_gaze_dataframe,
        )

        gaze = df_to_gaze_dataframe(sample_gaze_df)
        gaze = apply_pix2deg(gaze)

        # Check that position column was added
        samples = gaze.samples if hasattr(gaze, "samples") else gaze.frame
        columns = samples.columns
        assert "position" in columns or any("deg" in str(c).lower() for c in columns)

    def test_apply_pos2vel(self, sample_gaze_df):
        """Test velocity computation from position."""
        from tobii_pipeline.adapters.pymovements_adapter import (
            apply_pix2deg,
            apply_pos2vel,
            df_to_gaze_dataframe,
        )

        gaze = df_to_gaze_dataframe(sample_gaze_df)
        gaze = apply_pix2deg(gaze)
        gaze = apply_pos2vel(gaze)

        # Check that velocity column was added
        samples = gaze.samples if hasattr(gaze, "samples") else gaze.frame
        columns = samples.columns
        assert "velocity" in columns

    def test_detect_events_ivt(self, sample_gaze_df):
        """Test I-VT event detection algorithm."""
        from tobii_pipeline.adapters.pymovements_adapter import (
            apply_pix2deg,
            apply_pos2vel,
            detect_events_ivt,
            df_to_gaze_dataframe,
        )

        gaze = df_to_gaze_dataframe(sample_gaze_df)
        gaze = apply_pix2deg(gaze)
        gaze = apply_pos2vel(gaze)
        gaze = detect_events_ivt(gaze, velocity_threshold=30.0, minimum_duration=100)

        # Check that events were detected
        assert hasattr(gaze, "events")

    def test_events_to_df_returns_dataframe(self, sample_gaze_df):
        """Test that events_to_df returns a pandas DataFrame."""
        from tobii_pipeline.adapters.pymovements_adapter import (
            apply_pix2deg,
            apply_pos2vel,
            detect_events_ivt,
            df_to_gaze_dataframe,
            events_to_df,
        )

        gaze = df_to_gaze_dataframe(sample_gaze_df)
        gaze = apply_pix2deg(gaze)
        gaze = apply_pos2vel(gaze)
        gaze = detect_events_ivt(gaze)

        events_df = events_to_df(gaze)

        assert isinstance(events_df, pd.DataFrame)

    def test_get_fixation_stats_empty(self):
        """Test fixation stats with empty DataFrame."""
        from tobii_pipeline.adapters.pymovements_adapter import get_fixation_stats

        empty_df = pd.DataFrame({"name": [], "duration": []})
        stats = get_fixation_stats(empty_df)

        assert stats["count"] == 0
        assert stats["duration_mean_ms"] is None

    def test_get_fixation_stats_with_data(self):
        """Test fixation stats with fixation data."""
        from tobii_pipeline.adapters.pymovements_adapter import get_fixation_stats

        events_df = pd.DataFrame(
            {
                "name": ["fixation", "fixation", "saccade", "fixation"],
                "duration": [200, 300, 50, 150],
                "dispersion": [0.5, 0.7, None, 0.4],
            }
        )

        stats = get_fixation_stats(events_df)

        assert stats["count"] == 3
        assert stats["duration_mean_ms"] == pytest.approx(216.67, rel=0.01)
        assert stats["duration_min_ms"] == 150
        assert stats["duration_max_ms"] == 300

    def test_get_saccade_stats_empty(self):
        """Test saccade stats with empty DataFrame."""
        from tobii_pipeline.adapters.pymovements_adapter import get_saccade_stats

        empty_df = pd.DataFrame({"name": [], "duration": []})
        stats = get_saccade_stats(empty_df)

        assert stats["count"] == 0
        assert stats["duration_mean_ms"] is None

    def test_get_saccade_stats_with_data(self):
        """Test saccade stats with saccade data."""
        from tobii_pipeline.adapters.pymovements_adapter import get_saccade_stats

        events_df = pd.DataFrame(
            {
                "name": ["saccade", "saccade", "fixation", "saccade"],
                "duration": [30, 50, 200, 40],
                "amplitude": [2.5, 3.0, None, 1.5],
                "peak_velocity": [200, 300, None, 150],
            }
        )

        stats = get_saccade_stats(events_df)

        assert stats["count"] == 3
        assert stats["duration_mean_ms"] == 40.0
        assert stats["amplitude_mean_deg"] == pytest.approx(2.33, rel=0.01)


class TestMNEAdapter:
    """Tests for MNE adapter functions."""

    @pytest.fixture
    def sample_gaze_df(self):
        """Create a sample DataFrame with gaze and validity data."""
        n_samples = 100
        timestamps = list(range(0, n_samples * 10000, 10000))

        return pd.DataFrame(
            {
                "Recording timestamp": timestamps,
                "Gaze point X": np.random.uniform(400, 1500, n_samples),
                "Gaze point Y": np.random.uniform(300, 800, n_samples),
                "Pupil diameter left": np.random.uniform(3.0, 5.0, n_samples),
                "Pupil diameter right": np.random.uniform(3.0, 5.0, n_samples),
                "Validity left": ["Valid"] * n_samples,
                "Validity right": ["Valid"] * n_samples,
            }
        )

    @pytest.fixture
    def df_with_blinks(self):
        """Create a DataFrame with simulated blink periods."""
        n_samples = 100
        timestamps = list(range(0, n_samples * 10000, 10000))

        # Create validity pattern with one blink (20 samples = 200ms)
        validity = ["Valid"] * 30 + ["Invalid"] * 20 + ["Valid"] * 50

        return pd.DataFrame(
            {
                "Recording timestamp": timestamps,
                "Gaze point X": np.random.uniform(400, 1500, n_samples),
                "Gaze point Y": np.random.uniform(300, 800, n_samples),
                "Pupil diameter left": np.random.uniform(3.0, 5.0, n_samples),
                "Pupil diameter right": np.random.uniform(3.0, 5.0, n_samples),
                "Validity left": validity,
                "Validity right": validity,
            }
        )

    def test_df_to_mne_raw(self, sample_gaze_df):
        """Test conversion from Tobii DataFrame to MNE Raw object."""
        from tobii_pipeline.adapters.mne_adapter import df_to_mne_raw

        raw = df_to_mne_raw(sample_gaze_df)

        # Check that Raw object was created
        assert raw is not None
        assert hasattr(raw, "get_data")
        assert len(raw.ch_names) >= 2  # At least gaze_x and gaze_y

    def test_df_to_mne_raw_channels(self, sample_gaze_df):
        """Test that expected channels are created."""
        from tobii_pipeline.adapters.mne_adapter import df_to_mne_raw

        raw = df_to_mne_raw(sample_gaze_df)

        ch_names = raw.ch_names
        assert "gaze_x" in ch_names
        assert "gaze_y" in ch_names
        assert "pupil_left" in ch_names
        assert "pupil_right" in ch_names

    def test_create_blink_annotations(self, df_with_blinks):
        """Test blink annotation creation."""
        from tobii_pipeline.adapters.mne_adapter import create_blink_annotations

        annotations = create_blink_annotations(df_with_blinks)

        # Should detect exactly one blink
        assert len(annotations) == 1
        assert annotations.description[0] == "blink"

    def test_create_blink_annotations_no_blinks(self, sample_gaze_df):
        """Test blink annotation with no blinks."""
        from tobii_pipeline.adapters.mne_adapter import create_blink_annotations

        annotations = create_blink_annotations(sample_gaze_df)

        # Should detect no blinks
        assert len(annotations) == 0

    def test_interpolate_blinks_mne(self, df_with_blinks):
        """Test blink interpolation using MNE."""
        from tobii_pipeline.adapters.mne_adapter import interpolate_blinks_mne

        result = interpolate_blinks_mne(df_with_blinks)

        # Should return a DataFrame with same length
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df_with_blinks)

    def test_interpolate_blinks_mne_preserves_columns(self, df_with_blinks):
        """Test that blink interpolation preserves columns."""
        from tobii_pipeline.adapters.mne_adapter import interpolate_blinks_mne

        result = interpolate_blinks_mne(df_with_blinks)

        # All original columns should be preserved
        for col in df_with_blinks.columns:
            assert col in result.columns

    def test_get_blink_statistics_mne(self, df_with_blinks):
        """Test blink statistics computation."""
        from tobii_pipeline.adapters.mne_adapter import get_blink_statistics_mne

        stats = get_blink_statistics_mne(df_with_blinks)

        assert isinstance(stats, dict)
        assert "count" in stats
        assert "mean_duration_ms" in stats
        assert stats["count"] == 1

    def test_get_blink_statistics_mne_no_blinks(self, sample_gaze_df):
        """Test blink statistics with no blinks."""
        from tobii_pipeline.adapters.mne_adapter import get_blink_statistics_mne

        stats = get_blink_statistics_mne(sample_gaze_df)

        assert stats["count"] == 0
        assert stats["mean_duration_ms"] is None

    def test_plot_gaze_heatmap(self, sample_gaze_df):
        """Test gaze heatmap plotting."""
        import matplotlib.pyplot as plt

        from tobii_pipeline.adapters.mne_adapter import plot_gaze_heatmap

        fig, ax = plot_gaze_heatmap(sample_gaze_df)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_gaze_heatmap_empty_data(self):
        """Test heatmap with no valid gaze data."""
        import matplotlib.pyplot as plt

        from tobii_pipeline.adapters.mne_adapter import plot_gaze_heatmap

        empty_df = pd.DataFrame(
            {
                "Gaze point X": [np.nan] * 10,
                "Gaze point Y": [np.nan] * 10,
            }
        )

        fig, ax = plot_gaze_heatmap(empty_df)

        # Should handle gracefully
        assert fig is not None
        plt.close(fig)


class TestAdaptersIntegration:
    """Integration tests for adapters working together."""

    @pytest.fixture
    def realistic_recording(self):
        """Create a more realistic recording with varying gaze patterns."""
        n_samples = 500
        timestamps = np.arange(0, n_samples * 10000, 10000)

        np.random.seed(42)

        # Create realistic gaze pattern with fixations and saccades
        x = np.zeros(n_samples)
        y = np.zeros(n_samples)

        # Start position
        x[0] = 960
        y[0] = 540

        # Generate movement
        for i in range(1, n_samples):
            # Mostly small movements (fixations) with occasional large movements (saccades)
            if np.random.random() > 0.95:  # 5% chance of saccade
                x[i] = x[i - 1] + np.random.uniform(-300, 300)
                y[i] = y[i - 1] + np.random.uniform(-200, 200)
            else:
                x[i] = x[i - 1] + np.random.normal(0, 2)
                y[i] = y[i - 1] + np.random.normal(0, 2)

        # Clip to screen bounds
        x = np.clip(x, 0, 1920)
        y = np.clip(y, 0, 1080)

        # Add some blink periods
        validity = ["Valid"] * n_samples
        validity[100:115] = ["Invalid"] * 15  # ~150ms blink
        validity[300:320] = ["Invalid"] * 20  # ~200ms blink

        return pd.DataFrame(
            {
                "Recording timestamp": timestamps,
                "Gaze point X": x,
                "Gaze point Y": y,
                "Pupil diameter left": np.random.uniform(3.5, 4.5, n_samples),
                "Pupil diameter right": np.random.uniform(3.5, 4.5, n_samples),
                "Validity left": validity,
                "Validity right": validity,
            }
        )

    def test_full_pymovements_pipeline(self, realistic_recording):
        """Test complete pymovements processing pipeline."""
        from tobii_pipeline.adapters.pymovements_adapter import (
            apply_pix2deg,
            apply_pos2vel,
            compute_event_properties,
            detect_events_ivt,
            df_to_gaze_dataframe,
            events_to_df,
            get_fixation_stats,
        )

        # Full pipeline
        gaze = df_to_gaze_dataframe(realistic_recording)
        gaze = apply_pix2deg(gaze)
        gaze = apply_pos2vel(gaze)
        gaze = detect_events_ivt(gaze)
        gaze = compute_event_properties(gaze)
        events_df = events_to_df(gaze)

        # Should have detected some events
        assert len(events_df) > 0

        # Compute stats
        fix_stats = get_fixation_stats(events_df)
        assert fix_stats["count"] >= 0

    def test_full_mne_pipeline(self, realistic_recording):
        """Test complete MNE processing pipeline."""
        from tobii_pipeline.adapters.mne_adapter import (
            get_blink_statistics_mne,
            interpolate_blinks_mne,
        )

        # Blink detection and interpolation
        stats = get_blink_statistics_mne(realistic_recording)
        assert stats["count"] == 2  # We added 2 blink periods

        # Interpolate blinks
        result = interpolate_blinks_mne(realistic_recording)
        assert len(result) == len(realistic_recording)
