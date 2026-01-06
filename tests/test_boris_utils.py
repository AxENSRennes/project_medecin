"""Tests for the boris_pipeline utils module."""

import pandas as pd

from boris_pipeline.utils import (
    get_data_summary,
    list_participants,
    list_recordings,
    load_parquet,
    save_parquet,
)


class TestSaveLoadParquet:
    """Tests for save_parquet and load_parquet functions."""

    def test_save_load_roundtrip(self, tmp_path):
        """Save and load DataFrame preserves data."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        filepath = tmp_path / "test.parquet"

        save_parquet(df, filepath)
        df_loaded = load_parquet(filepath)

        pd.testing.assert_frame_equal(df, df_loaded)

    def test_creates_parent_dirs(self, tmp_path):
        """save_parquet creates parent directories."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        filepath = tmp_path / "subdir1" / "subdir2" / "test.parquet"

        save_parquet(df, filepath)

        assert filepath.exists()


class TestListRecordings:
    """Tests for list_recordings function."""

    def test_list_with_metadata(self, tmp_path, create_boris_files):
        """List recordings includes parsed metadata."""
        create_boris_files(tmp_path)

        recordings = list_recordings(tmp_path)

        assert len(recordings) == 2
        assert "participant" in recordings[0]
        assert "group" in recordings[0]
        assert "filepath" in recordings[0]
        assert "file_size_kb" in recordings[0]

    def test_filter_by_type(self, tmp_path, create_boris_files):
        """Filter recordings by type."""
        create_boris_files(tmp_path)

        time_budget = list_recordings(tmp_path, file_type="time_budget")
        aggregated = list_recordings(tmp_path, file_type="aggregated")

        assert len(time_budget) == 1
        assert len(aggregated) == 1
        assert not time_budget[0]["is_aggregated"]
        assert aggregated[0]["is_aggregated"]


class TestListParticipants:
    """Tests for list_participants function."""

    def test_unique_participants(self, tmp_path, create_boris_files):
        """Get unique participant codes."""
        create_boris_files(tmp_path, "G213_FAUJea_SDS2_P_M36_V4_25062025")
        create_boris_files(tmp_path, "G214_BENNaw_SDS2_C_M0_V1_01012025")

        participants = list_participants(tmp_path)

        assert len(participants) == 2
        assert "FAUJea" in participants
        assert "BENNaw" in participants

    def test_multiple_directories(self, tmp_path, create_boris_files):
        """List participants from multiple directories."""
        dir1 = tmp_path / "data_G" / "Boris"
        dir2 = tmp_path / "data_L" / "Boris"

        create_boris_files(dir1, "G213_FAUJea_SDS2_P_M36_V4_25062025")
        create_boris_files(dir2, "L001_BENNaw_SDS2_C_M0_V1_01012025")

        participants = list_participants([dir1, dir2])

        assert len(participants) == 2
        assert "FAUJea" in participants
        assert "BENNaw" in participants


class TestGetDataSummary:
    """Tests for get_data_summary function."""

    def test_summary_counts(self, tmp_path, create_boris_files):
        """Summary includes correct counts."""
        create_boris_files(tmp_path, "G213_FAUJea_SDS2_P_M36_V4_25062025")
        create_boris_files(tmp_path, "G214_BENNaw_SDS2_C_M0_V1_01012025")

        summary = get_data_summary(tmp_path)

        assert summary["total_files"] == 4
        assert summary["time_budget_files"] == 2
        assert summary["aggregated_files"] == 2
        assert summary["participant_count"] == 2
        assert "Patient" in summary["groups"]
        assert "Control" in summary["groups"]

    def test_summary_months(self, tmp_path, create_boris_files):
        """Summary includes month breakdown."""
        create_boris_files(tmp_path, "G213_FAUJea_SDS2_P_M36_V4_25062025")
        create_boris_files(tmp_path, "G214_BENNaw_SDS2_C_M0_V1_01012025")

        summary = get_data_summary(tmp_path)

        assert "M0" in summary["months"]
        assert "M36" in summary["months"]
