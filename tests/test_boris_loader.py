"""Tests for the boris_pipeline loader module."""

from pathlib import Path

import pandas as pd
import pytest

from boris_pipeline.loader import (
    get_file_type,
    get_recording_files,
    is_aggregated_file,
    load_aggregated_events,
    load_boris_file,
    load_recordings_from_dir,
    load_time_budget,
)


class TestIsAggregatedFile:
    """Tests for is_aggregated_file function."""

    def test_aggregated_underscore(self):
        """Detect _agregated suffix."""
        assert is_aggregated_file("G213_FAUJea_SDS2_P_M36_V4_25062025_agregated.xlsx")

    def test_aggregated_dash(self):
        """Detect -agregated suffix."""
        assert is_aggregated_file("L266_BAREli_SDS2_C_M0_V1_23072025-agregated.xlsx")

    def test_aggregated_english_underscore(self):
        """Detect English spelling _aggregated."""
        assert is_aggregated_file("G229_HEUPas_SDS2_C_M0_V1_02092025_aggregated.xlsx")

    def test_aggregated_english_dash(self):
        """Detect English spelling -aggregated."""
        assert is_aggregated_file("G229_HEUPas_SDS2_C_M0_V1_02092025-aggregated.xlsx")

    def test_not_aggregated(self):
        """Regular file is not aggregated."""
        assert not is_aggregated_file("G213_FAUJea_SDS2_P_M36_V4_25062025.xlsx")

    def test_path_object(self):
        """Works with Path objects."""
        assert is_aggregated_file(Path("data/G213_FAUJea_SDS2_P_M36_V4_25062025_agregated.xlsx"))


class TestGetFileType:
    """Tests for get_file_type function."""

    def test_time_budget_sheet(self, tmp_path, create_time_budget_excel):
        """Detect time budget file from sheet name."""
        filepath = create_time_budget_excel(tmp_path / "test.xlsx")
        assert get_file_type(filepath) == "time_budget"

    def test_aggregated_sheet(self, tmp_path, create_aggregated_excel):
        """Detect aggregated file from sheet name."""
        filepath = create_aggregated_excel(tmp_path / "test_agregated.xlsx")
        assert get_file_type(filepath) == "aggregated"

    def test_unknown_sheet_raises(self, tmp_path):
        """Raise ValueError for unknown sheet."""
        filepath = tmp_path / "unknown.xlsx"
        pd.DataFrame({"col": [1, 2, 3]}).to_excel(filepath, sheet_name="Unknown", index=False)
        with pytest.raises(ValueError, match="Could not detect file type"):
            get_file_type(filepath)


class TestLoadTimeBudget:
    """Tests for load_time_budget function."""

    def test_load_time_budget(self, tmp_path, create_time_budget_excel):
        """Load time budget file successfully."""
        filepath = create_time_budget_excel(tmp_path / "test.xlsx")
        df = load_time_budget(filepath)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "Subject" in df.columns
        assert "Behavior" in df.columns
        assert "Total duration (s)" in df.columns


class TestLoadAggregatedEvents:
    """Tests for load_aggregated_events function."""

    def test_load_aggregated(self, tmp_path, create_aggregated_excel):
        """Load aggregated events file successfully."""
        filepath = create_aggregated_excel(tmp_path / "test_agregated.xlsx")
        df = load_aggregated_events(filepath)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "Start (s)" in df.columns
        assert "Stop (s)" in df.columns
        assert "Duration (s)" in df.columns


class TestLoadBorisFile:
    """Tests for load_boris_file function."""

    def test_auto_detect_time_budget(self, tmp_path, create_time_budget_excel):
        """Auto-detect and load time budget file."""
        filepath = create_time_budget_excel(tmp_path / "test.xlsx")
        df = load_boris_file(filepath, file_type="auto")

        assert isinstance(df, pd.DataFrame)
        assert "Total duration (s)" in df.columns

    def test_auto_detect_aggregated(self, tmp_path, create_aggregated_excel):
        """Auto-detect and load aggregated file."""
        filepath = create_aggregated_excel(tmp_path / "test_agregated.xlsx")
        df = load_boris_file(filepath, file_type="auto")

        assert isinstance(df, pd.DataFrame)
        assert "Start (s)" in df.columns

    def test_explicit_type(self, tmp_path, create_time_budget_excel):
        """Load with explicit file type."""
        filepath = create_time_budget_excel(tmp_path / "test.xlsx")
        df = load_boris_file(filepath, file_type="time_budget")

        assert isinstance(df, pd.DataFrame)

    def test_invalid_type_raises(self, tmp_path, create_time_budget_excel):
        """Raise ValueError for invalid file type."""
        filepath = create_time_budget_excel(tmp_path / "test.xlsx")
        with pytest.raises(ValueError, match="Invalid file_type"):
            load_boris_file(filepath, file_type="invalid")


class TestGetRecordingFiles:
    """Tests for get_recording_files function."""

    def test_get_all_files(self, tmp_path, create_boris_files):
        """Get all Excel files."""
        create_boris_files(tmp_path)

        result = get_recording_files(tmp_path, file_type="all")
        assert len(result) == 2

    def test_filter_time_budget_only(self, tmp_path, create_boris_files):
        """Get only non-aggregated files."""
        create_boris_files(tmp_path)

        result = get_recording_files(tmp_path, file_type="time_budget")
        assert len(result) == 1
        assert "agregated" not in result[0].stem.lower()

    def test_filter_aggregated_only(self, tmp_path, create_boris_files):
        """Get only aggregated files."""
        create_boris_files(tmp_path)

        result = get_recording_files(tmp_path, file_type="aggregated")
        assert len(result) == 1
        assert "agregated" in result[0].stem.lower()

    def test_nonexistent_dir_raises(self, tmp_path):
        """Raise FileNotFoundError for nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            get_recording_files(tmp_path / "nonexistent")


class TestLoadRecordingsFromDir:
    """Tests for load_recordings_from_dir function."""

    def test_load_all_with_metadata(self, tmp_path, create_boris_files):
        """Load all files with metadata columns."""
        create_boris_files(tmp_path)

        df = load_recordings_from_dir(tmp_path, file_type="all", add_metadata=True, progress=False)

        assert isinstance(df, pd.DataFrame)
        assert "source_file" in df.columns
        assert "recording_id" in df.columns
        assert "participant_code" in df.columns
        assert "file_type" in df.columns

    def test_load_time_budget_only(self, tmp_path, create_boris_files):
        """Load only time budget files."""
        create_boris_files(tmp_path)

        df = load_recordings_from_dir(
            tmp_path, file_type="time_budget", add_metadata=True, progress=False
        )

        assert "Total duration (s)" in df.columns
        # Should only have time_budget file type
        assert (df["file_type"] == "time_budget").all()

    def test_empty_dir_raises(self, tmp_path):
        """Raise ValueError for empty directory."""
        tmp_path.mkdir(exist_ok=True)
        with pytest.raises(ValueError, match="No BORIS files found"):
            load_recordings_from_dir(tmp_path, progress=False)
