"""Tests for the loader module."""

from pathlib import Path

import pandas as pd

from tobii_pipeline.loader import get_recording_files, load_recording


class TestLoadRecording:
    """Tests for load_recording function."""

    def test_load_recording(self, tmp_path):
        """Load a single TSV file."""
        tsv_content = "col1\tcol2\tcol3\n1\t2\t3\n4\t5\t6\n"
        tsv_file = tmp_path / "test_recording.tsv"
        tsv_file.write_text(tsv_content)

        result = load_recording(tsv_file)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["col1", "col2", "col3"]

    def test_load_recording_with_chunksize(self, tmp_path):
        """Load with chunking returns iterator."""
        tsv_content = "col1\tcol2\n" + "\n".join(f"{i}\t{i + 1}" for i in range(10))
        tsv_file = tmp_path / "test_recording.tsv"
        tsv_file.write_text(tsv_content)

        result = load_recording(tsv_file, chunksize=3)

        chunks = list(result)
        assert len(chunks) == 4
        assert all(isinstance(chunk, pd.DataFrame) for chunk in chunks)

    def test_load_recording_path_object(self, tmp_path):
        """Load with Path object."""
        tsv_file = tmp_path / "test.tsv"
        tsv_file.write_text("a\tb\n1\t2\n")

        result = load_recording(Path(tsv_file))

        assert isinstance(result, pd.DataFrame)


class TestGetRecordingFiles:
    """Tests for get_recording_files function."""

    def test_get_recording_files(self, tmp_path):
        """List files matching pattern."""
        (tmp_path / "file1.tsv").write_text("data")
        (tmp_path / "file2.tsv").write_text("data")
        (tmp_path / "file3.csv").write_text("data")

        result = get_recording_files(tmp_path, pattern="*.tsv")

        assert len(result) == 2
        assert all(f.suffix == ".tsv" for f in result)

    def test_get_recording_files_sorted(self, tmp_path):
        """Files are returned sorted."""
        (tmp_path / "c_file.tsv").write_text("data")
        (tmp_path / "a_file.tsv").write_text("data")
        (tmp_path / "b_file.tsv").write_text("data")

        result = get_recording_files(tmp_path)

        names = [f.name for f in result]
        assert names == ["a_file.tsv", "b_file.tsv", "c_file.tsv"]

    def test_get_recording_files_empty(self, tmp_path):
        """Empty list when no files match."""
        result = get_recording_files(tmp_path, pattern="*.tsv")

        assert result == []
