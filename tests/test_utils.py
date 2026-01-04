"""Tests for the utils module."""

import pandas as pd

from tobii_pipeline.utils import load_parquet, save_parquet


class TestParquetIO:
    """Tests for Parquet save/load functions."""

    def test_save_and_load_parquet(self, tmp_path):
        """Round-trip save and load."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.5, 3.5],
                "str_col": ["a", "b", "c"],
            }
        )
        output_path = tmp_path / "test.parquet"

        save_parquet(df, output_path)
        result = load_parquet(output_path)

        pd.testing.assert_frame_equal(result, df)

    def test_save_parquet_creates_directory(self, tmp_path):
        """Creates parent directories if needed."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        output_path = tmp_path / "nested" / "dirs" / "test.parquet"

        save_parquet(df, output_path)

        assert output_path.exists()

    def test_save_parquet_compression(self, tmp_path):
        """Save with different compression options."""
        df = pd.DataFrame({"col": list(range(1000))})

        for compression in ["snappy", "gzip", None]:
            output_path = tmp_path / f"test_{compression}.parquet"
            save_parquet(df, output_path, compression=compression)

            result = load_parquet(output_path)
            pd.testing.assert_frame_equal(result, df)
