"""Tests for the parser module."""

from datetime import date

from tobii_pipeline.parser import parse_filename


class TestParseFilename:
    """Tests for parse_filename function."""

    def test_parse_filename_valid(self):
        """Parse a valid filename with all components."""
        result = parse_filename("G213_FAUJea_SDS2_P_M36_V4_25062025")

        assert result["id"] == "G213"
        assert result["participant"] == "FAUJea"
        assert result["study"] == "SDS2"
        assert result["group"] == "Patient"
        assert result["group_code"] == "P"
        assert result["month"] == 36
        assert result["month_code"] == "M36"
        assert result["visit"] == 4
        assert result["visit_code"] == "V4"
        assert result["date"] == date(2025, 6, 25)
        assert result["date_str"] == "25062025"
        assert result["raw_filename"] == "G213_FAUJea_SDS2_P_M36_V4_25062025"

    def test_parse_filename_with_extension(self):
        """Parse filename with .tsv extension."""
        result = parse_filename("G213_FAUJea_SDS2_P_M36_V4_25062025.tsv")

        assert result["id"] == "G213"
        assert result["participant"] == "FAUJea"

    def test_parse_filename_with_data_export_suffix(self):
        """Parse filename with ' Data Export' suffix."""
        result = parse_filename("G213_FAUJea_SDS2_P_M36_V4_25062025 Data Export.tsv")

        assert result["id"] == "G213"
        assert result["participant"] == "FAUJea"
        assert result["raw_filename"] == "G213_FAUJea_SDS2_P_M36_V4_25062025"

    def test_parse_filename_invalid(self):
        """Invalid filename returns None values."""
        result = parse_filename("invalid_filename.tsv")

        assert result["id"] is None
        assert result["participant"] is None
        assert result["study"] is None
        assert result["group"] is None
        assert result["raw_filename"] == "invalid_filename"

    def test_parse_filename_control_group(self):
        """Test Control group parsing."""
        result = parse_filename("L001_BENNaw_SDS2_C_M0_V1_01012024")

        assert result["group"] == "Control"
        assert result["group_code"] == "C"

    def test_parse_filename_patient_group(self):
        """Test Patient group parsing."""
        result = parse_filename("G100_ABCDef_SDS2_P_M12_V2_15032024")

        assert result["group"] == "Patient"
        assert result["group_code"] == "P"

    def test_parse_filename_various_months(self):
        """Test different month values."""
        test_cases = [
            ("G001_ABCDef_SDS2_P_M0_V1_01012024", 0),
            ("G002_ABCDef_SDS2_P_M12_V2_01012024", 12),
            ("G003_ABCDef_SDS2_P_M24_V3_01012024", 24),
            ("G004_ABCDef_SDS2_P_M36_V4_01012024", 36),
        ]

        for filename, expected_month in test_cases:
            result = parse_filename(filename)
            assert result["month"] == expected_month, f"Failed for {filename}"

    def test_parse_filename_l_series(self):
        """Test L-series recording ID."""
        result = parse_filename("L456_XYZabc_SDS2_C_M24_V3_10102024")

        assert result["id"] == "L456"
        assert result["participant"] == "XYZabc"

    def test_parse_filename_full_path(self):
        """Parse filename from a full path."""
        result = parse_filename("/path/to/Data/G213_FAUJea_SDS2_P_M36_V4_25062025.tsv")

        assert result["id"] == "G213"
        assert result["participant"] == "FAUJea"
