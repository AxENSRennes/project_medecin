"""Tests for the boris_pipeline parser module."""

from boris_pipeline.parser import parse_boris_filename, strip_aggregated_suffix


class TestStripAggregatedSuffix:
    """Tests for strip_aggregated_suffix function."""

    def test_underscore_agregated(self):
        """Strip _agregated suffix."""
        result = strip_aggregated_suffix("G213_FAUJea_SDS2_P_M36_V4_25062025_agregated.xlsx")
        assert result == "G213_FAUJea_SDS2_P_M36_V4_25062025.xlsx"

    def test_dash_agregated(self):
        """Strip -agregated suffix."""
        result = strip_aggregated_suffix("L266_BAREli_SDS2_C_M0_V1_23072025-agregated.xlsx")
        assert result == "L266_BAREli_SDS2_C_M0_V1_23072025.xlsx"

    def test_underscore_aggregated_english(self):
        """Strip _aggregated suffix (English spelling)."""
        result = strip_aggregated_suffix("G229_HEUPas_SDS2_C_M0_V1_02092025_aggregated.xlsx")
        assert result == "G229_HEUPas_SDS2_C_M0_V1_02092025.xlsx"

    def test_dash_aggregated_english(self):
        """Strip -aggregated suffix (English spelling)."""
        result = strip_aggregated_suffix("G229_HEUPas_SDS2_C_M0_V1_02092025-aggregated.xlsx")
        assert result == "G229_HEUPas_SDS2_C_M0_V1_02092025.xlsx"

    def test_no_suffix_unchanged(self):
        """Return unchanged if no suffix."""
        result = strip_aggregated_suffix("G213_FAUJea_SDS2_P_M36_V4_25062025.xlsx")
        assert result == "G213_FAUJea_SDS2_P_M36_V4_25062025.xlsx"

    def test_no_extension(self):
        """Handle filename without extension."""
        result = strip_aggregated_suffix("G213_FAUJea_SDS2_P_M36_V4_25062025_agregated")
        assert result == "G213_FAUJea_SDS2_P_M36_V4_25062025"

    def test_case_insensitive(self):
        """Handle uppercase variants."""
        result = strip_aggregated_suffix("G213_FAUJea_SDS2_P_M36_V4_25062025_AGREGATED.xlsx")
        assert result == "G213_FAUJea_SDS2_P_M36_V4_25062025.xlsx"


class TestParsBorisFilename:
    """Tests for parse_boris_filename function."""

    def test_parse_time_budget_file(self):
        """Parse standard time budget filename."""
        result = parse_boris_filename("G213_FAUJea_SDS2_P_M36_V4_25062025.xlsx")

        assert result["id"] == "G213"
        assert result["participant"] == "FAUJea"
        assert result["study"] == "SDS2"
        assert result["group"] == "Patient"
        assert result["month"] == 36
        assert result["visit"] == 4
        assert result["is_aggregated"] is False

    def test_parse_aggregated_file(self):
        """Parse aggregated filename with suffix stripped."""
        result = parse_boris_filename("G213_FAUJea_SDS2_P_M36_V4_25062025_agregated.xlsx")

        assert result["id"] == "G213"
        assert result["participant"] == "FAUJea"
        assert result["group"] == "Patient"
        assert result["is_aggregated"] is True

    def test_parse_control_group(self):
        """Parse control group filename."""
        result = parse_boris_filename("L266_BAREli_SDS2_C_M0_V1_23072025.xlsx")

        assert result["id"] == "L266"
        assert result["group"] == "Control"
        assert result["month"] == 0
        assert result["visit"] == 1

    def test_parse_different_months(self):
        """Parse different month values."""
        for month_str, expected in [("M0", 0), ("M12", 12), ("M24", 24), ("M36", 36)]:
            filename = f"G001_ABCDef_SDS2_P_{month_str}_V1_01012025.xlsx"
            result = parse_boris_filename(filename)
            assert result["month"] == expected

    def test_invalid_filename_returns_none_values(self):
        """Invalid filename returns dict with None values."""
        result = parse_boris_filename("invalid_filename.xlsx")

        assert result["id"] is None
        assert result["participant"] is None
        assert result["is_aggregated"] is False
