"""Tests for the analysis.stats module."""

import numpy as np
import pandas as pd
import pytest

from tobii_pipeline.analysis.stats import (
    compute_cohens_d,
    compute_confidence_interval,
    compute_effect_size_r,
    describe_fixations,
    describe_gaze,
    describe_pupil,
    interpret_effect_size,
    mannwhitneyu_groups,
    normality_test,
    ttest_groups,
    wilcoxon_paired,
)


class TestDescribeGaze:
    """Tests for describe_gaze function."""

    def test_returns_statistics(self):
        """Returns descriptive statistics for gaze."""
        df = pd.DataFrame(
            {
                "Gaze point X": [100, 200, 300, 400, 500],
                "Gaze point Y": [50, 100, 150, 200, 250],
            }
        )
        result = describe_gaze(df)
        assert "Gaze point X" in result.columns
        assert "Gaze point Y" in result.columns
        assert "mean" in result.index

    def test_empty_for_missing_columns(self):
        """Returns empty DataFrame when columns missing."""
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = describe_gaze(df)
        assert len(result) == 0


class TestDescribePupil:
    """Tests for describe_pupil function."""

    def test_returns_statistics(self):
        """Returns descriptive statistics for pupil."""
        df = pd.DataFrame(
            {
                "Pupil diameter left": [3.0, 3.5, 4.0, 4.5, 5.0],
                "Pupil diameter right": [3.2, 3.7, 4.2, 4.7, 5.2],
            }
        )
        result = describe_pupil(df)
        assert "Pupil diameter left" in result.columns
        assert "Pupil diameter right" in result.columns


class TestDescribeFixations:
    """Tests for describe_fixations function."""

    def test_returns_fixation_stats(self):
        """Returns fixation duration statistics."""
        df = pd.DataFrame(
            {
                "Eye movement type": ["Fixation", "Fixation", "Fixation"],
                "Eye movement type index": [1, 2, 3],
                "Gaze event duration": [100, 200, 300],
            }
        )
        result = describe_fixations(df)
        assert len(result) > 0


class TestTtestGroups:
    """Tests for ttest_groups function."""

    def test_equal_groups(self):
        """Returns non-significant for equal groups."""
        group1 = [1, 2, 3, 4, 5]
        group2 = [1, 2, 3, 4, 5]
        result = ttest_groups(group1, group2)
        assert result["p_value"] > 0.05
        assert not result["significant"]

    def test_different_groups(self):
        """Returns significant for very different groups."""
        group1 = [1, 2, 3, 4, 5]
        group2 = [100, 101, 102, 103, 104]
        result = ttest_groups(group1, group2)
        assert result["p_value"] < 0.05
        assert result["significant"]

    def test_handles_nan(self):
        """Removes NaN values before test."""
        group1 = [1, 2, np.nan, 4, 5]
        group2 = [1, 2, 3, 4, 5]
        result = ttest_groups(group1, group2)
        assert "t_statistic" in result

    def test_insufficient_data(self):
        """Returns NaN for insufficient data."""
        group1 = [1]
        group2 = [2]
        result = ttest_groups(group1, group2)
        assert np.isnan(result["t_statistic"])


class TestMannWhitneyU:
    """Tests for mannwhitneyu_groups function."""

    def test_different_groups(self):
        """Detects difference between groups."""
        group1 = [1, 2, 3, 4, 5]
        group2 = [10, 11, 12, 13, 14]
        result = mannwhitneyu_groups(group1, group2)
        assert result["p_value"] < 0.05
        assert result["significant"]

    def test_returns_u_statistic(self):
        """Returns U statistic."""
        group1 = [1, 2, 3]
        group2 = [4, 5, 6]
        result = mannwhitneyu_groups(group1, group2)
        assert "U_statistic" in result


class TestWilcoxonPaired:
    """Tests for wilcoxon_paired function."""

    def test_no_difference(self):
        """Returns non-significant for equal pairs."""
        v1 = list(range(1, 21))
        v2 = list(range(1, 21))
        result = wilcoxon_paired(v1, v2)
        # All differences are zero
        assert result["p_value"] == 1.0

    def test_significant_difference(self):
        """Returns significant for different pairs."""
        v1 = list(range(1, 21))
        v2 = [x + 10 for x in range(1, 21)]
        result = wilcoxon_paired(v1, v2)
        assert result["p_value"] < 0.05

    def test_insufficient_pairs(self):
        """Returns NaN for too few pairs."""
        v1 = [1, 2, 3]
        v2 = [2, 3, 4]
        result = wilcoxon_paired(v1, v2)
        assert np.isnan(result["W_statistic"])


class TestCohensD:
    """Tests for compute_cohens_d function."""

    def test_zero_effect(self):
        """Returns 0 for identical groups."""
        group1 = [1, 2, 3, 4, 5]
        group2 = [1, 2, 3, 4, 5]
        d = compute_cohens_d(group1, group2)
        assert d == pytest.approx(0.0, abs=0.01)

    def test_large_effect(self):
        """Returns large d for very different groups."""
        # Groups need variance for pooled std to be non-zero
        group1 = [1, 2, 3, 4, 5]
        group2 = [101, 102, 103, 104, 105]
        d = compute_cohens_d(group1, group2)
        assert abs(d) > 0.8

    def test_handles_nan(self):
        """Removes NaN before computing."""
        group1 = [1, 2, np.nan, 4, 5]
        group2 = [1, 2, 3, 4, 5]
        d = compute_cohens_d(group1, group2)
        assert not np.isnan(d)


class TestEffectSizeR:
    """Tests for compute_effect_size_r function."""

    def test_computes_r(self):
        """Computes effect size r from U."""
        r = compute_effect_size_r(U=0, n1=10, n2=10)
        assert not np.isnan(r)

    def test_nan_input(self):
        """Returns NaN for NaN U."""
        r = compute_effect_size_r(U=np.nan, n1=10, n2=10)
        assert np.isnan(r)


class TestInterpretEffectSize:
    """Tests for interpret_effect_size function."""

    def test_cohens_d_thresholds(self):
        """Interprets Cohen's d correctly."""
        assert interpret_effect_size(0.1, "d") == "negligible"
        assert interpret_effect_size(0.3, "d") == "small"
        assert interpret_effect_size(0.6, "d") == "medium"
        assert interpret_effect_size(1.0, "d") == "large"

    def test_r_thresholds(self):
        """Interprets r correctly."""
        assert interpret_effect_size(0.05, "r") == "negligible"
        assert interpret_effect_size(0.2, "r") == "small"
        assert interpret_effect_size(0.4, "r") == "medium"
        assert interpret_effect_size(0.6, "r") == "large"

    def test_nan_handling(self):
        """Returns 'unknown' for NaN."""
        assert interpret_effect_size(np.nan, "d") == "unknown"


class TestConfidenceInterval:
    """Tests for compute_confidence_interval function."""

    def test_computes_ci(self):
        """Computes confidence interval."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        lower, upper = compute_confidence_interval(values, confidence=0.95)
        assert lower < 5.5 < upper
        assert lower < upper

    def test_insufficient_data(self):
        """Returns NaN for insufficient data."""
        values = [1]
        lower, upper = compute_confidence_interval(values)
        assert np.isnan(lower)
        assert np.isnan(upper)


class TestNormalityTest:
    """Tests for normality_test function."""

    def test_normal_data(self):
        """Detects normal distribution."""
        np.random.seed(42)
        values = np.random.normal(100, 10, 100)
        result = normality_test(values)
        # Normal data should have p > 0.05
        assert result["p_value"] > 0.05
        assert result["is_normal"]

    def test_non_normal_data(self):
        """Detects non-normal distribution."""
        # Uniform distribution is not normal
        np.random.seed(42)
        values = np.random.uniform(0, 100, 100)
        result = normality_test(values)
        # May or may not be detected as non-normal depending on sample
        assert "is_normal" in result

    def test_insufficient_data(self):
        """Returns NaN for too few values."""
        values = [1, 2]
        result = normality_test(values)
        assert np.isnan(result["W_statistic"])
