"""Statistical analysis functions for eye-tracking data."""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# =============================================================================
# Descriptive Statistics
# =============================================================================


def describe_gaze(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics for gaze coordinates.

    Args:
        df: Input DataFrame with gaze columns

    Returns:
        DataFrame with statistics (count, mean, std, min, max, etc.)
        for Gaze point X and Y
    """
    gaze_cols = ["Gaze point X", "Gaze point Y"]
    available_cols = [c for c in gaze_cols if c in df.columns]

    if not available_cols:
        return pd.DataFrame()

    return df[available_cols].describe()


def describe_pupil(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics for pupil diameter.

    Args:
        df: Input DataFrame with pupil columns

    Returns:
        DataFrame with statistics for left and right pupil
    """
    pupil_cols = ["Pupil diameter left", "Pupil diameter right"]
    available_cols = [c for c in pupil_cols if c in df.columns]

    if not available_cols:
        return pd.DataFrame()

    return df[available_cols].describe()


def describe_fixations(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics for fixation metrics.

    Args:
        df: Input DataFrame with eye movement data

    Returns:
        DataFrame with fixation duration statistics
    """
    if "Eye movement type" not in df.columns or "Gaze event duration" not in df.columns:
        return pd.DataFrame()

    if "Eye movement type index" not in df.columns:
        return pd.DataFrame()

    fixations = df[df["Eye movement type"] == "Fixation"]
    durations = fixations.groupby("Eye movement type index")["Gaze event duration"].first()

    return durations.describe().to_frame(name="Fixation Duration (ms)")


# =============================================================================
# Comparative Statistics
# =============================================================================


def ttest_groups(
    group1_values: np.ndarray | pd.Series | list,
    group2_values: np.ndarray | pd.Series | list,
) -> dict:
    """Perform independent samples t-test.

    Args:
        group1_values: Values from first group
        group2_values: Values from second group

    Returns:
        Dict with test results
    """
    group1 = np.asarray(group1_values)
    group2 = np.asarray(group2_values)

    # Remove NaN values
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    if len(group1) < 2 or len(group2) < 2:
        return {
            "t_statistic": np.nan,
            "p_value": np.nan,
            "df": np.nan,
            "significant": False,
            "n1": len(group1),
            "n2": len(group2),
        }

    result = scipy_stats.ttest_ind(group1, group2)

    # Degrees of freedom for Welch's t-test
    n1, n2 = len(group1), len(group2)
    v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    if v1 == 0 and v2 == 0:
        df = n1 + n2 - 2
    else:
        df = ((v1 / n1 + v2 / n2) ** 2) / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))

    return {
        "t_statistic": result.statistic,
        "p_value": result.pvalue,
        "df": df,
        "significant": result.pvalue < 0.05,
        "n1": n1,
        "n2": n2,
    }


def mannwhitneyu_groups(
    group1_values: np.ndarray | pd.Series | list,
    group2_values: np.ndarray | pd.Series | list,
) -> dict:
    """Perform Mann-Whitney U test (non-parametric).

    Args:
        group1_values: Values from first group
        group2_values: Values from second group

    Returns:
        Dict with test results
    """
    group1 = np.asarray(group1_values)
    group2 = np.asarray(group2_values)

    # Remove NaN values
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    if len(group1) < 1 or len(group2) < 1:
        return {
            "U_statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
            "n1": len(group1),
            "n2": len(group2),
        }

    result = scipy_stats.mannwhitneyu(group1, group2, alternative="two-sided")

    return {
        "U_statistic": result.statistic,
        "p_value": result.pvalue,
        "significant": result.pvalue < 0.05,
        "n1": len(group1),
        "n2": len(group2),
    }


def wilcoxon_paired(
    values1: np.ndarray | pd.Series | list,
    values2: np.ndarray | pd.Series | list,
) -> dict:
    """Perform Wilcoxon signed-rank test for paired samples.

    Useful for longitudinal within-subject comparisons.

    Args:
        values1: First set of paired values
        values2: Second set of paired values

    Returns:
        Dict with test results
    """
    v1 = np.asarray(values1)
    v2 = np.asarray(values2)

    # Remove pairs with NaN in either value
    valid = ~(np.isnan(v1) | np.isnan(v2))
    v1 = v1[valid]
    v2 = v2[valid]

    if len(v1) < 10:  # Wilcoxon requires at least ~10 pairs for reliability
        return {
            "W_statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
            "n_pairs": len(v1),
        }

    # Check if all differences are zero
    diff = v1 - v2
    if np.all(diff == 0):
        return {
            "W_statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "n_pairs": len(v1),
        }

    result = scipy_stats.wilcoxon(v1, v2)

    return {
        "W_statistic": result.statistic,
        "p_value": result.pvalue,
        "significant": result.pvalue < 0.05,
        "n_pairs": len(v1),
    }


# =============================================================================
# Effect Sizes
# =============================================================================


def compute_cohens_d(
    group1: np.ndarray | pd.Series | list,
    group2: np.ndarray | pd.Series | list,
) -> float:
    """Compute Cohen's d effect size for group comparison.

    Args:
        group1: Values from first group
        group2: Values from second group

    Returns:
        Cohen's d value
        - Small: |d| < 0.2
        - Medium: 0.2 <= |d| < 0.8
        - Large: |d| >= 0.8
    """
    g1 = np.asarray(group1)
    g2 = np.asarray(group2)

    # Remove NaN values
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]

    if len(g1) < 2 or len(g2) < 2:
        return np.nan

    n1, n2 = len(g1), len(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.nan

    d = (np.mean(g1) - np.mean(g2)) / pooled_std
    return d


def compute_effect_size_r(
    U: float,
    n1: int,
    n2: int,
) -> float:
    """Compute effect size r from Mann-Whitney U statistic.

    Args:
        U: Mann-Whitney U statistic
        n1: Sample size of group 1
        n2: Sample size of group 2

    Returns:
        Effect size r
        - Small: r < 0.1
        - Medium: 0.1 <= r < 0.3
        - Large: r >= 0.5
    """
    if np.isnan(U) or n1 == 0 or n2 == 0:
        return np.nan

    # Z approximation
    mu_U = n1 * n2 / 2
    sigma_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

    if sigma_U == 0:
        return 0.0

    z = (U - mu_U) / sigma_U

    # Effect size r = z / sqrt(N)
    r = z / np.sqrt(n1 + n2)

    return abs(r)


def interpret_effect_size(
    effect_size: float,
    metric: str = "d",
) -> str:
    """Interpret effect size magnitude.

    Args:
        effect_size: The effect size value
        metric: "d" for Cohen's d, "r" for correlation-based

    Returns:
        String interpretation ("small", "medium", "large", "negligible")
    """
    if np.isnan(effect_size):
        return "unknown"

    abs_es = abs(effect_size)

    if metric == "d":
        # Cohen's d thresholds
        if abs_es < 0.2:
            return "negligible"
        if abs_es < 0.5:
            return "small"
        if abs_es < 0.8:
            return "medium"
        return "large"

    # r thresholds
    if abs_es < 0.1:
        return "negligible"
    if abs_es < 0.3:
        return "small"
    if abs_es < 0.5:
        return "medium"
    return "large"


# =============================================================================
# Additional Utilities
# =============================================================================


def compute_confidence_interval(
    values: np.ndarray | pd.Series | list,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute confidence interval for mean.

    Args:
        values: Sample values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    values = np.asarray(values)
    values = values[~np.isnan(values)]

    if len(values) < 2:
        return (np.nan, np.nan)

    n = len(values)
    mean = np.mean(values)
    se = scipy_stats.sem(values)

    h = se * scipy_stats.t.ppf((1 + confidence) / 2, n - 1)

    return (mean - h, mean + h)


def normality_test(
    values: np.ndarray | pd.Series | list,
) -> dict:
    """Test for normality using Shapiro-Wilk test.

    Args:
        values: Sample values

    Returns:
        Dict with test results
    """
    values = np.asarray(values)
    values = values[~np.isnan(values)]

    if len(values) < 3:
        return {
            "W_statistic": np.nan,
            "p_value": np.nan,
            "is_normal": False,
        }

    # Shapiro-Wilk has a limit of 5000 samples
    if len(values) > 5000:
        values = np.random.choice(values, 5000, replace=False)

    result = scipy_stats.shapiro(values)

    return {
        "W_statistic": result.statistic,
        "p_value": result.pvalue,
        "is_normal": result.pvalue >= 0.05,
    }
