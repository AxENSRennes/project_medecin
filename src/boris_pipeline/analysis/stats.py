"""Statistical analysis functions for BORIS behavioral data.

Reuses statistical functions from tobii_pipeline where applicable,
and adds categorical/behavioral-specific statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# =============================================================================
# Re-export from tobii_pipeline.analysis.stats
# =============================================================================
# Import these for convenience - users can access via boris_pipeline.analysis.stats
from tobii_pipeline.analysis.stats import (
    compute_cohens_d,
    compute_confidence_interval,
    compute_effect_size_r,
    interpret_effect_size,
    mannwhitneyu_groups,
    normality_test,
    ttest_groups,
    wilcoxon_paired,
)

# =============================================================================
# Descriptive Statistics (Boris-specific)
# =============================================================================


def describe_behaviors(
    df: pd.DataFrame,
    behavior_col: str = "Behavior",
    duration_col: str = "Duration (s)",
) -> pd.DataFrame:
    """Generate summary statistics per behavior.

    Args:
        df: Aggregated events DataFrame.
        behavior_col: Column name for behavior labels.
        duration_col: Column name for event durations.

    Returns:
        DataFrame with rows=behaviors, cols=count, duration_mean, duration_std,
        duration_min, duration_max, total_duration.
    """
    if behavior_col not in df.columns or duration_col not in df.columns:
        return pd.DataFrame()

    if len(df) == 0:
        return pd.DataFrame()

    result = (
        df.groupby(behavior_col)[duration_col]
        .agg(["count", "mean", "std", "min", "max", "sum"])
        .rename(
            columns={
                "count": "count",
                "mean": "duration_mean",
                "std": "duration_std",
                "min": "duration_min",
                "max": "duration_max",
                "sum": "total_duration",
            }
        )
    )

    return result


def describe_recording(
    df: pd.DataFrame,
    behavior_col: str = "Behavior",
    duration_col: str = "Duration (s)",
    start_col: str = "Start (s)",
    stop_col: str = "Stop (s)",
) -> dict:
    """Overall recording statistics.

    Args:
        df: Aggregated events DataFrame.
        behavior_col: Column name for behavior labels.
        duration_col: Column name for event durations.
        start_col: Column name for event start times.
        stop_col: Column name for event stop times.

    Returns:
        Dict with: total_events, total_behaviors, recording_duration,
                   total_behavior_duration, mean_event_duration, etc.
    """
    if len(df) == 0:
        return {
            "total_events": 0,
            "total_behaviors": 0,
            "recording_duration": 0.0,
            "total_behavior_duration": 0.0,
            "mean_event_duration": np.nan,
            "std_event_duration": np.nan,
        }

    total_events = len(df)
    total_behaviors = df[behavior_col].nunique() if behavior_col in df.columns else 0

    # Recording duration
    if start_col in df.columns and stop_col in df.columns:
        recording_duration = df[stop_col].max() - df[start_col].min()
    else:
        recording_duration = np.nan

    # Total and mean event duration
    if duration_col in df.columns:
        durations = df[duration_col].dropna()
        total_behavior_duration = durations.sum()
        mean_event_duration = durations.mean()
        std_event_duration = durations.std()
    else:
        total_behavior_duration = np.nan
        mean_event_duration = np.nan
        std_event_duration = np.nan

    return {
        "total_events": total_events,
        "total_behaviors": total_behaviors,
        "recording_duration": recording_duration,
        "total_behavior_duration": total_behavior_duration,
        "mean_event_duration": mean_event_duration,
        "std_event_duration": std_event_duration,
    }


# =============================================================================
# Categorical Statistics
# =============================================================================


def chi_square_independence(
    contingency_table: pd.DataFrame,
) -> dict:
    """Chi-square test for independence of categorical variables.

    Args:
        contingency_table: Cross-tabulation of two categorical variables.
            Can be created with pd.crosstab().

    Returns:
        Dict with: chi2, p_value, dof, expected, significant, cramers_v.
    """
    if contingency_table.empty:
        return {
            "chi2": np.nan,
            "p_value": np.nan,
            "dof": np.nan,
            "expected": None,
            "significant": False,
            "cramers_v": np.nan,
        }

    chi2, p_value, dof, expected = scipy_stats.chi2_contingency(contingency_table)

    # Compute Cramer's V
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    if min_dim > 0 and n > 0:
        cramers_v = np.sqrt(chi2 / (n * min_dim))
    else:
        cramers_v = np.nan

    return {
        "chi2": chi2,
        "p_value": p_value,
        "dof": dof,
        "expected": expected,
        "significant": p_value < 0.05,
        "cramers_v": cramers_v,
    }


def chi_square_behavior_frequency(
    group1_df: pd.DataFrame,
    group2_df: pd.DataFrame,
    behavior_col: str = "Behavior",
) -> dict:
    """Chi-square test comparing behavior frequency distributions.

    Tests whether two groups (e.g., Patient vs Control) have
    different behavior frequency profiles.

    Args:
        group1_df: Aggregated events DataFrame for group 1.
        group2_df: Aggregated events DataFrame for group 2.
        behavior_col: Column name for behavior labels.

    Returns:
        Dict with: chi2, p_value, dof, significant, cramers_v,
                   group1_counts, group2_counts.
    """
    if behavior_col not in group1_df.columns or behavior_col not in group2_df.columns:
        return {
            "chi2": np.nan,
            "p_value": np.nan,
            "dof": np.nan,
            "significant": False,
            "cramers_v": np.nan,
            "group1_counts": {},
            "group2_counts": {},
        }

    # Get frequency counts
    counts1 = group1_df[behavior_col].value_counts()
    counts2 = group2_df[behavior_col].value_counts()

    # Align behaviors (use union of all behaviors)
    all_behaviors = sorted(set(counts1.index) | set(counts2.index))

    if len(all_behaviors) < 2:
        return {
            "chi2": np.nan,
            "p_value": np.nan,
            "dof": np.nan,
            "significant": False,
            "cramers_v": np.nan,
            "group1_counts": counts1.to_dict(),
            "group2_counts": counts2.to_dict(),
        }

    # Create contingency table
    contingency = pd.DataFrame(
        {
            "group1": [counts1.get(b, 0) for b in all_behaviors],
            "group2": [counts2.get(b, 0) for b in all_behaviors],
        },
        index=all_behaviors,
    )

    result = chi_square_independence(contingency)
    result["group1_counts"] = counts1.to_dict()
    result["group2_counts"] = counts2.to_dict()

    return result


def fisher_exact_test(
    table: np.ndarray | list[list],
) -> dict:
    """Fisher's exact test for 2x2 contingency tables.

    Useful when sample sizes are small (expected frequencies < 5).

    Args:
        table: 2x2 contingency table as array or nested list.

    Returns:
        Dict with: odds_ratio, p_value, significant.
    """
    table = np.asarray(table)

    if table.shape != (2, 2):
        return {
            "odds_ratio": np.nan,
            "p_value": np.nan,
            "significant": False,
        }

    odds_ratio, p_value = scipy_stats.fisher_exact(table)

    return {
        "odds_ratio": odds_ratio,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def compute_cramers_v(
    contingency_table: pd.DataFrame,
) -> float:
    """Compute Cramer's V effect size for chi-square test.

    Interpretation:
    - Negligible: V < 0.1
    - Small: 0.1 <= V < 0.2
    - Medium: 0.2 <= V < 0.3
    - Large: V >= 0.3

    Args:
        contingency_table: Cross-tabulation of categorical variables.

    Returns:
        Cramer's V value between 0 and 1.
    """
    if contingency_table.empty:
        return np.nan

    chi2, _, _, _ = scipy_stats.chi2_contingency(contingency_table)

    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1

    if min_dim <= 0 or n == 0:
        return np.nan

    return np.sqrt(chi2 / (n * min_dim))


def interpret_cramers_v(v: float) -> str:
    """Interpret Cramer's V effect size.

    Args:
        v: Cramer's V value.

    Returns:
        String interpretation.
    """
    if np.isnan(v):
        return "unknown"

    if v < 0.1:
        return "negligible"
    if v < 0.2:
        return "small"
    if v < 0.3:
        return "medium"
    return "large"


# =============================================================================
# Sequence Analysis
# =============================================================================


def compute_sequence_similarity(
    seq1: list[str],
    seq2: list[str],
    method: str = "jaccard",
) -> float:
    """Compute similarity between two behavior sequences.

    Args:
        seq1: First behavior sequence (list of behavior labels).
        seq2: Second behavior sequence.
        method: "jaccard" (set similarity) or "bigram" (bigram overlap).

    Returns:
        Similarity score (0 to 1, higher = more similar).
    """
    if not seq1 or not seq2:
        return 0.0

    if method == "jaccard":
        # Jaccard similarity: |intersection| / |union|
        set1 = set(seq1)
        set2 = set(seq2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    if method == "bigram":
        # Bigram (transition) similarity
        def get_bigrams(seq):
            return set(zip(seq[:-1], seq[1:], strict=False))

        bigrams1 = get_bigrams(seq1)
        bigrams2 = get_bigrams(seq2)

        if not bigrams1 and not bigrams2:
            return 1.0 if seq1 == seq2 else 0.0

        intersection = len(bigrams1 & bigrams2)
        union = len(bigrams1 | bigrams2)
        return intersection / union if union > 0 else 0.0

    raise ValueError(f"Unknown method: {method}. Use 'jaccard' or 'bigram'.")


def compare_transition_matrices(
    matrix1: pd.DataFrame,
    matrix2: pd.DataFrame,
) -> dict:
    """Compare two transition matrices statistically.

    Uses chi-square test on the transition count differences.

    Args:
        matrix1: First transition matrix (from compute_transition_matrix).
        matrix2: Second transition matrix.

    Returns:
        Dict with: chi2, p_value, dof, significant, matrix_correlation.
    """
    if matrix1.empty or matrix2.empty:
        return {
            "chi2": np.nan,
            "p_value": np.nan,
            "dof": np.nan,
            "significant": False,
            "matrix_correlation": np.nan,
        }

    # Align matrices to have same behaviors
    all_behaviors = sorted(set(matrix1.index) | set(matrix2.index))

    # Reindex both matrices
    m1 = matrix1.reindex(index=all_behaviors, columns=all_behaviors, fill_value=0)
    m2 = matrix2.reindex(index=all_behaviors, columns=all_behaviors, fill_value=0)

    # Flatten for correlation
    flat1 = m1.values.flatten()
    flat2 = m2.values.flatten()

    # Pearson correlation between matrices
    if np.std(flat1) > 0 and np.std(flat2) > 0:
        matrix_correlation, _ = scipy_stats.pearsonr(flat1, flat2)
    else:
        matrix_correlation = np.nan

    # Chi-square test on combined contingency
    # Create a contingency table: rows = transitions, cols = group
    combined = pd.DataFrame({"group1": flat1, "group2": flat2})
    combined = combined[(combined["group1"] > 0) | (combined["group2"] > 0)]

    if len(combined) < 2:
        return {
            "chi2": np.nan,
            "p_value": np.nan,
            "dof": np.nan,
            "significant": False,
            "matrix_correlation": matrix_correlation,
        }

    try:
        chi2, p_value, dof, _ = scipy_stats.chi2_contingency(combined.T)
    except ValueError:
        return {
            "chi2": np.nan,
            "p_value": np.nan,
            "dof": np.nan,
            "significant": False,
            "matrix_correlation": matrix_correlation,
        }

    return {
        "chi2": chi2,
        "p_value": p_value,
        "dof": dof,
        "significant": p_value < 0.05,
        "matrix_correlation": matrix_correlation,
    }


def test_markov_property(
    df: pd.DataFrame,
    behavior_col: str = "Behavior",
) -> dict:
    """Test whether behavior sequence follows first-order Markov property.

    Compares first-order transition probabilities (P(B|A)) with
    second-order probabilities (P(C|A,B)) to test if history matters.

    Args:
        df: Aggregated events DataFrame.
        behavior_col: Column name for behavior labels.

    Returns:
        Dict with: is_markov, chi2, p_value, first_order_entropy,
                   second_order_entropy.
    """
    if behavior_col not in df.columns or len(df) < 3:
        return {
            "is_markov": None,
            "chi2": np.nan,
            "p_value": np.nan,
            "first_order_entropy": np.nan,
            "second_order_entropy": np.nan,
        }

    # Sort by time
    if "Start (s)" in df.columns:
        df = df.sort_values("Start (s)")

    behaviors = df[behavior_col].values

    # First-order transitions
    first_order = {}
    for i in range(len(behaviors) - 1):
        key = behaviors[i]
        target = behaviors[i + 1]
        if key not in first_order:
            first_order[key] = {}
        first_order[key][target] = first_order[key].get(target, 0) + 1

    # Second-order transitions
    second_order = {}
    for i in range(len(behaviors) - 2):
        key = (behaviors[i], behaviors[i + 1])
        target = behaviors[i + 2]
        if key not in second_order:
            second_order[key] = {}
        second_order[key][target] = second_order[key].get(target, 0) + 1

    # Compute entropies
    def compute_transition_entropy(trans_dict):
        total_entropy = 0.0
        total_count = 0
        for _source, targets in trans_dict.items():
            count = sum(targets.values())
            total_count += count
            for _target, n in targets.items():
                p = n / count
                if p > 0:
                    total_entropy -= p * np.log2(p) * count
        return total_entropy / total_count if total_count > 0 else 0.0

    first_order_entropy = compute_transition_entropy(first_order)
    second_order_entropy = compute_transition_entropy(second_order)

    # Simple heuristic: if second-order entropy is much lower,
    # the sequence has memory beyond first order
    entropy_diff = first_order_entropy - second_order_entropy
    is_markov = entropy_diff < 0.1  # Threshold for "approximately Markov"

    return {
        "is_markov": is_markov,
        "chi2": np.nan,  # Full chi-square test would require more complex analysis
        "p_value": np.nan,
        "first_order_entropy": first_order_entropy,
        "second_order_entropy": second_order_entropy,
        "entropy_reduction": entropy_diff,
    }


# =============================================================================
# Behavior-Specific Comparisons
# =============================================================================


def compare_behavior_duration(
    group1_df: pd.DataFrame,
    group2_df: pd.DataFrame,
    behavior: str,
    behavior_col: str = "Behavior",
    duration_col: str = "Duration (s)",
    test: str = "mannwhitneyu",
) -> dict:
    """Compare duration of a specific behavior between two groups.

    Args:
        group1_df: Aggregated events DataFrame for group 1.
        group2_df: Aggregated events DataFrame for group 2.
        behavior: Behavior to compare.
        behavior_col: Column name for behavior labels.
        duration_col: Column name for event durations.
        test: "mannwhitneyu" or "ttest".

    Returns:
        Dict with test results, means, and effect size.
    """
    # Filter for target behavior
    g1_durations = group1_df[group1_df[behavior_col] == behavior][duration_col].dropna()
    g2_durations = group2_df[group2_df[behavior_col] == behavior][duration_col].dropna()

    result = {
        "behavior": behavior,
        "group1_n": len(g1_durations),
        "group2_n": len(g2_durations),
        "group1_mean": g1_durations.mean() if len(g1_durations) > 0 else np.nan,
        "group2_mean": g2_durations.mean() if len(g2_durations) > 0 else np.nan,
        "group1_std": g1_durations.std() if len(g1_durations) > 1 else np.nan,
        "group2_std": g2_durations.std() if len(g2_durations) > 1 else np.nan,
    }

    if len(g1_durations) < 2 or len(g2_durations) < 2:
        result.update(
            {
                "test": test,
                "statistic": np.nan,
                "p_value": np.nan,
                "significant": False,
                "effect_size": np.nan,
            }
        )
        return result

    if test == "mannwhitneyu":
        test_result = mannwhitneyu_groups(g1_durations, g2_durations)
        effect_size = compute_effect_size_r(
            test_result["U_statistic"],
            test_result["n1"],
            test_result["n2"],
        )
        result.update(
            {
                "test": "mannwhitneyu",
                "statistic": test_result["U_statistic"],
                "p_value": test_result["p_value"],
                "significant": test_result["significant"],
                "effect_size": effect_size,
                "effect_size_type": "r",
            }
        )
    else:
        test_result = ttest_groups(g1_durations, g2_durations)
        effect_size = compute_cohens_d(g1_durations, g2_durations)
        result.update(
            {
                "test": "ttest",
                "statistic": test_result["t_statistic"],
                "p_value": test_result["p_value"],
                "significant": test_result["significant"],
                "effect_size": effect_size,
                "effect_size_type": "cohens_d",
            }
        )

    return result


# =============================================================================
# Export list for __all__
# =============================================================================

__all__ = [
    # Re-exported from tobii_pipeline
    "ttest_groups",
    "mannwhitneyu_groups",
    "wilcoxon_paired",
    "compute_cohens_d",
    "compute_effect_size_r",
    "interpret_effect_size",
    "compute_confidence_interval",
    "normality_test",
    # Boris-specific descriptive
    "describe_behaviors",
    "describe_recording",
    # Categorical statistics
    "chi_square_independence",
    "chi_square_behavior_frequency",
    "fisher_exact_test",
    "compute_cramers_v",
    "interpret_cramers_v",
    # Sequence analysis
    "compute_sequence_similarity",
    "compare_transition_matrices",
    "test_markov_property",
    # Behavior comparisons
    "compare_behavior_duration",
]
