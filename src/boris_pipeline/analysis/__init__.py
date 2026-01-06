"""Analysis module for BORIS behavioral observation data.

Provides metrics calculation, visualization, statistical analysis,
and group comparisons for behavioral observation data.

Example:
    from boris_pipeline.analysis import (
        compute_recording_summary,
        plot_recording_summary,
        compare_patient_vs_control,
    )

    # Load data
    from boris_pipeline import load_boris_file
    df = load_boris_file("path/to/recording.xlsx")

    # Compute metrics
    summary = compute_recording_summary(df)

    # Create visualizations
    fig = plot_recording_summary(df)

    # Compare groups
    results = compare_patient_vs_control(data_dirs, compute_sequence_entropy)
"""

from .compare import (
    compare_behavior_duration_by_group,
    compare_behavior_frequency_by_group,
    compare_behavior_profiles,
    compare_patient_vs_control,
    compute_longitudinal_metrics,
    generate_summary_report,
    load_all_recordings,
    load_recordings_by_group,
    load_recordings_by_month,
    plot_group_comparison,
    plot_longitudinal_trend,
    test_group_difference,
    test_longitudinal_change,
)
from .metrics import (
    compute_behavior_bout_stats,
    compute_behavior_durations,
    compute_behavior_frequency,
    compute_behavior_latency,
    compute_behavior_rate,
    compute_inter_event_intervals,
    compute_recording_summary,
    compute_sequence_entropy,
    compute_time_budget,
    compute_transition_counts,
    compute_transition_matrix,
)
from .plots import (
    plot_behavior_rate_over_time,
    plot_behavior_timeline,
    plot_cumulative_duration,
    plot_duration_boxplot,
    plot_duration_histogram,
    plot_ethogram,
    plot_frequency_bars,
    plot_inter_event_intervals,
    plot_recording_summary,
    plot_time_budget_pie,
    plot_transition_diagram,
    plot_transition_matrix,
)
from .stats import (
    chi_square_behavior_frequency,
    chi_square_independence,
    compare_behavior_duration,
    compare_transition_matrices,
    compute_cohens_d,
    compute_confidence_interval,
    compute_cramers_v,
    compute_effect_size_r,
    compute_sequence_similarity,
    describe_behaviors,
    describe_recording,
    fisher_exact_test,
    interpret_cramers_v,
    interpret_effect_size,
    mannwhitneyu_groups,
    normality_test,
    test_markov_property,
    ttest_groups,
    wilcoxon_paired,
)

__all__ = [
    # metrics
    "compute_behavior_frequency",
    "compute_behavior_durations",
    "compute_behavior_rate",
    "compute_time_budget",
    "compute_inter_event_intervals",
    "compute_behavior_latency",
    "compute_transition_matrix",
    "compute_transition_counts",
    "compute_sequence_entropy",
    "compute_behavior_bout_stats",
    "compute_recording_summary",
    # plots
    "plot_behavior_timeline",
    "plot_ethogram",
    "plot_duration_histogram",
    "plot_duration_boxplot",
    "plot_frequency_bars",
    "plot_time_budget_pie",
    "plot_transition_matrix",
    "plot_transition_diagram",
    "plot_inter_event_intervals",
    "plot_cumulative_duration",
    "plot_behavior_rate_over_time",
    "plot_recording_summary",
    # stats (reused from tobii)
    "ttest_groups",
    "mannwhitneyu_groups",
    "wilcoxon_paired",
    "compute_cohens_d",
    "compute_effect_size_r",
    "interpret_effect_size",
    "compute_confidence_interval",
    "normality_test",
    # stats (boris-specific)
    "describe_behaviors",
    "describe_recording",
    "chi_square_independence",
    "chi_square_behavior_frequency",
    "fisher_exact_test",
    "compute_cramers_v",
    "interpret_cramers_v",
    "compute_sequence_similarity",
    "compare_transition_matrices",
    "test_markov_property",
    "compare_behavior_duration",
    # compare
    "load_all_recordings",
    "load_recordings_by_group",
    "load_recordings_by_month",
    "compare_patient_vs_control",
    "compare_behavior_profiles",
    "plot_group_comparison",
    "test_group_difference",
    "compare_behavior_duration_by_group",
    "compare_behavior_frequency_by_group",
    "compute_longitudinal_metrics",
    "plot_longitudinal_trend",
    "test_longitudinal_change",
    "generate_summary_report",
]
