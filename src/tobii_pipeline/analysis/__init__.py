"""Analysis module for Tobii eye-tracking data.

Provides metrics calculation, visualization, statistical analysis,
and group comparisons for eye-tracking data.

Event detection (fixations, saccades) uses pymovements library.
Heatmap visualizations use MNE-Python for smoother plots.

Publication-quality visualizations available via pub_plots and group_viz modules.

Example:
    from tobii_pipeline.analysis import (
        compute_recording_summary,
        plot_recording_summary,
        compare_patient_vs_control,
    )

    # Compute metrics
    summary = compute_recording_summary(df)

    # Create visualizations
    fig = plot_recording_summary(df)

    # Compare groups
    results = compare_patient_vs_control(data_dirs, compute_fixation_stats)

    # Publication-quality figures
    from tobii_pipeline.analysis import (
        apply_publication_style,
        create_group_comparison_figure,
        plot_longitudinal_ci,
    )
"""

from .compare import (
    compare_patient_vs_control,
    compute_longitudinal_metrics,
    generate_summary_report,
    load_all_recordings,
    load_recordings_by_group,
    load_recordings_by_month,
    plot_group_comparison,
    plot_longitudinal_trend,
    test_group_difference,
)
from .group_viz import (
    compute_group_heatmap,
    compute_group_metrics_by_behavior,
    create_behavioral_figure,
    create_group_comparison_figure,
    create_longitudinal_figure,
    plot_behavior_group_comparison,
    plot_group_heatmap_comparison,
    plot_longitudinal_ci,
    plot_metric_violin,
    plot_multi_metric_comparison,
)
from .metrics import (
    compute_events,
    compute_fixation_stats,
    compute_gaze_center,
    compute_gaze_dispersion,
    compute_gaze_quadrant_distribution,
    compute_pupil_over_time,
    compute_pupil_stats,
    compute_pupil_variability,
    compute_recording_summary,
    compute_saccade_stats,
    compute_tracking_ratio,
    compute_validity_rate,
)
from .plots import (
    plot_eye_movement_timeline,
    plot_fixation_durations,
    plot_gaze_heatmap,
    plot_gaze_on_image,
    plot_gaze_scatter,
    plot_gaze_trajectory,
    plot_pupil_comparison,
    plot_pupil_distribution,
    plot_pupil_timeseries,
    plot_recording_summary,
    plot_scanpath,
)

# Publication-quality visualization modules
from .pub_plots import (
    CATEGORY_COLORS,
    FIGURE_SIZES,
    GROUP_COLORS,
    PUB_STYLE,
    add_panel_label,
    add_significance_bar,
    apply_publication_style,
    create_figure,
    despine,
    format_axis_labels,
    format_p_value,
    save_figure,
)
from .stats import (
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

__all__ = [
    # metrics
    "compute_validity_rate",
    "compute_tracking_ratio",
    "compute_gaze_center",
    "compute_gaze_dispersion",
    "compute_gaze_quadrant_distribution",
    "compute_pupil_stats",
    "compute_pupil_variability",
    "compute_pupil_over_time",
    "compute_events",
    "compute_fixation_stats",
    "compute_saccade_stats",
    "compute_recording_summary",
    # plots
    "plot_gaze_scatter",
    "plot_gaze_heatmap",
    "plot_gaze_on_image",
    "plot_gaze_trajectory",
    "plot_scanpath",
    "plot_pupil_timeseries",
    "plot_pupil_distribution",
    "plot_pupil_comparison",
    "plot_fixation_durations",
    "plot_eye_movement_timeline",
    "plot_recording_summary",
    # stats
    "describe_gaze",
    "describe_pupil",
    "describe_fixations",
    "ttest_groups",
    "mannwhitneyu_groups",
    "wilcoxon_paired",
    "compute_cohens_d",
    "compute_effect_size_r",
    "interpret_effect_size",
    "compute_confidence_interval",
    "normality_test",
    # compare
    "load_all_recordings",
    "load_recordings_by_group",
    "load_recordings_by_month",
    "compare_patient_vs_control",
    "plot_group_comparison",
    "test_group_difference",
    "compute_longitudinal_metrics",
    "plot_longitudinal_trend",
    "generate_summary_report",
    # pub_plots (publication styling)
    "PUB_STYLE",
    "GROUP_COLORS",
    "CATEGORY_COLORS",
    "FIGURE_SIZES",
    "apply_publication_style",
    "create_figure",
    "save_figure",
    "add_significance_bar",
    "format_p_value",
    "format_axis_labels",
    "add_panel_label",
    "despine",
    # group_viz (group-level visualizations)
    "compute_group_heatmap",
    "plot_group_heatmap_comparison",
    "plot_longitudinal_ci",
    "create_longitudinal_figure",
    "plot_metric_violin",
    "plot_multi_metric_comparison",
    "create_group_comparison_figure",
    "compute_group_metrics_by_behavior",
    "plot_behavior_group_comparison",
    "create_behavioral_figure",
]
