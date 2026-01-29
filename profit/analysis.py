"""
Statistical analysis functions for ProFiT results.
"""
import math
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from scipy import stats


def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Computes the confidence interval for a list of values.

    Args:
        values: List of numeric values.
        confidence: Confidence level (default: 0.95 for 95% CI).

    Returns:
        Tuple of (mean, lower_bound, upper_bound).
    """
    if not values or len(values) < 2:
        mean = values[0] if values else 0.0
        return (mean, mean, mean)

    n = len(values)
    mean = np.mean(values)
    std_err = stats.sem(values)  # Standard error of the mean

    # Use t-distribution for small samples
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_value * std_err

    return (float(mean), float(mean - margin), float(mean + margin))


def paired_t_test(
    baseline: List[float],
    treatment: List[float]
) -> Tuple[float, float]:
    """
    Performs a paired t-test to compare baseline vs treatment.

    Args:
        baseline: List of baseline values (one per fold/run).
        treatment: List of treatment values (one per fold/run).

    Returns:
        Tuple of (t_statistic, p_value).
    """
    if len(baseline) != len(treatment):
        raise ValueError("Baseline and treatment must have same length")

    if len(baseline) < 2:
        return (0.0, 1.0)  # Cannot compute with < 2 samples

    t_stat, p_value = stats.ttest_rel(treatment, baseline)
    return (float(t_stat), float(p_value))


def wilcoxon_signed_rank(
    baseline: List[float],
    treatment: List[float]
) -> Tuple[float, float]:
    """
    Performs Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Args:
        baseline: List of baseline values.
        treatment: List of treatment values.

    Returns:
        Tuple of (statistic, p_value).
    """
    if len(baseline) != len(treatment):
        raise ValueError("Baseline and treatment must have same length")

    if len(baseline) < 5:
        return (0.0, 1.0)  # Wilcoxon needs at least 5 samples for meaningful results

    try:
        stat, p_value = stats.wilcoxon(treatment, baseline)
        return (float(stat), float(p_value))
    except ValueError:
        # Can happen if all differences are zero
        return (0.0, 1.0)


def compute_effect_size(
    baseline: List[float],
    treatment: List[float]
) -> float:
    """
    Computes Cohen's d effect size for paired samples.

    Args:
        baseline: List of baseline values.
        treatment: List of treatment values.

    Returns:
        Cohen's d effect size.
    """
    if len(baseline) != len(treatment) or len(baseline) < 2:
        return 0.0

    differences = [t - b for t, b in zip(treatment, baseline)]
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)

    if std_diff == 0:
        return 0.0

    return float(mean_diff / std_diff)


def interpret_effect_size(d: float) -> str:
    """Interprets Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def interpret_p_value(p: float, alpha: float = 0.05) -> str:
    """Interprets p-value for statistical significance."""
    if p < 0.001:
        return "highly significant (p < 0.001)"
    elif p < 0.01:
        return "very significant (p < 0.01)"
    elif p < alpha:
        return f"significant (p < {alpha})"
    else:
        return "not significant"


class StatisticalSummary:
    """
    Generates statistical summary for batch results.
    """

    def __init__(
        self,
        treatment_values: List[float],
        baseline_values: Optional[List[float]] = None,
        metric_name: str = "metric"
    ):
        self.treatment = treatment_values
        self.baseline = baseline_values
        self.metric_name = metric_name

    def compute(self) -> Dict[str, Any]:
        """Computes full statistical summary."""
        result = {
            "metric": self.metric_name,
            "n": len(self.treatment),
        }

        if not self.treatment:
            return result

        # Basic statistics
        result["mean"] = float(np.mean(self.treatment))
        result["std"] = float(np.std(self.treatment, ddof=1)) if len(self.treatment) > 1 else 0.0
        result["min"] = float(np.min(self.treatment))
        result["max"] = float(np.max(self.treatment))
        result["median"] = float(np.median(self.treatment))

        # Confidence interval
        mean, ci_lower, ci_upper = compute_confidence_interval(self.treatment)
        result["ci_95_lower"] = ci_lower
        result["ci_95_upper"] = ci_upper

        # Comparison with baseline (if provided)
        if self.baseline and len(self.baseline) == len(self.treatment):
            result["baseline_mean"] = float(np.mean(self.baseline))
            result["baseline_std"] = float(np.std(self.baseline, ddof=1)) if len(self.baseline) > 1 else 0.0

            # Improvement
            result["improvement"] = result["mean"] - result["baseline_mean"]

            # Statistical tests
            t_stat, t_pvalue = paired_t_test(self.baseline, self.treatment)
            result["t_statistic"] = t_stat
            result["t_pvalue"] = t_pvalue
            result["t_interpretation"] = interpret_p_value(t_pvalue)

            # Effect size
            effect = compute_effect_size(self.baseline, self.treatment)
            result["cohens_d"] = effect
            result["effect_interpretation"] = interpret_effect_size(effect)

            # Non-parametric test (for small samples or non-normal data)
            if len(self.treatment) >= 5:
                w_stat, w_pvalue = wilcoxon_signed_rank(self.baseline, self.treatment)
                result["wilcoxon_statistic"] = w_stat
                result["wilcoxon_pvalue"] = w_pvalue

        return result

    def format_summary(self) -> str:
        """Formats statistical summary as human-readable text."""
        stats = self.compute()
        lines = []

        lines.append(f"=== {self.metric_name} ===")
        lines.append(f"N = {stats['n']} runs")
        lines.append("")

        if stats['n'] == 0:
            lines.append("No data available")
            return "\n".join(lines)

        lines.append("Treatment (ProFiT):")
        lines.append(f"  Mean:   {stats['mean']:.4f}")
        lines.append(f"  Std:    {stats['std']:.4f}")
        lines.append(f"  95% CI: [{stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}]")
        lines.append(f"  Range:  [{stats['min']:.4f}, {stats['max']:.4f}]")
        lines.append(f"  Median: {stats['median']:.4f}")

        if 'baseline_mean' in stats:
            lines.append("")
            lines.append("Baseline:")
            lines.append(f"  Mean:   {stats['baseline_mean']:.4f}")
            lines.append(f"  Std:    {stats['baseline_std']:.4f}")
            lines.append("")
            lines.append("Comparison:")
            lines.append(f"  Improvement: {stats['improvement']:+.4f}")
            lines.append(f"  Effect size: {stats['cohens_d']:.3f} ({stats['effect_interpretation']})")
            lines.append(f"  t-test:      {stats['t_interpretation']} (p={stats['t_pvalue']:.4f})")

            if 'wilcoxon_pvalue' in stats:
                lines.append(f"  Wilcoxon:    {interpret_p_value(stats['wilcoxon_pvalue'])} (p={stats['wilcoxon_pvalue']:.4f})")

        return "\n".join(lines)
