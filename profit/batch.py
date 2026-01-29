"""
Batch execution for running multiple fold/seed combinations.
"""
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from profit.config import Config
from profit.optimizer import Optimizer
from profit.results import Results
from profit.analysis import StatisticalSummary, compute_confidence_interval


@dataclass
class RunResult:
    """Result from a single fold/seed run."""
    fold_idx: int
    seed_idx: int
    results: Optional[Results]
    error: Optional[str] = None


class BatchResults:
    """
    Aggregates results from multiple fold/seed optimization runs.
    """

    def __init__(self, run_results: List[RunResult], config: Config):
        self.run_results = run_results
        self.config = config
        self.successful_runs = [r for r in run_results if r.results is not None]
        self.failed_runs = [r for r in run_results if r.error is not None]

    def get_best_test_fitness_per_run(self) -> List[Dict[str, Any]]:
        """Returns the best test fitness from each successful run."""
        results = []
        for run in self.successful_runs:
            if run.results and run.results.test_results:
                # Get best test result (first one, already sorted)
                best = run.results.test_results[0] if run.results.test_results else {}
                results.append({
                    "fold": run.fold_idx,
                    "seed": run.seed_idx,
                    "train_fitness": best.get("train_fitness"),
                    "test_fitness": best.get("test_fitness"),
                    "train_sharpe": best.get("train_sharpe"),
                    "test_sharpe": best.get("test_sharpe"),
                })
        return results

    def aggregate_metrics(self) -> Dict[str, Any]:
        """Computes aggregate statistics across all runs."""
        import numpy as np

        best_per_run = self.get_best_test_fitness_per_run()
        if not best_per_run:
            return {}

        train_fitness = [r["train_fitness"] for r in best_per_run if r["train_fitness"] is not None]
        test_fitness = [r["test_fitness"] for r in best_per_run if r["test_fitness"] is not None]
        train_sharpe = [r["train_sharpe"] for r in best_per_run if r["train_sharpe"] is not None]
        test_sharpe = [r["test_sharpe"] for r in best_per_run if r["test_sharpe"] is not None]

        result = {
            "num_runs": len(self.run_results),
            "successful_runs": len(self.successful_runs),
            "failed_runs": len(self.failed_runs),
            "train_fitness_mean": float(np.mean(train_fitness)) if train_fitness else None,
            "train_fitness_std": float(np.std(train_fitness)) if train_fitness else None,
            "test_fitness_mean": float(np.mean(test_fitness)) if test_fitness else None,
            "test_fitness_std": float(np.std(test_fitness)) if test_fitness else None,
            "train_sharpe_mean": float(np.mean(train_sharpe)) if train_sharpe else None,
            "train_sharpe_std": float(np.std(train_sharpe)) if train_sharpe else None,
            "test_sharpe_mean": float(np.mean(test_sharpe)) if test_sharpe else None,
            "test_sharpe_std": float(np.std(test_sharpe)) if test_sharpe else None,
        }

        # Add 95% confidence intervals
        if test_fitness:
            mean, ci_lower, ci_upper = compute_confidence_interval(test_fitness)
            result["test_fitness_ci_lower"] = ci_lower
            result["test_fitness_ci_upper"] = ci_upper

        if test_sharpe:
            mean, ci_lower, ci_upper = compute_confidence_interval(test_sharpe)
            result["test_sharpe_ci_lower"] = ci_lower
            result["test_sharpe_ci_upper"] = ci_upper

        return result

    def statistical_summary(self) -> Dict[str, Any]:
        """
        Generates detailed statistical summary with confidence intervals
        and significance tests comparing train vs test performance.
        """
        best_per_run = self.get_best_test_fitness_per_run()
        if not best_per_run:
            return {}

        train_fitness = [r["train_fitness"] for r in best_per_run if r["train_fitness"] is not None]
        test_fitness = [r["test_fitness"] for r in best_per_run if r["test_fitness"] is not None]
        train_sharpe = [r["train_sharpe"] for r in best_per_run if r["train_sharpe"] is not None]
        test_sharpe = [r["test_sharpe"] for r in best_per_run if r["test_sharpe"] is not None]

        summaries = {}

        # Test fitness analysis
        if test_fitness:
            fitness_summary = StatisticalSummary(
                treatment_values=test_fitness,
                baseline_values=train_fitness if len(train_fitness) == len(test_fitness) else None,
                metric_name="Annualized Return (%)"
            )
            summaries["fitness"] = fitness_summary.compute()
            summaries["fitness_text"] = fitness_summary.format_summary()

        # Test Sharpe analysis
        if test_sharpe:
            sharpe_summary = StatisticalSummary(
                treatment_values=test_sharpe,
                baseline_values=train_sharpe if len(train_sharpe) == len(test_sharpe) else None,
                metric_name="Sharpe Ratio"
            )
            summaries["sharpe"] = sharpe_summary.compute()
            summaries["sharpe_text"] = sharpe_summary.format_summary()

        return summaries

    def generate_batch_report(self, output_dir: str):
        """Generates a comprehensive report for all batch runs."""
        import pandas as pd
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)

        # 1. Summary statistics
        agg = self.aggregate_metrics()
        summary_path = os.path.join(output_dir, "batch_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("=== Batch Run Summary ===\n\n")
            f.write(f"Total runs: {agg.get('num_runs', 0)}\n")
            f.write(f"Successful: {agg.get('successful_runs', 0)}\n")
            f.write(f"Failed: {agg.get('failed_runs', 0)}\n\n")

            f.write("=== Aggregate Metrics (Best Strategy per Run) ===\n\n")
            if agg.get('train_fitness_mean') is not None:
                f.write(f"Train Fitness: {agg['train_fitness_mean']:.2f}% +/- {agg['train_fitness_std']:.2f}%\n")
            if agg.get('test_fitness_mean') is not None:
                f.write(f"Test Fitness:  {agg['test_fitness_mean']:.2f}% +/- {agg['test_fitness_std']:.2f}%\n")
                if agg.get('test_fitness_ci_lower') is not None:
                    f.write(f"  95% CI:      [{agg['test_fitness_ci_lower']:.2f}%, {agg['test_fitness_ci_upper']:.2f}%]\n")
            if agg.get('train_sharpe_mean') is not None:
                f.write(f"Train Sharpe:  {agg['train_sharpe_mean']:.2f} +/- {agg['train_sharpe_std']:.2f}\n")
            if agg.get('test_sharpe_mean') is not None:
                f.write(f"Test Sharpe:   {agg['test_sharpe_mean']:.2f} +/- {agg['test_sharpe_std']:.2f}\n")
                if agg.get('test_sharpe_ci_lower') is not None:
                    f.write(f"  95% CI:      [{agg['test_sharpe_ci_lower']:.2f}, {agg['test_sharpe_ci_upper']:.2f}]\n")

            if self.failed_runs:
                f.write("\n=== Failed Runs ===\n\n")
                for run in self.failed_runs:
                    f.write(f"Fold {run.fold_idx}, Seed {run.seed_idx}: {run.error}\n")

        # 2. Detailed statistical analysis
        stats = self.statistical_summary()
        if stats:
            stats_path = os.path.join(output_dir, "statistical_analysis.txt")
            with open(stats_path, 'w') as f:
                f.write("=== Statistical Analysis ===\n\n")
                f.write("This analysis compares Test performance (out-of-sample)\n")
                f.write("against Train performance (in-sample) to assess generalization.\n\n")

                if "fitness_text" in stats:
                    f.write(stats["fitness_text"])
                    f.write("\n\n")

                if "sharpe_text" in stats:
                    f.write(stats["sharpe_text"])
                    f.write("\n")

        # 3. Per-run results CSV
        best_per_run = self.get_best_test_fitness_per_run()
        if best_per_run:
            df = pd.DataFrame(best_per_run)
            df.to_csv(os.path.join(output_dir, "per_run_results.csv"), index=False)

        # 4. Train vs Test comparison plot
        if best_per_run:
            self._plot_train_vs_test(output_dir, best_per_run)

        # 5. Confidence interval plot
        if best_per_run and len(best_per_run) >= 2:
            self._plot_confidence_intervals(output_dir)

        # 6. Generate individual run reports
        runs_dir = os.path.join(output_dir, "runs")
        os.makedirs(runs_dir, exist_ok=True)
        for run in self.successful_runs:
            if run.results:
                run_dir = os.path.join(runs_dir, f"fold{run.fold_idx}_seed{run.seed_idx}")
                run.results.generate_report(run_dir, self.config.objectives)

        print(f"Batch report generated in '{output_dir}'")

    def _plot_train_vs_test(self, output_dir: str, best_per_run: List[Dict]):
        """Plots train vs test fitness comparison across runs."""
        import matplotlib.pyplot as plt

        train = [r["train_fitness"] for r in best_per_run if r["train_fitness"] is not None]
        test = [r["test_fitness"] for r in best_per_run if r["test_fitness"] is not None]
        labels = [f"F{r['fold']}S{r['seed']}" for r in best_per_run]

        if not train or not test:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(train))
        width = 0.35

        ax.bar([i - width/2 for i in x], train, width, label='Train', alpha=0.8)
        ax.bar([i + width/2 for i in x], test, width, label='Test', alpha=0.8)

        ax.set_xlabel("Run (Fold/Seed)")
        ax.set_ylabel("Annualized Return (%)")
        ax.set_title("Train vs Test Performance Across Runs")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "train_vs_test_comparison.png"))
        plt.close(fig)

    def _plot_confidence_intervals(self, output_dir: str):
        """Plots confidence intervals for key metrics."""
        import matplotlib.pyplot as plt
        import numpy as np

        best_per_run = self.get_best_test_fitness_per_run()
        if not best_per_run or len(best_per_run) < 2:
            return

        train_fitness = [r["train_fitness"] for r in best_per_run if r["train_fitness"] is not None]
        test_fitness = [r["test_fitness"] for r in best_per_run if r["test_fitness"] is not None]
        train_sharpe = [r["train_sharpe"] for r in best_per_run if r["train_sharpe"] is not None]
        test_sharpe = [r["test_sharpe"] for r in best_per_run if r["test_sharpe"] is not None]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Fitness (Annualized Return)
        ax1 = axes[0]
        metrics = []
        means = []
        ci_errors = []

        if train_fitness:
            mean, ci_low, ci_high = compute_confidence_interval(train_fitness)
            metrics.append("Train")
            means.append(mean)
            ci_errors.append([mean - ci_low, ci_high - mean])

        if test_fitness:
            mean, ci_low, ci_high = compute_confidence_interval(test_fitness)
            metrics.append("Test")
            means.append(mean)
            ci_errors.append([mean - ci_low, ci_high - mean])

        if means:
            x = np.arange(len(metrics))
            ci_errors = np.array(ci_errors).T
            ax1.bar(x, means, yerr=ci_errors, capsize=5, alpha=0.7, color=['steelblue', 'darkorange'][:len(metrics)])
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics)
            ax1.set_ylabel("Annualized Return (%)")
            ax1.set_title("Fitness with 95% CI")
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax1.grid(True, axis='y', alpha=0.3)

        # Plot 2: Sharpe Ratio
        ax2 = axes[1]
        metrics = []
        means = []
        ci_errors = []

        if train_sharpe:
            mean, ci_low, ci_high = compute_confidence_interval(train_sharpe)
            metrics.append("Train")
            means.append(mean)
            ci_errors.append([mean - ci_low, ci_high - mean])

        if test_sharpe:
            mean, ci_low, ci_high = compute_confidence_interval(test_sharpe)
            metrics.append("Test")
            means.append(mean)
            ci_errors.append([mean - ci_low, ci_high - mean])

        if means:
            x = np.arange(len(metrics))
            ci_errors = np.array(ci_errors).T
            ax2.bar(x, means, yerr=ci_errors, capsize=5, alpha=0.7, color=['steelblue', 'darkorange'][:len(metrics)])
            ax2.set_xticks(x)
            ax2.set_xticklabels(metrics)
            ax2.set_ylabel("Sharpe Ratio")
            ax2.set_title("Sharpe Ratio with 95% CI")
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confidence_intervals.png"))
        plt.close(fig)


class BatchRunner:
    """
    Runs optimization across multiple fold/seed combinations.
    """

    def __init__(
        self,
        config: Config,
        folds: List[int],
        seeds: List[int],
        parallel: bool = False
    ):
        self.config = config
        self.folds = folds
        self.seeds = seeds
        self.parallel = parallel

    def run_single(self, fold_idx: int, seed_idx: int) -> RunResult:
        """Runs optimization for a single fold/seed combination."""
        print(f"\n{'='*60}")
        print(f"Running Fold {fold_idx}, Seed {seed_idx}")
        print('='*60)

        try:
            optimizer = Optimizer(self.config, fold_idx=fold_idx, seed_idx=seed_idx)
            results = optimizer.run()
            return RunResult(fold_idx=fold_idx, seed_idx=seed_idx, results=results)
        except Exception as e:
            import traceback
            print(f"Run failed: {e}")
            traceback.print_exc()
            return RunResult(fold_idx=fold_idx, seed_idx=seed_idx, results=None, error=str(e))

    def run_all(self) -> BatchResults:
        """Runs all fold/seed combinations."""
        total_runs = len(self.folds) * len(self.seeds)
        print(f"\n{'#'*60}")
        print(f"# Batch Run: {len(self.folds)} folds x {len(self.seeds)} seeds = {total_runs} runs")
        print(f"# Parallel: {self.parallel}")
        print('#'*60)

        if self.parallel:
            return self._run_parallel()
        else:
            return self._run_sequential()

    def _run_sequential(self) -> BatchResults:
        """Runs all combinations sequentially."""
        run_results = []
        total = len(self.folds) * len(self.seeds)
        current = 0

        for fold_idx in self.folds:
            for seed_idx in self.seeds:
                current += 1
                print(f"\n[{current}/{total}] ", end="")
                result = self.run_single(fold_idx, seed_idx)
                run_results.append(result)

        return BatchResults(run_results, self.config)

    def _run_parallel(self) -> BatchResults:
        """Runs all combinations in parallel using Ray."""
        try:
            import ray
        except ImportError:
            print("Ray not available, falling back to sequential execution")
            return self._run_sequential()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        @ray.remote
        def run_single_remote(config: Config, fold_idx: int, seed_idx: int) -> RunResult:
            """Remote function for parallel execution."""
            try:
                optimizer = Optimizer(config, fold_idx=fold_idx, seed_idx=seed_idx)
                results = optimizer.run()
                return RunResult(fold_idx=fold_idx, seed_idx=seed_idx, results=results)
            except Exception as e:
                return RunResult(fold_idx=fold_idx, seed_idx=seed_idx, results=None, error=str(e))

        # Submit all tasks
        futures = []
        for fold_idx in self.folds:
            for seed_idx in self.seeds:
                future = run_single_remote.remote(self.config, fold_idx, seed_idx)
                futures.append(future)

        # Collect results
        print(f"Submitted {len(futures)} parallel tasks...")
        run_results = ray.get(futures)

        return BatchResults(run_results, self.config)
