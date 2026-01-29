"""
The Results object for storing and analyzing optimization runs.
"""
import os
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel, ConfigDict

from profit.ga.individual import Individual


class Results(BaseModel):
    """
    Represents the final results of an optimization run.

    Args:
        final_population (List[Individual]): The list of individuals in the
            last generation of the optimization.
        history (List[Dict[str, Any]]): A list of dictionaries, where each
            dictionary holds the fitness statistics for a generation (training metrics).
        validation_history (List[Dict[str, Any]]): Validation set metrics tracked
            during evolution for monitoring overfitting.
        test_results (List[Dict[str, Any]]): Final test set evaluation results
            for each individual in the final population.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    final_population: List[Individual]
    history: List[Dict[str, Any]]
    validation_history: List[Dict[str, Any]] = []
    test_results: List[Dict[str, Any]] = []

    def generate_report(self, output_dir: str, objectives: List[str]):
        """
        Generates a collection of static report files (plots, tables) in the
        specified output directory.
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. Fitness Growth Plot
        self._plot_fitness_growth(output_dir)

        # 2. Train vs Validation Plot (overfitting detection)
        self._plot_train_vs_val(output_dir)

        # 3. Save Test Results Table
        self._save_test_results(output_dir)

        # 4. Save Top Strategies
        self._save_top_strategies(output_dir)

    def _plot_fitness_growth(self, output_dir: str):
        """Plots the fitness of accepted strategies over time."""
        if not self.history:
            return

        df = pd.DataFrame(self.history)
        if "train_fitness" not in df.columns and "fitness" not in df.columns:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        # Support both old format (fitness) and new format (train_fitness)
        fitness_col = "train_fitness" if "train_fitness" in df.columns else "fitness"
        ax.plot(df.index, df[fitness_col], marker='o', linestyle='-')
        ax.set_xlabel("Accepted Mutation Event")
        ax.set_ylabel("Annualized Return (%)")
        ax.set_title("Fitness Growth (Training Set)")
        ax.grid(True)

        plt.savefig(os.path.join(output_dir, "fitness_growth.png"))
        plt.close(fig)

    def _plot_train_vs_val(self, output_dir: str):
        """Plots training vs validation fitness to detect overfitting."""
        if not self.validation_history:
            return

        df = pd.DataFrame(self.validation_history)
        if "train_fitness" not in df.columns or "val_fitness" not in df.columns:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["generation"], df["train_fitness"], marker='o', linestyle='-', label='Training')
        ax.plot(df["generation"], df["val_fitness"], marker='s', linestyle='--', label='Validation')
        ax.set_xlabel("Generation")
        ax.set_ylabel("Annualized Return (%)")
        ax.set_title("Training vs Validation Performance")
        ax.legend()
        ax.grid(True)

        plt.savefig(os.path.join(output_dir, "train_vs_val.png"))
        plt.close(fig)

    def _save_test_results(self, output_dir: str):
        """Saves test set evaluation results as a CSV table."""
        if not self.test_results:
            return

        df = pd.DataFrame(self.test_results)
        df.to_csv(os.path.join(output_dir, "test_results.csv"), index=False)

        # Also save a summary text file
        summary_path = os.path.join(output_dir, "test_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("=== Test Set Evaluation Summary ===\n\n")
            for i, result in enumerate(self.test_results):
                f.write(f"Strategy {i+1}:\n")
                for key, value in result.items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
                f.write("\n")

    def _save_top_strategies(self, output_dir: str, top_n: int = 5):
        """Save the code of top N strategies."""
        strategies_dir = os.path.join(output_dir, "strategies")
        os.makedirs(strategies_dir, exist_ok=True)
        
        # Sort by fitness (Annualized Return)
        sorted_pop = sorted(self.final_population, key=lambda ind: ind.fitness.get("annualized_return", -100), reverse=True)
        
        for i, ind in enumerate(sorted_pop[:top_n]):
            filepath = os.path.join(strategies_dir, f"strategy_{i+1}.py")
            with open(filepath, 'w') as f:
                f.write(f"# Fitness: {ind.fitness}\n")
                f.write(f"# Rank: {i+1}\n\n")
                f.write(ind.chromosome.code)