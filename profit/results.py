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
    
    This object holds the final population of individuals and provides methods
    for analyzing and reporting on the results.
    
    Args:
        final_population (List[Individual]): The list of individuals in the
            last generation of the optimization.
        history (List[Dict[str, Any]]): A list of dictionaries, where each
            dictionary holds the fitness statistics for a generation.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    final_population: List[Individual]
    history: List[Dict[str, Any]]

    def generate_report(self, output_dir: str, objectives: List[str]):
        """
        Generates a collection of static report files (plots, tables) in the
        specified output directory.
        
        Args:
            output_dir (str): The path to the directory where the report
                files should be saved.
            objectives (List[str]): The list of objective names that were optimized.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Pareto Front Plot
        self._plot_pareto_front(output_dir, objectives)
        
        # 2. Fitness Convergence Plot
        self._plot_fitness_convergence(output_dir, objectives)
        
        # 3. Strategy Tearsheets (simplified)
        self._generate_tearsheets(output_dir)

    def _plot_pareto_front(self, output_dir: str, objectives: List[str]):
        """Plots the 3D Pareto front."""
        if len(objectives) != 3:
            print("Pareto front plot is only supported for 3 objectives.")
            return
            
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        x = [ind.fitness.get(objectives[0], 0) for ind in self.final_population]
        y = [ind.fitness.get(objectives[1], 0) for ind in self.final_population]
        z = [ind.fitness.get(objectives[2], 0) for ind in self.final_population]
        
        ax.scatter(x, y, z)
        ax.set_xlabel(objectives[0])
        ax.set_ylabel(objectives[1])
        ax.set_zlabel(objectives[2])
        ax.set_title("Pareto Front")
        
        plt.savefig(os.path.join(output_dir, "pareto_front.png"))
        plt.close(fig)

    def _plot_fitness_convergence(self, output_dir: str, objectives: List[str]):
        """Plots the convergence of fitness values over generations."""
        df_history = pd.DataFrame(self.history)
        
        fig, axes = plt.subplots(len(objectives), 1, figsize=(10, 6 * len(objectives)), sharex=True)
        if len(objectives) == 1:
            axes = [axes]

        for i, obj in enumerate(objectives):
            ax = axes[i]
            ax.plot(df_history.index, df_history[f"max_{obj}"], label=f"Max {obj}")
            ax.plot(df_history.index, df_history[f"avg_{obj}"], label=f"Avg {obj}")
            ax.set_ylabel("Fitness")
            ax.set_title(f"Convergence for {obj}")
            ax.legend()
            
        axes[-1].set_xlabel("Generation")
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, "fitness_convergence.png"))
        plt.close(fig)

    def _generate_tearsheets(self, output_dir: str, top_n: int = 5):
        """Generates simplified tearsheets for the top N strategies."""
        strategies_dir = os.path.join(output_dir, "strategies")
        os.makedirs(strategies_dir, exist_ok=True)
        
        # For simplicity, we just print the chromosome to a file.
        # A real implementation would plot the equity curve, drawdowns, etc.
        for i, ind in enumerate(self.final_population[:top_n]):
            filepath = os.path.join(strategies_dir, f"strategy_{i+1}.txt")
            with open(filepath, 'w') as f:
                f.write("Strategy Chromosome:\n")
                f.write(str(ind.chromosome.model_dump_json(indent=2)))
                f.write("\n\nFitness:\n")
                f.write(str(ind.fitness))

