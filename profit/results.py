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
            dictionary holds the fitness statistics for a generation.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    final_population: List[Individual]
    history: List[Dict[str, Any]]

    def generate_report(self, output_dir: str, objectives: List[str]):
        """
        Generates a collection of static report files (plots, tables) in the
        specified output directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Fitness Growth Plot
        self._plot_fitness_growth(output_dir)
        
        # 2. Save Top Strategies
        self._save_top_strategies(output_dir)

    def _plot_fitness_growth(self, output_dir: str):
        """Plots the fitness of accepted strategies over time."""
        if not self.history:
            return
            
        df = pd.DataFrame(self.history)
        if "fitness" not in df.columns:
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index, df["fitness"], marker='o', linestyle='-')
        ax.set_xlabel("Accepted Mutation Event")
        ax.set_ylabel("Annualized Return (%)")
        ax.set_title("Fitness Growth")
        ax.grid(True)
        
        plt.savefig(os.path.join(output_dir, "fitness_growth.png"))
        plt.close(fig)

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