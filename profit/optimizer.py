from typing import List, Dict, Any
import ray
from profit.backtester.base import BaseBacktester
from profit.backtester.results import BacktestResult
from profit.config import Config
from profit.data.provider import CSVProvider, ParquetProvider, DataProvider
from profit.ga.individual import Individual
from profit.llm.client import get_client
from profit.strategy import Chromosome, Rule
from profit.backtester.vectorized import VectorizedBacktester
from profit.ga import crossover, mutation, selection
from profit.results import Results
import random


@ray.remote
def _run_backtest_remote(backtester: BaseBacktester, chromosome: Chromosome) -> BacktestResult:
    """Remote function to run a backtest in a separate process."""
    return backtester.run(chromosome)


def _create_random_chromosome() -> Chromosome:
    """Creates a single random chromosome for the initial population."""
    # This is a placeholder. A real implementation would have more
    # sophisticated random generation of rules and features.
    fast_ma = random.randint(5, 20)
    slow_ma = random.randint(25, 50)
    return Chromosome(
        parameters={"fast_ma": fast_ma, "slow_ma": slow_ma},
        rules=[Rule(condition="c1", action="enter_long")],
        features={"c1": "sma(close, fast_ma) > sma(close, slow_ma)"},
    )


class Optimizer:
    """
    Manages the main training loop and orchestrates the GA process.
    """

    def __init__(self, config: Config):
        """
        Initializes the Optimizer.

        Args:
            config (Config): The configuration object for the run.
        """
        self.config = config
        self.data_provider: DataProvider
 
        # Initialize data provider
        if config.data.path.endswith(".parquet"):
            self.data_provider = ParquetProvider(path=config.data.path)
        else:
            self.data_provider = CSVProvider(path=config.data.path)
            
        self.data = self.data_provider.load()
        
        # Initialize backtester
        self.backtester = VectorizedBacktester(data=self.data)
        
        # Initialize LLM Client
        self.llm_client = get_client(config.llm.client) if config.llm.client else None

        self.initial_population: List[Individual] = []
        self.history: List[Dict[str, Any]] = []

    def _evaluate_population(self, population: List[Individual]):
        """
        Evaluates the fitness of each individual in the population in parallel.
        """
        # Put the backtester in the Ray object store to be accessed by remote workers
        backtester_ref = ray.put(self.backtester)
        
        # Launch remote backtests
        futures = [
            _run_backtest_remote.remote(backtester_ref, ind.chromosome)
            for ind in population
        ]
        
        # Retrieve results
        results = ray.get(futures)
        
        # Assign fitness scores
        for ind, result in zip(population, results):
            ind.fitness = result.metrics

    def run(self) -> Results:
        """
        Executes the main GA optimization loop.

        Returns:
            Results: An object containing the results of the optimization.
        """
        ray.init(ignore_reinit_error=True)
        
        try:
            # 1. Initialization
            self.initial_population = [
                Individual(chromosome=_create_random_chromosome())
                for _ in range(self.config.ga.population_size)
            ]
            self._evaluate_population(self.initial_population)
            
            population = self.initial_population
            
            for gen in range(self.config.ga.generations):
                print(f"Generation {gen+1}/{self.config.ga.generations}...")

                # Record history
                gen_history = {}
                for obj in self.config.objectives:
                    fitness_values = [ind.fitness.get(obj, 0) for ind in population]
                    gen_history[f"max_{obj}"] = max(fitness_values)
                    gen_history[f"avg_{obj}"] = sum(fitness_values) / len(fitness_values)
                self.history.append(gen_history)

                # 2. Selection
                parents = selection.select_nsga2(population, self.config.ga.population_size, self.config.objectives)
                
                # 3. Crossover & Mutation
                offspring: List[Individual] = []
                for i in range(0, len(parents), 2):
                    if i + 1 >= len(parents):
                        continue
                    p1, p2 = parents[i], parents[i+1]
                    
                    child1_chrom, child2_chrom = crossover.crossover_one_point(p1.chromosome, p2.chromosome)
                    
                    # Apply standard mutation
                    child1_chrom = mutation.mutate(child1_chrom, mutation_rate=0.1)
                    child2_chrom = mutation.mutate(child2_chrom, mutation_rate=0.1)
                    
                    offspring.extend([
                        Individual(chromosome=child1_chrom),
                        Individual(chromosome=child2_chrom),
                    ])
                
                # 4. Evaluate new offspring
                self._evaluate_population(offspring)
                
                # 5. Form new population for next generation
                combined_population = parents + offspring
                population = selection.select_nsga2(combined_population, self.config.ga.population_size, self.config.objectives)

            return Results(final_population=population, history=self.history)
        finally:
            ray.shutdown()
