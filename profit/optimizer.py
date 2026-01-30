import random
import traceback
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime

from profit.backtester.base import BaseBacktester
from profit.backtester.lucit_adapter import LucitBacktester
from profit.config import Config
from profit.data.provider import CSVProvider, ParquetProvider, DataProvider
from profit.ga.individual import Individual
from profit.ga.selection import select_nsga2, dominates, crowding_distance_assignment, fast_non_dominated_sort
from profit.llm.client import get_client
from profit.strategy import Chromosome
from profit.results import Results
from profit.seed_strategies import ALL_SEEDS

# Table 1 from the paper
FOLDS = [
    {
        "train": ("2008-01-02", "2010-06-30"),
        "val": ("2010-06-30", "2011-01-11"),
        "test": ("2011-01-21", "2011-07-25")
    },
    {
        "train": ("2011-07-25", "2014-01-20"),
        "val": ("2014-01-20", "2014-08-03"),
        "test": ("2014-08-13", "2015-02-14")
    },
    {
        "train": ("2015-02-14", "2017-08-13"),
        "val": ("2017-08-13", "2018-02-24"),
        "test": ("2018-03-06", "2018-09-07")
    },
    {
        "train": ("2018-09-07", "2021-03-05"),
        "val": ("2021-03-05", "2021-09-16"),
        "test": ("2021-09-26", "2022-03-30")
    },
    {
        "train": ("2022-03-30", "2024-09-25"),
        "val": ("2024-09-25", "2025-04-08"),
        "test": ("2025-04-18", "2025-10-20")
    },
    # Demo Fold for available sample data
    {
        "train": ("2024-10-30", "2025-06-30"),
        "val": ("2025-07-01", "2025-09-30"),
        "test": ("2025-10-01", "2025-12-02")
    }
]

class Optimizer:
    """
    Manages the ProFiT evolutionary loop.
    """

    def __init__(self, config: Config, fold_idx: int = 0, seed_idx: int = 0):
        """
        Initializes the Optimizer.

        Args:
            config (Config): The configuration object.
            fold_idx (int): The fold index (0-4).
            seed_idx (int): The seed strategy index (0-4).
        """
        self.config = config
        self.fold_idx = fold_idx
        self.seed_idx = seed_idx
        
        # Initialize data provider
        if config.data.path.endswith(".parquet"):
            self.data_provider = ParquetProvider(path=config.data.path)
        else:
            self.data_provider = CSVProvider(path=config.data.path)
            
        self.full_data = self.data_provider.load()
        
        # Slice Data for this Fold
        self.train_data, self.val_data, self.test_data = self._slice_data(self.full_data, fold_idx)

        # Get fold date boundaries for warm-up periods
        fold = FOLDS[fold_idx]
        val_start = fold["val"][0]
        test_start = fold["test"][0]

        # Initialize backtesters with warm-up data for proper indicator initialization
        # Training: just train data (indicators warm up during training)
        self.train_backtester = LucitBacktester(data=self.train_data)

        # Validation: train + val data, but only measure performance from val_start
        # This allows indicators with long lookbacks to properly initialize
        train_val_data = pd.concat([self.train_data, self.val_data])
        self.val_backtester = LucitBacktester(data=train_val_data, eval_start=val_start)

        # Test: train + val + test data, but only measure performance from test_start
        train_val_test_data = pd.concat([self.train_data, self.val_data, self.test_data])
        self.test_backtester = LucitBacktester(data=train_val_test_data, eval_start=test_start)
        
        # Initialize LLM Client
        client_name = config.llm.client if config.llm.client else "openai"
        self.llm_client = get_client(client_name, model=config.llm.model)

        self.population: List[Individual] = []
        self.history: List[Dict[str, Any]] = []
        self.validation_history: List[Dict[str, Any]] = []
        self.mas: float = -100.0  # Minimum Acceptable Score

    def _slice_data(self, data: pd.DataFrame, fold_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Slices the data into Train, Val, Test based on fold definition."""
        if fold_idx < 0 or fold_idx >= len(FOLDS):
            raise ValueError(f"Invalid fold index {fold_idx}. Must be 0-4.")
            
        fold = FOLDS[fold_idx]
        
        train = data.loc[fold["train"][0] : fold["train"][1]]
        val = data.loc[fold["val"][0] : fold["val"][1]]
        test = data.loc[fold["test"][0] : fold["test"][1]]
        
        return train, val, test

    def _evaluate(self, backtester: LucitBacktester, chromosome: Chromosome) -> Dict[str, float]:
        """Runs backtest and returns metrics."""
        result = backtester.run(chromosome)
        return result.metrics

    def _evaluate_on_validation(self, individual: Individual) -> Dict[str, float]:
        """Evaluates an individual on the validation set."""
        return self._evaluate(self.val_backtester, individual.chromosome)

    def _evaluate_on_test(self, individual: Individual) -> Dict[str, float]:
        """Evaluates an individual on the test set."""
        return self._evaluate(self.test_backtester, individual.chromosome)

    def _get_best_individual(self) -> Individual:
        """Returns the best individual in the population.

        For single objective: returns individual with highest annualized_return.
        For multi-objective: returns a random individual from the first Pareto front.
        """
        if len(self.config.objectives) == 1:
            return max(self.population, key=lambda ind: ind.fitness.get("annualized_return", -100))
        else:
            # Multi-objective: return random from first Pareto front
            fronts = fast_non_dominated_sort(self.population, self.config.objectives)
            if fronts and fronts[0]:
                return random.choice(fronts[0])
            return self.population[0]

    def _is_multi_objective(self) -> bool:
        """Returns True if using multi-objective optimization."""
        return len(self.config.objectives) > 1

    def _select_parent(self) -> Individual:
        """Selects a parent for mutation.

        Single objective: uniform random selection from population.
        Multi-objective: binary tournament using Pareto rank and crowding distance.
        """
        if not self._is_multi_objective():
            return random.choice(self.population)

        # Multi-objective: binary tournament selection
        return self._tournament_select()

    def _tournament_select(self, tournament_size: int = 2) -> Individual:
        """Binary tournament selection using NSGA-II criteria.

        Compares individuals by:
        1. Pareto rank (lower is better)
        2. Crowding distance (higher is better, for diversity)
        """
        # Ensure ranks and crowding distances are computed
        fronts = fast_non_dominated_sort(self.population, self.config.objectives)
        for front in fronts:
            crowding_distance_assignment(front, self.config.objectives)

        # Select tournament_size random individuals
        candidates = random.sample(self.population, min(tournament_size, len(self.population)))

        # Compare and return the best
        def is_better(ind1: Individual, ind2: Individual) -> bool:
            """Returns True if ind1 is better than ind2 by NSGA-II criteria."""
            if ind1.rank != ind2.rank:
                return ind1.rank < ind2.rank
            return ind1.crowding_distance > ind2.crowding_distance

        best = candidates[0]
        for candidate in candidates[1:]:
            if is_better(candidate, best):
                best = candidate
        return best

    def _should_accept(self, candidate: Individual, parent: Individual) -> bool:
        """Determines if a candidate should be accepted into the population.

        Single objective: accept if fitness >= MAS (Minimum Acceptable Score).
        Multi-objective: accept if candidate is non-dominated by any existing individual,
                        or if it dominates the parent.
        """
        if not self._is_multi_objective():
            return candidate.fitness["annualized_return"] >= self.mas

        # Multi-objective acceptance criteria:
        # Accept if the candidate is not dominated by everyone in the population
        # (i.e., it would be in some Pareto front, not completely dominated)
        dominated_by_all = all(
            dominates(ind, candidate, self.config.objectives)
            for ind in self.population
        )
        return not dominated_by_all

    def _should_accept_with_validation(
        self, candidate: Individual, parent: Individual, max_overfit_ratio: float = 3.0
    ) -> Tuple[bool, Optional[Dict[str, float]], str]:
        """Determines if a candidate should be accepted, considering validation performance.

        This prevents severe overfitting by rejecting strategies where:
        1. Training return is positive but validation return is very negative
        2. The gap between train and validation exceeds max_overfit_ratio

        Args:
            candidate: The candidate individual to evaluate
            parent: The parent individual (for reference)
            max_overfit_ratio: Maximum allowed ratio of |train - val| / |train|

        Returns:
            Tuple of (should_accept, validation_metrics, rejection_reason)
        """
        # First check basic acceptance criteria
        basic_accept = self._should_accept(candidate, parent)
        if not basic_accept:
            return False, None, "dominated" if self._is_multi_objective() else "below_mas"

        # Evaluate on validation set
        try:
            val_metrics = self._evaluate_on_validation(candidate)
        except Exception as e:
            # If validation fails, reject the strategy
            return False, None, f"validation_error: {e}"

        train_return = candidate.fitness.get("annualized_return", 0)
        val_return = val_metrics.get("annualized_return", 0)

        # Check for severe overfitting patterns
        # Pattern 1: Positive train, significantly negative validation
        if train_return > 5 and val_return < -20:
            return False, val_metrics, "overfit_positive_train_negative_val"

        # Pattern 2: Large train-validation gap relative to train magnitude
        if abs(train_return) > 1:  # Avoid division issues with near-zero returns
            gap_ratio = abs(train_return - val_return) / abs(train_return)
            if gap_ratio > max_overfit_ratio and val_return < 0:
                return False, val_metrics, f"overfit_gap_ratio_{gap_ratio:.1f}"

        # Pattern 3: Strategy doesn't trade (0% return on both)
        # This is acceptable but we track it
        if train_return == 0 and val_return == 0:
            # Accept but note it's a non-trading strategy
            pass

        return True, val_metrics, "accepted"

    def _evaluate_final_test(self) -> List[Dict[str, Any]]:
        """Evaluates all individuals in the final population on the test set."""
        test_results = []

        # Sort population by training fitness (best first)
        sorted_pop = sorted(
            self.population,
            key=lambda ind: ind.fitness.get("annualized_return", -100),
            reverse=True
        )

        for i, ind in enumerate(sorted_pop):
            try:
                test_metrics = self._evaluate_on_test(ind)

                # Get validation metrics (cached or evaluate)
                val_fitness = None
                val_sharpe = None
                if ind.validation_fitness:
                    val_fitness = ind.validation_fitness.get("annualized_return")
                    val_sharpe = ind.validation_fitness.get("sharpe_ratio")

                result = {
                    "rank": i + 1,
                    "train_fitness": ind.fitness["annualized_return"],
                    "train_sharpe": ind.fitness["sharpe_ratio"],
                    "val_fitness": val_fitness,
                    "val_sharpe": val_sharpe,
                    "test_fitness": test_metrics["annualized_return"],
                    "test_sharpe": test_metrics["sharpe_ratio"],
                    "test_expectancy": test_metrics.get("expectancy", 0.0)
                }
                test_results.append(result)

                val_str = f"{val_fitness:.2f}%" if val_fitness is not None else "N/A"
                print(f"  Strategy {i+1}: Train={ind.fitness['annualized_return']:.2f}%, Val={val_str}, Test={test_metrics['annualized_return']:.2f}%")
            except Exception as e:
                print(f"  Strategy {i+1}: Test evaluation failed - {e}")
                test_results.append({
                    "rank": i + 1,
                    "train_fitness": ind.fitness["annualized_return"],
                    "train_sharpe": ind.fitness["sharpe_ratio"],
                    "val_fitness": None,
                    "val_sharpe": None,
                    "test_fitness": None,
                    "test_sharpe": None,
                    "test_expectancy": None,
                    "error": str(e)
                })

        return test_results

    def run(self) -> Results:
        """
        Executes the ProFiT Evolutionary Loop (Algorithm 1).
        """
        # 1. Initialize Seed Strategy S0
        seed_code = ALL_SEEDS[self.seed_idx]
        seed_chromosome = Chromosome(code=seed_code)
        
        print(f"Initializing with Seed Strategy {self.seed_idx} on Fold {self.fold_idx}...")
        
        # Compute baseline annualized return P0 on TRAIN data
        try:
            metrics = self._evaluate(self.train_backtester, seed_chromosome)
            p0 = metrics["annualized_return"]
        except Exception as e:
            print(f"Seed strategy failed: {e}")
            traceback.print_exc()
            return Results(final_population=[], history=[])

        seed_ind = Individual(chromosome=seed_chromosome, fitness=metrics)
        
        # 2. Set MAS <- P0
        self.mas = p0
        print(f"Baseline Fitness (MAS): {self.mas:.2f}%")
        
        # 3. Add seed to population
        self.population = [seed_ind]
        
        # 4. Evolutionary Loop
        multi_obj_mode = self._is_multi_objective()
        if multi_obj_mode:
            print(f"Multi-objective mode: optimizing {self.config.objectives}")

        for gen in range(self.config.ga.generations):
            print(f"\n--- Generation {gen+1}/{self.config.ga.generations} ---")

            # Select Parent (St)
            parent = self._select_parent()
            if multi_obj_mode:
                obj_str = ", ".join(f"{obj}={parent.fitness.get(obj, 0):.2f}" for obj in self.config.objectives)
                print(f"Selected parent: {obj_str}")
            else:
                print("Selected parent with fitness: {:.2f}%".format(parent.fitness["annualized_return"]))
            
            # LLM Analysis
            print("Requesting LLM Analysis...")
            analysis = self.llm_client.analyze_strategy(parent.chromosome, parent.fitness)
            
            # LLM Improvement
            print("Requesting LLM Improvement...")
            improved_code = self.llm_client.improve_strategy(parent.chromosome, analysis)
            
            # Repair Loop
            new_ind: Optional[Individual] = None
            
            for attempt in range(10): # up to 10 attempts
                print(f"  Repair Attempt {attempt+1}/10...")
                try:
                    # Try to compile and backtest
                    candidate_chrom = Chromosome(code=improved_code)
                    
                    # Backtest on TRAIN
                    metrics = self._evaluate(self.train_backtester, candidate_chrom)
                    candidate_ind = Individual(chromosome=candidate_chrom, fitness=metrics)

                    if multi_obj_mode:
                        obj_str = ", ".join(f"{obj}={metrics.get(obj, 0):.2f}" for obj in self.config.objectives)
                        print(f"  Candidate: {obj_str}")
                    else:
                        print(f"  Candidate Fitness: {metrics['annualized_return']:.2f}% (MAS: {self.mas:.2f}%)")

                    # Use validation-aware acceptance to prevent overfitting
                    accepted, val_metrics, reason = self._should_accept_with_validation(
                        candidate_ind, parent
                    )

                    if accepted:
                        new_ind = candidate_ind
                        # Store validation metrics with the individual for tracking
                        if val_metrics:
                            new_ind.validation_fitness = val_metrics
                        print("  >>> Strategy Accepted!")
                        if val_metrics:
                            print(f"      Validation: {val_metrics['annualized_return']:.2f}%")
                    else:
                        if reason.startswith("overfit"):
                            val_ret = val_metrics["annualized_return"] if val_metrics else "N/A"
                            print(f"  >>> Strategy Rejected (overfitting detected)")
                            print(f"      Train: {metrics['annualized_return']:.2f}%, Val: {val_ret}%")
                        elif multi_obj_mode:
                            print("  >>> Strategy Rejected (dominated by population).")
                        else:
                            print("  >>> Strategy Rejected (Fitness < MAS).")
                    
                    # Break loop if compiled and ran successfully (even if rejected)
                    # Wait, paper says: "If the LLM produces non-functional code... repair loop continues."
                    # If it produces functional code but low fitness, it is discarded (step 18).
                    # So we break here.
                    break
                    
                except Exception as e:
                    # Failure: Prompt LLM to fix
                    print(f"  Backtest/Compile Failed: {e}")
                    tb = traceback.format_exc()
                    improved_code = self.llm_client.repair_strategy(improved_code, tb)
            
            # Add to population if valid and good
            if new_ind:
                self.population.append(new_ind)

                # Trim population if it exceeds max size (using NSGA-II selection)
                max_pop_size = self.config.ga.population_size
                if len(self.population) > max_pop_size:
                    self.population = select_nsga2(
                        self.population,
                        max_pop_size,
                        self.config.objectives
                    )
                    print(f"  Population trimmed to {max_pop_size} (NSGA-II selection)")

                # Record History (training metrics)
                self.history.append({
                    "generation": gen,
                    "train_fitness": new_ind.fitness["annualized_return"],
                    "train_sharpe": new_ind.fitness["sharpe_ratio"],
                    "parent_fitness": parent.fitness["annualized_return"]
                })

            # Track validation metrics for the best strategy (for overfitting monitoring)
            best_ind = self._get_best_individual()
            # Use cached validation if available, otherwise evaluate
            if best_ind.validation_fitness:
                val_metrics = best_ind.validation_fitness
            else:
                try:
                    val_metrics = self._evaluate_on_validation(best_ind)
                except Exception as e:
                    print(f"  Validation evaluation failed: {e}")
                    val_metrics = None

            if val_metrics:
                self.validation_history.append({
                    "generation": gen,
                    "train_fitness": best_ind.fitness["annualized_return"],
                    "train_sharpe": best_ind.fitness["sharpe_ratio"],
                    "val_fitness": val_metrics["annualized_return"],
                    "val_sharpe": val_metrics["sharpe_ratio"]
                })
                print(f"  Best: Train={best_ind.fitness['annualized_return']:.2f}%, Val={val_metrics['annualized_return']:.2f}%")

        # Final test set evaluation
        print("\n--- Final Test Set Evaluation ---")
        test_results = self._evaluate_final_test()

        return Results(
            final_population=self.population,
            history=self.history,
            validation_history=self.validation_history,
            test_results=test_results
        )