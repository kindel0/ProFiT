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
        
        # Initialize backtesters (one for each split if needed, but mainly Train for evolution)
        self.train_backtester = LucitBacktester(data=self.train_data)
        self.val_backtester = LucitBacktester(data=self.val_data)
        # Test backtester is used only at end
        self.test_backtester = LucitBacktester(data=self.test_data)
        
        # Initialize LLM Client
        client_name = config.llm.client if config.llm.client else "openai"
        self.llm_client = get_client(client_name, model=config.llm.model)

        self.population: List[Individual] = []
        self.history: List[Dict[str, Any]] = []
        self.mas: float = -100.0 # Minimum Acceptable Score

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
        for gen in range(self.config.ga.generations):
            print(f"\n--- Generation {gen+1}/{self.config.ga.generations} ---")
            
            # Select Parent (St)
            # Paper suggests various methods (roulette, etc.). Simple uniform or best for now.
            # "Since the population only contains strategies > MAS, various selection methods... including uniform."
            parent = random.choice(self.population)
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
                    fitness = metrics["annualized_return"]
                    
                    print(f"  Candidate Fitness: {fitness:.2f}% (MAS: {self.mas:.2f}%)")
                    
                    if fitness >= self.mas:
                        new_ind = Individual(chromosome=candidate_chrom, fitness=metrics)
                        print("  >>> Strategy Accepted!")
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
                
                # Record History
                self.history.append({
                    "generation": gen,
                    "fitness": new_ind.fitness["annualized_return"],
                    "sharpe": new_ind.fitness["sharpe_ratio"],
                    "parent_fitness": parent.fitness["annualized_return"]
                })

        return Results(final_population=self.population, history=self.history)