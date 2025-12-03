"""
Tests for the Optimizer class.
"""
import pytest
import yaml
import pandas as pd

from profit.config import Config
from profit.optimizer import Optimizer
from profit.backtester.results import BacktestResult
from profit.strategy import Chromosome


@pytest.fixture
def mock_config() -> Config:
    """
    Provides a mock Config object for testing the optimizer.
    """
    yaml_config = """
data:
  path: "tests/data/long_sample_data.csv"
  asset: "BTC-USD"
ga:
  population_size: 10
  generations: 3
objectives:
  - "sharpe_ratio"
  - "annualized_return"
llm:
  client: "mock"
  model: "mock-model"
"""
    return Config.model_validate(yaml.safe_load(yaml_config))


def test_optimizer_run(mock_config, monkeypatch):
    """
    Tests the main `run` loop of the optimizer in an integration-like fashion,
    but with a mocked backtester for speed and predictability.
    """
    # Mock the backtester's run method
    def mock_backtest_run(self, chromosome: Chromosome):
        # Create a mock fitness score based on a chromosome's parameter.
        # This simulates "better" chromosomes getting better scores.
        fast_ma = chromosome.parameters.get("fast_ma", 10)
        return BacktestResult(
            trades=[],
            equity_curve=pd.Series([100000.0]),
            metrics={
                "sharpe_ratio": fast_ma / 10.0, # Better score for smaller fast_ma
                "annualized_return": 0.1, # Keep this one constant
            },
        )

    # We need to patch the path to the class used by the optimizer
    monkeypatch.setattr(
        "profit.optimizer.VectorizedBacktester.run",
        mock_backtest_run
    )
    
    optimizer = Optimizer(config=mock_config)
    results = optimizer.run()
    
    # Check the output
    assert results is not None
    assert len(results.final_population) == mock_config.ga.population_size
    
    # Check that the population has evolved. The average fitness should ideally
    # improve. We check the 'sharpe_ratio' which we designed to be evolvable.
    initial_pop_fitness = [
        ind.fitness.get("sharpe_ratio", 0) for ind in optimizer.initial_population
    ]
    final_pop_fitness = [
        ind.fitness.get("sharpe_ratio", 0) for ind in results.final_population
    ]

    # This is not a guarantee in a real GA, but with our mock backtester and
    # a few generations, the average fitness should increase.
    assert sum(final_pop_fitness) / len(final_pop_fitness) > sum(initial_pop_fitness) / len(initial_pop_fitness)
