"""
Tests for the Results class and report generation.
"""
import os
import shutil
import pytest

from profit.results import Results
from profit.ga.individual import Individual
from profit.strategy import Chromosome


@pytest.fixture
def sample_results() -> Results:
    """
    Provides a sample Results object for testing.
    """
    population = [
        Individual(
            chromosome=Chromosome(parameters={}, rules=[], features={}),
            fitness={"sharpe": 0.8, "return": 0.6, "expectancy": 0.2},
        ),
        Individual(
            chromosome=Chromosome(parameters={}, rules=[], features={}),
            fitness={"sharpe": 0.6, "return": 0.8, "expectancy": 0.3},
        ),
    ]
    history = [
        {"max_sharpe": 0.5, "avg_sharpe": 0.3, "max_return": 0.5, "avg_return": 0.2, "max_expectancy": 0.1, "avg_expectancy": 0.05},
        {"max_sharpe": 0.8, "avg_sharpe": 0.5, "max_return": 0.8, "avg_return": 0.4, "max_expectancy": 0.3, "avg_expectancy": 0.15},
    ]
    return Results(final_population=population, history=history)


def test_generate_report(sample_results):
    """
    Tests that `generate_report` creates the expected output files.
    """
    output_dir = "tests/temp_report"
    objectives = ["sharpe", "return", "expectancy"]
    
    # Ensure the directory is clean before the test
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    sample_results.generate_report(output_dir=output_dir, objectives=objectives)
    
    # Check for expected files
    assert os.path.exists(os.path.join(output_dir, "pareto_front.png"))
    assert os.path.exists(os.path.join(output_dir, "fitness_convergence.png"))
    assert os.path.exists(os.path.join(output_dir, "strategies", "strategy_1.txt"))
    
    # Clean up the directory after the test
    shutil.rmtree(output_dir)
