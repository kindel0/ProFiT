import random
import pytest
import jsonpatch

from profit.ga import crossover, mutation
from profit.llm.client import MockLLMClient
from profit.strategy import Chromosome, Rule


@pytest.fixture
def sample_parents() -> tuple[Chromosome, Chromosome]:
    """
    Provides two sample parent Chromosomes for testing.
    """
    parent1 = Chromosome(
        parameters={"ma_period": 20, "rsi_period": 14, "use_volume": True},
        rules=[Rule(condition="c1", action="enter_long")],
        features={"c1": "rsi > 70"},
    )
    parent2 = Chromosome(
        parameters={"ma_period": 50, "rsi_period": 7, "use_volume": False},
        rules=[Rule(condition="c2", action="exit_long")],
        features={"c2": "rsi < 30"},
    )
    return parent1, parent2


def test_crossover_one_point(sample_parents):
    """
    Tests the one-point crossover operator.
    """
    parent1, parent2 = sample_parents
    random.seed(42) # for reproducible tests

    offspring1, offspring2 = crossover.crossover_one_point(parent1, parent2)

    # Check that offspring are different from parents
    assert offspring1.model_dump() != parent1.model_dump()
    assert offspring2.model_dump() != parent2.model_dump()
    
    # Check parameter crossover
    # Based on seed 42, the crossover point for params will be predictable
    assert offspring1.parameters["ma_period"] == 50
    assert offspring2.parameters["ma_period"] == 20

    # Check that offspring have valid structure
    assert isinstance(offspring1, Chromosome)
    assert isinstance(offspring2, Chromosome)


def test_mutate(sample_parents):
    """
    Tests the mutation operator.
    """
    parent1, _ = sample_parents
    random.seed(42)

    mutated_child = mutation.mutate(parent1, mutation_rate=1.0) # 100% mutation rate for testing

    # Check that mutation has occurred
    assert mutated_child.model_dump() != parent1.model_dump()

    # Check parameter mutation (numeric)
    assert mutated_child.parameters["rsi_period"] != parent1.parameters["rsi_period"]

    # Check feature mutation
    assert mutated_child.features["c1"] != parent1.features["c1"]

    # Test with zero mutation rate
    not_mutated_child = mutation.mutate(parent1, mutation_rate=0.0)
    assert not_mutated_child.model_dump() == parent1.model_dump()


def test_mutate_with_llm(sample_parents):
    """
    Tests the LLM-powered mutation.
    """
    parent1, _ = sample_parents
    
    # Define a patch to be returned by the mock client
    patch_data = [{"op": "replace", "path": "/parameters/ma_period", "value": 100}]
    patch = jsonpatch.JsonPatch(patch_data)
    mock_client = MockLLMClient(patch=patch)
    
    # Dummy performance metrics
    metrics = {"sharpe": 1.2}
    
    mutated_child = mutation.mutate_with_llm(parent1, metrics, mock_client)
    
    # Check that the mutation was applied correctly
    assert mutated_child.parameters["ma_period"] == 100
    
    # Check that other parts of the chromosome are unchanged
    assert mutated_child.parameters["rsi_period"] == parent1.parameters["rsi_period"]
    assert mutated_child.rules == parent1.rules
