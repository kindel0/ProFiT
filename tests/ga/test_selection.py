"""
Tests for the NSGA-II selection algorithm.
"""
import pytest
from profit.ga import selection
from profit.ga.individual import Individual
from profit.strategy import Chromosome


@pytest.fixture
def sample_population() -> list[Individual]:
    """
    Provides a sample population of Individuals with 2 objectives to maximize.
    - p1 dominates p3
    - p2 dominates p3
    - p1 and p2 are non-dominated
    - p4 is dominated by all
    - p5 and p6 are non-dominated and on a better front than p1,p2
    """
    # Dummy chromosome, not important for these tests
    dummy_chrom = Chromosome(parameters={}, rules=[], features={})
    
    p1 = Individual(chromosome=dummy_chrom, fitness={"sharpe": 0.8, "return": 0.6})
    p2 = Individual(chromosome=dummy_chrom, fitness={"sharpe": 0.6, "return": 0.8})
    p3 = Individual(chromosome=dummy_chrom, fitness={"sharpe": 0.5, "return": 0.5})
    p4 = Individual(chromosome=dummy_chrom, fitness={"sharpe": 0.2, "return": 0.2})
    p5 = Individual(chromosome=dummy_chrom, fitness={"sharpe": 0.9, "return": 0.9})
    p6 = Individual(chromosome=dummy_chrom, fitness={"sharpe": 1.0, "return": 0.85})
    
    return [p1, p2, p3, p4, p5, p6]


OBJECTIVES = ["sharpe", "return"]


def test_dominates(sample_population):
    """
    Tests the `dominates` function.
    """
    p1, p2, p3, p4, p5, p6 = sample_population

    assert selection.dominates(p1, p3, OBJECTIVES)
    assert selection.dominates(p2, p3, OBJECTIVES)
    assert not selection.dominates(p1, p2, OBJECTIVES)
    assert not selection.dominates(p2, p1, OBJECTIVES)
    assert selection.dominates(p5, p1, OBJECTIVES)
    assert selection.dominates(p6, p2, OBJECTIVES)


def test_fast_non_dominated_sort(sample_population):
    """
    Tests that the population is correctly sorted into domination fronts.
    """
    p1, p2, p3, p4, p5, p6 = sample_population
    fronts = selection.fast_non_dominated_sort(sample_population, OBJECTIVES)

    assert len(fronts) == 4
    
    # Front 0 should contain p5 and p6
    assert set(fronts[0]) == {p5, p6}
    assert all(p.rank == 0 for p in fronts[0])

    # Front 1 should contain p1 and p2
    assert set(fronts[1]) == {p1, p2}
    assert all(p.rank == 1 for p in fronts[1])
    
    # Front 2 should contain p3
    assert set(fronts[2]) == {p3}
    assert all(p.rank == 2 for p in fronts[2])

    # Front 3 should contain p4
    assert set(fronts[3]) == {p4}
    assert all(p.rank == 3 for p in fronts[3])


def test_crowding_distance_assignment(sample_population):
    """
    Tests the calculation of crowding distance.
    """
    p1, p2, _, _, _, _ = sample_population
    front = [p1, p2] # A simple front with two individuals
    selection.crowding_distance_assignment(front, OBJECTIVES)

    # Boundary points should have infinite distance
    # The order depends on the objective used for sorting
    front.sort(key=lambda x: x.fitness[OBJECTIVES[0]])
    assert front[0].crowding_distance == float('inf')
    assert front[-1].crowding_distance == float('inf')


def test_select_nsga2(sample_population):
    """
    Tests the main NSGA-II selection process.
    """
    # Select the top 3 individuals
    selected = selection.select_nsga2(sample_population, 3, OBJECTIVES)
    p1, p2, p3, p4, p5, p6 = sample_population

    assert len(selected) == 3
    
    # It must select the entire best front (p5, p6)
    assert p5 in selected
    assert p6 in selected
    
    # It must select one from the second front (p1, p2)
    # The one with the larger crowding distance will be chosen.
    # Since there are only two points, they are both boundary points and have
    # infinite distance, so the choice between them is arbitrary.
    assert (p1 in selected) or (p2 in selected)
    
    # It must not select anyone from the worst fronts
    assert p3 not in selected
    assert p4 not in selected
