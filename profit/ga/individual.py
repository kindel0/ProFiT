"""
Data structure for representing an individual in the population.
"""
from typing import Dict

from profit.strategy import Chromosome
from pydantic import BaseModel, Field


class Individual(BaseModel):
    """
    Represents a single individual in the population, which includes its
    genetic material (chromosome) and its calculated fitness scores.

    Args:
        chromosome (Chromosome): The chromosome defining the strategy.
        fitness (Dict[str, float]): A dictionary of fitness scores, where keys
            are the names of the objectives.
    """
    chromosome: Chromosome
    fitness: Dict[str, float] = Field(default_factory=dict)
    
    # Attributes for NSGA-II
    rank: int = 0
    crowding_distance: float = 0.0
    
    def __hash__(self):
        # Make individual hashable for use in sets
        return id(self)

    def __eq__(self, other):
        # Make individual comparable for use in sets
        return id(self) == id(other)
