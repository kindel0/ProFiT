"""
Crossover operators for the Genetic Algorithm.
"""
import random
from copy import deepcopy
from typing import Tuple

from profit.strategy import Chromosome


def crossover_one_point(
    parent1: Chromosome,
    parent2: Chromosome,
) -> Tuple[Chromosome, Chromosome]:
    """
    Performs a one-point crossover on the parameters, rules, and features
    of two parent chromosomes.

    Args:
        parent1 (Chromosome): The first parent.
        parent2 (Chromosome): The second parent.

    Returns:
        Tuple[Chromosome, Chromosome]: A tuple containing two new offspring
        chromosomes.
    """
    offspring1 = deepcopy(parent1)
    offspring2 = deepcopy(parent2)

    # Crossover on parameters
    params1, params2 = _crossover_dict(offspring1.parameters, offspring2.parameters)
    offspring1.parameters, offspring2.parameters = params1, params2

    # Crossover on rules
    rules1, rules2 = _crossover_list(offspring1.rules, offspring2.rules)
    offspring1.rules, offspring2.rules = rules1, rules2

    # Crossover on features
    features1, features2 = _crossover_dict(offspring1.features, offspring2.features)
    offspring1.features, offspring2.features = features1, features2
    
    return offspring1, offspring2


def _crossover_dict(d1: dict, d2: dict) -> Tuple[dict, dict]:
    """Helper for one-point crossover on dictionaries."""
    new_d1, new_d2 = deepcopy(d1), deepcopy(d2)
    
    # Crossover on intersection of keys
    common_keys = sorted(list(set(d1.keys()) & set(d2.keys())))
    if len(common_keys) < 2:
        return new_d1, new_d2

    point = random.randint(1, len(common_keys) - 1)
    keys_to_swap = common_keys[:point]
    
    for key in keys_to_swap:
        new_d1[key], new_d2[key] = new_d2[key], new_d1[key]
        
    return new_d1, new_d2


def _crossover_list(l1: list, l2: list) -> Tuple[list, list]:
    """Helper for one-point crossover on lists."""
    if not l1 or not l2:
        return l1, l2

    min_len = min(len(l1), len(l2))
    if min_len < 2:
        return l1, l2
        
    point = random.randint(1, min_len - 1)
    
    new_l1 = l1[:point] + l2[point:]
    new_l2 = l2[:point] + l1[point:]

    return new_l1, new_l2
