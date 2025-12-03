"""
Selection operators for the Genetic Algorithm, including NSGA-II.
"""
from typing import List, Dict

from profit.ga.individual import Individual


def dominates(ind1: Individual, ind2: Individual, objectives: List[str]) -> bool:
    """
    Checks if individual 1 dominates individual 2.

    An individual `ind1` dominates `ind2` if it is at least as good on all
    objectives and strictly better on at least one objective. Assumes all
    objectives are to be maximized.

    Args:
        ind1 (Individual): The first individual.
        ind2 (Individual): The second individual.
        objectives (List[str]): The list of objective names.

    Returns:
        bool: True if ind1 dominates ind2, False otherwise.
    """
    is_better = [ind1.fitness[obj] >= ind2.fitness[obj] for obj in objectives]
    is_strictly_better = [ind1.fitness[obj] > ind2.fitness[obj] for obj in objectives]
    return all(is_better) and any(is_strictly_better)


def fast_non_dominated_sort(
    population: List[Individual],
    objectives: List[str],
) -> List[List[Individual]]:
    """
    Sorts a population into fronts based on non-domination.

    Args:
        population (List[Individual]): The population to sort.
        objectives (List[str]): The list of objective names.

    Returns:
        List[List[Individual]]: A list of fronts, where each front is a list
        of non-dominated individuals.
    """
    fronts: List[List[Individual]] = [[]]
    domination_info: Dict[int, Dict] = {
        id(p): {"n": 0, "S": []} for p in population
    }

    for p in population:
        for q in population:
            if p == q:
                continue
            if dominates(p, q, objectives):
                domination_info[id(p)]["S"].append(q)
            elif dominates(q, p, objectives):
                domination_info[id(p)]["n"] += 1
        
        if domination_info[id(p)]["n"] == 0:
            p.rank = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: List[Individual] = []
        for p in fronts[i]:
            for q in domination_info[id(p)]["S"]:
                domination_info[id(q)]["n"] -= 1
                if domination_info[id(q)]["n"] == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
        
    return fronts[:-1] # The last front is always empty


def crowding_distance_assignment(front: List[Individual], objectives: List[str]):
    """
    Calculates the crowding distance for each individual in a front.

    Args:
        front (List[Individual]): A list of individuals in the same front.
        objectives (List[str]): The list of objective names.
    """
    if not front:
        return

    for ind in front:
        ind.crowding_distance = 0.0

    for obj in objectives:
        front.sort(key=lambda x: x.fitness[obj])
        
        min_val = front[0].fitness[obj]
        max_val = front[-1].fitness[obj]
        
        if max_val == min_val:
            continue # Avoid division by zero

        # Assign infinite distance to boundary individuals
        front[0].crowding_distance = float('inf')
        front[-1].crowding_distance = float('inf')
        
        # Calculate for intermediate individuals
        for i in range(1, len(front) - 1):
            distance = front[i+1].fitness[obj] - front[i-1].fitness[obj]
            front[i].crowding_distance += distance / (max_val - min_val)


def select_nsga2(
    population: List[Individual],
    num_to_select: int,
    objectives: List[str],
) -> List[Individual]:
    """
    Selects the best individuals from a population using the NSGA-II algorithm.

    Args:
        population (List[Individual]): The population to select from.
        num_to_select (int): The number of individuals to select.
        objectives (List[str]): The list of objective names.

    Returns:
        List[Individual]: A new list of selected individuals.
    """
    fronts = fast_non_dominated_sort(population, objectives)
    
    new_population: List[Individual] = []
    front_idx = 0
    
    while len(new_population) + len(fronts[front_idx]) <= num_to_select:
        crowding_distance_assignment(fronts[front_idx], objectives)
        new_population.extend(fronts[front_idx])
        front_idx += 1
        if front_idx >= len(fronts):
            break

    if len(new_population) < num_to_select and front_idx < len(fronts):
        crowding_distance_assignment(fronts[front_idx], objectives)
        # Sort by crowding distance (descending) to get the most diverse solutions
        fronts[front_idx].sort(key=lambda x: x.crowding_distance, reverse=True)
        num_needed = num_to_select - len(new_population)
        new_population.extend(fronts[front_idx][:num_needed])
        
    return new_population
