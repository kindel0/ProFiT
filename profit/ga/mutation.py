"""
Mutation operators for the Genetic Algorithm.
"""
import random
from copy import deepcopy
from typing import Any, Dict

from profit.llm.client import LLMClient
from profit.strategy import Chromosome


def mutate_with_llm(
    chromosome: Chromosome,
    performance_metrics: Dict[str, Any],
    llm_client: LLMClient,
) -> Chromosome:
    """
    Mutates a chromosome using suggestions from an LLM.

    Args:
        chromosome (Chromosome): The chromosome to mutate.
        performance_metrics (Dict[str, Any]): The performance of the strategy.
        llm_client (LLMClient): The LLM client to use for generating the patch.

    Returns:
        Chromosome: The mutated chromosome.
    """
    patch = llm_client.generate_patch(chromosome, performance_metrics)
    
    # Pydantic models are not directly mutable like dicts.
    # We convert to dict, apply patch, then parse back into a model.
    chromosome_dict = chromosome.model_dump()
    mutated_dict = patch.apply(chromosome_dict)
    
    return Chromosome.model_validate(mutated_dict)


def mutate(chromosome: Chromosome, mutation_rate: float) -> Chromosome:
    """
    Applies mutation to a chromosome.

    Each part of the chromosome (parameters, rules, features) has a chance
    to be mutated.

    Args:
        chromosome (Chromosome): The chromosome to mutate.
        mutation_rate (float): The probability of each gene being mutated.

    Returns:
        Chromosome: The mutated chromosome.
    """
    mutated_chromosome = deepcopy(chromosome)

    # Mutate parameters
    for key, value in mutated_chromosome.parameters.items():
        if random.random() < mutation_rate:
            mutated_chromosome.parameters[key] = _mutate_value(value)

    # Mutate rules
    for rule in mutated_chromosome.rules:
        if random.random() < mutation_rate:
            # For simplicity, we'll just swap a condition with a feature key
            rule.condition = random.choice(list(mutated_chromosome.features.keys()))
        if random.random() < mutation_rate:
            # And swap an action with a predefined list of actions
            rule.action = random.choice(["enter_long", "exit_long", "enter_short", "exit_short"])

    # Mutate features (simple string-based mutation)
    for key, value in mutated_chromosome.features.items():
        if random.random() < mutation_rate:
            mutated_chromosome.features[key] = _mutate_feature_string(value)
            
    return mutated_chromosome


def _mutate_value(value: Any) -> Any:
    """
    Mutates a single parameter value based on its type.
    """
    if isinstance(value, int):
        # Add or subtract a small integer value, ensuring it stays positive
        return max(1, value + random.randint(-3, 3))
    elif isinstance(value, float):
        # Apply a small perturbation
        return value * (1 + random.uniform(-0.1, 0.1))
    elif isinstance(value, str):
        # For strings, we can't do much without more context.
        # A real implementation might have a predefined list of choices.
        # For now, we'll just return the original value.
        return value
    # Add more type-specific mutations here if needed
    return value


def _mutate_feature_string(feature_str: str) -> str:
    """
    Performs a more robust mutation on a feature expression string.
    
    It identifies all possible mutation points (operators, digits) and
    randomly picks one to mutate.
    """
    replacements = {
        '>': '<', '<': '>',
        '+': '-', '-': '+',
        '*': '/', '/': '*',
    }
    
    mutable_indices = []
    for i, char in enumerate(feature_str):
        if char in replacements or char.isdigit():
            mutable_indices.append(i)
            
    if not mutable_indices:
        return feature_str # No place to mutate

    idx_to_mutate = random.choice(mutable_indices)
    char_to_mutate = feature_str[idx_to_mutate]

    if char_to_mutate in replacements:
        new_char = replacements[char_to_mutate]
    elif char_to_mutate.isdigit():
        new_char = str((int(char_to_mutate) + random.randint(1, 5)) % 10)
    else:
        # Should not happen given the logic above
        return feature_str

    return feature_str[:idx_to_mutate] + new_char + feature_str[idx_to_mutate+1:]
