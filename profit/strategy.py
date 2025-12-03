"""
Core data structures for representing a trading strategy's genetic makeup.

This module defines the Pydantic models that constitute a "Chromosome", which is
the complete definition of a trading strategy. This structure is what the
Genetic Algorithm will operate on.
"""
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class Rule(BaseModel):
    """
    Represents a single trading rule, combining a condition and a resulting action.

    Args:
        condition (str): A boolean expression of features (e.g., 'c1 and c2').
        action (str): The action to take if the condition is true (e.g.,
            'enter_long').
    """
    condition: str = Field(
        ...,
        description="A boolean expression of features (e.g., 'c1 and c2')."
    )
    action: str = Field(
        ...,
        description="The action to take if the condition is true (e.g., 'enter_long')."
    )


class Chromosome(BaseModel):
    """
    Represents the complete genetic makeup of a trading strategy.

    This multi-part structure allows for complex evolution, where each part can
    be mutated or crossed over independently.

    Args:
        parameters (Dict[str, Any]): A dictionary of tunable parameters for
            pre-defined logic, such as indicator periods (e.g.,
            `{"rsi_period": 14, "ma_period": 200}`).
        rules (List[Rule]): A list of rule objects that define the strategy's
            entry and exit logic.
        features (Dict[str, str]): A dictionary where keys are feature names
            (e.g., `c1`) and values are expression strings that define how the
            feature is calculated from OHLCV data. This allows the GP engine to
            invent novel features. For example: `{"c1": "close > sma(close, 200)"}`.
    """
    parameters: Dict[str, Any] = Field(
        ...,
        description="Parameters for pre-defined logic (e.g., {'rsi_period': 14})."
    )
    rules: List[Rule] = Field(
        ...,
        description="A list of trading rules."
    )
    features: Dict[str, str] = Field(
        ...,
        description="Feature definitions as expression strings (e.g., {'c1': 'close > sma(close, 200)'})."
    )

    @classmethod
    def from_json(cls, json_str: str) -> "Chromosome":
        """
        Creates a Chromosome instance from a JSON string.

        Args:
            json_str (str): A JSON string representing a Chromosome.

        Returns:
            Chromosome: A Chromosome instance.
        """
        return cls.model_validate_json(json_str)

