"""
Core data structures for representing a trading strategy's genetic makeup.
"""
from pydantic import BaseModel, Field


class Chromosome(BaseModel):
    """
    Represents the complete genetic makeup of a trading strategy as Python source code.

    Args:
        code (str): The valid Python source code defining the strategy class.
    """
    code: str = Field(
        ...,
        description="The valid Python source code defining the strategy class."
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