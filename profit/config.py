"""
Configuration models for the ProFiT framework.

This module defines the Pydantic models for validating and managing the
framework's configuration, which is typically loaded from a YAML file.
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """
    Configuration for data loading.

    Args:
        path (str): The file path to the dataset (e.g., './data/my_data.parquet').
        asset (str): The specific asset to be used from the data
            (e.g., 'BTC-USD').
    """
    path: str = Field(..., description="Path to the dataset file.")
    asset: str = Field(..., description="The asset to be used from the data.")


class GAConfig(BaseModel):
    """
    Configuration for the Genetic Algorithm.

    Args:
        population_size (int): The number of individuals in the GA population.
        generations (int): The number of generations to run the evolution for.
    """
    population_size: int = Field(..., gt=0, description="Number of individuals in the population.")
    generations: int = Field(..., gt=0, description="Number of generations to run.")


class LLMConfig(BaseModel):
    """
    Configuration for the Large Language Model client.

    Args:
        client (str): The identifier for the LLM client to use (e.g., 'openai', 'ollama').
        model (str): The specific model name to be used (e.g., 'gpt-4-turbo', 'qwen2.5-coder:14b').
        base_url (Optional[str]): Custom API endpoint URL (e.g., for Ollama on a different host).
    """
    client: str = Field(..., description="Identifier for the LLM client.")
    model: str = Field(..., description="The specific LLM model to use.")
    base_url: Optional[str] = Field(None, description="Custom API endpoint URL.")


class Config(BaseModel):
    """
    Top-level configuration object for a ProFiT optimization run.

    Args:
        data (DataConfig): Data loading configuration.
        ga (GAConfig): Genetic Algorithm configuration.
        objectives (List[str]): A list of objective function names to optimize for.
        llm (LLMConfig): Language Model configuration.
    """
    data: DataConfig
    ga: GAConfig
    objectives: List[str]
    llm: LLMConfig
