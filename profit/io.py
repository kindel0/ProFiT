"""
Input/Output operations for the ProFiT framework.

This module provides utility functions for loading and saving framework
objects, such as configurations.
"""

import yaml
from profit.config import Config


def load_config(path: str) -> Config:
    """
    Loads a YAML configuration file and parses it into a strongly-typed
    Config object.

    Args:
        path (str): The path to the YAML configuration file.

    Returns:
        Config: A Pydantic Config object with the validated configuration.
    """
    with open(path, 'r') as f:
        raw_config = yaml.safe_load(f)
    return Config(**raw_config)
