"""
Tests for the configuration loading and validation logic.
"""
import os
import pytest
from pydantic import ValidationError
import yaml

from profit.io import load_config
from profit.config import Config, DataConfig, GAConfig, LLMConfig

# Path to the test YAML file
TEST_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')


def test_load_valid_config():
    """
    Tests that a valid YAML configuration file is loaded correctly into a
    Pydantic Config object.
    """
    config = load_config(TEST_CONFIG_PATH)

    assert isinstance(config, Config)
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.ga, GAConfig)
    assert isinstance(config.llm, LLMConfig)

    assert config.data.path == "./data/my_clean_btc_data.parquet"
    assert config.data.asset == "BTC-USD"
    assert config.ga.population_size == 100
    assert config.ga.generations == 50
    assert config.llm.client == "openai"
    assert config.llm.model == "gpt-4-turbo"
    assert config.objectives == ["sharpe_ratio", "annualized_return", "expectancy"]


def test_load_missing_file():
    """
    Tests that trying to load a non-existent file raises a FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_file.yaml")


def test_invalid_config_missing_field():
    """
    Tests that a configuration with a missing required field raises a
    ValidationError.
    """
    with open(TEST_CONFIG_PATH, 'r') as f:
        good_config = yaml.safe_load(f)

    # Remove a required field
    del good_config['data']

    # Create a temporary invalid config file
    invalid_path = "tests/invalid_config.yaml"
    with open(invalid_path, 'w') as f:
        yaml.dump(good_config, f)

    with pytest.raises(ValidationError):
        load_config(invalid_path)

    os.remove(invalid_path)


def test_invalid_config_wrong_type():
    """
    Tests that a config with an incorrect data type for a field raises a
    ValidationError.
    """
    with open(TEST_CONFIG_PATH, 'r') as f:
        good_config = yaml.safe_load(f)

    # Set a field to an incorrect type
    good_config['ga']['population_size'] = "one hundred" # Should be int

    # Create a temporary invalid config file
    invalid_path = "tests/invalid_config_type.yaml"
    with open(invalid_path, 'w') as f:
        yaml.dump(good_config, f)

    with pytest.raises(ValidationError):
        load_config(invalid_path)

    os.remove(invalid_path)


def test_ga_config_validation():
    """
    Tests that the validation constraints (e.g., gt=0) on GAConfig are enforced.
    """
    with pytest.raises(ValidationError):
        # Population size must be greater than 0
        GAConfig(population_size=0, generations=50)

    with pytest.raises(ValidationError):
        # Generations must be greater than 0
        GAConfig(population_size=100, generations=-10)
