"""
Tests for the core strategy data structures.
"""

import pytest
from pydantic import ValidationError

from profit.strategy import Chromosome, Rule


def test_create_valid_rule():
    """
    Tests the successful creation of a Rule object with valid data.
    """
    rule = Rule(condition="c1 and c2", action="enter_long")
    assert rule.condition == "c1 and c2"
    assert rule.action == "enter_long"


def test_create_invalid_rule():
    """
    Tests that creating a Rule object with missing fields raises a validation error.
    """
    with pytest.raises(ValidationError):
        # Missing 'action'
        Rule(condition="c1 and c2")

    with pytest.raises(ValidationError):
        # Missing 'condition'
        Rule(action="enter_long")

    with pytest.raises(ValidationError):
        # Wrong data type
        Rule(condition=123, action="enter_long")


def test_create_valid_chromosome():
    """
    Tests the successful creation of a Chromosome object with valid data.
    """
    chromosome = Chromosome(
        parameters={"rsi_period": 14, "ma_period": 200},
        rules=[
            Rule(condition="c1 and c2", action="enter_long"),
            Rule(condition="c3", action="exit_long"),
        ],
        features={
            "c1": "rsi(14) > 70",
            "c2": "close > sma(close, 200)",
            "c3": "rsi(14) < 30",
        },
    )
    assert chromosome.parameters["ma_period"] == 200
    assert len(chromosome.rules) == 2
    assert chromosome.rules[0].action == "enter_long"
    assert "c1" in chromosome.features


def test_create_invalid_chromosome():
    """
    Tests that creating a Chromosome with invalid data raises a validation error.
    """
    # Missing 'features'
    with pytest.raises(ValidationError):
        Chromosome(
            parameters={"rsi_period": 14},
            rules=[Rule(condition="c1", action="enter_long")],
        )

    # Invalid type for 'rules'
    with pytest.raises(ValidationError):
        Chromosome(
            parameters={"rsi_period": 14},
            rules="not a list",
            features={"c1": "close > 100"},
        )

    # Invalid content in 'rules' list
    with pytest.raises(ValidationError):
        Chromosome(
            parameters={"rsi_period": 14},
            rules=[{"condition": "c1"}],  # Missing 'action' in rule
            features={"c1": "close > 100"},
        )
