import pandas as pd
import pytest

from profit.strategy import Chromosome
from profit.backtester.vectorized import VectorizedBacktester, BacktestResult
from profit.backtester.expression_parser import parse_expression, Literal, Variable, FunctionCall, BinaryOperation


# Sample data for testing
@pytest.fixture
def sample_data():
    dates = pd.to_datetime([f"2023-01-{i:02d}" for i in range(1, 21)])
    data = pd.DataFrame({
        "open": [i + 100 for i in range(20)],
        "high": [i + 102 for i in range(20)],
        "low": [i + 98 for i in range(20)],
        "close": [i + 101 for i in range(20)],
        "volume": [i * 100 for i in range(20)],
    }, index=dates)
    return data

@pytest.fixture
def mock_backtester(sample_data):
    return VectorizedBacktester(
        data=sample_data,
        initial_equity=10000.0
    )

# --- Tests for Chromosome.from_json ---
def test_chromosome_from_json_example1():
    json_str = """
    {
      "parameters": {
        "fast_ma": 10,
        "slow_ma": 30,
        "rsi_period": 14
      },
      "rules": [
        {
          "condition": "signal_long and not overbought",
          "action": "enter_long"
        },
        {
          "condition": "signal_short and not oversold",
          "action": "enter_short"
        },
        {
          "condition": "take_profit or stop_loss",
          "action": "exit_position"
        }
      ],
      "features": {
        "signal_long": "sma(close, fast_ma) > sma(close, slow_ma)",
        "signal_short": "sma(close, fast_ma) < sma(close, slow_ma)",
        "overbought": "rsi(close, rsi_period) > 70",
        "oversold": "rsi(close, rsi_period) < 30",
        "take_profit": "current_profit_percentage > 0.05",
        "stop_loss": "current_loss_percentage > 0.03"
      }
    }
    """
    chromosome = Chromosome.from_json(json_str)
    assert chromosome.parameters == {"fast_ma": 10, "slow_ma": 30, "rsi_period": 14}
    assert len(chromosome.rules) == 3
    assert chromosome.rules[0].condition == "signal_long and not overbought"
    assert "signal_long" in chromosome.features
    assert "sma(close, fast_ma) > sma(close, slow_ma)" == chromosome.features["signal_long"]

# --- Tests for expression_parser ---
def test_parse_expression_literal():
    expr = parse_expression("123.45")
    assert isinstance(expr, Literal)
    assert expr.value == 123.45

def test_parse_expression_variable():
    expr = parse_expression("close")
    assert isinstance(expr, Variable)
    assert expr.name == "close"

def test_parse_expression_function_call():
    expr = parse_expression("sma(close, 20)")
    assert isinstance(expr, FunctionCall)
    assert expr.func_name == "sma"
    assert len(expr.args) == 2
    assert isinstance(expr.args[0], Variable)
    assert expr.args[0].name == "close"
    assert isinstance(expr.args[1], Literal)
    assert expr.args[1].value == 20

def test_parse_expression_binary_operation():
    expr = parse_expression("a > b")
    assert isinstance(expr, BinaryOperation)
    assert expr.operator == ">"
    assert isinstance(expr.left, Variable)
    assert expr.left.name == "a"
    assert isinstance(expr.right, Variable)
    assert expr.right.name == "b"

def test_parse_expression_complex():
    expr = parse_expression("sma(close, fast_ma) > sma(close, slow_ma) and not (rsi(close, rsi_period) > 70)")
    assert isinstance(expr, BinaryOperation) # 'and' is the top-level operator
    assert expr.operator == 'and'

    # Test evaluation of a simple expression
def test_expression_evaluation_literal(sample_data):
    expr = parse_expression("100")
    result = expr.evaluate(sample_data, {})
    assert result == 100

def test_expression_evaluation_variable(sample_data):
    expr = parse_expression("close")
    result = expr.evaluate(sample_data, {})
    pd.testing.assert_series_equal(sample_data["close"], result)

def test_expression_evaluation_function_call(sample_data):
    # Mock indicator_factory.sma for predictable results
    # For actual tests, ensure indicator_factory methods are correctly implemented
    # For this test, we'll just check if it can call it.
    class MockIndicatorFactory:
        def sma(self, series, length):
            return series.rolling(window=length).mean()
    
    original_indicator_factory = FunctionCall.evaluate.__globals__['indicator_factory']
    FunctionCall.evaluate.__globals__['indicator_factory'] = MockIndicatorFactory()

    expr = parse_expression("sma(close, 5)")
    result = expr.evaluate(sample_data, {})
    expected_sma = sample_data["close"].rolling(window=5).mean()
    pd.testing.assert_series_equal(expected_sma, result)

    FunctionCall.evaluate.__globals__['indicator_factory'] = original_indicator_factory


def test_expression_evaluation_binary_operation(sample_data):
    expr = parse_expression("close > open")
    result = expr.evaluate(sample_data, {})
    pd.testing.assert_series_equal(sample_data["close"] > sample_data["open"], result)

def test_expression_evaluation_with_parameters(sample_data):
    parameters = {"length": 5}
    expr = parse_expression("close > close.shift(length)")
    # This requires 'shift' to be a recognized function or the expression to be structured differently
    # For now, let's test a simpler parameter usage, e.g., in an indicator.
    # We'll use a variable that represents a parameter value.
    expr = parse_expression("close + length") # simple example of using a parameter
    result = expr.evaluate(sample_data, parameters)
    pd.testing.assert_series_equal(sample_data["close"] + 5, result)


# --- Integration tests for VectorizedBacktester with JSON strategies ---

def test_backtester_run_example1_integration(mock_backtester):
    # This test assumes 'sma' and 'rsi' are available in indicator_factory
    # and the logic for 'current_profit_percentage', 'current_loss_percentage'
    # would need to be implemented for full functionality.
    # For this test, we'll create a simplified version that relies only on MA cross.
    json_str_simplified = """
    {
      "parameters": {
        "fast_ma": 2,
        "slow_ma": 5
      },
      "rules": [
        {
          "condition": "signal_long",
          "action": "enter_long"
        },
        {
          "condition": "signal_short",
          "action": "exit_position"
        }
      ],
      "features": {
        "signal_long": "sma(close, fast_ma) > sma(close, slow_ma)",
        "signal_short": "sma(close, fast_ma) < sma(close, slow_ma)"
      }
    }
    """
    chromosome = Chromosome.from_json(json_str_simplified)
    result = mock_backtester.run(chromosome)

    assert result is not None
    assert isinstance(result, BacktestResult)
    # Add more specific assertions based on expected trades and equity curve.
    # This will require manual calculation of expected results or a more robust
    # comparison method. For now, just check if it runs and returns a result.
    assert len(result.trades) > 0 or len(result.equity_curve) > 0 # Expect some activity

def test_backtester_run_no_features_or_rules(mock_backtester):
    json_str = """
    {
      "parameters": {},
      "rules": [],
      "features": {}
    }
    """
    chromosome = Chromosome.from_json(json_str)
    result = mock_backtester.run(chromosome)
    assert result is not None
    assert len(result.trades) == 0
    assert result.equity_curve.iloc[-1] == mock_backtester._initial_equity

def test_backtester_run_undefined_variable_in_feature_expression_raises_error(mock_backtester):
    json_str = """
    {
      "parameters": {},
      "rules": [],
      "features": {
        "invalid_feature": "non_existent_column > 10"
      }
    }
    """
    chromosome = Chromosome.from_json(json_str)
    with pytest.raises(ValueError, match="Undefined variable or column: non_existent_column"):
        mock_backtester.run(chromosome)

def test_backtester_run_undefined_function_in_feature_expression_raises_error(mock_backtester):
    json_str = """
    {
      "parameters": {},
      "rules": [],
      "features": {
        "invalid_feature": "nonExistentFunc(close, 10)"
      }
    }
    """
    chromosome = Chromosome.from_json(json_str)
    with pytest.raises(ValueError, match="Unknown function: nonExistentFunc"):
        mock_backtester.run(chromosome)
