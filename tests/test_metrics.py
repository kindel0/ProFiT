"""
Tests for the performance metric calculation functions.
"""
import numpy as np
import pandas as pd
import pytest

from profit import metrics


@pytest.fixture
def sample_returns() -> pd.Series:
    """
    Provides a sample Series of daily returns.
    """
    return pd.Series([0.01, -0.005, 0.02, 0.015, -0.01, 0.005, 0.01, 0.002])


@pytest.fixture
def sample_equity_curve() -> pd.Series:
    """
    Provides a sample equity curve over 2 years (504 days).
    """
    initial_equity = 100000
    np_returns = np.random.normal(loc=0.0005, scale=0.01, size=504)
    # Add a bit of a positive drift
    pd_returns = pd.Series(np_returns) + 0.0001
    return initial_equity * (1 + pd_returns).cumprod()


@pytest.fixture
def sample_trade_returns() -> pd.Series:
    """
    Provides a sample Series of individual trade returns.
    """
    return pd.Series([0.05, -0.02, 0.10, 0.03, -0.04, 0.07])


def test_calculate_sharpe_ratio(sample_returns):
    """
    Tests the Sharpe ratio calculation.
    """
    # Pre-calculated values
    mean_returns = sample_returns.mean()
    std_returns = sample_returns.std()
    expected_sharpe = (mean_returns / std_returns) * np.sqrt(252)

    sharpe = metrics.calculate_sharpe_ratio(sample_returns)
    assert isinstance(sharpe, float)
    assert pytest.approx(sharpe, 0.001) == expected_sharpe

    # Test with zero standard deviation
    zero_std_returns = pd.Series([0.01, 0.01, 0.01])
    assert metrics.calculate_sharpe_ratio(zero_std_returns) == 0.0


def test_calculate_annualized_return(sample_equity_curve):
    """
    Tests the annualized return calculation.
    """
    start_equity = sample_equity_curve.iloc[0]
    end_equity = sample_equity_curve.iloc[-1]
    num_days = len(sample_equity_curve)

    total_return = (end_equity / start_equity) - 1
    expected_annual_return = (1 + total_return) ** (252.0 / num_days) - 1

    annual_return = metrics.calculate_annualized_return(sample_equity_curve)
    assert isinstance(annual_return, float)
    assert pytest.approx(annual_return) == expected_annual_return

    # Test with short equity curve
    short_curve = pd.Series([100, 101])
    assert metrics.calculate_annualized_return(short_curve) > 0


def test_calculate_expectancy(sample_trade_returns):
    """
    Tests the expectancy calculation.
    """
    wins = sample_trade_returns[sample_trade_returns > 0]
    losses = sample_trade_returns[sample_trade_returns <= 0]
    win_rate = len(wins) / len(sample_trade_returns)
    loss_rate = len(losses) / len(sample_trade_returns)
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    expected_expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

    expectancy = metrics.calculate_expectancy(sample_trade_returns)
    assert isinstance(expectancy, float)
    assert pytest.approx(expectancy) == expected_expectancy

    # Test with no trades
    assert metrics.calculate_expectancy(pd.Series([], dtype=np.float64)) == 0.0

    # Test with only wins
    only_wins = pd.Series([0.1, 0.2])
    assert metrics.calculate_expectancy(only_wins) == pytest.approx(0.15)

    # Test with only losses
    only_losses = pd.Series([-0.1, -0.2])
    assert metrics.calculate_expectancy(only_losses) == pytest.approx(-0.15)
