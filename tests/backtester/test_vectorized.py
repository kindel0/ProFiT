"""
Tests for the VectorizedBacktester.
"""
import numpy as np
import pandas as pd
import pytest

from profit.backtester.results import BacktestResult
from profit.backtester.vectorized import VectorizedBacktester
from profit.strategy import Chromosome, Rule


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """
    Provides a sample DataFrame of OHLCV data for testing.
    This data is designed to have a clear crossover event.
    """
    dates = pd.to_datetime(pd.date_range('2023-01-01', periods=40))
    close = np.concatenate([
        np.linspace(100, 110, 15),  # slow rise
        np.linspace(110, 100, 10),  # dip
        np.linspace(100, 120, 15),  # fast rise
    ])
    open_ = close - np.random.uniform(0, 1, 40)
    high = close + np.random.uniform(0, 1, 40)
    low = close - np.random.uniform(1, 2, 40)
    volume = np.random.uniform(1000, 2000, 40)
    
    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }, index=dates)


@pytest.fixture
def sample_chromosome() -> Chromosome:
    """
    Provides a sample Chromosome for a simple MA crossover strategy.
    """
    return Chromosome(
        parameters={"fast_ma": 5, "slow_ma": 20},
        rules=[Rule(condition="c1", action="enter_long")], # Rules not used by this simple backtester yet
        features={"c1": "sma(5) > sma(20)"}, # Features not used by this simple backtester yet
    )


def test_run_vectorized_backtest(sample_ohlcv_data, sample_chromosome):
    """
    Tests the main `run` method of the VectorizedBacktester.
    """
    initial_equity = 100000.0
    backtester = VectorizedBacktester(data=sample_ohlcv_data, initial_equity=initial_equity)
    result = backtester.run(chromosome=sample_chromosome)

    # Check result types
    assert isinstance(result, BacktestResult)
    assert isinstance(result.equity_curve, pd.Series)
    assert isinstance(result.metrics, dict)
    assert isinstance(result.trades, list)

    # Check equity curve
    assert result.equity_curve.iloc[0] == initial_equity
    assert not result.equity_curve.isna().any()

    # Check metrics
    expected_metrics = ["sharpe_ratio", "annualized_return", "expectancy"]
    for metric in expected_metrics:
        assert metric in result.metrics
        assert isinstance(result.metrics[metric], float)

    # Check trades
    # With the given data, there should be at least one crossover and therefore trades
    assert len(result.trades) > 0
    trade = result.trades[0]
    assert trade.entry_price > 0
    assert trade.exit_price > 0
    assert trade.entry_time < trade.exit_time
