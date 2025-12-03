"""
Tests for the indicator factory functions.
"""
import numpy as np
import pandas as pd
import pytest

from profit.indicators import factory


@pytest.fixture
def sample_close_prices() -> pd.Series:
    """
    Provides a sample Series of close prices for testing.
    """
    return pd.Series(
        np.array([
            100.0, 101.0, 102.5, 101.75, 103.0, 104.25, 103.5, 105.0,
            106.5, 105.75, 107.0, 108.5, 107.75, 109.0, 110.0, 111.5,
            112.5, 111.75, 113.25, 114.0, 115.5, 116.0, 115.25, 117.0,
            118.5, 117.75, 119.0, 120.5, 119.75, 121.0, 122.5, 121.75,
            123.0, 124.5, 123.75, 125.0, 126.5, 125.75, 127.0, 128.5
        ]),
        dtype=np.float64
    )


def test_sma_calculation(sample_close_prices):
    """
    Tests the SMA calculation with a known output.
    """
    sma_series = factory.sma(close=sample_close_prices, length=5)
    assert isinstance(sma_series, pd.Series)
    assert sma_series.notna().sum() == (len(sample_close_prices) - 5 + 1)
    # Compare with a pre-calculated value
    assert pytest.approx(sma_series.iloc[-1], 0.001) == 126.55