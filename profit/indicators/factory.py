"""
A factory for creating financial indicators.

This module provides simple, consistent wrappers for calculating
common financial technical indicators.
"""
from typing import Optional

import pandas as pd


def sma(
    close: pd.Series,
    length: int = 20,
    **kwargs,
) -> Optional[pd.Series]:
    """
    Calculates the Simple Moving Average (SMA).

    Args:
        close (pd.Series): A Series of closing prices.
        length (int): The time period.

    Returns:
        Optional[pd.Series]: A Series containing the SMA, or None if the input
        is not long enough.
    """
    if len(close) < length:
        return None
    return close.rolling(window=length).mean()