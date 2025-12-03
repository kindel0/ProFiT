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


def _get_Wilder_SMMA(series: pd.Series, length: int) -> pd.Series:
    """
    Calculates the Wilder's Smoothing Moving Average (SMMA).

    Args:
        series (pd.Series): The input series.
        length (int): The period for the SMMA.

    Returns:
        pd.Series: A Series containing the SMMA.
    """
    return series.ewm(alpha=1/length, adjust=False).mean()


def rsi(
    close: pd.Series,
    length: int = 14,
    **kwargs,
) -> Optional[pd.Series]:
    """
    Calculates the Relative Strength Index (RSI).

    Args:
        close (pd.Series): A Series of closing prices.
        length (int): The time period.

    Returns:
        Optional[pd.Series]: A Series containing the RSI, or None if the input
        is not long enough.
    """
    if len(close) < length:
        return None

    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = _get_Wilder_SMMA(gain, length)
    avg_loss = _get_Wilder_SMMA(loss, length)

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, pd.NA) 
    rsi = 100 - (100 / (1 + rs))

    return rsi


def current_profit_percentage(
    trade_entry: float,
    current_price: pd.Series,
    **kwargs,
) -> Optional[pd.Series]:
    """
    Calculates the current profit percentage.

    Args:
        trade_entry (float): The price at which the trade was entered.
        current_price (pd.Series): A Series of current prices.

    Returns:
        Optional[pd.Series]: A Series containing the profit percentage, or None
        if the trade_entry is zero.
    """
    if trade_entry == 0:
        return None
    return ((current_price - trade_entry) / trade_entry) * 100

