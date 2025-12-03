"""
Functions for calculating performance metrics of trading strategies.

These functions are used as the objective functions in the multi-objective
optimization process.
"""
from typing import Callable, Dict

import numpy as np
import pandas as pd

# Registry for objective functions
OBJECTIVE_REGISTRY: Dict[str, Callable[..., float]] = {}


def register_objective(name: str, func: Callable[..., float]):
    """
    Registers a new objective function for use by the optimizer.

    Args:
        name (str): The name of the objective function.
        func (Callable[..., float]): The function to be registered.
    """
    if name in OBJECTIVE_REGISTRY:
        raise ValueError(f"Objective '{name}' is already registered.")
    OBJECTIVE_REGISTRY[name] = func


def get_objective(name: str) -> Callable[..., float]:
    """
    Retrieves an objective function from the registry.

    Args:
        name (str): The name of the objective function to retrieve.

    Returns:
        Callable[..., float]: The requested objective function.
    """
    if name not in OBJECTIVE_REGISTRY:
        raise ValueError(f"Objective '{name}' is not registered. Available: {list(OBJECTIVE_REGISTRY.keys())}")
    return OBJECTIVE_REGISTRY[name]


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculates the annualized Sharpe ratio from a series of periodic returns.

    Args:
        returns (pd.Series): A Series of periodic returns (e.g., daily).
        risk_free_rate (float): The annualized risk-free rate.
        periods_per_year (int): The number of trading periods in a year.

    Returns:
        float: The calculated annualized Sharpe ratio. Returns 0.0 if the
        standard deviation of returns is zero.
    """
    std_dev = returns.std()
    if std_dev == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    sharpe_ratio = excess_returns.mean() / std_dev
    annualized_sharpe = sharpe_ratio * np.sqrt(periods_per_year)
    return float(annualized_sharpe)


def calculate_annualized_return(
    equity_curve: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Calculates the annualized return from an equity curve.

    Args:
        equity_curve (pd.Series): A Series representing the portfolio's equity
            over time.
        periods_per_year (int): The number of trading periods in a year.

    Returns:
        float: The calculated annualized return. Returns 0.0 if the equity
        curve has fewer than 2 data points.
    """
    if len(equity_curve) < 2:
        return 0.0

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    num_periods = len(equity_curve)
    annualized_return = (1 + total_return) ** (periods_per_year / num_periods) - 1
    return float(annualized_return)


def calculate_expectancy(trade_returns: pd.Series) -> float:
    """
    Calculates the expectancy per trade.

    Expectancy is the (average win * win rate) - (average loss * loss rate).

    Args:
        trade_returns (pd.Series): A Series of returns for each individual trade.

    Returns:
        float: The calculated expectancy. Returns 0.0 if there are no trades.
    """
    if trade_returns.empty:
        return 0.0

    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns <= 0]

    win_rate = len(wins) / len(trade_returns)
    loss_rate = len(losses) / len(trade_returns)

    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = abs(losses.mean()) if not losses.empty else 0.0

    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
    return float(expectancy)

# Register the default objective functions
register_objective("sharpe_ratio", calculate_sharpe_ratio)
register_objective("annualized_return", calculate_annualized_return)
register_objective("expectancy", calculate_expectancy)
