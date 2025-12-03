"""
Data structures for holding the results of a backtest.
"""
from datetime import datetime
from typing import List, Dict

import pandas as pd
from pydantic import BaseModel, Field, ConfigDict


class Trade(BaseModel):
    """
    Represents a single trade executed by a strategy.

    Args:
        entry_time (datetime): The timestamp of the trade entry.
        exit_time (datetime): The timestamp of the trade exit.
        entry_price (float): The price at which the trade was entered.
        exit_price (float): The price at which the trade was exited.
        return_pct (float): The percentage return of the trade.
        size (float): The size or quantity of the trade.
    """
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    return_pct: float
    size: float


class BacktestResult(BaseModel):
    """
    Holds all the results from a single backtest run of a strategy.

    Args:
        trades (List[Trade]): A list of all trades executed.
        equity_curve (pd.Series): The portfolio's equity over time.
        metrics (Dict[str, float]): A dictionary of calculated performance
            metrics (e.g., {'sharpe_ratio': 1.5, 'annualized_return': 0.12}).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    trades: List[Trade] = Field(..., description="A list of all trades executed.")
    equity_curve: pd.Series = Field(..., description="The portfolio's equity over time.")
    metrics: Dict[str, float] = Field(..., description="Calculated performance metrics.")

