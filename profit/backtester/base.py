"""
Abstract base class for backtesting engines.
"""
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from profit.backtester.results import BacktestResult
from profit.strategy import Chromosome


class BaseBacktester(ABC):
    """
    Abstract base class for all backtesting engines.

    It defines the common interface for running a backtest against a given
    strategy and dataset.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_equity: float = 200000.0,
        eval_start: Optional[str] = None,
    ):
        """
        Initializes the backtester.

        Args:
            data (pd.DataFrame): The OHLCV data to be used for the backtest.
                This can include warm-up data before the evaluation period.
            initial_equity (float): The starting equity for the portfolio.
            eval_start (str, optional): The start date of the evaluation period.
                If provided, metrics are calculated only from this date onwards,
                but indicators can warm up on earlier data. Format: 'YYYY-MM-DD'
        """
        self._data = data
        self._initial_equity = initial_equity
        self._eval_start = eval_start

    @abstractmethod
    def run(self, chromosome: Chromosome) -> BacktestResult:
        """
        Runs a backtest for the given strategy chromosome.

        Args:
            chromosome (Chromosome): The strategy to be backtested.

        Returns:
            BacktestResult: An object containing the results of the backtest.
        """
        raise NotImplementedError
