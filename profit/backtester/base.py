"""
Abstract base class for backtesting engines.
"""
from abc import ABC, abstractmethod

import pandas as pd

from profit.backtester.results import BacktestResult
from profit.strategy import Chromosome


class BaseBacktester(ABC):
    """
    Abstract base class for all backtesting engines.

    It defines the common interface for running a backtest against a given
    strategy and dataset.
    """

    def __init__(self, data: pd.DataFrame, initial_equity: float = 100000.0):
        """
        Initializes the backtester.

        Args:
            data (pd.DataFrame): The OHLCV data to be used for the backtest.
            initial_equity (float): The starting equity for the portfolio.
        """
        self._data = data
        self._initial_equity = initial_equity

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
