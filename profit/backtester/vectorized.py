"""
A simplified, vectorized backtesting engine.

This backtester is fast but is not suitable for all strategies. It works well
for strategies that are not path-dependent, meaning the decision at any given
time step only depends on the data at that time step, not on the sequence of
trades that came before.
"""
import numpy as np
import pandas as pd

from profit.backtester.base import BaseBacktester
from profit.backtester.results import BacktestResult, Trade
from profit.indicators import factory as indicator_factory
from profit.metrics import (
    calculate_annualized_return,
    calculate_expectancy,
    calculate_sharpe_ratio,
)
from profit.strategy import Chromosome


class VectorizedBacktester(BaseBacktester):
    """
    A vectorized backtesting engine for non-path-dependent strategies.
    """

    def run(self, chromosome: Chromosome) -> BacktestResult:
        """
        Runs a vectorized backtest for the given strategy chromosome.

        This implementation is a simplified example that assumes a strategy
        based on a moving average crossover. It is a placeholder for the more
        complex "expression tree" evaluation that will be needed for the full
        GP engine.

        Args:
            chromosome (Chromosome): The strategy to be backtested. It is
                expected to have 'fast_ma' and 'slow_ma' in its 'parameters'.

        Returns:
            BacktestResult: An object containing the results of the backtest.
        """
        # 1. Generate Signals (Placeholder for real strategy evaluation)
        # This is a simplified example of generating a signal from features.
        # The real implementation will evaluate the 'features' and 'rules'
        # from the chromosome.
        fast_ma_period = chromosome.parameters.get("fast_ma", 10)
        slow_ma_period = chromosome.parameters.get("slow_ma", 30)

        fast_ma = indicator_factory.sma(self._data["close"], length=fast_ma_period)
        slow_ma = indicator_factory.sma(self._data["close"], length=slow_ma_period)

        if fast_ma is None or slow_ma is None:
            # Not enough data to compute indicators, return empty result
            return BacktestResult(
                trades=[],
                equity_curve=pd.Series(
                    [self._initial_equity] * len(self._data), index=self._data.index
                ),
                metrics={
                    "sharpe_ratio": 0.0,
                    "annualized_return": 0.0,
                    "expectancy": 0.0,
                },
            )

        # A simple signal: 1 for long, -1 for short
        signal = pd.Series(np.where(fast_ma > slow_ma, 1, -1), index=self._data.index)
        signal = signal.ffill().fillna(0) # Forward fill NaNs and then fill remaining with 0

        # 2. Determine Positions
        # We assume we are always in the market based on the signal.
        # A real backtester would handle entries/exits more granularly.
        positions = signal.shift(1) # Shift to avoid lookahead bias

        # 3. Calculate Returns
        asset_returns = self._data["close"].pct_change()
        strategy_returns = positions * asset_returns

        # 4. Generate Equity Curve
        equity_curve = self._initial_equity * (1 + strategy_returns).cumprod()
        equity_curve.iloc[0] = self._initial_equity # Start with initial equity

        # 5. Extract Trades
        trades = self._extract_trades(positions)

        # 6. Calculate Metrics
        trade_returns = pd.Series([t.return_pct for t in trades])
        metrics = {
            "sharpe_ratio": calculate_sharpe_ratio(strategy_returns),
            "annualized_return": calculate_annualized_return(equity_curve),
            "expectancy": calculate_expectancy(trade_returns),
        }

        # 7. Return BacktestResult
        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics,
        )

    def _extract_trades(self, positions: pd.Series) -> list[Trade]:
        """
        Extracts a list of Trade objects from a series of positions.
        """
        trades = []
        last_pos = 0.0
        entry_time, entry_price = None, None

        for time in positions.index:
            pos = positions.loc[time]
            if pos != last_pos:
                # Position has changed, a trade occurred
                if last_pos != 0 and entry_time is not None:
                    assert entry_price is not None
                    # Exit previous trade
                    exit_price = float(self._data.loc[time, "open"]) # Exit at next bar's open
                    return_pct = (exit_price / entry_price - 1) * last_pos
                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        return_pct=return_pct,
                        size=last_pos, # Simplified size
                    ))
                    entry_time, entry_price = None, None

                if pos != 0:
                    # Enter new trade
                    entry_time = time
                    entry_price = float(self._data.loc[time, "open"]) # Enter at current bar's open

            last_pos = pos

        # Close any open trade at the end of the data
        if entry_time is not None and last_pos != 0:
            assert entry_price is not None
            exit_time = self._data.index[-1]
            exit_price = float(self._data.loc[exit_time, "close"]) # Exit at last bar's close
            return_pct = (exit_price / entry_price - 1) * last_pos
            trades.append(Trade(
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                return_pct=return_pct,
                size=last_pos,
            ))
        return trades
