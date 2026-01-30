import pandas as pd
import numpy as np
from typing import List, Type, Optional
from backtesting import Backtest, Strategy
from profit.backtester.base import BaseBacktester
from profit.backtester.results import BacktestResult, Trade
from profit.strategy import Chromosome
import inspect

class LucitBacktester(BaseBacktester):
    """
    A backtesting engine using 'lucit-backtesting' (fork of backtesting.py).

    Supports warm-up periods: pass data including historical warm-up data,
    and specify eval_start to measure performance only from that date onwards.
    This allows indicators with long lookback periods to properly initialize.
    """

    def run(self, chromosome: Chromosome) -> BacktestResult:
        """
        Runs a backtest using lucit-backtesting.

        Args:
            chromosome (Chromosome): The strategy to be backtested.

        Returns:
            BacktestResult: An object containing the results of the backtest.
        """
        # 1. Prepare Data
        # lucit-backtesting expects Capitalized columns: Open, High, Low, Close, Volume
        data = self._data.copy()
        data.columns = [c.capitalize() for c in data.columns]

        # 2. Load Dynamic Strategy Class
        StrategyClass = self._load_strategy_class(chromosome.code)

        # 3. Initialize Backtest
        # cash is initial_equity.
        # Commission = 0.2% = 0.002
        # Exclusive Orders = True
        bt = Backtest(
            data,
            StrategyClass,
            cash=self._initial_equity,
            commission=0.002,
            exclusive_orders=True,
            trade_on_close=False
        )

        # 4. Run Backtest
        try:
            stats = bt.run()
        except Exception as e:
            # If backtest fails (runtime error in strategy), return empty result with error or re-raise
            # For the repair loop, we might need to capture this.
            # For now, let's re-raise so the optimizer catches it?
            # Or return a failed result? The optimizer expects valid metrics.
            raise e


        # 5. Extract Results
        trades_df = stats['_trades']
        equity_curve_df = stats['_equity_curve']

        # 6. If eval_start is specified, filter to evaluation period only
        if self._eval_start is not None:
            metrics = self._calculate_eval_period_metrics(
                trades_df, equity_curve_df, self._eval_start
            )
            # Filter trades to those entered during eval period
            trades = self._convert_trades(trades_df, eval_start=self._eval_start)
            equity_curve = equity_curve_df.loc[self._eval_start:, 'Equity']
        else:
            trades = self._convert_trades(trades_df)
            equity_curve = equity_curve_df['Equity']
            metrics = {
                "sharpe_ratio": stats['Sharpe Ratio'] if not pd.isna(stats['Sharpe Ratio']) else 0.0,
                "annualized_return": stats['Return (Ann.) [%]'] if not pd.isna(stats['Return (Ann.) [%]']) else -100.0,
                "expectancy": stats['Avg. Trade [%]'] if not pd.isna(stats['Avg. Trade [%]']) else 0.0,
                "max_drawdown": stats['Max. Drawdown [%]'] if not pd.isna(stats['Max. Drawdown [%]']) else 0.0,
            }

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics
        )

    def _calculate_eval_period_metrics(
        self,
        trades_df: pd.DataFrame,
        equity_curve_df: pd.DataFrame,
        eval_start: str
    ) -> dict:
        """
        Calculate performance metrics for only the evaluation period.

        Args:
            trades_df: DataFrame of all trades from backtest
            equity_curve_df: Full equity curve from backtest
            eval_start: Start date of evaluation period

        Returns:
            Dictionary of metrics calculated for the evaluation period only
        """
        # Get equity curve for eval period
        eval_equity = equity_curve_df.loc[eval_start:]

        if len(eval_equity) < 2:
            return {
                "sharpe_ratio": 0.0,
                "annualized_return": 0.0,
                "expectancy": 0.0,
                "max_drawdown": 0.0,
            }

        # Starting equity at eval_start
        start_equity = eval_equity['Equity'].iloc[0]
        end_equity = eval_equity['Equity'].iloc[-1]

        # Calculate returns
        total_return = (end_equity - start_equity) / start_equity

        # Calculate trading days in eval period
        trading_days = len(eval_equity)
        annual_factor = 365 / trading_days if trading_days > 0 else 1

        # Annualized return
        annualized_return = ((1 + total_return) ** annual_factor - 1) * 100

        # Daily returns for Sharpe ratio
        daily_equity = eval_equity['Equity']
        daily_returns = daily_equity.pct_change().dropna()

        # Sharpe ratio (assuming 0 risk-free rate)
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)
        else:
            sharpe_ratio = 0.0

        # Max drawdown during eval period
        rolling_max = daily_equity.expanding().max()
        drawdown = (daily_equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100  # As percentage

        # Expectancy: average trade return for trades entered during eval period
        if not trades_df.empty:
            eval_trades = trades_df[trades_df['EntryTime'] >= eval_start]
            if len(eval_trades) > 0:
                expectancy = eval_trades['ReturnPct'].mean()
            else:
                expectancy = 0.0
        else:
            expectancy = 0.0

        return {
            "sharpe_ratio": sharpe_ratio if not pd.isna(sharpe_ratio) else 0.0,
            "annualized_return": annualized_return if not pd.isna(annualized_return) else 0.0,
            "expectancy": expectancy if not pd.isna(expectancy) else 0.0,
            "max_drawdown": max_drawdown if not pd.isna(max_drawdown) else 0.0,
        }

    def _load_strategy_class(self, code: str) -> Type[Strategy]:
        """
        Dynamically loads the strategy class from the provided Python code.
        """
        namespace = {}
        exec(code, namespace)
        
        # Find the class that inherits from Strategy (but is not Strategy itself)
        # Note: We might need to inject 'backtesting' dependencies into the namespace or rely on the code importing them.
        # The seeds import what they need.
        
        strategy_class = None
        for name, obj in namespace.items():
            if inspect.isclass(obj) and issubclass(obj, Strategy) and obj is not Strategy:
                strategy_class = obj
                break
        
        if strategy_class is None:
            raise ValueError("No valid Strategy class found in the provided code.")
            
        return strategy_class

    def _convert_trades(
        self, trades_df: pd.DataFrame, eval_start: Optional[str] = None
    ) -> List[Trade]:
        """
        Converts lucit-backtesting trades DataFrame to our Trade objects.

        Args:
            trades_df: DataFrame of trades from backtesting.py
            eval_start: If provided, only include trades entered on or after this date
        """
        trades: List[Trade] = []
        if trades_df.empty:
            return trades

        # Filter to eval period if specified
        if eval_start is not None:
            trades_df = trades_df[trades_df['EntryTime'] >= eval_start]

        for _, row in trades_df.iterrows():
            trades.append(Trade(
                entry_time=row['EntryTime'],
                exit_time=row['ExitTime'],
                entry_price=row['EntryPrice'],
                exit_price=row['ExitPrice'],
                return_pct=row['ReturnPct'],
                size=row['Size']
            ))
        return trades