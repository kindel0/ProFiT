import pandas as pd
from typing import List, Type
from backtesting import Backtest, Strategy
from profit.backtester.base import BaseBacktester
from profit.backtester.results import BacktestResult, Trade
from profit.strategy import Chromosome
import inspect

class LucitBacktester(BaseBacktester):
    """
    A backtesting engine using 'lucit-backtesting' (fork of backtesting.py).
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
        
        trades = self._convert_trades(trades_df)
        equity_curve = equity_curve_df['Equity']
        
        metrics = {
            "sharpe_ratio": stats['Sharpe Ratio'] if not pd.isna(stats['Sharpe Ratio']) else 0.0,
            "annualized_return": stats['Return (Ann.) [%]'] if not pd.isna(stats['Return (Ann.) [%]']) else -100.0, 
            "expectancy": stats['Avg. Trade [%]'] if not pd.isna(stats['Avg. Trade [%]']) else 0.0,
            # Add Max Drawdown as it's often useful
            "max_drawdown": stats['Max. Drawdown [%]'] if not pd.isna(stats['Max. Drawdown [%]']) else 0.0,
        }

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics
        )

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

    def _convert_trades(self, trades_df: pd.DataFrame) -> List[Trade]:
        """
        Converts lucit-backtesting trades DataFrame to our Trade objects.
        """
        trades: List[Trade] = []
        if trades_df.empty:
            return trades

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