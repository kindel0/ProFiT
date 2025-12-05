"""
A simplified, vectorized backtesting engine.

This backtester is fast but is not suitable for all strategies. It works well
for strategies that are not path-dependent, meaning the decision at any given
time step only depends on the data at that time step, not on the sequence of
trades that came before.
"""
from typing import Any, Optional
import numpy as np
import pandas as pd

from profit.backtester.base import BaseBacktester
from profit.backtester.expression_parser import parse_expression
from profit.backtester.results import BacktestResult, Trade
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

        This implementation dynamically evaluates features and rules defined
        in the chromosome to generate trading signals.

        Args:
            chromosome (Chromosome): The strategy to be backtested.

        Returns:
            BacktestResult: An object containing the results of the backtest.
        """
        # Initialize a dictionary to store evaluated features
        # Ensure it has an index for aligning with _data
        evaluated_features = pd.DataFrame(index=self._data.index)

        # 1. Evaluate Features
        for feature_name, expression_str in chromosome.features.items():
            parsed_expression = parse_expression(expression_str)
            evaluated_features[feature_name] = parsed_expression.evaluate(self._data, chromosome.parameters)
        
        # Add basic price data to evaluated_features for rule evaluation
        evaluated_features['close'] = self._data['close']
        evaluated_features['open'] = self._data['open']
        evaluated_features['high'] = self._data['high']
        evaluated_features['low'] = self._data['low']
        evaluated_features['volume'] = self._data['volume']

        # Initialize signal series (1 for long, -1 for short, 0 for flat)
        signal = pd.Series(0, index=self._data.index, dtype=int)
        
        # 2. Evaluate Rules and Generate Signals
        # This is a simplified approach. A more sophisticated system would handle
        # multiple conflicting rules, position sizing, etc.
        # For now, we apply rules sequentially.
        for rule in chromosome.rules:
            parsed_condition = parse_expression(rule.condition)
            condition_result = parsed_condition.evaluate(evaluated_features, chromosome.parameters)

            # Apply action based on condition
            if rule.action == 'enter_long':
                signal = pd.Series(np.where(condition_result, 1, signal), index=self._data.index)
            elif rule.action == 'enter_short':
                signal = pd.Series(np.where(condition_result, -1, signal), index=self._data.index)
            elif rule.action == 'exit_position':
                signal = pd.Series(np.where(condition_result, 0, signal), index=self._data.index)
            # Add other actions as needed (e.g., exit_long, exit_short)

        # Check for Stop Loss or Take Profit parameters
        sl_pct = chromosome.parameters.get("stop_loss_pct")
        tp_pct = chromosome.parameters.get("take_profit_pct")

        if sl_pct is not None or tp_pct is not None:
            return self._run_iterative(signal, sl_pct, tp_pct)
        
        # Shift signal to avoid lookahead bias and fill any NaNs
        positions = signal.shift(1).ffill().fillna(0)
        
        # Ensure we start with 0 position if the first signal was not explicitly set
        if not positions.empty and positions.iloc[0] != 0:
            positions.iloc[0] = 0

        # Handle cases where indicators might not have enough data to compute
        if positions.isnull().all():
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

        # 3. Calculate Returns
        asset_returns = self._data["close"].pct_change()
        strategy_returns = positions * asset_returns
        strategy_returns.fillna(0, inplace=True) # Fill NaNs from shifting or initial periods

        # 4. Generate Equity Curve
        equity_curve = self._initial_equity * (1 + strategy_returns).cumprod()
        if not equity_curve.empty:
            equity_curve.iloc[0] = self._initial_equity # Start with initial equity

        # 5. Extract Trades
        trades = self._extract_trades(positions)

        # 6. Calculate Metrics
        trade_returns = pd.Series([t.return_pct for t in trades])
        metrics = {
            "sharpe_ratio": calculate_sharpe_ratio(strategy_returns) if not strategy_returns.empty else 0.0,
            "annualized_return": calculate_annualized_return(equity_curve) if not equity_curve.empty else 0.0,
            "expectancy": calculate_expectancy(trade_returns) if not trade_returns.empty else 0.0,
        }

        # 7. Return BacktestResult
        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics,
        )

    def _run_iterative(self, signal: pd.Series, sl_pct: Optional[float], tp_pct: Optional[float]) -> BacktestResult:
        """
        Runs an iterative backtest to support path-dependent logic like Stop Loss/Take Profit.
        """
        # Prepare data for fast iteration
        opens = self._data['open'].values
        highs = self._data['high'].values
        lows = self._data['low'].values
        closes = self._data['close'].values
        times = self._data.index
        n = len(self._data)

        # State
        cash = self._initial_equity
        shares = 0.0
        entry_price = 0.0
        entry_time: Any = None
        
        trades: list[Trade] = []
        equity_curve = np.zeros(n)
        
        # Iterate through bars
        for i in range(n):
            current_time = times[i]
            current_open = opens[i]
            
            # 1. Signal Execution (at Open)
            # We look at the signal generated at i-1 to decide action at Open[i]
            target_pos = 0
            if i > 0:
                target_pos = signal.iloc[i-1]
            
            # Current position direction (1, -1, 0)
            current_dir = 1 if shares > 0 else (-1 if shares < 0 else 0)
            
            # If signal changed, execute at Open
            if current_dir != target_pos:
                # Close existing
                if current_dir != 0:
                    exit_price = current_open
                    position_value = shares * exit_price
                    cash += position_value # Close position, convert to cash
                    
                    # Trade Record
                    ret_pct = (exit_price / entry_price - 1) * current_dir
                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=current_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        return_pct=ret_pct,
                        size=abs(shares)
                    ))
                    
                    shares = 0.0
                    current_dir = 0
                
                # Open new
                if target_pos != 0:
                    # Calculate total equity to size position
                    # Equity = cash (since shares is 0)
                    total_equity = cash 
                    
                    # Simple sizing: 100% equity
                    position_value = total_equity
                    
                    # Calculate shares
                    # Long: shares = value / price
                    # Short: shares = - value / price
                    shares = (position_value / current_open) * target_pos
                    
                    # Update Cash
                    # Long: Pay cash. cash -= value.
                    # Short: Receive cash. cash += value.
                    # Note: shares has sign.
                    # cash -= shares * current_open
                    # Long: cash -= (val/p)*p = -val. Correct.
                    # Short: cash -= (-val/p)*p = +val. Correct.
                    cash -= shares * current_open
                    
                    entry_price = current_open
                    entry_time = current_time
                    current_dir = target_pos

            # 2. Intra-bar Checks (SL/TP)
            # Only if we have a position
            if shares != 0:
                hit_exit = False
                exit_price = 0.0
                
                if shares > 0: # Long
                    sl_price = entry_price * (1 - sl_pct) if sl_pct else -np.inf
                    tp_price = entry_price * (1 + tp_pct) if tp_pct else np.inf
                    
                    if lows[i] <= sl_price:
                        exit_price = min(current_open, sl_price) # Gap check
                        hit_exit = True
                    elif highs[i] >= tp_price:
                        exit_price = max(current_open, tp_price)
                        hit_exit = True
                        
                elif shares < 0: # Short
                    sl_price = entry_price * (1 + sl_pct) if sl_pct else np.inf
                    tp_price = entry_price * (1 - tp_pct) if tp_pct else -np.inf
                    
                    if highs[i] >= sl_price:
                        exit_price = max(current_open, sl_price) # Gap check (buy stop)
                        hit_exit = True
                    elif lows[i] <= tp_price:
                        exit_price = min(current_open, tp_price)
                        hit_exit = True

                if hit_exit:
                    # Execute Exit
                    # Update Cash
                    cash += shares * exit_price
                    
                    # Record Trade
                    ret_pct = (exit_price / entry_price - 1) * (1 if shares > 0 else -1)
                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=current_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        return_pct=ret_pct,
                        size=abs(shares)
                    ))
                    
                    shares = 0.0
                    current_dir = 0
                    # We are now flat for the remainder of the bar (Mark to Market will use Cash)

            # 3. Mark to Market (End of Bar)
            # Equity = Cash + Market Value of Shares
            # Long: Val = shares * close.
            # Short: Val = shares * close. (shares is neg, close is pos -> neg value, liability).
            # Equity = Cash + (shares * close).
            # Example Short: Cash = 2000 (1000 init + 1000 proceeds). Shares = -10. Close = 90.
            # Val = -900. Equity = 2000 - 900 = 1100. Profit 100. Correct.
            
            current_equity = cash + (shares * closes[i])
            equity_curve[i] = current_equity
        
        # Close open trade at end
        if shares != 0:
            exit_price = closes[-1]
            cash += shares * exit_price
            ret_pct = (exit_price / entry_price - 1) * (1 if shares > 0 else -1)
            trades.append(Trade(
                entry_time=entry_time,
                exit_time=times[-1],
                entry_price=entry_price,
                exit_price=exit_price,
                return_pct=ret_pct,
                size=abs(shares)
            ))
            shares = 0.0

        # Create result objects
        equity_series = pd.Series(equity_curve, index=times)
        returns_series = equity_series.pct_change().fillna(0)
        
        metrics = {
            "sharpe_ratio": calculate_sharpe_ratio(returns_series) if not returns_series.empty else 0.0,
            "annualized_return": calculate_annualized_return(equity_series) if not equity_series.empty else 0.0,
            "expectancy": calculate_expectancy(pd.Series([t.return_pct for t in trades])) if trades else 0.0,
        }

        return BacktestResult(
            trades=trades,
            equity_curve=equity_series,
            metrics=metrics,
        )

    def _extract_trades(self, positions: pd.Series) -> list[Trade]:
        """
        Extracts a list of Trade objects from a series of positions.
        """
        trades = []
        last_pos = 0.0
        entry_time: Any = None
        entry_price: Optional[float] = None

        # Iterate through the index of positions, ensuring we use the original data's index for prices
        for i in range(len(positions)):
            time = positions.index[i]
            pos = positions.iloc[i]

            if pos != last_pos:
                # Position has changed, a trade occurred
                if last_pos != 0 and entry_time is not None and entry_price is not None:
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
            if entry_price is None:
                raise ValueError("entry_price cannot be None for an open trade.")
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
