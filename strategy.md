# Strategy JSON Structure (Chromosome)

The `Chromosome` represents the complete genetic makeup of a trading strategy within the ProFiT framework. It is a Pydantic model with three primary fields, allowing for a flexible and evolvable definition of a strategy.

## `Chromosome` Fields:

1.  **`parameters` (Type: `Dict[str, Any]`)**
    *   **Description:** A dictionary of tunable parameters for pre-defined logic within the strategy. These typically control aspects of indicators or other static components.
    *   **Example:**
        ```json
        {
          "rsi_period": 14,
          "ma_period": 200,
          "some_threshold": 0.7
        }
        ```

2.  **`rules` (Type: `List[Rule]`)**
    *   **Description:** A list of `Rule` objects that define the strategy's entry, exit, and other operational logic. Each rule specifies a condition and a corresponding action.
    *   **`Rule` Object Structure:**
        *   **`condition` (Type: `str`)**: A boolean expression composed of defined features (e.g., `"c1 and not c2"`).
        *   **`action` (Type: `str`)**: The trading action to take if the condition is met (e.g., `"enter_long"`, `"exit_short"`, `"hold"`).
    *   **Example:**
        ```json
        [
          {
            "condition": "c1 and c2",
            "action": "enter_long"
          },
          {
            "condition": "c3 or c4",
            "action": "exit_long"
          },
          {
            "condition": "volume_spike",
            "action": "enter_short"
          }
        ]
        ```

3.  **`features` (Type: `Dict[str, str]`)**
    *   **Description:** A dictionary where keys are feature names (e.g., `"c1"`, `"volume_spike"`) and values are expression strings that define how the feature is calculated from OHLCV (Open, High, Low, Close, Volume) data or other existing indicators. This allows for the dynamic creation of new features.
    *   **Example:**
        ```json
        {
          "c1": "close > sma(close, 20)",
          "c2": "rsi(close, 14) < 30",
          "volume_spike": "volume > sma(volume, 50) * 2"
        }
        ```

---

## Complete Example of a Chromosome (Strategy JSON):

```json
{
  "parameters": {
    "fast_ma": 10,
    "slow_ma": 30,
    "rsi_period": 14
  },
  "rules": [
    {
      "condition": "signal_long and not overbought",
      "action": "enter_long"
    },
    {
      "condition": "signal_short and not oversold",
      "action": "enter_short"
    },
    {
      "condition": "take_profit or stop_loss",
      "action": "exit_position"
    }
  ],
  "features": {
    "signal_long": "sma(close, fast_ma) > sma(close, slow_ma)",
    "signal_short": "sma(close, fast_ma) < sma(close, slow_ma)",
    "overbought": "rsi(close, rsi_period) > 70",
    "oversold": "rsi(close, rsi_period) < 30",
    "take_profit": "current_profit_percentage > 0.05",
    "stop_loss": "current_loss_percentage > 0.03"
  }
}
```