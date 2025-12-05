# ProFiT: Program Search for Financial Trading

This repository contains a reproduction of the **ProFiT framework** as described in the research paper [*"ProFiT: Program Search for Financial Trading"*](https://www.researchgate.net/publication/398248186_ProFiT_Program_Search_for_Financial_Trading).<br>
ProFiT is an LLM-driven evolutionary search algorithm that autonomously discovers and improves algorithmic trading strategies. It uses a closed feedback loop where an LLM analyzes strategy performance on backtests and proposes code-level improvements.

## Key Features

*   **LLM-Driven Evolution**: Uses Large Language Models (e.g., GPT-4) to act as both "Analyst" and "Developer" to mutate trading strategies.
*   **Code-Level Search**: Strategies are represented as executable Python code, not abstract parameters.
*   **Self-Repair**: Includes a repair loop where the LLM fixes syntax or runtime errors based on tracebacks.
*   **Walk-Forward Validation**: Implements the specific 5-fold time-series split described in the paper.
*   **Robust Backtesting**: Uses `backtesting.py` (via `lucit-backtesting`) with realistic constraints ( 0.2% commission, exclusive orders).

## Installation

This project is managed with `uv`.

1.  **Install dependencies:**
    ```bash
    uv sync
    ```

2.  **Set up Environment:**
    If using the OpenAI client, set your API key:
    ```bash
    export OPENAI_API_KEY="sk-..."
    ```

## Configuration

Edit `config.yaml` to control the experiment parameters:

```yaml
data:
  path: "tests/data/Bitcoin_historical_data_coinmarketcap.csv"
ga:
  generations: 15  # Number of evolutionary steps
llm:
  client: "openai" # "openai" or "mock"
  model: "gpt-4o"
```

## Running the Reproduction

To run the standard experiment (Seed: Bollinger, Fold: 0):

```bash
python main.py
```

Results will be saved to `runs/reproduction_run`, including the generated strategy code and fitness plots.

## Project Structure

*   `profit/optimizer.py`: The core ProFiT evolutionary loop (Algorithm 1).
*   `profit/llm/client.py`: LLM interaction logic (Analysis, Improvement, Repair prompts).
*   `profit/seed_strategies.py`: The 5 initial seed strategies and baselines.
*   `profit/backtester/lucit_adapter.py`: Adapter for the backtesting engine.
*   `profit/strategy.py`: Definition of the code-based Chromosome.

## Differences from Original Paper

*   This implementation currently runs on a single Fold per execution (configurable in `main.py`).
*   Default config uses a Mock LLM for safety; switch to "openai" for real reproduction.

## License

MIT