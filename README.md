# ProFiT: Program Search for Financial Trading

This repository contains a reproduction of the **ProFiT framework** as described in the research paper [*"ProFiT: Program Search for Financial Trading"*](https://www.researchgate.net/publication/398248186_ProFiT_Program_Search_for_Financial_Trading).

ProFiT is an LLM-driven evolutionary search algorithm that autonomously discovers and improves algorithmic trading strategies. It uses a closed feedback loop where an LLM analyzes strategy performance on backtests and proposes code-level improvements.

## Key Features

- **LLM-Driven Evolution**: Uses Large Language Models to act as both "Analyst" and "Developer" to mutate trading strategies
- **Multiple LLM Backends**: Supports OpenAI, DeepSeek, and Ollama (local models)
- **Code-Level Search**: Strategies are represented as executable Python code, not abstract parameters
- **Self-Repair**: Includes a repair loop where the LLM fixes syntax or runtime errors based on tracebacks
- **Walk-Forward Validation**: Implements 5-fold time-series split with train/validation/test evaluation
- **Multi-Objective Optimization**: NSGA-II support for optimizing multiple objectives simultaneously
- **Batch Execution**: Run multiple fold/seed combinations with parallel execution support
- **Statistical Analysis**: Confidence intervals, significance tests, and effect size calculations
- **Robust Backtesting**: Uses `backtesting.py` (via `lucit-backtesting`) with realistic constraints

## Installation

This project is managed with `uv`.

```bash
# Install dependencies
uv sync

# Install package in editable mode
uv pip install -e .
```

## Environment Setup

Set up API keys based on your chosen LLM provider:

```bash
# For OpenAI
export OPENAI_API_KEY="sk-..."

# For DeepSeek
export DEEPSEEK_API_KEY="sk-..."

# For Ollama (local) - no API key needed, just run:
ollama serve
ollama pull qwen2.5-coder:14b
```

## Configuration

Edit `config.yaml` to control experiment parameters:

```yaml
# Data configuration
data:
  path: "tests/data/Bitcoin_historical_data_coinmarketcap.csv"
  asset: "BTC-USD"

# Genetic Algorithm parameters
ga:
  population_size: 10  # Max population size (trimmed via NSGA-II)
  generations: 15      # Number of evolutionary steps

# Optimization objectives
objectives:
  - "annualized_return"
  # Uncomment for multi-objective optimization (NSGA-II):
  # - "sharpe_ratio"

# LLM configuration
llm:
  client: "openai"     # Options: "openai", "deepseek", "ollama", "mock"
  model: "gpt-4o"      # Model name for the chosen client
  # base_url: "..."    # Optional: custom API endpoint
```

### Configuration Options

| Section | Field | Description |
|---------|-------|-------------|
| `data` | `path` | Path to OHLCV data file (CSV or Parquet) |
| `data` | `asset` | Asset identifier (for logging) |
| `ga` | `population_size` | Maximum population size |
| `ga` | `generations` | Number of evolutionary iterations |
| `objectives` | - | List of metrics to optimize: `annualized_return`, `sharpe_ratio`, `expectancy` |
| `llm` | `client` | LLM provider: `openai`, `deepseek`, `ollama`, `mock` |
| `llm` | `model` | Model name (e.g., `gpt-4o`, `deepseek-coder`, `qwen2.5-coder:14b`) |
| `llm` | `base_url` | Optional custom API endpoint |

### LLM Client Options

| Client | Model Examples | API Key |
|--------|---------------|---------|
| `openai` | `gpt-4o`, `gpt-4-turbo` | `OPENAI_API_KEY` |
| `deepseek` | `deepseek-coder`, `deepseek-chat` | `DEEPSEEK_API_KEY` |
| `ollama` | `qwen2.5-coder:14b`, `codellama` | None (local) |
| `mock` | `mock` | None (for testing) |

## Running ProFiT

### Basic Usage

```bash
# Single run with default settings (demo fold, seed 0)
uv run python main.py

# Specify fold and seed
uv run python main.py --fold 0 --seed 0

# Custom config file
uv run python main.py --config my_config.yaml
```

### Batch Execution

Run multiple fold/seed combinations:

```bash
# Run all 5 seeds on fold 0
uv run python main.py --fold 0 --seeds 0,1,2,3,4

# Run all 5 folds with seed 0
uv run python main.py --folds 0,1,2,3,4 --seed 0

# Full batch: 5 folds x 5 seeds = 25 runs
uv run python main.py --folds 0,1,2,3,4 --seeds 0,1,2,3,4

# Parallel execution (requires Ray)
uv run python main.py --folds 0,1,2,3,4 --seeds 0,1,2,3,4 --parallel
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--config`, `-c` | Path to config file (default: `config.yaml`) |
| `--fold` | Single fold index (0-4, or 5 for demo fold) |
| `--folds` | Comma-separated fold indices (e.g., `0,1,2,3,4`) |
| `--seed` | Single seed strategy index (0-4) |
| `--seeds` | Comma-separated seed indices (e.g., `0,1,2,3,4`) |
| `--parallel` | Run fold/seed combinations in parallel |
| `--output`, `-o` | Output directory (default: `runs`) |

### Fold Definitions

| Fold | Train Period | Validation | Test Period |
|------|--------------|------------|-------------|
| 0 | 2008-01 to 2010-06 | 2010-06 to 2011-01 | 2011-01 to 2011-07 |
| 1 | 2011-07 to 2014-01 | 2014-01 to 2014-08 | 2014-08 to 2015-02 |
| 2 | 2015-02 to 2017-08 | 2017-08 to 2018-02 | 2018-03 to 2018-09 |
| 3 | 2018-09 to 2021-03 | 2021-03 to 2021-09 | 2021-09 to 2022-03 |
| 4 | 2022-03 to 2024-09 | 2024-09 to 2025-04 | 2025-04 to 2025-10 |
| 5 | Demo fold for testing with sample data |

### Seed Strategies

| Index | Strategy | Description |
|-------|----------|-------------|
| 0 | Bollinger Bands | Mean reversion based on Bollinger Bands |
| 1 | CCI | Commodity Channel Index momentum |
| 2 | EMA Crossover | Exponential moving average crossover |
| 3 | MACD | MACD signal line crossover |
| 4 | Williams %R | Williams Relative Strength oscillator |

## Output

### Single Run Output

```
runs/fold0_seed0/
├── fitness_growth.png      # Training fitness over generations
├── train_vs_val.png        # Train vs validation comparison
├── test_results.csv        # Test set metrics
├── test_summary.txt        # Human-readable test summary
└── strategies/             # Top 5 strategy code files
    ├── strategy_1.py
    ├── strategy_2.py
    └── ...
```

### Batch Run Output

```
runs/batch_run/
├── batch_summary.txt           # Aggregate statistics
├── statistical_analysis.txt    # Significance tests & effect sizes
├── per_run_results.csv         # Per-run metrics table
├── train_vs_test_comparison.png
├── confidence_intervals.png    # 95% CI visualization
└── runs/                       # Individual run reports
    ├── fold0_seed0/
    ├── fold0_seed1/
    └── ...
```

### Statistical Analysis

Batch runs include comprehensive statistical analysis:

- **Descriptive Statistics**: Mean, std, median, min, max
- **Confidence Intervals**: 95% CI using t-distribution
- **Significance Tests**: Paired t-test, Wilcoxon signed-rank
- **Effect Size**: Cohen's d with interpretation

## Project Structure

```
profit/
├── __init__.py          # Public API exports
├── optimizer.py         # Core ProFiT evolutionary loop (Algorithm 1)
├── batch.py             # Batch execution and BatchResults
├── analysis.py          # Statistical analysis functions
├── results.py           # Results and report generation
├── config.py            # Pydantic configuration models
├── strategy.py          # Code-based Chromosome definition
├── metrics.py           # Fitness metrics (return, Sharpe, expectancy)
├── seed_strategies.py   # 5 initial seed strategies
├── llm/
│   └── client.py        # LLM clients (OpenAI, DeepSeek, Ollama)
├── ga/
│   ├── individual.py    # Individual representation
│   ├── selection.py     # NSGA-II selection operators
│   ├── crossover.py     # Crossover operators (future use)
│   └── mutation.py      # Mutation operators (future use)
├── backtester/
│   ├── base.py          # Abstract backtester interface
│   └── lucit_adapter.py # backtesting.py adapter
└── data/
    └── provider.py      # CSV/Parquet data loading
```

## Multi-Objective Optimization

Enable NSGA-II by specifying multiple objectives:

```yaml
objectives:
  - "annualized_return"
  - "sharpe_ratio"
```

In multi-objective mode:
- Parent selection uses binary tournament with Pareto rank and crowding distance
- Acceptance uses Pareto dominance (non-dominated candidates are accepted)
- Population trimming preserves diversity via NSGA-II selection

## Running Tests

```bash
uv run pytest -v
```

## License

MIT
