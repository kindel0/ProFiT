# ProFiT: A Hybrid GA-LLM Framework for Financial Strategy Optimization

![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)
![Code Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)

ProFiT (PROgrammatic FInancial-strategy-Tuning) is a desktop-first Python framework designed for quantitative researchers to discover, backtest, and optimize sophisticated financial trading strategies. It leverages a hybrid approach combining the evolutionary power of Genetic Algorithms (GA) and Genetic Programming (GP) with the contextual, pattern-matching capabilities of Large Language Models (LLM).

This framework is built for **power users** who need a robust, reproducible, and highly customizable environment for serious quantitative research.

## Core Concepts

*   **Hybrid Evolutionary Engine:** ProFiT doesn't just tune parameters. It evolves the very logic of trading strategies. The GA/GP engine mutates rules and invents new features, while the LLM acts as a "domain expert," suggesting intelligent mutations based on a strategy's performance profile.
*   **Genetic Programming for Feature Discovery:** Instead of being limited to a fixed set of indicators, ProFiT’s GP engine can generate novel features by creating mathematical expression trees from raw `OHLCV` data (e.g., `(high - low) / close`).
*   **Safe LLM Integration:** The LLM does not generate executable code. It returns a structured **JSON Patch** containing proposed modifications. This makes a strategy's evolution safe, inspectable, and deterministic.
*   **Reproducibility First:** Every component—from data handling to configuration—is designed for perfect run reproducibility. Experiments are defined in version-controllable YAML files and results are tied to specific run IDs.

## Features

*   **Multi-Objective Optimization:** Optimize strategies across a user-defined Pareto front using the NSGA-II algorithm. The default objectives are Sharpe Ratio, Annualized Return, and Expectancy.
*   **Event-Driven Backtester:** A realistic simulator that can model transaction costs, slippage, and latency.
*   **LLM Agnostic:** Use any LLM (OpenAI, an open-source model via Hugging Face, etc.) through a simple client interface.
*   **Parallelized Performance:** Leverages `Ray` or `multiprocessing` to distribute backtesting across all available CPU cores for rapid evaluation of GA populations.
*   **Extensible by Design:** A simple plugin architecture allows users to easily add their own indicators, fitness objectives, and LLM clients.
*   **Comprehensive Reporting:** Automatically generates static `.png` reports for each run, including:
    *   3D Pareto Front plots
    *   Fitness convergence charts
    *   Detailed strategy tearsheets with equity curves, drawdown plots, and performance metrics.
*   **Detailed Observability:** Structured logging, metrics, and strategy lineage tracking provide deep insight into the evolutionary process.

## Installation

ProFiT uses `uv` for ultra-fast project and virtual environment management.

1.  **Install `uv`** (if you haven't already):
    ```bash
    # On macOS / Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # On Windows
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-org/ProFiT.git
    cd ProFiT
    ```

3.  **Create and Activate Virtual Environment:**
    ```bash
    uv venv
    source .venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    uv 
    ```

## Quick Start

1.  **Configure Your Run:** Create a `config.yaml` file to define your experiment. Set your LLM API keys as environment variables (e.g., `OPENAI_API_KEY`).

    ```yaml
    # config.yaml
    data:
      path: "./data/my_clean_btc_data.parquet"
      asset: "BTC-USD"
    ga:
      population_size: 100
      generations: 50
    objectives:
      - "sharpe_ratio"
      - "annualized_return"
      - "expectancy"
    llm:
      client: "openai"
      model: "gpt-4-turbo"
    ```

2.  **Run the Optimizer:** Create a `main.py` file to launch the framework.

    ```python
    # main.py
    import profit
    import os

    # Set API key from environment variable
    # The client will automatically pick it up
    # os.environ["OPENAI_API_KEY"] = "sk-..."

    # 1. Load configuration from file
    config = profit.load_config("config.yaml")

    # 2. Create and run the optimizer
    optimizer = profit.Optimizer(config)
    results = optimizer.run()

    # 3. Generate the final report
    print("Optimization complete. Generating report...")
    results.generate_report(output_dir="runs/my_first_run")
    print("Report generated in 'runs/my_first_run'.")
    ```

3.  **Execute the Run:**
    ```bash
    python main.py
    ```

This will start the optimization process. Upon completion, you will find all logs, metrics, and visualization charts in the `runs/my_first_run/` directory.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
