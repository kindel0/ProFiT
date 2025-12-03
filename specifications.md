Final Technical Specification: ProFiT Framework
Project Name: ProFiT (PROgrammatic FInancial-strategy-Tuning)
Version: 1.0
Date: December 3, 2025
Core Vision: A Python framework for quantitative researchers to discover and optimize financial trading strategies through a hybrid approach of Genetic Algorithms (GA), Genetic Programming (GP), and Large Language Models (LLM).
1. Framework Core & Architecture
Architectural Paradigm: Desktop-first, API-driven, with a focus on historical research and reproducibility. The system prioritizes computational power and correctness over real-time capabilities or polished user interfaces.

Financial Environment/Simulator:

Data Interface: A standardized DataProvider class will be implemented. The system will ship with a default ParquetProvider and CSVProvider. It is the user's responsibility to provide clean, timestamp-indexed, and corporate-action-adjusted data. The framework will perform validation (e.g., checking for missing columns) but not cleaning.
Backtesting Engine: The primary backtester will be event-driven to realistically model transaction costs, but a simplified vectorized mode will be available for rapid testing of simple, non-path-dependent strategies.
Walk-Forward Validation: A core utility that programmatically splits the user's dataset into sequential training and out-of-sample test sets to prevent overfitting and produce robust performance metrics.
Genetic Algorithm (GA) & Genetic Programming (GP) Engine:

Chromosome Representation: A strategy chromosome is a multi-part dictionary structure, enabling complex evolution:
/parameters: A dictionary of values for pre-defined template logic (e.g., {"rsi_period": 14, "ma_period": 200}).
/rules: A list of rule objects, where each rule has a condition and an action (e.g., {"condition": "c1 and c2", "action": "enter_long"}).
/features: A dictionary where keys are feature names (e.g., c1) and values are expression trees representing mathematical formulas (e.g., (close - open) / (high - low)). This allows the GP engine to invent novel features from OHLCV data.
Evolutionary Operations:
Crossover: Context-aware crossover will swap entire rules, feature trees, or parameter blocks between parent chromosomes.
Mutation: A two-pronged approach:
Standard Mutation: Random tweaks to parameters, operators, and expression tree nodes.
LLM-Powered Mutation: High-performing strategies are selected. Their chromosome structure and performance metrics are serialized. An LLM is prompted to analyze this data and return a structured JSON Patch (RFC 6902) to suggest a targeted mutation.
LLM Integration Layer:

LLM Client Interface: A simple abstract base class LLMClient defines a generate_patch method. The user instantiates a concrete client (e.g., OpenAIClient, HuggingFaceLocalClient) and passes it into the configuration. This makes the framework LLM-agnostic.
Prompt Orchestrator: Manages a library of YAML-based prompt templates. It dynamically injects the strategy's data and performance metrics to generate the final prompt.
Security Sandbox: The returned JSON Patch is validated for structural correctness and applied in a controlled manner. This entirely avoids executing LLM-generated code, providing maximum security.
2. Data Management & Preprocessing
Data Expectation: The framework assumes the user is a power user who provides high-quality, pre-cleaned data (e.g., in .parquet or .csv format). All timestamps are expected to be in UTC.
Feature Engineering Library: ProFiT will ship with a profit.indicators module containing a comprehensive set of common financial indicators (e.g., RSI, MACD, Bollinger Bands, ATR) built on performant libraries like TA-Lib or pandas-ta.
Reproducibility: The framework will strongly encourage users to version their experiments using Git for code and DVC (Data Version Control) for datasets. The run configuration itself (YAML file) should be committed to Git.
3. Training & Optimization Features
GA Training Loop & Orchestrator: A central profit.Optimizer class manages the main training loop: initialization, evaluation, selection, evolution. It will handle checkpointing (saving state to disk periodically) so long runs can be resumed.
Multi-Objective Optimization: The core fitness evaluation will use the NSGA-II algorithm to find the Pareto optimal front for three specific, hardcoded objectives:
Sharpe Ratio (Maximize)
Annualized Return (Maximize)
Expectancy per Trade (Maximize)
Parallelization & Resource Management: The evaluation of the GA population will be parallelized across all available local CPU cores using libraries like Ray or Python's native multiprocessing. The framework is designed for a single powerful machine (e.g., a 16-core desktop), not a distributed cluster.
Hyperparameter Optimization: Integration with Optuna will allow users to run meta-optimizations on the GA/LLM parameters themselves (e.g., population size, mutation rates, LLM temperature).
4. Observability & Diagnostics
Logging: All logs will be structured (JSON format) and written to a simple local file named {run_id}.log. Each log entry will contain a run_id and generation_id for easy filtering and analysis with tools like grep or jq.
Metrics: Key time-series metrics (e.g., population fitness, LLM latency) will be appended to a simple text file, {run_id}_metrics.txt, in the OpenMetrics format for potential future parsing.
Strategy Lineage: During a run, a file named {run_id}_lineage.csv will be generated. Each row will contain child_id, parent1_id, parent2_id, mutation_type, providing a raw but complete traceability map of every strategy's origin.
5. Visualization & Reporting Tools
Workflow: Visualization is a post-processing step, not a live dashboard. After a run completes, the user will call results.generate_report(output_dir).
Static Chart Generation: The report generator will use Matplotlib to create and save several .png image files to the specified output directory.
Mandatory Visualizations (MVP):
3D Pareto Front Plot (pareto_front.png): A 3D scatter plot visualizing the final non-dominated set of strategies across the Sharpe/Return/Expectancy axes.
Fitness Convergence Plot (fitness_convergence.png): A 2D line chart showing the max and average fitness for each of the three objectives across all generations.
Strategy Tearsheets (/strategies/): A subdirectory will be created. For each of the top N strategies on the Pareto front, a detailed tearsheet image (strategy_{id}_tearsheet.png) will be generated. This tearsheet will include:
Equity Curve
Drawdown Plot
A table of key performance metrics (Calmar, win rate, etc.)
A text-based representation of the strategy's chromosome (parameters, rules, and features).
6. API Design & Extensibility
Primary Interaction: The user interacts with the framework through a clean, programmatic Python API. The core workflow is Configure -> Optimize -> Analyze.
Configuration Management: Runs are configured using a YAML file. This file is loaded into a Pydantic Configuration object at runtime, which provides validation, type safety, and editor auto-completion. This ensures experiments are reproducible and easy to manage.
# config.yaml
data:
  path: "./data/clean_data.parquet"
  asset: "BTC-USD"
ga:
  population_size: 200
  generations: 50
llm:
  client: "openai" # Corresponds to a registered client
  model: "gpt-4-turbo-preview"
  # api_key is loaded from environment variables, not stored here
Copy
Extensibility (Plugin Model): The framework will be extensible by design. Users can register their own custom components:
Indicators: By placing a file in a designated plugins folder.
Fitness Objectives: Via a profit.register_objective() function.
LLM Clients: By subclassing profit.clients.LLMClient and registering the new class.