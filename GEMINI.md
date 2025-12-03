# AI Agent Development Protocol for the ProFiT Framework

## 1. Mission Statement

This document outlines the strict, non-negotiable rules and development protocol for the AI agent responsible for writing the source code of the **ProFiT** framework. The agent's primary directive is to produce code that is robust, maintainable, testable, and perfectly aligned with the technical specification. Adherence to this protocol is mandatory for every code generation task.

## 2. Core Mandates

### Rule #1: Python Language Exclusivity
The Agent MUST write all application and test code in Python 3.11 or newer. No other programming languages are permitted.

### Rule #2: Environment and Dependency Management (`uv` Only)
The Agent MUST use `uv` exclusively for all environment and dependency management tasks.
*   **Forbidden Tools:** `pip`, `venv`, `virtualenv`, `poetry`, `pipenv`, or any other package manager are strictly forbidden. The only exception is the initial bootstrap installation of `uv` itself.
*   **Creating Environments:** A virtual environment must be created with `uv venv`.
*   **Activating Environments:** The environment must be activated with `source .venv/bin/activate` (for Unix-like systems).
*   **Managing Dependencies:**
    *   To add a new dependency: `uv pip install <package_name>`
    *   To install all dependencies for the project: `uv pip sync requirements.txt`
    *   After adding a new dependency, the `requirements.txt` file MUST be updated with: `uv pip freeze > requirements.txt`

### Rule #3: Rigorous and Comprehensive Testing
Code without tests is considered broken.
*   **Test Framework:** The Agent MUST use the `pytest` framework for all unit and integration tests.
*   **Test-Driven Mindset:** For any new function or class, the test file MUST be created first or concurrently. The agent should follow a "Red-Green-Refactor" thought process.
*   **Test Location:** All test files must reside in the `/tests` directory and mirror the source code's directory structure. For example, the tests for `profit/core/optimizer.py` MUST be located at `tests/core/test_optimizer.py`.
*   **Coverage:** Code generation is not complete until the new code is covered by tests. The goal is to maintain 100% test coverage.

### Rule #4: Comprehensive Documentation via Docstrings
Every functional component MUST be documented.
*   **Docstring Format:** The Agent MUST write docstrings for every module, class, method, and function using the **Google Style** Python Docstrings format.
*   **Content:** Docstrings must include a one-line summary, a more detailed description, `Args:` for all parameters with their types, and a `Returns:` section describing the output and its type.

### Rule #5: Robust and Explicit Static Typing
Code must be clear about the data it handles.
*   **Type Hinting:** The Agent MUST use Python's `typing` module extensively for all function signatures, variables, and class attributes.
*   **No `Any`:** The use of `typing.Any` is strictly forbidden. The Agent must always endeavor to use more specific types, such as `list[str]`, `dict[str, float]`, `Callable[[int], str]`, or custom `TypeDict` and `NewType` definitions. If a generic type is absolutely necessary, `object` is preferred over `Any`.

## 3. Development Workflow Example

This is the standard operating procedure for adding a new function, `calculate_sharpe_ratio`, to a `metrics.py` module.

1.  **Define the Interface (Code and Test):**
    *   Create the file `profit/analyzer/metrics.py`.
    *   Write the function signature with types and a complete docstring *first*:
        ```python
        # profit/analyzer/metrics.py
        from typing import List

        def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float) -> float:
            """Calculates the annualized Sharpe ratio from a list of returns.

            Args:
                returns (List[float]): A list of periodic returns (e.g., daily).
                risk_free_rate (float): The annualized risk-free rate.

            Returns:
                float: The calculated annualized Sharpe ratio.
            """
            # Implementation to be added
            pass
        ```
    *   Create the test file `tests/analyzer/test_metrics.py`.
    *   Write a test case that will initially fail:
        ```python
        # tests/analyzer/test_metrics.py
        from profit.analyzer import metrics
        import pytest

        def test_calculate_sharpe_ratio():
            """Tests the Sharpe ratio calculation with a sample case."""
            # This test will fail until the function is implemented
            sample_returns = [0.01, -0.005, 0.02, 0.015]
            risk_free = 0.02
            # The expected value needs to be pre-calculated
            expected_sharpe = 1.234 # Placeholder
            assert metrics.calculate_sharpe_ratio(sample_returns, risk_free) == pytest.approx(expected_sharpe)
        ```

2.  **Implement the Functionality:**
    *   Write the business logic inside the `calculate_sharpe_ratio` function until the test `test_calculate_sharpe_ratio` passes.

3.  **Refactor and Add Edge Case Tests:**
    *   Add more tests to `test_metrics.py` covering edge cases: empty returns list, all zero returns, etc. Ensure they raise appropriate exceptions or return expected values (e.g., 0.0).
    *   Refactor the implementation for clarity and performance if needed.

4.  **Final Verification:**
    *   Before concluding the task, run the full verification suite from the project root:
        ```bash
        # 1. Run all tests
        uv run pytest

        # 2. Run the static type checker
        uv run mypy .

        # 3. Run the linter
        uv run ruff check .
        ```
    *   Only if all three commands pass without error is the task considered complete.

The Agent's performance will be judged on its strict adherence to this protocol.
