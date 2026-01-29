"""
Abstract base class and implementation for LLM clients using ProFiT prompts.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Type, Optional
import os
from openai import OpenAI
from profit.strategy import Chromosome

# Registry for LLM clients
CLIENT_REGISTRY: Dict[str, Type["LLMClient"]] = {}


def register_client(name: str, client_class: Type["LLMClient"]):
    if name in CLIENT_REGISTRY:
        raise ValueError(f"LLM Client '{name}' is already registered.")
    CLIENT_REGISTRY[name] = client_class


def get_client(name: str, **kwargs) -> "LLMClient":
    if name not in CLIENT_REGISTRY:
        raise ValueError(
            f"LLM Client '{name}' is not registered. Available: {list(CLIENT_REGISTRY.keys())}"
        )
    client_class = CLIENT_REGISTRY[name]
    return client_class(**kwargs)


class LLMClient(ABC):
    """
    Abstract base class for all Large Language Model clients in ProFiT.
    """

    @abstractmethod
    def analyze_strategy(
        self,
        chromosome: Chromosome,
        performance_metrics: Dict[str, Any],
    ) -> str:
        """
        Analyzes the strategy and proposes improvements (textual).
        """
        raise NotImplementedError

    @abstractmethod
    def improve_strategy(
        self,
        chromosome: Chromosome,
        analysis: str,
    ) -> str:
        """
        Rewrites the strategy code based on the analysis/proposals.
        Returns the new valid Python code.
        """
        raise NotImplementedError
    
    @abstractmethod
    def repair_strategy(
        self,
        code: str,
        error_traceback: str,
    ) -> str:
        """
        Repairs the strategy code based on the error traceback.
        """
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """
    A mock LLM client for testing purposes.
    """

    def __init__(self, **kwargs):
        pass

    def analyze_strategy(self, chromosome: Chromosome, performance_metrics: Dict[str, Any]) -> str:
        return "Generic improvement: Buy low, sell high."

    def improve_strategy(self, chromosome: Chromosome, analysis: str) -> str:
        # Just return the original code to keep it valid
        return chromosome.code

    def repair_strategy(self, code: str, error_traceback: str) -> str:
        return code


class OpenAIClient(LLMClient):
    """
    Implementation of LLM client using OpenAI's API with ProFiT prompts.
    """
    
    # Analysis Prompts
    _ANALYSIS_SYSTEM = "You are an expert quantitative strategist."
    _ANALYSIS_USER = """
Analyze the following trading strategy code and its recent backtest results.

Your task:
1. Identify the most significant weaknesses, inefficiencies, or sources of poor performance in the strategy.
2. Propose no more than 2-3 concrete, high-impact improvements that could realistically improve out-of-sample returns and robustness.

Guidelines:
- Be specific and actionable (for example: "replace SMA with EMA to improve responsiveness", "add a volatility filter to reduce false signals").
- Focus on core logic changes, not cosmetic or minor tweaks.
- Avoid generic advice (for example: "tune parameters", "improve risk management") unless you specify exactly how.
- Keep the response concise, in plain English, suitable for direct implementation in the next rewrite step.
- Do not output code here; only short textual suggestions.

Backtest summary:
{backtest_results}

Strategy code:
{strategy_code}
"""

    # Improvement Prompts
    _IMPROVEMENT_SYSTEM = "You are a quantitative trading developer."
    _IMPROVEMENT_USER = """
Using the original strategy code and the improvement proposals provided below, rewrite the strategy to incorporate the requested enhancements.

Requirements:
- Output only valid Python code.
- Do not include any explanations, comments, or extra text.
- The rewritten class must:
    - Keep the same class name.
    - Remain compatible with the backtesting.py library.
    - Compile and run inside a call such as: Backtest(data, StrategyClass, cash=10000, commission=0.002, exclusive_orders=True).run(**params)
    - Retain the existing structure unless a change is explicitly needed for the improvements.
    - Include all tunable hyperparameters as class variables with sensible defaults.

Improvement proposals:
{improvement_proposals}

Strategy code:
{strategy_code}
"""

    # Repair Prompt
    _REPAIR_SYSTEM = "You are an expert Python developer debugging a trading strategy."
    _REPAIR_USER = """
The following trading strategy code failed to compile or run. 

Error Traceback:
{traceback}

Strategy Code:
{strategy_code}

Please fix the error and return the full, valid Python code. 
Requirements:
- Output only valid Python code.
- Do not include any explanations or markdown formatting (unless code block).
- Ensure compatibility with backtesting.py.
"""

    def __init__(self, model: str = "gpt-4o"):
        self.env_name = "OPENAI_API_KEY"
        self.api_key = None
        if self.env_name in os.environ:
            self.api_key = os.environ[self.env_name]
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def _clean_code_response(self, response_content: str) -> str:
        """Helper to strip markdown code blocks if present."""
        content = response_content.strip()
        if content.startswith("```python"):
            content = content[len("```python"):]
        elif content.startswith("```"):
            content = content[len("```"):]
        
        if content.endswith("```"):
            content = content[:-3]
        
        return content.strip()

    def analyze_strategy(
        self,
        chromosome: Chromosome,
        performance_metrics: Dict[str, Any],
    ) -> str:
        metrics_str = ", ".join([f"{k}: {v}" for k, v in performance_metrics.items()])
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._ANALYSIS_SYSTEM},
                {"role": "user", "content": self._ANALYSIS_USER.format(
                    backtest_results=metrics_str,
                    strategy_code=chromosome.code
                )},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content

    def improve_strategy(
        self,
        chromosome: Chromosome,
        analysis: str,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._IMPROVEMENT_SYSTEM},
                {"role": "user", "content": self._IMPROVEMENT_USER.format(
                    improvement_proposals=analysis,
                    strategy_code=chromosome.code
                )},
            ],
            temperature=0.2, # Lower temp for code generation
        )
        return self._clean_code_response(response.choices[0].message.content)
    
    def repair_strategy(
        self,
        code: str,
        error_traceback: str,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._REPAIR_SYSTEM},
                {"role": "user", "content": self._REPAIR_USER.format(
                    traceback=error_traceback,
                    strategy_code=code
                )},
            ],
            temperature=0.2,
        )
        return self._clean_code_response(response.choices[0].message.content)


class OllamaClient(LLMClient):
    """
    Implementation of LLM client using Ollama's OpenAI-compatible API.
    """

    # Reuse the same prompts as OpenAIClient
    _ANALYSIS_SYSTEM = OpenAIClient._ANALYSIS_SYSTEM
    _ANALYSIS_USER = OpenAIClient._ANALYSIS_USER
    _IMPROVEMENT_SYSTEM = OpenAIClient._IMPROVEMENT_SYSTEM
    _IMPROVEMENT_USER = OpenAIClient._IMPROVEMENT_USER
    _REPAIR_SYSTEM = OpenAIClient._REPAIR_SYSTEM
    _REPAIR_USER = OpenAIClient._REPAIR_USER

    def __init__(self, model: str = "qwen2.5-coder:14b", base_url: str = None):
        self.model = model
        self.base_url = base_url or "http://localhost:11434/v1"
        # Ollama doesn't require an API key, but the OpenAI client needs one
        self.client = OpenAI(base_url=self.base_url, api_key="ollama")

    def _clean_code_response(self, response_content: str) -> str:
        """Helper to strip markdown code blocks if present."""
        content = response_content.strip()
        if content.startswith("```python"):
            content = content[len("```python"):]
        elif content.startswith("```"):
            content = content[len("```"):]

        if content.endswith("```"):
            content = content[:-3]

        return content.strip()

    def analyze_strategy(
        self,
        chromosome: Chromosome,
        performance_metrics: Dict[str, Any],
    ) -> str:
        metrics_str = ", ".join([f"{k}: {v}" for k, v in performance_metrics.items()])

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._ANALYSIS_SYSTEM},
                {"role": "user", "content": self._ANALYSIS_USER.format(
                    backtest_results=metrics_str,
                    strategy_code=chromosome.code
                )},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content

    def improve_strategy(
        self,
        chromosome: Chromosome,
        analysis: str,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._IMPROVEMENT_SYSTEM},
                {"role": "user", "content": self._IMPROVEMENT_USER.format(
                    improvement_proposals=analysis,
                    strategy_code=chromosome.code
                )},
            ],
            temperature=0.2,
        )
        return self._clean_code_response(response.choices[0].message.content)

    def repair_strategy(
        self,
        code: str,
        error_traceback: str,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._REPAIR_SYSTEM},
                {"role": "user", "content": self._REPAIR_USER.format(
                    traceback=error_traceback,
                    strategy_code=code
                )},
            ],
            temperature=0.2,
        )
        return self._clean_code_response(response.choices[0].message.content)


class DeepSeekClient(LLMClient):
    """
    Implementation of LLM client using DeepSeek's OpenAI-compatible API.
    """

    # Reuse the same prompts as OpenAIClient
    _ANALYSIS_SYSTEM = OpenAIClient._ANALYSIS_SYSTEM
    _ANALYSIS_USER = OpenAIClient._ANALYSIS_USER
    _IMPROVEMENT_SYSTEM = OpenAIClient._IMPROVEMENT_SYSTEM
    _IMPROVEMENT_USER = OpenAIClient._IMPROVEMENT_USER
    _REPAIR_SYSTEM = OpenAIClient._REPAIR_SYSTEM
    _REPAIR_USER = OpenAIClient._REPAIR_USER

    def __init__(self, model: str = "deepseek-coder", base_url: str = None):
        self.env_name = "DEEPSEEK_API_KEY"
        self.api_key = os.environ.get(self.env_name)
        self.model = model
        self.base_url = base_url or "https://api.deepseek.com"
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _clean_code_response(self, response_content: str) -> str:
        """Helper to strip markdown code blocks if present."""
        content = response_content.strip()
        if content.startswith("```python"):
            content = content[len("```python"):]
        elif content.startswith("```"):
            content = content[len("```"):]

        if content.endswith("```"):
            content = content[:-3]

        return content.strip()

    def analyze_strategy(
        self,
        chromosome: Chromosome,
        performance_metrics: Dict[str, Any],
    ) -> str:
        metrics_str = ", ".join([f"{k}: {v}" for k, v in performance_metrics.items()])

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._ANALYSIS_SYSTEM},
                {"role": "user", "content": self._ANALYSIS_USER.format(
                    backtest_results=metrics_str,
                    strategy_code=chromosome.code
                )},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content

    def improve_strategy(
        self,
        chromosome: Chromosome,
        analysis: str,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._IMPROVEMENT_SYSTEM},
                {"role": "user", "content": self._IMPROVEMENT_USER.format(
                    improvement_proposals=analysis,
                    strategy_code=chromosome.code
                )},
            ],
            temperature=0.2,
        )
        return self._clean_code_response(response.choices[0].message.content)

    def repair_strategy(
        self,
        code: str,
        error_traceback: str,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._REPAIR_SYSTEM},
                {"role": "user", "content": self._REPAIR_USER.format(
                    traceback=error_traceback,
                    strategy_code=code
                )},
            ],
            temperature=0.2,
        )
        return self._clean_code_response(response.choices[0].message.content)


register_client("mock", MockLLMClient)
register_client("openai", OpenAIClient)
register_client("ollama", OllamaClient)
register_client("deepseek", DeepSeekClient)