"""
Abstract base class and mock implementation for LLM clients.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Type

import jsonpatch

from openai import OpenAI
import os
from profit.strategy import Chromosome

# Registry for LLM clients
CLIENT_REGISTRY: Dict[str, Type["LLMClient"]] = {}


def register_client(name: str, client_class: Type["LLMClient"]):
    """
    Registers a new LLM client class.

    Args:
        name (str): The identifier for the client.
        client_class (Type["LLMClient"]): The client class to register.
    """
    if name in CLIENT_REGISTRY:
        return # Client already registered, do nothing.
    CLIENT_REGISTRY[name] = client_class


def get_client(name: str, **kwargs) -> "LLMClient":
    """
    Retrieves an instance of a registered LLM client.

    Args:
        name (str): The identifier of the client to retrieve.
        **kwargs: Keyword arguments to pass to the client's constructor.

    Returns:
        LLMClient: An instance of the requested client.
    """
    if name not in CLIENT_REGISTRY:
        raise ValueError(
            f"LLM Client '{name}' is not registered. Available: {list(CLIENT_REGISTRY.keys())}"
        )
    client_class = CLIENT_REGISTRY[name]
    return client_class(**kwargs)


class LLMClient(ABC):
    """
    Abstract base class for all Large Language Model clients.

    This class defines the interface for generating a JSON patch to mutate a
    strategy's chromosome based on its performance.
    """

    @abstractmethod
    def generate_patch(
        self,
        chromosome: Chromosome,
        performance_metrics: Dict[str, Any],
    ) -> jsonpatch.JsonPatch:
        """
        Generates a JSON patch to mutate a chromosome.

        Args:
            chromosome (Chromosome): The chromosome of the strategy to mutate.
            performance_metrics (Dict[str, Any]): A dictionary of performance
                metrics for the given strategy.

        Returns:
            jsonpatch.JsonPatch: A JSON patch object with proposed mutations.
        """
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """
    A mock LLM client for testing purposes.

    This client returns a predefined JSON patch to facilitate predictable
    tests without making actual LLM API calls.
    """

    def __init__(self, patch: jsonpatch.JsonPatch = None):
        """
        Initializes the mock client with a specific patch to return.

        Args:
            patch (jsonpatch.JsonPatch): The patch to be returned by
                `generate_patch`. Defaults to an empty patch.
        """
        self._patch = patch if patch is not None else jsonpatch.JsonPatch([])

    def generate_patch(
        self,
        chromosome: Chromosome,
        performance_metrics: Dict[str, Any],
    ) -> jsonpatch.JsonPatch:
        """
        Returns the predefined JSON patch.

        Args:
            chromosome (Chromosome): Ignored.
            performance_metrics (Dict[str, Any]): Ignored.

        Returns:
            jsonpatch.JsonPatch: The predefined patch.
        """
        return self._patch


class OpenAIClient(LLMClient):
    """
    An example implementation of an LLM client using OpenAI's API.

    Note: This is a placeholder implementation. Actual API calls and logic
    to generate patches based on model responses should be implemented.
    """

    _SYSTEM_PROMPT_TEMPLATE = (
        "You are an expert in financial trading strategies and genetic algorithms. "
        "Your task is to analyze a strategy's performance and propose improvements "
        "by generating a JSON patch (RFC 6902) that mutates its chromosome. "
        "The goal is to optimize for Sharpe Ratio, Annualized Return, and Expectancy per Trade. "
        "The output MUST be a valid JSON object representing the JSON patch."
    )

    _USER_PROMPT_TEMPLATE = (
        "Given the following strategy chromosome and its performance metrics, "
        "propose mutations to improve its performance according to the stated objectives. "
        "Return the mutations as a JSON patch (RFC 6902). "
        "You MUST respond with a JSON object containing the patch. \n\n"
        "Here are a few examples of valid JSON strategy (Chromosome) structures:\n\n"
        "Example 1:\n"
        "```json\n"
        "{\n"
        "  \"parameters\": {\n"
        "    \"fast_ma\": 10,\n"
        "    \"slow_ma\": 30,\n"
        "    \"rsi_period\": 14\n"
        "  },\n"
        "  \"rules\": [\n"
        "    {\n"
        "      \"condition\": \"signal_long and not overbought\",\n"
        "      \"action\": \"enter_long\"\n"
        "    },\n"
        "    {\n"
        "      \"condition\": \"signal_short and not oversold\",\n"
        "      \"action\": \"enter_short\"\n"
        "    },\n"
        "    {\n"
        "      \"condition\": \"take_profit or stop_loss\",\n"
        "      \"action\": \"exit_position\"\n"
        "    }\n"
        "  ],\n"
        "  \"features\": {\n"
        "    \"signal_long\": \"sma(close, fast_ma) > sma(close, slow_ma)\",\n"
        "    \"signal_short\": \"sma(close, fast_ma) < sma(close, slow_ma)\",\n"
        "    \"overbought\": \"rsi(close, rsi_period) > 70\",\n"
        "    \"oversold\": \"rsi(close, rsi_period) < 30\",\n"
        "    \"take_profit\": \"current_profit_percentage > 0.05\",\n"
        "    \"stop_loss\": \"current_loss_percentage > 0.03\"\n"
        "  }\n"
        "}\n"
        "```\n\n"
        "Example 2:\n"
        "```json\n"
        "{\n"
        "  \"parameters\": {\n"
        "    \"macd_fast\": 12,\n"
        "    \"macd_slow\": 26,\n"
        "    \"macd_signal\": 9\n"
        "  },\n"
        "  \"rules\": [\n"
        "    {\n"
        "      \"condition\": \"macd_cross_up\",\n"
        "      \"action\": \"enter_long\"\n"
        "    },\n"
        "    {\n"
        "      \"condition\": \"macd_cross_down\",\n"
        "      \"action\": \"exit_long\"\n"
        "    }\n"
        "  ],\n"
        "  \"features\": {\n"
        "    \"macd_cross_up\": \"macd_line > macd_signal_line and macd_line[-1] < macd_signal_line[-1]\",\n"
        "    \"macd_cross_down\": \"macd_line < macd_signal_line and macd_line[-1] > macd_signal_line[-1]\"\n"
        "  }\n"
        "}\n"
        "```\n\n"
        "Current Chromosome: {chromosome_json}\n"
        "Current Performance Metrics: {performance_metrics_json}"
    )

    def __init__(self, model: str = "gpt-4"):
        """
        Initializes the OpenAI client.

        Args:
            api_key (str): The API key for authenticating with OpenAI.
            model (str): The model to use for generating responses.
        """
        self.env_name = "OPENAI_API_KEY"
        self.api_key = None
        if self.env_name in os.environ:
            self.api_key = os.environ[self.env_name]
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def generate_patch(
        self,
        chromosome: Chromosome,
        performance_metrics: Dict[str, Any],
    ) -> jsonpatch.JsonPatch:
        """
        Generates a JSON patch using OpenAI's API.

        Args:
            chromosome (Chromosome): The chromosome of the strategy to mutate.
            performance_metrics (Dict[str, Any]): A dictionary of performance
                metrics for the given strategy.

        Returns:
            jsonpatch.JsonPatch: A JSON patch object with proposed mutations.
        """
        responses = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},  # type: ignore
            messages=[
                {
                    "role": "system",
                    "content": self._SYSTEM_PROMPT_TEMPLATE,
                },
                {
                    "role": "user",
                    "content": self._USER_PROMPT_TEMPLATE.format(
                        chromosome_json=chromosome.model_dump_json(),
                        performance_metrics_json=performance_metrics,
                    ),
                },
            ],
        )
        # The content is already a JSON string due to response_format
        json_patch_str = responses.choices[0].message.content
        try:
            patch = jsonpatch.JsonPatch.from_string(json_patch_str)
            return patch
        except jsonpatch.JsonPatchException as e:
            raise ValueError(
                f"Received invalid JSON patch from OpenAI response: {e}\nResponse content: {json_patch_str}"
            )


# Register the mock client for testing and demonstration
register_client("mock", MockLLMClient)
register_client("openai", OpenAIClient)
register_client("openai", OpenAIClient)
