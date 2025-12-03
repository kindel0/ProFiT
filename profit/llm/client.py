"""
Abstract base class and mock implementation for LLM clients.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Type

import jsonpatch

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
        raise ValueError(f"LLM Client '{name}' is already registered.")
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
        raise ValueError(f"LLM Client '{name}' is not registered. Available: {list(CLIENT_REGISTRY.keys())}")
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

# Register the mock client for testing and demonstration
register_client("mock", MockLLMClient)
