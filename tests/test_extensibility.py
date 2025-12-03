"""
Tests for the extensibility and plugin registration mechanism.
"""
import pytest
from profit import metrics
from profit.llm import client as llm_client


def test_objective_registration():
    """
    Tests the registration of a custom objective function.
    """
    def custom_objective(returns) -> float:
        return returns.mean()
        
    # Check that it's not already there
    with pytest.raises(ValueError):
        metrics.get_objective("custom_objective")
        
    # Register and retrieve
    metrics.register_objective("custom_objective", custom_objective)
    retrieved_func = metrics.get_objective("custom_objective")
    assert retrieved_func == custom_objective
    
    # Check for duplicate registration
    with pytest.raises(ValueError):
        metrics.register_objective("custom_objective", custom_objective)


def test_llm_client_registration():
    """
    Tests the registration of a custom LLM client.
    """
    class CustomLLMClient(llm_client.LLMClient):
        def generate_patch(self, *args, **kwargs):
            return None # Dummy implementation
            
    # Check that it's not already there
    with pytest.raises(ValueError):
        llm_client.get_client("custom_client")
        
    # Register and retrieve
    llm_client.register_client("custom_client", CustomLLMClient)
    retrieved_client = llm_client.get_client("custom_client")
    assert isinstance(retrieved_client, CustomLLMClient)
    
    # Check for duplicate registration
    with pytest.raises(ValueError):
        llm_client.register_client("custom_client", CustomLLMClient)
