"""
Tests for the LLMClient and its mock implementation.
"""
import jsonpatch

from profit.llm.client import MockLLMClient
from profit.strategy import Chromosome, Rule


def test_mock_llm_client():
    """
    Tests that the MockLLMClient returns the patch it was initialized with.
    """
    # A sample patch to replace a parameter
    patch_data = [{"op": "replace", "path": "/parameters/rsi_period", "value": 21}]
    patch = jsonpatch.JsonPatch(patch_data)
    
    client = MockLLMClient(patch=patch)
    
    # Create dummy inputs (they should be ignored by the mock client)
    dummy_chromosome = Chromosome(
        parameters={"rsi_period": 14},
        rules=[Rule(condition="c1", action="enter_long")],
        features={"c1": "rsi > 70"},
    )
    dummy_metrics = {"sharpe": 1.0}
    
    # The client should return the exact patch it was given
    returned_patch = client.generate_patch(dummy_chromosome, dummy_metrics)
    assert returned_patch == patch
    assert returned_patch.patch[0]["value"] == 21
