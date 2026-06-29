"""
Reproducing bounty finding: Memanto /answer endpoint drops sources and
hallucinates confidence, while /{agent_id}/answer correctly extracts
sources from the same Moorcheh SDK response.

Bounty issue: #770 (Memanto Bug & Exploit Challenge)
Author: Yzgaming005
"""

import pytest
from unittest.mock import MagicMock, PropertyMock, patch


@pytest.fixture
def mock_client_with_sources():
    """Mock Moorcheh client whose answer.generate returns real sources."""
    client = MagicMock()
    client.answer.generate.return_value = {
        "answer": "Photosynthesis converts CO2 + sunlight into glucose.",
        "sources": [
            {"id": "mem_001", "score": 0.92, "text": "Chloroplasts absorb sunlight..."},
            {"id": "mem_002", "score": 0.85, "text": "CO2 enters via stomata..."},
            {"id": "mem_003", "score": 0.71, "text": "Glucose is stored as starch..."},
        ],
    }
    return client


def test_legacy_answer_endpoint_returns_real_sources(mock_client_with_sources):
    """
    POST /answer (memanto/app/legacy/memory.py) MUST return sources
    extracted from the Moorcheh response, not an empty list.
    """
    from memanto.app.services.memory_read_service import MemoryReadService

    service = MemoryReadService(mock_client_with_sources)

    # Call generate_answer (which the legacy endpoint wraps)
    result = service.generate_answer(
        query="How does photosynthesis work?",
        scope_type="agent",
        scope_id="test",
    )

    # Assert sources are forwarded from SDK response
    assert "sources" in result, "generate_answer() must return 'sources'"
    assert len(result["sources"]) == 3, f"Expected 3 sources, got {len(result['sources'])}"
    assert result["sources"][0]["id"] == "mem_001"
    assert result["sources"][0]["score"] == 0.92

    # Assert confidence is computed from sources, not hardcoded
    assert "confidence" in result
    expected_confidence = round((0.92 + 0.85 + 0.71) / 3, 3)
    assert result["confidence"] == expected_confidence, f"Expected {expected_confidence}, got {result['confidence']}"
    assert result["confidence"] != 0.8, "confidence must not be hardcoded to 0.8"


def test_generate_answer_propagates_sources_from_sdk(mock_client_with_sources):
    """
    MemoryReadService.generate_answer() must include 'sources' in its
    return dict so downstream endpoints can surface them.
    """
    from memanto.app.services.memory_read_service import MemoryReadService

    service = MemoryReadService(mock_client_with_sources)

    result = service.generate_answer(
        query="How does photosynthesis work?",
        scope_type="agent",
        scope_id="test",
    )

    assert "sources" in result, "generate_answer() must surface 'sources'"
    assert len(result["sources"]) == 3, "should pass through all 3 SDK sources"
    assert "confidence" in result, "generate_answer() must include 'confidence'"


def test_confidence_is_derived_from_sources_not_hardcoded(mock_client_with_sources):
    """
    The endpoint must NOT hardcode confidence=0.8. If sources exist,
    confidence must reflect them (average relevance score).
    """
    from memanto.app.services.memory_read_service import MemoryReadService

    service = MemoryReadService(mock_client_with_sources)

    result = service.generate_answer(
        query="test query",
        scope_type="agent",
        scope_id="test",
    )

    # Expected confidence from real source scores: (0.92 + 0.85 + 0.71) / 3 = 0.827
    expected_avg = round((0.92 + 0.85 + 0.71) / 3, 3)

    # Assert production code computes confidence from sources
    assert result["confidence"] == expected_avg, f"Expected {expected_avg}, got {result['confidence']}"
    assert result["confidence"] != 0.8, "confidence must not be hardcoded to 0.8"


def test_malformed_score_does_not_crash(mock_client_with_sources):
    """
    If a source returns a non-numeric score, generate_answer must skip it
    instead of raising ValueError/TypeError.
    """
    client = MagicMock()
    client.answer.generate.return_value = {
        "answer": "Test",
        "sources": [
            {"id": "a", "score": "invalid", "text": "bad score"},
            {"id": "b", "score": 0.9, "text": "good score"},
        ],
    }

    from memanto.app.services.memory_read_service import MemoryReadService

    service = MemoryReadService(client)

    # Should not raise — malformed score is skipped
    result = service.generate_answer(
        query="test",
        scope_type="agent",
        scope_id="test",
    )

    # Only the valid score (0.9) should be used
    assert result["confidence"] == 0.9, f"Expected 0.9 (only valid score), got {result['confidence']}"
