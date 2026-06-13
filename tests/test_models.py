"""
test_models.py — Tests for Pydantic v2 data models.

Tests field validators, default values, and serialization.
"""

import pytest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    ConversationTurn, Example, ExampleType, Generator,
    Instruction, SeedInstruction, ToolCall, ToolSchema,
    ScoreResult, AxisScore, PreferencePair, Difficulty, FailureMode
)


def test_tool_call_model():
    """ToolCall should store name and arguments."""
    tc = ToolCall(name="web_search", arguments={"query": "OpenAI", "num_results": 5})
    assert tc.name == "web_search"
    assert tc.arguments["query"] == "OpenAI"


def test_example_has_auto_uuid():
    """Example should auto-generate a UUID id."""
    ex = Example(
        instruction_id="test-instr-123",
        instruction_text="Find the weather",
        domain="weather",
        example_type=1,
        difficulty="easy",
        generator=Generator.GROQ,
        tool_calls=[],
        conversation=[],
    )
    assert ex.id is not None
    assert len(ex.id) > 0


def test_example_serializes_to_json():
    """Example should serialize cleanly to JSON."""
    tc = ToolCall(name="get_current_weather", arguments={"location": "Paris"})
    ex = Example(
        instruction_id="test-instr-123",
        instruction_text="Weather in Paris?",
        domain="weather",
        example_type=1,
        difficulty="easy",
        generator=Generator.GEMINI,
        tool_calls=[tc],
        conversation=[ConversationTurn(role="user", content="Weather in Paris?")],
    )
    data = ex.model_dump(mode="json")
    serialized = json.dumps(data)
    assert "get_current_weather" in serialized
    assert "Weather in Paris" in serialized


def test_generator_enum_values():
    """Generator enum should have GROQ and GEMINI values."""
    assert Generator.GROQ.value == "groq"
    assert Generator.GEMINI.value == "gemini"


def test_example_type_enum_values():
    """ExampleType should have all 4 canonical types."""
    assert ExampleType.SINGLE == 1
    assert ExampleType.CHAIN == 2
    assert ExampleType.AMBIGUOUS == 3
    assert ExampleType.PARALLEL == 4


def test_score_result_valid():
    """ScoreResult should hold axis scores and composite."""
    score = ScoreResult(
        example_id="test-id",
        schema_correctness=AxisScore(score=1.0, reasoning="perfect", is_deterministic=True),
        argument_completeness=AxisScore(score=0.9, reasoning="good", is_deterministic=False),
        intent_alignment=AxisScore(score=0.8, reasoning="mostly", is_deterministic=False),
        hallucination_score=AxisScore(score=1.0, reasoning="none", is_deterministic=True),
        chain_coherence=AxisScore(score=0.75, reasoning="coherent", is_deterministic=True),
        composite_score=0.0,
        quality_tier="low",
    )
    assert score.composite_score == 0.905
    assert score.quality_tier == "high"
    assert score.schema_correctness.score == 1.0


def test_seed_instruction_parses_correctly():
    """SeedInstruction should parse from dict."""
    data = {
        "id": "calendar_t1_001",
        "text": "Schedule a meeting tomorrow at 3pm",
        "type": 1,
        "difficulty": "easy",
        "domain": "calendar",
        "expected_tools": ["create_event"],
        "notes": "Test note",
    }
    seed = SeedInstruction(**data)
    assert seed.id == "calendar_t1_001"
    assert seed.type == 1
    assert seed.domain == "calendar"


def test_failure_mode_enum_all_present():
    """All expected failure modes should exist."""
    expected = [
        "HALLUCINATED_TOOL", "HALLUCINATED_PARAM", "TYPE_MISMATCH",
        "MISSING_REQUIRED", "ENUM_VIOLATION", "LOW_SEMANTIC_SIMILARITY",
        "BROKEN_CHAIN", "MALFORMED_JSON", "API_ERROR", "DUPLICATE"
    ]
    for mode in expected:
        assert hasattr(FailureMode, mode), f"FailureMode missing: {mode}"
