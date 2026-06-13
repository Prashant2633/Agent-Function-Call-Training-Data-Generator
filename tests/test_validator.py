"""
test_validator.py — Tests for the three-stage validation engine.

Tests each validation stage independently:
  Stage 1: Schema validation (jsonschema)
  Stage 2: Semantic similarity (embedding cosine sim)
  Stage 3: Multi-turn chain coherence
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    ConversationTurn, Example, ExampleType, Generator,
    Instruction, ToolCall, Difficulty
)
from src.registry import load_registry
from src.validator import Validator


@pytest.fixture(scope="module")
def registry():
    return load_registry()


@pytest.fixture(scope="module")
def validator(registry):
    return Validator(registry)


def make_example(
    instruction_text: str,
    tool_name: str,
    arguments: dict,
    domain: str = "calendar",
    example_type: int = 1,
) -> Example:
    """Helper to build a minimal Example for testing."""
    tool_call = ToolCall(name=tool_name, arguments=arguments)
    return Example(
        instruction_id="test-instr-123",
        instruction_text=instruction_text,
        domain=domain,
        example_type=example_type,
        difficulty="easy",
        generator=Generator.GROQ,
        tool_calls=[tool_call],
        conversation=[
            ConversationTurn(role="user", content=instruction_text),
            ConversationTurn(role="assistant", tool_calls=[tool_call]),
        ],
    )


def test_valid_calendar_example_passes(validator: Validator):
    """A correctly formed create_event call should pass validation."""
    example = make_example(
        instruction_text="Schedule a team meeting tomorrow at 10am for 1 hour",
        tool_name="create_event",
        arguments={
            "title": "Team Meeting",
            "start_time": "2024-01-16T10:00:00Z",
            "end_time": "2024-01-16T11:00:00Z",
            "attendees": ["alice@example.com"],
            "timezone": "UTC",
            "recurrence": "none",
        },
        domain="calendar",
    )
    result = validator.validate(example)
    assert result.is_valid, f"Expected valid but got failures: {result.failure_details}"


def test_hallucinated_tool_fails(validator: Validator):
    """A tool name not in the registry should trigger HALLUCINATED_TOOL."""
    example = make_example(
        instruction_text="Book a flight to Paris",
        tool_name="book_flight",  # Not in registry
        arguments={"destination": "Paris"},
        domain="calendar",
    )
    result = validator.validate(example)
    assert not result.is_valid
    assert "HALLUCINATED_TOOL" in result.failure_modes


def test_missing_required_param_fails(validator: Validator):
    """Missing a required parameter should fail with MISSING_REQUIRED."""
    example = make_example(
        instruction_text="Delete the event",
        tool_name="delete_event",
        arguments={"notify_attendees": True},  # Missing required 'event_id'
        domain="calendar",
    )
    result = validator.validate(example)
    assert not result.is_valid
    assert "MISSING_REQUIRED" in result.failure_modes


def test_enum_violation_fails(validator: Validator):
    """Passing an invalid enum value should fail with ENUM_VIOLATION."""
    example = make_example(
        instruction_text="Get directions by spaceship",
        tool_name="get_directions",
        arguments={
            "origin": "New York",
            "destination": "Los Angeles",
            "mode": "spaceship",  # Not in enum: driving/walking/cycling/transit
        },
        domain="maps",
    )
    result = validator.validate(example)
    assert not result.is_valid
    assert "ENUM_VIOLATION" in result.failure_modes


def test_hallucinated_param_fails(validator: Validator):
    """A parameter not in the schema should fail with HALLUCINATED_PARAM."""
    example = make_example(
        instruction_text="Search for python tutorials",
        tool_name="web_search",
        arguments={
            "query": "python tutorials",
            "quantum_filter": True,  # Hallucinated param
        },
        domain="search",
    )
    result = validator.validate(example)
    assert not result.is_valid
    assert "HALLUCINATED_PARAM" in result.failure_modes


def test_empty_example_type3_is_valid(validator: Validator):
    """Type 3 (ambiguous) examples with no tool calls are valid — they ask for clarification."""
    example = Example(
        instruction_id="test-instr-123",
        instruction_text="Send an email.",
        domain="email",
        example_type=3,
        difficulty="easy",
        generator=Generator.GEMINI,
        tool_calls=[],  # Intentionally empty
        conversation=[
            ConversationTurn(role="user", content="Send an email."),
            ConversationTurn(role="assistant", content="I need more information. What is the recipient's email address?"),
        ],
    )
    result = validator.validate(example)
    assert result.is_valid, f"Type 3 with clarification should be valid: {result.failure_details}"


def test_weather_valid_example(validator: Validator):
    """Valid get_current_weather call should pass."""
    example = make_example(
        instruction_text="What's the weather in Tokyo right now?",
        tool_name="get_current_weather",
        arguments={"location": "Tokyo, Japan", "units": "metric"},
        domain="weather",
    )
    result = validator.validate(example)
    assert result.is_valid
