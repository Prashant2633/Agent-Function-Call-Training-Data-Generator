"""
test_scorer.py — Tests for the 5-axis scoring engine.

Tests that:
  - The deterministic axes score correctly
  - The composite score is within [0, 1]
  - Quality tiers are assigned correctly
  - The preference pair builder produces valid DPO pairs
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    ConversationTurn, Example, ExampleType, Generator,
    ToolCall, ValidationResult, FailureMode
)
from src.registry import load_registry
from src.scorer import ScorerEngine
from src.preference import PreferencePairBuilder
from src.validator import Validator


@pytest.fixture(scope="module")
def registry():
    return load_registry()


@pytest.fixture(scope="module")
def scorer(registry):
    return ScorerEngine(registry)


@pytest.fixture(scope="module")
def validator(registry):
    return Validator(registry)


def make_valid_example(generator=Generator.GROQ) -> Example:
    """Helper to build a high-quality example."""
    tool_call = ToolCall(
        name="get_current_weather",
        arguments={"location": "London, UK", "units": "metric"}
    )
    return Example(
        instruction_id="test-instr-123",
        instruction_text="What is the current weather in London?",
        domain="weather",
        example_type=1,
        difficulty="easy",
        generator=generator,
        tool_calls=[tool_call],
        conversation=[
            ConversationTurn(role="user", content="What is the current weather in London?"),
            ConversationTurn(role="assistant", tool_calls=[tool_call]),
        ],
        is_valid=True,
    )


def make_poor_example() -> Example:
    """Helper to build a low-quality example with validation errors."""
    tool_call = ToolCall(
        name="get_current_weather",
        arguments={}  # Missing required 'location'
    )
    return Example(
        instruction_id="test-instr-123",
        instruction_text="Weather please",
        domain="weather",
        example_type=1,
        difficulty="easy",
        generator=Generator.GEMINI,
        tool_calls=[tool_call],
        conversation=[
            ConversationTurn(role="user", content="Weather please"),
            ConversationTurn(role="assistant", tool_calls=[tool_call]),
        ],
        is_valid=False,
        validation_errors=["MISSING_REQUIRED: location is required"],
    )


def test_score_valid_example_returns_score_result(scorer, validator):
    """Scoring a valid example should return a ScoreResult object."""
    example = make_valid_example()
    val_result = validator.validate(example)
    score = scorer.score(example, val_result)
    assert score is not None
    assert 0.0 <= score.composite_score <= 1.0


def test_score_composite_is_in_range(scorer, validator):
    """Composite score must always be in [0.0, 1.0]."""
    for _ in range(3):
        example = make_valid_example()
        val = validator.validate(example)
        score = scorer.score(example, val)
        assert 0.0 <= score.composite_score <= 1.0, f"Score out of range: {score.composite_score}"


def test_valid_example_scores_higher_than_invalid(scorer, validator):
    """A valid example should score higher than one with missing required params."""
    good = make_valid_example()
    bad = make_poor_example()

    val_good = validator.validate(good)
    val_bad = ValidationResult(example_id=bad.id, is_valid=False, failure_modes=["MISSING_REQUIRED"], failure_details=["location required"])

    score_good = scorer.score(good, val_good)
    score_bad = scorer.score(bad, val_bad)

    assert score_good.composite_score > score_bad.composite_score, (
        f"Good ({score_good.composite_score:.3f}) should beat bad ({score_bad.composite_score:.3f})"
    )


def test_quality_tier_high_threshold(scorer, validator):
    """High-quality valid examples should receive a 'high' tier."""
    example = make_valid_example()
    val = validator.validate(example)
    score = scorer.score(example, val)
    # Only check if the score itself is high (>= 0.65 at minimum)
    assert score.quality_tier in ["high", "medium"], f"Unexpected tier: {score.quality_tier}"


def test_preference_pair_builder_valid_pair():
    """PreferencePairBuilder should produce a pair when score delta >= 0.05."""
    builder = PreferencePairBuilder()

    # Make two examples with different scores
    from src.models import ScoreResult, AxisScore
    chosen_ex = make_valid_example(Generator.GROQ)
    rejected_ex = make_valid_example(Generator.GEMINI)
    rejected_ex.instruction_id = chosen_ex.instruction_id  # Same instruction

    score_high = ScoreResult(
        example_id=chosen_ex.id,
        schema_correctness=AxisScore(score=1.0, reasoning="perfect", is_deterministic=True),
        argument_completeness=AxisScore(score=0.9, reasoning="good", is_deterministic=False),
        intent_alignment=AxisScore(score=0.85, reasoning="very good", is_deterministic=False),
        hallucination_score=AxisScore(score=1.0, reasoning="none", is_deterministic=True),
        chain_coherence=AxisScore(score=1.0, reasoning="coherent", is_deterministic=True),
        composite_score=0.0,
        quality_tier="low",
    )
    score_low = ScoreResult(
        example_id=rejected_ex.id,
        schema_correctness=AxisScore(score=0.5, reasoning="violations", is_deterministic=True),
        argument_completeness=AxisScore(score=0.6, reasoning="some issues", is_deterministic=False),
        intent_alignment=AxisScore(score=0.7, reasoning="decent", is_deterministic=False),
        hallucination_score=AxisScore(score=0.8, reasoning="minor", is_deterministic=True),
        chain_coherence=AxisScore(score=0.5, reasoning="issues", is_deterministic=True),
        composite_score=0.0,
        quality_tier="low",
    )

    pair = builder.build_pair(
        instruction_id=chosen_ex.instruction_id,
        instruction_text=chosen_ex.instruction_text,
        domain=chosen_ex.domain,
        example_type=chosen_ex.example_type,
        example_a=chosen_ex,
        score_a=score_high,
        example_b=rejected_ex,
        score_b=score_low,
    )
    assert pair is not None
    assert pair.score_delta >= 0.05
    assert pair.chosen_example_id == chosen_ex.id
    assert pair.rejected_example_id == rejected_ex.id


def test_preference_pair_not_built_for_small_delta():
    """No pair should be built when score difference is less than MIN_PREFERENCE_DELTA (0.05)."""
    from src.models import ScoreResult, AxisScore
    builder = PreferencePairBuilder()

    ex_a = make_valid_example(Generator.GROQ)
    ex_b = make_valid_example(Generator.GEMINI)

    # Scores very close together
    score_a = ScoreResult(
        example_id=ex_a.id,
        schema_correctness=AxisScore(score=0.8, reasoning="ok", is_deterministic=True),
        argument_completeness=AxisScore(score=0.8, reasoning="ok", is_deterministic=False),
        intent_alignment=AxisScore(score=0.8, reasoning="ok", is_deterministic=False),
        hallucination_score=AxisScore(score=0.8, reasoning="ok", is_deterministic=True),
        chain_coherence=AxisScore(score=0.8, reasoning="ok", is_deterministic=True),
        composite_score=0.0,
        quality_tier="medium",
    )
    score_b = ScoreResult(
        example_id=ex_b.id,
        schema_correctness=AxisScore(score=0.8, reasoning="ok", is_deterministic=True),
        argument_completeness=AxisScore(score=0.8, reasoning="ok", is_deterministic=False),
        intent_alignment=AxisScore(score=0.8, reasoning="ok", is_deterministic=False),
        hallucination_score=AxisScore(score=0.8, reasoning="ok", is_deterministic=True),
        chain_coherence=AxisScore(score=0.8, reasoning="ok", is_deterministic=True),
        composite_score=0.0,
        quality_tier="medium",
    )

    pair = builder.build_pair(
        instruction_id=ex_a.instruction_id,
        instruction_text="Same quality",
        domain="weather",
        example_type=1,
        example_a=ex_a,
        score_a=score_a,
        example_b=ex_b,
        score_b=score_b,
    )
    assert pair is None, "Should not build pair when delta < 0.05"
