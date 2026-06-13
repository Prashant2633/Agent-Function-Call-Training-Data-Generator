"""
models.py — Pydantic v2 data models shared across all pipeline components.

Every model has:
  - Full type annotations
  - Field validators where needed
  - JSON serialization helpers
  - Docstrings explaining purpose

Design decision: Using Pydantic v2 (not v1) for 5-10x faster validation,
which matters when processing 6,000+ examples. model_config replaces class Config.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class ExampleType(int, Enum):
    """The 4 canonical example types in the training data taxonomy."""
    SINGLE = 1        # Single tool call
    CHAIN = 2         # Multi-turn chaining
    AMBIGUOUS = 3     # Underspecified instruction
    PARALLEL = 4      # Multiple simultaneous tools


class Difficulty(str, Enum):
    """Difficulty level — used for curriculum learning and analysis."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Generator(str, Enum):
    """Which LLM generated the example."""
    GROQ = "groq"
    GEMINI = "gemini"


class FailureMode(str, Enum):
    """Tagged failure modes for failure analysis and debugging."""
    HALLUCINATED_TOOL = "HALLUCINATED_TOOL"         # Tool name not in registry
    HALLUCINATED_PARAM = "HALLUCINATED_PARAM"       # Param name not in schema
    TYPE_MISMATCH = "TYPE_MISMATCH"                 # Wrong Python type for param
    MISSING_REQUIRED = "MISSING_REQUIRED"           # Required param absent
    ENUM_VIOLATION = "ENUM_VIOLATION"               # Value not in allowed enum
    LOW_SEMANTIC_SIMILARITY = "LOW_SEMANTIC_SIMILARITY"  # Instruction ↔ tool misaligned
    BROKEN_CHAIN = "BROKEN_CHAIN"                   # Multi-turn output/input mismatch
    MALFORMED_JSON = "MALFORMED_JSON"               # Generator returned invalid JSON
    API_ERROR = "API_ERROR"                         # API call failed after retries
    DUPLICATE = "DUPLICATE"                         # Embedding dedup rejected this


class QualityTier(str, Enum):
    """Quality tier based on composite score thresholds."""
    HIGH = "high"       # >= 0.85
    MEDIUM = "medium"   # 0.65 - 0.84
    LOW = "low"         # < 0.65 (rejected)


# ─────────────────────────────────────────────────────────────────────────────
# Tool Schema Models
# ─────────────────────────────────────────────────────────────────────────────

class ToolParameter(BaseModel):
    """Single parameter definition within a tool schema."""
    model_config = ConfigDict(extra="allow")  # Allow JSON Schema extra fields ($comment etc.)

    type: str | list[str]
    description: str
    examples: list[Any] = Field(default_factory=list)
    enum: Optional[list[Any]] = None
    default: Optional[Any] = None
    items: Optional[dict[str, Any]] = None      # For array types
    properties: Optional[dict[str, Any]] = None  # For object types
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    minLength: Optional[int] = None
    maxLength: Optional[int] = None


class ToolSchema(BaseModel):
    """
    Complete tool definition loaded from schemas/*.json.

    Stored in the registry and used by:
    - Generator (to build tool_definitions for API calls)
    - Validator (to check generated calls against schema)
    - Scorer (to detect hallucinated parameters)
    """
    model_config = ConfigDict(extra="allow")

    name: str
    domain: str
    description: str
    parameters: dict[str, Any]  # Raw JSON Schema object (validated by jsonschema lib)

    @property
    def required_params(self) -> list[str]:
        """Returns list of required parameter names."""
        return self.parameters.get("required", [])

    @property
    def param_names(self) -> set[str]:
        """Returns set of all parameter names (required + optional)."""
        return set(self.parameters.get("properties", {}).keys())

    def to_openai_tool(self) -> dict[str, Any]:
        """
        Convert to OpenAI/Grok function-calling format.
        Used by GrokGenerator and also for export JSONL.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_gemini_tool(self) -> dict[str, Any]:
        """
        Convert to Gemini function declaration format.
        Gemini uses a slightly different structure (no outer 'type' key).
        """
        def clean_schema(val: Any) -> Any:
            if isinstance(val, dict):
                return {
                    k: clean_schema(v)
                    for k, v in val.items()
                    if k not in ("$schema", "$comment")
                }
            elif isinstance(val, list):
                return [clean_schema(item) for item in val]
            return val

        return {
            "name": self.name,
            "description": self.description,
            "parameters": clean_schema(self.parameters),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Instruction Models
# ─────────────────────────────────────────────────────────────────────────────

class SeedInstruction(BaseModel):
    """
    A seed instruction loaded from instructions/*.json.
    Each seed is paraphrased 10x to generate the full 6,000-instruction pool.
    """
    id: str
    text: str
    type: ExampleType
    difficulty: Difficulty
    domain: str
    expected_tools: list[str]
    notes: str = ""


class Instruction(BaseModel):
    """
    A fully resolved instruction (seed + paraphrase variant) ready for generation.
    Stored in the `instructions` DB table.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    domain: str
    example_type: ExampleType
    difficulty: Difficulty
    seed_id: str  # References the original SeedInstruction.id
    embedding: Optional[list[float]] = None  # Stored after embedding computation
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(use_enum_values=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tool Call Models
# ─────────────────────────────────────────────────────────────────────────────

class ToolCall(BaseModel):
    """A single tool call produced by a generator."""
    name: str
    arguments: dict[str, Any]

    def to_openai_format(self) -> dict[str, Any]:
        """Format for OpenAI fine-tuning JSONL."""
        return {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }


class ConversationTurn(BaseModel):
    """
    A single turn in a conversation.
    For multi-turn (Type 2) examples, multiple turns are chained.
    """
    role: str  # "system" | "user" | "assistant" | "tool"
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None  # For tool result turns
    name: Optional[str] = None  # Tool name for result turns

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed = {"system", "user", "assistant", "tool"}
        if v not in allowed:
            raise ValueError(f"role must be one of {allowed}, got '{v}'")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Example Models
# ─────────────────────────────────────────────────────────────────────────────

class Example(BaseModel):
    """
    A complete generated example: instruction + tool call(s) + conversation.
    This is the central object that flows through the entire pipeline:
    Generator → Validator → Scorer → DB → Exporter
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    instruction_id: str
    instruction_text: str
    domain: str
    example_type: ExampleType
    difficulty: Difficulty
    generator: Generator

    # The actual tool call(s) generated
    tool_calls: list[ToolCall]

    # Full conversation including system prompt, user message, assistant response
    conversation: list[ConversationTurn]

    # Raw API response (stored for debugging)
    raw_response: Optional[dict[str, Any]] = None

    # Set by validator
    is_valid: bool = False
    validation_errors: list[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(use_enum_values=True)


# ─────────────────────────────────────────────────────────────────────────────
# Validation Models
# ─────────────────────────────────────────────────────────────────────────────

class ValidationResult(BaseModel):
    """
    Output of the Validator for a single example.
    Passed directly to ScorerEngine — deterministic axes are computed here.
    """
    example_id: str
    is_valid: bool
    failure_modes: list[FailureMode] = Field(default_factory=list)
    failure_details: list[str] = Field(default_factory=list)

    # Axis 1: Schema Correctness (deterministic)
    schema_correctness: float = 1.0  # Degraded per violation found

    # Axis 4: Hallucination Detection (deterministic)
    hallucination_score: float = 1.0  # 1.0 = clean, 0.0 = hallucination detected

    # Semantic similarity score (instruction ↔ tool description)
    semantic_similarity: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Score Models
# ─────────────────────────────────────────────────────────────────────────────

class AxisScore(BaseModel):
    """Score on a single rubric axis with reasoning."""
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    is_deterministic: bool  # True = computed from code, False = LLM judge


class ScoreResult(BaseModel):
    """
    5-axis scoring result for one example.
    Stored in the `scores` table with full reasoning per axis.
    """
    example_id: str

    # The 5 axes
    schema_correctness: AxisScore      # weight 0.25 — deterministic
    argument_completeness: AxisScore   # weight 0.20 — Gemini judge
    intent_alignment: AxisScore        # weight 0.25 — Gemini judge
    hallucination_score: AxisScore     # weight 0.20 — deterministic
    chain_coherence: AxisScore         # weight 0.10 — rule-based / auto 1.0

    # Computed composite (weighted sum)
    composite_score: float = Field(ge=0.0, le=1.0)
    quality_tier: QualityTier

    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(use_enum_values=True)

    @model_validator(mode="after")
    def compute_composite(self) -> "ScoreResult":
        """
        Compute composite score as weighted sum of 5 axes.
        Weights: schema=0.25, completeness=0.20, alignment=0.25,
                 hallucination=0.20, chain=0.10
        """
        weights = {
            "schema_correctness": 0.25,
            "argument_completeness": 0.20,
            "intent_alignment": 0.25,
            "hallucination_score": 0.20,
            "chain_coherence": 0.10,
        }
        composite = (
            self.schema_correctness.score * weights["schema_correctness"]
            + self.argument_completeness.score * weights["argument_completeness"]
            + self.intent_alignment.score * weights["intent_alignment"]
            + self.hallucination_score.score * weights["hallucination_score"]
            + self.chain_coherence.score * weights["chain_coherence"]
        )
        self.composite_score = round(composite, 4)

        # Assign quality tier
        if composite >= 0.85:
            self.quality_tier = QualityTier.HIGH
        elif composite >= 0.65:
            self.quality_tier = QualityTier.MEDIUM
        else:
            self.quality_tier = QualityTier.LOW

        return self


# ─────────────────────────────────────────────────────────────────────────────
# Preference Pair Models
# ─────────────────────────────────────────────────────────────────────────────

class PreferencePair(BaseModel):
    """
    An RLHF/DPO preference pair.

    When both Grok and Gemini generate valid examples for the same instruction,
    the higher-scoring one becomes 'chosen' and lower becomes 'rejected'.
    The pair is used for DPO (Direct Preference Optimization) fine-tuning.

    Design decision: We only create pairs where score_delta >= 0.05 to ensure
    the preference signal is meaningful (not noise from near-identical outputs).
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    instruction_id: str
    instruction_text: str
    domain: str
    example_type: ExampleType

    chosen_example_id: str
    rejected_example_id: str
    chosen_generator: Generator
    rejected_generator: Generator

    chosen_score: float
    rejected_score: float
    score_delta: float  # chosen_score - rejected_score

    # DPO-ready format
    prompt: str
    chosen: dict[str, Any]   # Full chosen conversation in OpenAI format
    rejected: dict[str, Any]  # Full rejected conversation in OpenAI format

    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(use_enum_values=True)


# ─────────────────────────────────────────────────────────────────────────────
# Failure Record
# ─────────────────────────────────────────────────────────────────────────────

class FailureRecord(BaseModel):
    """
    Logged for every failed example — never discarded.
    Failures are valuable training signal (negative examples, edge cases).
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    instruction_id: str
    example_id: Optional[str] = None  # None if generation itself failed
    failure_mode: FailureMode
    failure_detail: str
    generator: Generator
    raw_response: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(use_enum_values=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Stats
# ─────────────────────────────────────────────────────────────────────────────

class PipelineStats(BaseModel):
    """Live stats updated throughout the pipeline run. Printed to terminal."""
    total_instructions: int = 0
    generated: int = 0
    validated: int = 0
    rejected: int = 0
    high_quality: int = 0
    medium_quality: int = 0
    preference_pairs: int = 0
    avg_composite_score: float = 0.0
    failures_by_mode: dict[str, int] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Export Models
# ─────────────────────────────────────────────────────────────────────────────

class ExportConfig(BaseModel):
    """Configuration for an export job."""
    format: str = "jsonl"           # jsonl | csv | parquet
    domain: str = "all"
    example_type: Optional[int] = None
    min_score: float = 0.65
    split: tuple[float, float, float] = (0.8, 0.1, 0.1)
    include_pairs: bool = False
    output_dir: str = "./exports"
