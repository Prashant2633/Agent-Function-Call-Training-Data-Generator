"""
validator.py — Validation Engine for generated function-calling examples.

Includes three stages of validation:
1. SchemaValidator   — jsonschema + type checks (using registry.validate_call)
2. SemanticValidator — checks semantic alignment between instruction and tool (Gemini embeddings)
3. ChainValidator    — checks parameter passing in multi-turn examples (Type 2)

Returns a ValidationResult with tagged failure modes if any checks fail.
"""

from __future__ import annotations

import logging
import os
import json
from typing import Any, Optional

import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

from src.models import (
    Example,
    ExampleType,
    FailureMode,
    ValidationResult,
)
from src.registry import SchemaRegistry, get_registry

load_dotenv(override=True)
logger = logging.getLogger(__name__)

# Configurable thresholds
SIMILARITY_THRESHOLD = float(os.getenv("EMBEDDING_SIMILARITY_THRESHOLD", "0.75"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

# Configure Gemini SDK if key is present
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)


class Validator:
    """
    Main validator class that orchestrates schema, semantic, and chain checks.
    """

    def __init__(self, registry: Optional[SchemaRegistry] = None) -> None:
        self.registry = registry or get_registry()

    def validate(self, example: Example) -> ValidationResult:
        """
        Runs all validation checks on the example.
        """
        result = ValidationResult(
            example_id=example.id,
            is_valid=True,
            schema_correctness=1.0,
            hallucination_score=1.0,
        )

        # ── 1. Schema Correctness & Hallucination Checks ──
        if example.example_type == ExampleType.AMBIGUOUS and not example.tool_calls:
            # Type 3 ambiguous examples with no tool calls are valid (model asked for clarification)
            pass
        else:
            if not example.tool_calls:
                result.is_valid = False
                result.failure_modes.append(FailureMode.MALFORMED_JSON)
                result.failure_details.append("Example type requires tool calls, but none were generated.")
                result.schema_correctness = 0.0
            else:
                total_violations = 0
                for tc in example.tool_calls:
                    # Registry existence check (hallucination detection)
                    if not self.registry.tool_exists(tc.name):
                        result.is_valid = False
                        if FailureMode.HALLUCINATED_TOOL not in result.failure_modes:
                            result.failure_modes.append(FailureMode.HALLUCINATED_TOOL)
                        result.failure_details.append(f"Tool '{tc.name}' is hallucinated (not in registry)")
                        result.hallucination_score = 0.0
                        result.schema_correctness = min(result.schema_correctness, 0.0)
                        continue

                    # Validate arguments against JSON schema
                    call_valid, errors = self.registry.validate_call(tc.name, tc.arguments)
                    if not call_valid:
                        result.is_valid = False
                        total_violations += len(errors)
                        for err in errors:
                            mode = self._map_error_to_failure_mode(err)
                            if mode not in result.failure_modes:
                                result.failure_modes.append(mode)
                            result.failure_details.append(f"[{tc.name}] {err}")

                if total_violations > 0:
                    # Degrade schema correctness score based on number of errors
                    # 1 error -> 0.75, 2 errors -> 0.5, 3+ errors -> 0.0
                    deduction = total_violations * 0.25
                    result.schema_correctness = max(0.0, 1.0 - deduction)

        # ── 2. Semantic Alignment Check ──
        if result.is_valid and example.tool_calls:
            try:
                similarity = self._check_semantic_similarity(
                    example.instruction_text, example.tool_calls
                )
                result.semantic_similarity = similarity
                if similarity < SIMILARITY_THRESHOLD:
                    result.is_valid = False
                    result.failure_modes.append(FailureMode.LOW_SEMANTIC_SIMILARITY)
                    result.failure_details.append(
                        f"Low semantic similarity ({similarity:.3f} < {SIMILARITY_THRESHOLD}) "
                        "between user instruction and selected tool(s)."
                    )
            except Exception as e:
                logger.warning(f"Failed to calculate semantic similarity: {e}")

        # ── 3. Multi-Turn Chain Reference Check (Type 2 only) ──
        if result.is_valid and example.example_type == ExampleType.CHAIN:
            chain_valid, chain_err = self._check_chain_coherence(example)
            if not chain_valid:
                result.is_valid = False
                result.failure_modes.append(FailureMode.BROKEN_CHAIN)
                result.failure_details.append(chain_err)

        return result

    def _map_error_to_failure_mode(self, error_msg: str) -> FailureMode:
        """Maps jsonschema validation errors to specific FailureMode enums."""
        msg = error_msg.lower()
        if "not found in registry" in msg:
            return FailureMode.HALLUCINATED_TOOL
        elif "required" in msg:
            return FailureMode.MISSING_REQUIRED
        elif "enum" in msg or "is not one of" in msg:
            return FailureMode.ENUM_VIOLATION
        elif "hallucinated parameter" in msg or "additionalproperties" in msg:
            return FailureMode.HALLUCINATED_PARAM
        elif "type" in msg or "is not of type" in msg:
            return FailureMode.TYPE_MISMATCH
        else:
            return FailureMode.TYPE_MISMATCH

    def _check_semantic_similarity(self, instruction: str, tool_calls: list[Any]) -> float:
        """
        Uses Gemini embeddings to check if the instruction aligns with the called tools.
        """
        if not GOOGLE_API_KEY:
            # Fallback when api key is not set (during tests)
            return 1.0

        # Gather tool descriptions
        descriptions = []
        for tc in tool_calls:
            tool = self.registry.get_tool(tc.name)
            if tool:
                descriptions.append(tool.description)

        if not descriptions:
            return 1.0

        combined_tool_desc = " ".join(descriptions)

        # Get embeddings
        inst_emb = self._get_embedding(instruction)
        tool_emb = self._get_embedding(combined_tool_desc)

        if inst_emb is None or tool_emb is None:
            return 1.0

        # Cosine similarity
        dot_product = np.dot(inst_emb, tool_emb)
        norm_inst = np.linalg.norm(inst_emb)
        norm_tool = np.linalg.norm(tool_emb)

        if norm_inst == 0 or norm_tool == 0:
            return 0.0

        return float(dot_product / (norm_inst * norm_tool))

    def _get_embedding(self, text: str) -> Optional[list[float]]:
        """Fetch embedding from Gemini API."""
        try:
            response = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document",
            )
            return response["embedding"]
        except Exception as e:
            logger.error(f"Error fetching embedding from Gemini: {e}")
            return None

    def _check_chain_coherence(self, example: Example) -> tuple[bool, str]:
        """
        Validates that parameters are passed from one turn's output to the next turn's input.
        """
        tool_results_values: set[str] = set()
        assistant_calls_inputs: list[dict[str, Any]] = []

        # Parse turns
        for turn in example.conversation:
            if turn.role == "tool" and turn.content:
                try:
                    data = json.loads(turn.content)
                    self._extract_primitive_values(data, tool_results_values)
                except json.JSONDecodeError:
                    pass
            elif turn.role == "assistant" and turn.tool_calls:
                assistant_calls_inputs.append(
                    {tc.name: tc.arguments for tc in turn.tool_calls}
                )

        # If there are no tool results at all, it's not a real chain or no simulation occurred
        if not tool_results_values:
            # If the example contains only one turn of tool calls, it's invalid for a chain
            if len(assistant_calls_inputs) <= 1:
                return False, "Chain example must contain multiple turns of tool calls."
            return True, ""  # No outputs captured, but multiple turns present

        # Check if subsequent calls use values from previous tool outputs
        # Skip the first turn's calls (idx 0) since they cannot depend on any outputs
        referenced_outputs = False
        for turn_idx, calls in enumerate(assistant_calls_inputs):
            if turn_idx == 0:
                continue

            for tool_name, args in calls.items():
                call_values: set[str] = set()
                self._extract_primitive_values(args, call_values)

                # See if there is any intersection with previous outputs
                # We filter out very short strings (like empty or 1-2 char identifiers) to avoid false positives
                clean_call_values = {v for v in call_values if len(str(v)) > 3}
                clean_tool_outputs = {v for v in tool_results_values if len(str(v)) > 3}

                if clean_call_values.intersection(clean_tool_outputs):
                    referenced_outputs = True
                    break

            if referenced_outputs:
                break

        # If there are subsequent tool calls, but they didn't reference any previous tool outputs,
        # it's a broken chain.
        if len(assistant_calls_inputs) > 1 and not referenced_outputs:
            return (
                False,
                "Broken chain: Subsequent tool calls do not use output parameters from previous tool runs.",
            )

        return True, ""

    def _extract_primitive_values(self, data: Any, values: set[str]) -> None:
        """Helper to recursively extract all string/int values from a nested structure."""
        if isinstance(data, dict):
            for k, v in data.items():
                self._extract_primitive_values(v, values)
        elif isinstance(data, list):
            for item in data:
                self._extract_primitive_values(item, values)
        elif isinstance(data, (str, int)):
            values.add(str(data))
