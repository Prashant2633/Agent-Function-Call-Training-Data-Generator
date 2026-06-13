"""
scorer.py — 5-Axis Rubric Scorer Engine.

Evaluates examples on a 0.0 - 1.0 scale across:
1. Schema Correctness (deterministic from validator)
2. Argument Completeness (Gemini judge)
3. Intent Alignment (Gemini judge)
4. Hallucination Detection (deterministic from validator)
5. Chain Coherence (rule-based for Type 2, auto 1.0 for others)

Computes the weighted composite score and assigns a quality tier (High, Medium, Low).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.models import (
    AxisScore,
    Example,
    ExampleType,
    QualityTier,
    ScoreResult,
    ValidationResult,
)
from src.registry import SchemaRegistry, get_registry

load_dotenv(override=True)
logger = logging.getLogger(__name__)

# Config from environment
SCORER_MODEL = os.getenv("SCORER_MODEL", "gemini-2.0-flash")
SCORING_TEMP = float(os.getenv("SCORING_TEMPERATURE", "0.1"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)


class ScorerEngine:
    """
    Evaluates generated examples using deterministic checks and Gemini LLM as a judge.
    """

    def __init__(self, registry: Optional[SchemaRegistry] = None) -> None:
        self.registry = registry or get_registry()
        # Initialize Gemini model for scoring if key exists
        if GOOGLE_API_KEY:
            self.model = genai.GenerativeModel(
                model_name=SCORER_MODEL,
                generation_config=genai.GenerationConfig(
                    temperature=SCORING_TEMP,
                    response_mime_type="application/json",
                ),
            )
        else:
            self.model = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_judge_api(self, prompt: str) -> str:
        """Call Gemini judge API with retry logic."""
        if not self.model:
            raise RuntimeError("Gemini Scorer model not initialized (missing API key)")
        response = self.model.generate_content(prompt)
        return response.text

    def score(
        self, example: Example, validation: ValidationResult
    ) -> ScoreResult:
        """
        Evaluate the example and return a ScoreResult.
        """
        # Axis 1: Schema Correctness (deterministic)
        schema_reasoning = "Valid JSON schema validation."
        if not validation.is_valid:
            schema_reasoning = "Schema violations found: " + "; ".join(
                [d for d in validation.failure_details if not d.startswith("Low semantic")]
            )
        schema_correctness = AxisScore(
            score=validation.schema_correctness,
            reasoning=schema_reasoning,
            is_deterministic=True,
        )

        # Axis 4: Hallucination Detection (deterministic)
        hallucination_reasoning = "No hallucinated tools or parameters detected."
        if validation.hallucination_score < 1.0:
            hallucination_reasoning = "Hallucination detected: " + "; ".join(
                [d for d in validation.failure_details if "hallucinated" in d.lower()]
            )
        hallucination_score = AxisScore(
            score=validation.hallucination_score,
            reasoning=hallucination_reasoning,
            is_deterministic=True,
        )

        # Axis 5: Chain Coherence (deterministic/rule-based)
        chain_score_val = 1.0
        chain_reasoning = "N/A for single-turn examples."
        if example.example_type == ExampleType.CHAIN:
            # Check if validation flagged BROKEN_CHAIN
            is_broken = any("broken chain" in d.lower() for d in validation.failure_details)
            if is_broken:
                chain_score_val = 0.0
                chain_reasoning = "Chain is broken: subsequent calls do not reference previous outputs."
            else:
                chain_score_val = 1.0
                chain_reasoning = "Chain coherence verified: parameters passed successfully across turns."

        chain_coherence = AxisScore(
            score=chain_score_val,
            reasoning=chain_reasoning,
            is_deterministic=True,
        )

        # subjective axes: default fallback if Gemini key is missing
        argument_completeness = AxisScore(
            score=1.0 if validation.is_valid else 0.5,
            reasoning="Fallback score: Gemini judge not available.",
            is_deterministic=False,
        )
        intent_alignment = AxisScore(
            score=1.0 if validation.is_valid else 0.5,
            reasoning="Fallback score: Gemini judge not available.",
            is_deterministic=False,
        )

        # subjective axes: evaluate with Gemini if available
        if GOOGLE_API_KEY and self.model:
            try:
                scores = self._evaluate_subjective_axes(example)
                if "argument_completeness" in scores:
                    argument_completeness = AxisScore(
                        score=float(scores["argument_completeness"]["score"]),
                        reasoning=str(scores["argument_completeness"]["reasoning"]),
                        is_deterministic=False,
                    )
                if "intent_alignment" in scores:
                    intent_alignment = AxisScore(
                        score=float(scores["intent_alignment"]["score"]),
                        reasoning=str(scores["intent_alignment"]["reasoning"]),
                        is_deterministic=False,
                    )
            except Exception as e:
                logger.warning(f"Gemini judge evaluation failed: {e}. Using fallback scores.")

        # ScoreResult will automatically compute the composite score in its validator
        result = ScoreResult(
            example_id=example.id,
            schema_correctness=schema_correctness,
            argument_completeness=argument_completeness,
            intent_alignment=intent_alignment,
            hallucination_score=hallucination_score,
            chain_coherence=chain_coherence,
            composite_score=0.0,  # Will be recalculated
            quality_tier=QualityTier.LOW,  # Will be recalculated
        )
        return result

    def _evaluate_subjective_axes(self, example: Example) -> dict[str, Any]:
        """
        Queries Gemini judge to evaluate argument completeness and intent alignment.
        """
        # Gather schemas of all tools in the domain to show the judge
        domain_tools = self.registry.get_domain(example.domain)
        tools_info = []
        for t in domain_tools:
            tools_info.append({
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters.get("properties", {}),
                "required": t.parameters.get("required", []),
            })

        # Format assistant response: tool calls or clarification text
        response_data: dict[str, Any] = {}
        if example.tool_calls:
            response_data["tool_calls"] = [
                {"name": tc.name, "arguments": tc.arguments} for tc in example.tool_calls
            ]
        # Include dialogue content (assistant turns)
        assistant_turns = [
            turn.content for turn in example.conversation
            if turn.role == "assistant" and turn.content
        ]
        if assistant_turns:
            response_data["assistant_text"] = assistant_turns[-1]

        prompt = f"""You are an expert evaluator of tool-use / function-calling datasets.
Your task is to evaluate a generated example consisting of a User Instruction, the Schemas of the tools available, and the Tool Calls or responses generated by the AI assistant.

You must score the following two axes on a scale from 0.0 to 1.0:

1. "argument_completeness":
   - How well did the assistant extract all necessary parameters from the user's instruction?
   - 1.0: All required and relevant optional parameters are perfectly extracted without guessing.
   - 0.5 - 0.9: Extracted required parameters, but missed some helpful optional parameters or made minor extraction errors.
   - 0.0: Missed required parameters, or completely guessed parameter values that were not in the instruction.
   - Note: For Type 3 (ambiguous) instructions where the assistant asked a clarifying question instead of making tool calls, if the clarification question was specific and correct, score this as 1.0.

2. "intent_alignment":
   - Does the assistant's response (tool calls or clarification) actually satisfy the user's intent?
   - 1.0: The assistant selected the correct tools and set the arguments to exactly match what the user wanted.
   - 0.5 - 0.9: Mostly correct, but slightly mismatched or chose sub-optimal tools.
   - 0.0: The assistant selected wrong tools or completely misaligned with the user's intent.

Return your evaluation in JSON format with this exact structure:
{{
  "argument_completeness": {{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<explanation of the score>"
  }},
  "intent_alignment": {{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<explanation of the score>"
  }}
}}

User Instruction: {example.instruction_text}
Available Tools: {json.dumps(tools_info, indent=2)}
Assistant Response: {json.dumps(response_data, indent=2)}
"""

        response_text = self._call_judge_api(prompt)
        return json.loads(response_text)
