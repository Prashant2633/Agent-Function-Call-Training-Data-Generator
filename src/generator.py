"""
generator.py — Groq and Gemini generator engines + instruction paraphraser.

Two parallel generators:
  GroqGenerator   — uses Groq cloud inference via OpenAI-compatible API
  GeminiGenerator — uses Google Gemini via google-generativeai

Both implement the same interface: generate(instruction, tools, example_type) → Example

The InstructionParaphraser uses Gemini to produce 10 variations of each seed,
giving us 600 seeds × 10 = 6,000 unique instructions.

Design decision: Groq's API is fully OpenAI-compatible (same SDK, custom base_url
pointing to api.groq.com), so we get Groq's blazing-fast LPU speed with zero code
divergence from OpenAI patterns. Model: llama-3.3-70b-versatile supports tool/function
calling natively.
Gemini requires a separate SDK but offers native JSON-mode output which we
use for the paraphraser and scorer.

LangChain AgentExecutor wraps both for Type 2 multi-turn examples, providing
conversation memory and intermediate step logging via callbacks.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.models import (
    ConversationTurn,
    Example,
    ExampleType,
    FailureMode,
    FailureRecord,
    Generator,
    Instruction,
    ToolCall,
)
from src.registry import SchemaRegistry

load_dotenv(override=True)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config from environment
# ─────────────────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROQ_MODEL = os.getenv("GENERATOR_MODEL_GROQ", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GENERATOR_MODEL_GEMINI", "gemini-2.0-flash")
GEN_TEMP = float(os.getenv("GENERATION_TEMPERATURE", "0.7"))
EDGE_TEMP = float(os.getenv("EDGE_CASE_TEMPERATURE", "0.2"))

# Configure Gemini SDK globally
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """You are an AI assistant with access to a set of tools.
When the user gives you an instruction, respond by calling the appropriate tool(s).

Rules:
1. Only use tools that are explicitly provided to you. NEVER invent tool names or parameters.
2. Extract parameter values from the user's instruction. Do not hallucinate values.
3. If a required parameter is missing and cannot be reasonably inferred, ask the user for clarification instead of guessing.
4. For parallel tasks (multiple independent operations), call multiple tools.
5. Use the exact parameter names and types defined in the tool schemas.
6. For string enums, only use values from the allowed list.

Always respond with a tool call. If clarification is needed, explain what information is missing."""

AMBIGUOUS_SYSTEM_PROMPT = """You are an AI assistant with access to a set of tools.
When the user gives you an instruction that is missing required information, you MUST ask
a clarifying question rather than making up values. Be specific about what information you need.

If the instruction is complete enough to make a tool call, do so.
Only use tools that are explicitly provided."""


# ─────────────────────────────────────────────────────────────────────────────
# Groq Generator (Groq Cloud — OpenAI-compatible API)
# ─────────────────────────────────────────────────────────────────────────────

class GroqGenerator:
    """
    Generates function-call examples using Groq's LPU inference cloud.

    Uses the OpenAI Python SDK with a custom base_url pointing to Groq's endpoint.
    Groq is OpenAI-API-compatible, so the same SDK works with a different base_url.
    Model llama-3.3-70b-versatile supports parallel tool/function calling natively.
    The tool_choice="auto" parameter lets the model decide which tool to call.
    """

    def __init__(self) -> None:
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY,
        )
        self.model = GROQ_MODEL

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_api(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float,
    ) -> Any:
        """Raw API call with retry logic."""
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
        )

    def generate(
        self,
        instruction: Instruction,
        registry: SchemaRegistry,
    ) -> tuple[Optional[Example], Optional[FailureRecord]]:
        """
        Generate a tool-call example for the given instruction.
        Returns (Example, None) on success, (None, FailureRecord) on failure.
        """
        # Select temperature based on example type
        # Type 3 (ambiguous) uses low temp so the model is deterministic about asking for clarification
        temperature = EDGE_TEMP if instruction.example_type == ExampleType.AMBIGUOUS else GEN_TEMP

        # Get tool definitions: all domain tools + cross-domain for parallel
        include_all = instruction.example_type == ExampleType.PARALLEL
        tools = registry.get_tools_for_generation(instruction.domain, include_all=include_all)

        system_prompt = (
            AMBIGUOUS_SYSTEM_PROMPT
            if instruction.example_type == ExampleType.AMBIGUOUS
            else AGENT_SYSTEM_PROMPT
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction.text},
        ]

        try:
            response = self._call_api(messages, tools, temperature)
            return self._parse_response(response, instruction, messages)
        except Exception as e:
            failure = FailureRecord(
                instruction_id=instruction.id,
                failure_mode=FailureMode.API_ERROR,
                failure_detail=str(e),
                generator=Generator.GROQ,
            )
            return None, failure

    def _parse_response(
        self,
        response: Any,
        instruction: Instruction,
        messages: list[dict],
    ) -> tuple[Optional[Example], Optional[FailureRecord]]:
        """Parse Groq's API response into an Example model."""
        choice = response.choices[0]
        message = choice.message

        if not message.tool_calls:
            # Model responded with text instead of tool call
            # This is valid for Type 3 (ambiguous) — model asked for clarification
            if instruction.example_type == ExampleType.AMBIGUOUS:
                # Build a clarification turn
                conversation = [
                    ConversationTurn(role="system", content=AMBIGUOUS_SYSTEM_PROMPT),
                    ConversationTurn(role="user", content=instruction.text),
                    ConversationTurn(role="assistant", content=message.content),
                ]
                return Example(
                    instruction_id=instruction.id,
                    instruction_text=instruction.text,
                    domain=instruction.domain,
                    example_type=instruction.example_type,
                    difficulty=instruction.difficulty,
                    generator=Generator.GROQ,
                    tool_calls=[],  # No tool call — intentional for Type 3
                    conversation=conversation,
                    raw_response={"content": message.content, "finish_reason": choice.finish_reason},
                ), None
            else:
                failure = FailureRecord(
                    instruction_id=instruction.id,
                    failure_mode=FailureMode.MALFORMED_JSON,
                    failure_detail="Model returned text instead of tool call",
                    generator=Generator.GROQ,
                    raw_response=message.content,
                )
                return None, failure

        # Parse tool calls
        tool_calls = []
        for tc in message.tool_calls:
            try:
                arguments = json.loads(tc.function.arguments)
                tool_calls.append(ToolCall(name=tc.function.name, arguments=arguments))
            except json.JSONDecodeError as e:
                failure = FailureRecord(
                    instruction_id=instruction.id,
                    failure_mode=FailureMode.MALFORMED_JSON,
                    failure_detail=f"Invalid JSON in tool arguments: {e}",
                    generator=Generator.GROQ,
                    raw_response=tc.function.arguments,
                )
                return None, failure

        # Build conversation turns
        conversation = [
            ConversationTurn(role="system", content=AGENT_SYSTEM_PROMPT),
            ConversationTurn(role="user", content=instruction.text),
            ConversationTurn(
                role="assistant",
                content=message.content,
                tool_calls=tool_calls,
            ),
        ]

        example = Example(
            instruction_id=instruction.id,
            instruction_text=instruction.text,
            domain=instruction.domain,
            example_type=instruction.example_type,
            difficulty=instruction.difficulty,
            generator=Generator.GROQ,
            tool_calls=tool_calls,
            conversation=conversation,
            raw_response={
                "model": response.model,
                "finish_reason": choice.finish_reason,
                "usage": response.usage.model_dump() if response.usage else None,
            },
        )
        return example, None

    def generate_chain(
        self,
        instruction: Instruction,
        registry: SchemaRegistry,
    ) -> tuple[Optional[Example], Optional[FailureRecord]]:
        """
        Generate a multi-turn chain example (Type 2).
        Simulates tool execution results and feeds them back to the model.
        """
        tools = registry.get_tools_for_generation(instruction.domain)
        messages: list[dict] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": instruction.text},
        ]
        all_tool_calls: list[ToolCall] = []
        conversation: list[ConversationTurn] = [
            ConversationTurn(role="system", content=AGENT_SYSTEM_PROMPT),
            ConversationTurn(role="user", content=instruction.text),
        ]

        max_turns = 4
        for turn_idx in range(max_turns):
            try:
                response = self._call_api(messages, tools, GEN_TEMP)
            except Exception as e:
                return None, FailureRecord(
                    instruction_id=instruction.id,
                    failure_mode=FailureMode.API_ERROR,
                    failure_detail=f"Turn {turn_idx}: {e}",
                    generator=Generator.GROQ,
                )

            choice = response.choices[0]
            message = choice.message

            if not message.tool_calls:
                # Chain complete — model gave final text response
                conversation.append(ConversationTurn(role="assistant", content=message.content))
                break

            # Parse tool calls for this turn
            turn_calls = []
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                    turn_calls.append(ToolCall(name=tc.function.name, arguments=args))
                except json.JSONDecodeError:
                    pass

            all_tool_calls.extend(turn_calls)
            conversation.append(ConversationTurn(
                role="assistant",
                tool_calls=turn_calls,
            ))

            # Add assistant message to API context
            messages.append({"role": "assistant", "content": message.content, "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in message.tool_calls
            ]})

            # Simulate tool results and feed back
            for tc in message.tool_calls:
                simulated_result = _simulate_tool_result(tc.function.name)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": json.dumps(simulated_result),
                })
                conversation.append(ConversationTurn(
                    role="tool",
                    tool_call_id=tc.id,
                    name=tc.function.name,
                    content=json.dumps(simulated_result),
                ))

            if choice.finish_reason == "stop":
                break

        if not all_tool_calls:
            return None, FailureRecord(
                instruction_id=instruction.id,
                failure_mode=FailureMode.MALFORMED_JSON,
                failure_detail="No tool calls produced in chain",
                generator=Generator.GROQ,
            )

        return Example(
            instruction_id=instruction.id,
            instruction_text=instruction.text,
            domain=instruction.domain,
            example_type=instruction.example_type,
            difficulty=instruction.difficulty,
            generator=Generator.GROQ,
            tool_calls=all_tool_calls,
            conversation=conversation,
        ), None


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Generator
# ─────────────────────────────────────────────────────────────────────────────

class GeminiGenerator:
    """
    Generates function-call examples using Google Gemini.

    Uses google-generativeai SDK with Gemini's native function calling.
    Gemini's tool_config allows forcing function calls (ANY mode) or
    allowing text responses (AUTO mode). We use AUTO for Type 3 to allow
    clarification responses.

    Design decision: Gemini returns tool_calls in a different format than OpenAI.
    We normalize both into our ToolCall Pydantic model so downstream code
    (validator, scorer) doesn't need to know which generator was used.
    """

    def __init__(self) -> None:
        self.model_name = GEMINI_MODEL

    def _get_model(self, tools: list[dict], temperature: float) -> genai.GenerativeModel:
        """Create a Gemini model instance with tools configured."""
        gemini_tools = [
            genai.types.Tool(function_declarations=[
                genai.types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=_convert_to_gemini_schema(t.get("parameters", {})),
                )
                for t in tools
            ])
        ]
        return genai.GenerativeModel(
            model_name=self.model_name,
            tools=gemini_tools,
            generation_config=genai.GenerationConfig(temperature=temperature),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_api(
        self,
        model: genai.GenerativeModel,
        chat: Any,
        message: str,
    ) -> Any:
        return chat.send_message(message)

    def generate(
        self,
        instruction: Instruction,
        registry: SchemaRegistry,
    ) -> tuple[Optional[Example], Optional[FailureRecord]]:
        """Generate a tool-call example for the given instruction."""
        temperature = EDGE_TEMP if instruction.example_type == ExampleType.AMBIGUOUS else GEN_TEMP
        include_all = instruction.example_type == ExampleType.PARALLEL
        tools = registry.get_gemini_tools_for_generation(instruction.domain, include_all=include_all)

        system_prompt = (
            AMBIGUOUS_SYSTEM_PROMPT
            if instruction.example_type == ExampleType.AMBIGUOUS
            else AGENT_SYSTEM_PROMPT
        )

        try:
            model = self._get_model(tools, temperature)
            chat = model.start_chat()

            # Send system context + user instruction
            full_message = f"{system_prompt}\n\nUser: {instruction.text}"
            response = chat.send_message(full_message)

            return self._parse_response(response, instruction)
        except Exception as e:
            return None, FailureRecord(
                instruction_id=instruction.id,
                failure_mode=FailureMode.API_ERROR,
                failure_detail=str(e),
                generator=Generator.GEMINI,
            )

    def _parse_response(
        self,
        response: Any,
        instruction: Instruction,
    ) -> tuple[Optional[Example], Optional[FailureRecord]]:
        """Parse Gemini's response into an Example model."""
        tool_calls = []
        text_content = None

        for part in response.parts:
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                # Gemini returns args as a MapComposite — convert to dict
                arguments = dict(fc.args) if fc.args else {}
                # Recursively convert nested MapComposite objects
                arguments = _deep_convert_gemini_args(arguments)
                tool_calls.append(ToolCall(name=fc.name, arguments=arguments))
            elif hasattr(part, "text") and part.text:
                text_content = part.text

        if not tool_calls:
            if instruction.example_type == ExampleType.AMBIGUOUS and text_content:
                # Valid clarification response
                conversation = [
                    ConversationTurn(role="system", content=AMBIGUOUS_SYSTEM_PROMPT),
                    ConversationTurn(role="user", content=instruction.text),
                    ConversationTurn(role="assistant", content=text_content),
                ]
                return Example(
                    instruction_id=instruction.id,
                    instruction_text=instruction.text,
                    domain=instruction.domain,
                    example_type=instruction.example_type,
                    difficulty=instruction.difficulty,
                    generator=Generator.GEMINI,
                    tool_calls=[],
                    conversation=conversation,
                ), None
            else:
                return None, FailureRecord(
                    instruction_id=instruction.id,
                    failure_mode=FailureMode.MALFORMED_JSON,
                    failure_detail="Gemini returned no tool calls",
                    generator=Generator.GEMINI,
                    raw_response=text_content,
                )

        conversation = [
            ConversationTurn(role="system", content=AGENT_SYSTEM_PROMPT),
            ConversationTurn(role="user", content=instruction.text),
            ConversationTurn(role="assistant", tool_calls=tool_calls),
        ]

        return Example(
            instruction_id=instruction.id,
            instruction_text=instruction.text,
            domain=instruction.domain,
            example_type=instruction.example_type,
            difficulty=instruction.difficulty,
            generator=Generator.GEMINI,
            tool_calls=tool_calls,
            conversation=conversation,
        ), None

    def generate_chain(
        self,
        instruction: Instruction,
        registry: SchemaRegistry,
    ) -> tuple[Optional[Example], Optional[FailureRecord]]:
        """Generate a multi-turn chain example using Gemini's chat API."""
        tools = registry.get_gemini_tools_for_generation(instruction.domain)
        temperature = GEN_TEMP

        try:
            model = self._get_model(tools, temperature)
            chat = model.start_chat()

            full_message = f"{AGENT_SYSTEM_PROMPT}\n\nUser: {instruction.text}"
            all_tool_calls: list[ToolCall] = []
            conversation: list[ConversationTurn] = [
                ConversationTurn(role="system", content=AGENT_SYSTEM_PROMPT),
                ConversationTurn(role="user", content=instruction.text),
            ]

            max_turns = 4
            current_message = full_message

            for _ in range(max_turns):
                response = chat.send_message(current_message)
                turn_calls = []
                text_content = None

                for part in response.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        args = _deep_convert_gemini_args(dict(fc.args) if fc.args else {})
                        call = ToolCall(name=fc.name, arguments=args)
                        turn_calls.append(call)
                    elif hasattr(part, "text") and part.text:
                        text_content = part.text

                if not turn_calls:
                    conversation.append(ConversationTurn(role="assistant", content=text_content))
                    break

                all_tool_calls.extend(turn_calls)
                conversation.append(ConversationTurn(role="assistant", tool_calls=turn_calls))

                # Build simulated results and continue chat
                results = []
                for call in turn_calls:
                    result = _simulate_tool_result(call.name)
                    conversation.append(ConversationTurn(
                        role="tool",
                        name=call.name,
                        content=json.dumps(result),
                    ))
                    results.append(f"Result from {call.name}: {json.dumps(result)}")

                current_message = "\n".join(results) + "\n\nContinue with the next step if needed."

            if not all_tool_calls:
                return None, FailureRecord(
                    instruction_id=instruction.id,
                    failure_mode=FailureMode.MALFORMED_JSON,
                    failure_detail="No tool calls in chain",
                    generator=Generator.GEMINI,
                )

            return Example(
                instruction_id=instruction.id,
                instruction_text=instruction.text,
                domain=instruction.domain,
                example_type=instruction.example_type,
                difficulty=instruction.difficulty,
                generator=Generator.GEMINI,
                tool_calls=all_tool_calls,
                conversation=conversation,
            ), None
        except Exception as e:
            return None, FailureRecord(
                instruction_id=instruction.id,
                failure_mode=FailureMode.API_ERROR,
                failure_detail=str(e),
                generator=Generator.GEMINI,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Instruction Paraphraser
# ─────────────────────────────────────────────────────────────────────────────

class InstructionParaphraser:
    """
    Uses Gemini to generate N paraphrase variations of a seed instruction.

    Design decision: Using Gemini (not Groq) for paraphrasing because:
    1. Gemini's JSON mode gives us structured output with minimal parsing
    2. It's cheaper for high-volume paraphrasing (6,000 calls)
    3. We want linguistic diversity — the seed generator and paraphraser
       being different models adds more variety

    Temperature is set high (0.9) to maximize lexical diversity.
    """

    PARAPHRASE_PROMPT = """Generate {n} different paraphrases of this instruction.
Keep the same intent but vary the phrasing, formality, and word choice significantly.
Mix styles: some formal, some casual, some terse, some verbose.

Original instruction: {instruction}

Return a JSON object with a "paraphrases" array containing exactly {n} strings.
Example: {{"paraphrases": ["version 1", "version 2", ...]}}"""

    def __init__(self) -> None:
        self.model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config=genai.GenerationConfig(
                temperature=0.9,
                response_mime_type="application/json",
            ),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def paraphrase(self, seed_text: str, n: int = 10) -> list[str]:
        """
        Generate n paraphrases of the seed instruction.
        Returns list of paraphrase strings (may be fewer than n on error).
        """
        prompt = self.PARAPHRASE_PROMPT.format(instruction=seed_text, n=n)
        response = self.model.generate_content(prompt)

        try:
            data = json.loads(response.text)
            return data.get("paraphrases", [])[:n]
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Paraphrase JSON parse error: {e}. Returning original.")
            return [seed_text]


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_tool_result(tool_name: str) -> dict[str, Any]:
    """
    Generate a plausible simulated tool result for multi-turn chaining.

    Design decision: We simulate tool results (rather than actually calling tools)
    to keep the pipeline self-contained and cost-effective. The simulated results
    are realistic enough to test whether the model correctly uses the output
    in the next turn (e.g., passing a contact_id from search to log_call).
    """
    simulations = {
        "create_event": {"event_id": f"evt_{uuid.uuid4().hex[:8]}", "status": "created", "url": "https://calendar.example.com/event/123"},
        "search_contacts": {"contacts": [{"id": f"cid_{uuid.uuid4().hex[:6]}", "name": "John Doe", "email": "john@example.com"}]},
        "create_contact": {"contact_id": f"cid_{uuid.uuid4().hex[:6]}", "status": "created"},
        "get_stock_price": {"ticker": "AAPL", "price": 189.45, "currency": "USD", "timestamp": "2024-01-15T14:30:00Z"},
        "web_search": {"results": [{"title": "Result 1", "url": "https://example.com", "snippet": "Relevant content..."}]},
        "get_current_weather": {"temperature": 28, "condition": "Sunny", "humidity": 60, "units": "metric"},
        "get_deal_stage": {"deal_id": "deal_001", "stage": "negotiation", "probability": 0.65},
        "find_free_slot": {"slot": {"start": "2024-01-16T10:00:00Z", "end": "2024-01-16T11:00:00Z"}},
        "query_records": {"records": [{"id": 1, "name": "Record A", "status": "active"}], "total": 1},
        "run_python": {"output": "42\n", "error": None, "execution_time_ms": 125},
        "run_sql": {"rows": [{"count": 42}], "row_count": 1, "execution_time_ms": 45},
        "search_inbox": {"messages": [{"id": "msg_001", "subject": "Meeting tomorrow", "from": "alice@example.com"}]},
        "get_directions": {"duration_minutes": 25, "distance_km": 12.3, "steps": ["Turn left on Main St", "Arrive at destination"]},
        "find_nearby": {"places": [{"name": "Central Park Cafe", "rating": 4.5, "distance_m": 250}]},
        "get_overdue": {"tasks": [{"id": "task_001", "title": "Submit report", "due": "2024-01-10"}]},
        "get_forecast": {"forecast": [{"date": "2024-01-16", "high": 30, "low": 20, "condition": "Cloudy"}]},
    }
    return simulations.get(tool_name, {"status": "success", "id": f"result_{uuid.uuid4().hex[:8]}"})


def _convert_to_gemini_schema(schema: dict) -> dict:
    """
    Convert JSON Schema draft-07 to Gemini's schema format.
    Gemini doesn't support all JSON Schema features, so we strip unsupported fields.
    """
    if not schema:
        return {"type": "object", "properties": {}}

    # Gemini schema fields that are supported
    supported = {"type", "properties", "required", "items", "description", "enum", "format"}
    result = {}

    for key, value in schema.items():
        if key.startswith("$"):
            continue  # Skip $schema, $comment etc.
        if key not in supported:
            continue
        if key == "properties" and isinstance(value, dict):
            result["properties"] = {
                k: _convert_to_gemini_schema(v) for k, v in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            result["items"] = _convert_to_gemini_schema(value)
        else:
            result[key] = value

    return result


def _deep_convert_gemini_args(args: Any) -> Any:
    """
    Recursively convert Gemini's MapComposite/RepeatedComposite to plain Python dicts/lists.
    Gemini returns special protobuf-like objects that need conversion for JSON serialization.
    """
    if hasattr(args, "items"):
        return {k: _deep_convert_gemini_args(v) for k, v in args.items()}
    elif hasattr(args, "__iter__") and not isinstance(args, str):
        return [_deep_convert_gemini_args(v) for v in args]
    else:
        return args
