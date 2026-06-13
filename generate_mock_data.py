import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Add project root to path so 'src' package is importable
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(override=True)

from src.models import (
    Example,
    ExampleType,
    Difficulty,
    Generator,
    ToolCall,
    ConversationTurn,
    Instruction,
    ScoreResult,
    AxisScore,
    QualityTier,
    PreferencePair,
    FailureRecord,
    ToolSchema,
    FailureMode
)
from src.database import (
    init_db,
    insert_instruction,
    insert_example,
    insert_score,
    insert_preference_pair,
    insert_failure,
    upsert_tool
)

def populate_mock_data():
    print("Initializing database schema...")
    init_db()

    # 1. Upsert mock tools
    print("Inserting mock tools...")
    mock_weather_tool = ToolSchema(
        name="get_current_weather",
        domain="weather",
        description="Get the current weather for a location.",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g. San Francisco, CA"
                },
                "units": {
                    "type": "string",
                    "enum": ["metric", "imperial"],
                    "default": "metric"
                }
            },
            "required": ["location"]
        }
    )
    mock_calendar_tool = ToolSchema(
        name="create_event",
        domain="calendar",
        description="Create a new event in the user's calendar.",
        parameters={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Event title"
                },
                "start_time": {
                    "type": "string",
                    "description": "ISO 8601 start timestamp"
                },
                "end_time": {
                    "type": "string",
                    "description": "ISO 8601 end timestamp"
                }
            },
            "required": ["title", "start_time", "end_time"]
        }
    )
    upsert_tool(mock_weather_tool)
    upsert_tool(mock_calendar_tool)

    # 2. Insert mock instructions, examples, scores, and preference pairs
    print("Inserting mock instructions, examples, scores, and preference pairs...")
    
    mock_scenarios = [
        {
            "domain": "weather",
            "instruction": "What is the temperature in Seattle right now in Fahrenheit?",
            "type": ExampleType.SINGLE,
            "difficulty": Difficulty.EASY,
            "tool_calls_groq": [ToolCall(name="get_current_weather", arguments={"location": "Seattle, WA", "units": "imperial"})],
            "tool_calls_gemini": [ToolCall(name="get_current_weather", arguments={"location": "Seattle", "units": "imperial"})],
            "groq_score_details": {
                "schema_correctness": 1.0,
                "argument_completeness": 1.0,
                "intent_alignment": 1.0,
                "hallucination_score": 1.0,
                "chain_coherence": 1.0,
                "composite": 1.0,
                "tier": QualityTier.HIGH
            },
            "gemini_score_details": {
                "schema_correctness": 1.0,
                "argument_completeness": 0.8,
                "intent_alignment": 1.0,
                "hallucination_score": 1.0,
                "chain_coherence": 1.0,
                "composite": 0.96,
                "tier": QualityTier.HIGH
            }
        },
        {
            "domain": "calendar",
            "instruction": "Book a team sync meeting tomorrow at 10am for 45 minutes",
            "type": ExampleType.SINGLE,
            "difficulty": Difficulty.EASY,
            "tool_calls_groq": [ToolCall(name="create_event", arguments={"title": "Team Sync", "start_time": "2026-06-14T10:00:00", "end_time": "2026-06-14T10:45:00"})],
            "tool_calls_gemini": [ToolCall(name="create_event", arguments={"title": "Meeting", "start_time": "2026-06-14T10:00:00", "end_time": "2026-06-14T10:45:00"})],
            "groq_score_details": {
                "schema_correctness": 1.0,
                "argument_completeness": 1.0,
                "intent_alignment": 1.0,
                "hallucination_score": 1.0,
                "chain_coherence": 1.0,
                "composite": 1.0,
                "tier": QualityTier.HIGH
            },
            "gemini_score_details": {
                "schema_correctness": 1.0,
                "argument_completeness": 0.6,
                "intent_alignment": 0.8,
                "hallucination_score": 1.0,
                "chain_coherence": 1.0,
                "composite": 0.82,
                "tier": QualityTier.MEDIUM
            }
        }
    ]

    for idx, scenario in enumerate(mock_scenarios):
        # Insert Instruction
        instr_id = f"mock_instr_{idx}"
        instruction = Instruction(
            id=instr_id,
            text=scenario["instruction"],
            domain=scenario["domain"],
            example_type=scenario["type"],
            difficulty=scenario["difficulty"],
            seed_id=f"seed_{scenario['domain']}_{idx}",
            embedding=[0.1] * 768,
            created_at=datetime.utcnow()
        )
        insert_instruction(instruction)

        # Helper to convert tool calls into list of dicts for conversation turn structure
        def get_openai_conv_format(generator_name, tool_calls):
            return [
                {"role": "user", "content": scenario["instruction"]},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{generator_name}_{idx}",
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments
                            }
                        } for tc in tool_calls
                    ]
                }
            ]

        # Insert Example Groq
        example_groq_id = f"mock_ex_groq_{idx}"
        conv_groq = [
            ConversationTurn(role="user", content=scenario["instruction"]),
            ConversationTurn(role="assistant", tool_calls=scenario["tool_calls_groq"])
        ]
        example_groq = Example(
            id=example_groq_id,
            instruction_id=instr_id,
            instruction_text=scenario["instruction"],
            domain=scenario["domain"],
            example_type=scenario["type"],
            difficulty=scenario["difficulty"],
            generator=Generator.GROQ,
            tool_calls=scenario["tool_calls_groq"],
            conversation=conv_groq,
            is_valid=True,
            validation_errors=[]
        )
        insert_example(example_groq)

        # Insert Score Groq
        score_groq = ScoreResult(
            example_id=example_groq_id,
            schema_correctness=AxisScore(score=scenario["groq_score_details"]["schema_correctness"], reasoning="Perfect match.", is_deterministic=True),
            argument_completeness=AxisScore(score=scenario["groq_score_details"]["argument_completeness"], reasoning="Perfect match.", is_deterministic=False),
            intent_alignment=AxisScore(score=scenario["groq_score_details"]["intent_alignment"], reasoning="Perfect match.", is_deterministic=False),
            hallucination_score=AxisScore(score=scenario["groq_score_details"]["hallucination_score"], reasoning="Perfect match.", is_deterministic=True),
            chain_coherence=AxisScore(score=scenario["groq_score_details"]["chain_coherence"], reasoning="Perfect match.", is_deterministic=True),
            composite_score=scenario["groq_score_details"]["composite"],
            quality_tier=scenario["groq_score_details"]["tier"],
            created_at=datetime.utcnow()
        )
        insert_score(score_groq)

        # Insert Example Gemini
        example_gemini_id = f"mock_ex_gemini_{idx}"
        conv_gemini = [
            ConversationTurn(role="user", content=scenario["instruction"]),
            ConversationTurn(role="assistant", tool_calls=scenario["tool_calls_gemini"])
        ]
        example_gemini = Example(
            id=example_gemini_id,
            instruction_id=instr_id,
            instruction_text=scenario["instruction"],
            domain=scenario["domain"],
            example_type=scenario["type"],
            difficulty=scenario["difficulty"],
            generator=Generator.GEMINI,
            tool_calls=scenario["tool_calls_gemini"],
            conversation=conv_gemini,
            is_valid=True,
            validation_errors=[]
        )
        insert_example(example_gemini)

        # Insert Score Gemini
        score_gemini = ScoreResult(
            example_id=example_gemini_id,
            schema_correctness=AxisScore(score=scenario["gemini_score_details"]["schema_correctness"], reasoning="Clean.", is_deterministic=True),
            argument_completeness=AxisScore(score=scenario["gemini_score_details"]["argument_completeness"], reasoning="Missed optional standard parameter formats.", is_deterministic=False),
            intent_alignment=AxisScore(score=scenario["gemini_score_details"]["intent_alignment"], reasoning="Aligned.", is_deterministic=False),
            hallucination_score=AxisScore(score=scenario["gemini_score_details"]["hallucination_score"], reasoning="Clean.", is_deterministic=True),
            chain_coherence=AxisScore(score=scenario["gemini_score_details"]["chain_coherence"], reasoning="Clean.", is_deterministic=True),
            composite_score=scenario["gemini_score_details"]["composite"],
            quality_tier=scenario["gemini_score_details"]["tier"],
            created_at=datetime.utcnow()
        )
        insert_score(score_gemini)

        # Insert Preference Pair (if delta >= 0.05)
        score_delta = abs(scenario["groq_score_details"]["composite"] - scenario["gemini_score_details"]["composite"])
        if score_delta >= 0.05:
            if scenario["groq_score_details"]["composite"] >= scenario["gemini_score_details"]["composite"]:
                chosen_ex, chosen_score = example_groq, scenario["groq_score_details"]["composite"]
                rejected_ex, rejected_score = example_gemini, scenario["gemini_score_details"]["composite"]
            else:
                chosen_ex, chosen_score = example_gemini, scenario["gemini_score_details"]["composite"]
                rejected_ex, rejected_score = example_groq, scenario["groq_score_details"]["composite"]

            pair = PreferencePair(
                id=f"mock_pair_{idx}",
                instruction_id=instr_id,
                instruction_text=scenario["instruction"],
                domain=scenario["domain"],
                example_type=scenario["type"],
                chosen_example_id=chosen_ex.id,
                rejected_example_id=rejected_ex.id,
                chosen_generator=chosen_ex.generator,
                rejected_generator=rejected_ex.generator,
                chosen_score=chosen_score,
                rejected_score=rejected_score,
                score_delta=score_delta,
                prompt=scenario["instruction"],
                chosen={"messages": get_openai_conv_format(chosen_ex.generator, chosen_ex.tool_calls)},
                rejected={"messages": get_openai_conv_format(rejected_ex.generator, rejected_ex.tool_calls)},
                created_at=datetime.utcnow()
            )
            insert_preference_pair(pair)

    # 3. Insert failure
    print("Inserting mock failures...")
    fail_record = FailureRecord(
        id="mock_fail_1",
        instruction_id="mock_instr_1",
        example_id="mock_ex_gemini_1",
        failure_mode=FailureMode.MISSING_REQUIRED,
        failure_detail="Required parameter 'end_time' is missing in the tool call.",
        generator=Generator.GEMINI,
        raw_response='{"name": "create_event", "arguments": {"title": "Meeting", "start_time": "2026-06-14T10:00:00"}}',
        created_at=datetime.utcnow()
    )
    insert_failure(fail_record)

    print("Data population complete! You can now view this data in the Streamlit Dashboard or export it.")

if __name__ == "__main__":
    populate_mock_data()
