# 🤖 Agent Function-Call Training Data Generator

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Groq](https://img.shields.io/badge/Generator%20A-Groq%20LPU-orange?logo=groq)
![Gemini](https://img.shields.io/badge/Generator%20B-Gemini-blue?logo=google)
![PostgreSQL](https://img.shields.io/badge/Storage-PostgreSQL%2016-336791?logo=postgresql)

A **production-grade, end-to-end pipeline** that automatically generates high-quality function-call / tool-use training data for fine-tuning LLMs.

The system uses **Groq** (Generator A — blazing-fast LPU inference) and **Google Gemini** (Generator B, Embeddings, and LLM Judge), LangChain for orchestration, PostgreSQL for storage, and a custom 5-axis scoring engine to produce **RLHF-ready preference pairs**.

**Output:** 6,000+ validated training samples exported as JSONL, ready to drop into any fine-tuning pipeline (OpenAI, Together AI, Axolotl, TRL).

> ⚠️ This is NOT a demo. Every component is functional, tested, and explainable in a technical interview.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Key Components](#-key-components)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Pipeline Stages](#-pipeline-stages)
- [Tool Schemas](#-tool-schemas-46-tools--12-domains)
- [Example Output](#-example-output)
- [Scoring Rubric](#-scoring-rubric)
- [Interview Q&A](#-interview-qa)
- [Testing](#-testing)
- [Project Structure](#-project-structure)

---

## 🔍 Overview

### What it produces

| Output | Format | Count |
|--------|--------|-------|
| Single tool-call examples (Type 1) | JSONL | ~1,500 |
| Multi-turn chain examples (Type 2) | JSONL | ~1,500 |
| Ambiguous / clarification examples (Type 3) | JSONL | ~1,500 |
| Parallel multi-tool examples (Type 4) | JSONL | ~1,500 |
| **DPO preference pairs** | JSONL (TRL/Axolotl) | ~3,000 |

### Why this approach

- **Dual-generator design**: Groq and Gemini generate responses independently for the *same* instruction. Their outputs are both validated and scored — the higher-scoring one becomes `chosen` and the lower-scoring becomes `rejected` in the DPO pair.
- **Groq's LPU speed**: ~10-100x faster inference than standard cloud APIs — critical for generating 6,000+ examples cost-effectively.
- **Gemini as judge**: Gemini evaluates `argument_completeness` and `intent_alignment` — two axes that require semantic understanding, not just rule checks.
- **Vector deduplication**: Gemini `text-embedding-004` embeddings + cosine similarity (> 0.95) ensure no near-duplicate instructions pollute the dataset.

---

## 🏗 Architecture

```
                    ┌─────────────────────────────────┐
                    │    Seed Instructions (600)       │
                    │  50 per domain × 12 domains      │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │   InstructionParaphraser (Gemini) │
                    │   600 seeds × 10 = 6,000 variants│
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │  Vector Deduplication             │
                    │  Gemini text-embedding-004        │
                    │  Cosine similarity > 0.95 → DROP  │
                    └──────────────┬──────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                                         │
  ┌───────────▼──────────┐              ┌──────────────▼──────────┐
  │  Generator A (Groq)   │              │  Generator B (Gemini)    │
  │  llama-3.3-70b-vers.  │              │  gemini-2.0-flash        │
  │  OpenAI-compatible    │              │  Native function calling  │
  └───────────┬──────────┘              └──────────────┬──────────┘
              │                                         │
              └────────────────────┬────────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │        Validator (3 stages)       │
                    │  1. JSON Schema (jsonschema)       │
                    │  2. Semantic similarity (Gemini)   │
                    │  3. Chain coherence (rule-based)   │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │     Scorer Engine (5 axes)        │
                    │  Gemini judge + deterministic     │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │   Preference Pair Builder (DPO)   │
                    │   chosen vs rejected (Δ ≥ 0.05)   │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │      PostgreSQL Database          │
                    │  instructions, examples, scores,  │
                    │  preference_pairs, failures        │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │         Exporters                 │
                    │  JSONL / DPO JSONL / CSV/Parquet  │
                    └─────────────────────────────────┘
```

---

## 🧩 Key Components

| Module | File | Purpose |
|--------|------|---------|
| Schema Registry | `src/registry.py` | Loads 46 JSON schemas, provides O(1) tool lookup |
| Groq Generator | `src/generator.py::GroqGenerator` | Generates via Groq LPU (OpenAI-compatible) |
| Gemini Generator | `src/generator.py::GeminiGenerator` | Generates via Gemini native function calling |
| Paraphraser | `src/generator.py::InstructionParaphraser` | Expands 600 seeds → 6,000 via Gemini JSON mode |
| Validator | `src/validator.py` | 3-stage: schema + semantic + chain checks |
| Scorer | `src/scorer.py` | 5-axis rubric, Gemini as judge for 2 axes |
| Preference Builder | `src/preference.py` | DPO pair construction (chosen/rejected) |
| Pipeline | `src/pipeline.py` | ThreadPool orchestrator, tqdm progress |
| Database | `src/database.py` | SQLAlchemy Core, PostgreSQL CRUD |
| Models | `src/models.py` | Pydantic v2 data structures |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourname/agent-function-call-generator
cd "agent-function-call-generator"
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
cp .env.example .env
# Edit .env and fill in:
#   GROQ_API_KEY   → from https://console.groq.com/keys
#   GOOGLE_API_KEY → from https://aistudio.google.com/apikey
```

### 3. Start PostgreSQL

```bash
# Requires Docker Desktop running
docker-compose up -d db
# Verify health:
docker-compose ps
```

### 4. Run the Pipeline

```bash
# Quick smoke test (50 samples)
python run_pipeline.py --samples 50 --workers 4

# Full run (6,000 samples)
python run_pipeline.py --samples 6000 --workers 8

# Specific domains only
python run_pipeline.py --samples 500 --domains calendar weather finance
```

### 5. Export Data

```bash
# Export training JSONL (OpenAI/Together format)
python export.py --format jsonl --output exports/train.jsonl

# Export DPO preference pairs (TRL/Axolotl format)
python export.py --format dpo --output exports/dpo_pairs.jsonl

# Export all formats at once
python export.py --all

# Only high-quality examples
python export.py --format jsonl --min-quality high
```

### 6. Launch Dashboard

```bash
streamlit run dashboard.py
# Opens at http://localhost:8501
```

---

## ⚙️ Configuration

All settings live in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(required)* | Groq API key from console.groq.com |
| `GOOGLE_API_KEY` | *(required)* | Gemini API key from aistudio.google.com |
| `DATABASE_URL` | `postgresql://agent:agentpass@localhost:5432/agent_training` | PostgreSQL connection string |
| `GENERATOR_MODEL_GROQ` | `llama-3.3-70b-versatile` | Groq model (supports tool calling) |
| `GENERATOR_MODEL_GEMINI` | `gemini-2.0-flash` | Gemini generator model |
| `SCORER_MODEL` | `gemini-2.0-flash` | Gemini model used as LLM judge |
| `EMBEDDING_MODEL` | `models/text-embedding-004` | Gemini embedding model |
| `GENERATION_TEMPERATURE` | `0.7` | Temperature for Types 1, 2, 4 |
| `EDGE_CASE_TEMPERATURE` | `0.2` | Temperature for Type 3 (ambiguous) |
| `SCORING_TEMPERATURE` | `0.1` | Temperature for Gemini judge (low = consistent) |
| `EMBEDDING_SIMILARITY_THRESHOLD` | `0.75` | Min cosine sim for semantic validation |
| `DEDUP_SIMILARITY_THRESHOLD` | `0.95` | Max cosine sim before dedup rejection |
| `HIGH_QUALITY_THRESHOLD` | `0.85` | Composite score ≥ this → "high" tier |
| `MEDIUM_QUALITY_THRESHOLD` | `0.65` | Composite score ≥ this → "medium" tier |
| `MIN_PREFERENCE_DELTA` | `0.05` | Min score gap to build a DPO pair |
| `PARAPHRASE_COUNT` | `10` | Paraphrase variations per seed |
| `DEFAULT_WORKERS` | `4` | ThreadPool worker count |
| `MAX_RETRIES` | `3` | API retry attempts (exponential backoff) |

---

## 🔄 Pipeline Stages

### Stage 1 — Schema Registry Init
All 46 JSON schemas (Draft-07) are loaded from `schemas/` into an in-memory dict. Each schema defines parameter types, descriptions, required fields, enums, and examples. The registry provides O(1) lookup by tool name.

### Stage 2 — Seed Instruction Loading
600 seed instructions (50 per domain × 12 domains) are loaded from `instructions/*.json`. Each seed has a type (1-4), difficulty (easy/medium/hard), expected tool list, and notes.

### Stage 3 — Paraphrasing (Gemini JSON Mode)
`InstructionParaphraser` calls Gemini with `response_mime_type="application/json"` and temperature 0.9 to generate 10 diverse rewrites of each seed — casual, formal, terse, verbose. This expands 600 seeds into up to 6,000 unique instructions.

### Stage 4 — Vector Deduplication
Each paraphrase is embedded with Gemini `text-embedding-004`. Cosine similarity is computed against all existing instruction embeddings using NumPy vectorized operations. If similarity > 0.95 with any existing instruction, the paraphrase is discarded and logged as a `DUPLICATE` failure.

### Stage 5 — Parallel Generation
Both generators process each instruction in a `ThreadPoolExecutor`:
- **Groq** uses the OpenAI SDK with `base_url="https://api.groq.com/openai/v1"` and `tool_choice="auto"`.
- **Gemini** uses the `google-generativeai` SDK with `FunctionDeclaration` converted from our JSON schemas.
- Type 2 (chain) instructions trigger `generate_chain()` which simulates tool results and feeds them back for multi-turn sequences.

### Stage 6 — Validation (3 stages)
1. **Schema Validation**: `jsonschema.Draft7Validator` checks each tool call's arguments against the registered schema. Tags: `HALLUCINATED_TOOL`, `HALLUCINATED_PARAM`, `MISSING_REQUIRED`, `TYPE_MISMATCH`, `ENUM_VIOLATION`.
2. **Semantic Validation**: Gemini embedding of the instruction is compared (cosine similarity) against the tool's description embedding. Similarity < 0.75 → `LOW_SEMANTIC_SIMILARITY`.
3. **Chain Coherence**: For Type 2, checks that IDs/values returned by tool N are referenced in tool N+1's arguments. Failure → `BROKEN_CHAIN`.

### Stage 7 — Scoring (5 axes)
| Axis | Method | Weight |
|------|--------|--------|
| Schema Correctness | Deterministic (from validator) | 30% |
| Argument Completeness | Gemini judge (0-1) | 25% |
| Intent Alignment | Gemini judge (0-1) | 25% |
| Hallucination Score | Deterministic (1 - hallucination rate) | 10% |
| Chain Coherence | Rule-based (Type 2 only, else 1.0) | 10% |

Composite = weighted sum → Quality tier: **high** (≥ 0.85), **medium** (0.65–0.84), **low** (< 0.65).

### Stage 8 — DPO Preference Pairs
For each instruction where both Groq and Gemini produced valid examples with a composite score delta ≥ 0.05, a preference pair is built: the higher-scoring example is `chosen`, the lower-scoring is `rejected`. The `reason` field explains which axes drove the difference.

### Stage 9 — Storage & Export
All data persists in PostgreSQL (5 tables). The exporter queries validated examples and converts them to:
- **JSONL**: OpenAI `{"messages": [...]}` fine-tune format
- **DPO JSONL**: TRL / Axolotl `{"prompt": ..., "chosen": [...], "rejected": [...]}` format
- **CSV**: Flat rows with all score axes for analysis
- **Parquet**: HuggingFace Datasets-compatible columnar format

---

## 🛠 Tool Schemas: 46 Tools × 12 Domains

| Domain | Tools | Count |
|--------|-------|-------|
| **calendar** | create_event, delete_event, reschedule_event, find_free_slot | 4 |
| **search** | web_search, image_search, news_search | 3 |
| **code_exec** | run_python, run_sql, lint_code, explain_code | 4 |
| **crm** | create_contact, update_contact, log_call, get_deal_stage, add_note | 5 |
| **weather** | get_current_weather, get_forecast, get_historical_weather | 3 |
| **finance** | get_stock_price, get_portfolio_summary, place_order, get_exchange_rate, calculate_roi | 5 |
| **email** | send_email, search_inbox, mark_read, get_thread | 4 |
| **files** | read_file, write_file, list_directory, move_file | 4 |
| **notifications** | send_push, send_sms, send_slack_message | 3 |
| **maps** | get_directions, find_nearby, calculate_distance | 3 |
| **tasks** | create_task, assign_task, update_status, get_overdue | 4 |
| **database** | query_records, insert_record, update_record, delete_record | 4 |
| **Total** | | **46** |

Every schema includes:
- `$schema`: JSON Schema Draft-07
- `description`: Tool purpose
- `parameters.properties`: Each parameter with `type`, `description`, `examples` (2-3), `$comment` (design rationale)
- `required`: Mandatory parameter list
- `enum`: Bounded value sets where applicable

---

## 📄 Example Output

### JSONL (Fine-tune format)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an AI assistant with access to a set of tools..."
    },
    {
      "role": "user",
      "content": "What's the weather like in Tokyo right now?"
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "type": "function",
          "function": {
            "name": "get_current_weather",
            "arguments": "{\"location\": \"Tokyo, Japan\", \"units\": \"metric\", \"include_aqi\": false}"
          }
        }
      ]
    }
  ],
  "metadata": {
    "domain": "weather",
    "example_type": 1,
    "difficulty": "easy",
    "generator": "groq",
    "composite_score": 0.9125,
    "quality_tier": "high"
  }
}
```

### DPO JSONL (Preference pair format)

```json
{
  "prompt": "Search for recent news about AI regulation in the EU",
  "chosen": [
    {"role": "user", "content": "Search for recent news about AI regulation in the EU"},
    {"role": "assistant", "tool_calls": [{"type": "function", "function": {
      "name": "news_search",
      "arguments": "{\"query\": \"AI regulation EU 2024\", \"num_results\": 5, \"language\": \"en\", \"category\": \"tech\", \"sort_by\": \"date\"}"
    }}]}
  ],
  "rejected": [
    {"role": "user", "content": "Search for recent news about AI regulation in the EU"},
    {"role": "assistant", "tool_calls": [{"type": "function", "function": {
      "name": "news_search",
      "arguments": "{\"query\": \"AI EU\"}"
    }}]}
  ],
  "metadata": {
    "domain": "search",
    "example_type": 1,
    "score_delta": 0.18,
    "chosen_score": 0.91,
    "rejected_score": 0.73,
    "reason": "chosen has better argument_completeness (includes category, sort_by, num_results)"
  }
}
```

---

## 📊 Scoring Rubric

The 5-axis composite score determines example quality and DPO pair selection:

| Axis | Weight | Method | What it measures |
|------|--------|--------|-----------------|
| **Schema Correctness** | 30% | Deterministic | Are all args valid per JSON Schema? No hallucinated/missing params? |
| **Argument Completeness** | 25% | Gemini judge | Did the model fill in all reasonable args given the instruction context? |
| **Intent Alignment** | 25% | Gemini judge | Does the tool call actually accomplish what the user asked? |
| **Hallucination Score** | 10% | Deterministic | `1 - (hallucinated_params / total_params)` |
| **Chain Coherence** | 10% | Rule-based | For Type 2: do output IDs from step N appear in step N+1's inputs? |

**Quality Tiers:**
- 🟢 **High** (≥ 0.85): Export-ready, preferred in DPO pairs
- 🟡 **Medium** (0.65–0.84): Acceptable for training
- 🔴 **Low** (< 0.65): Rejected — logged as failure, excluded from exports

---

## 🎙 Interview Q&A

### Q1: Why use Groq + Gemini instead of a single model?

**A:** Two reasons — quality through diversity and cost optimization.

Using **two different model architectures** as generators means each will make different mistakes. When we score both and select the better one, we're implicitly doing model-based rejection sampling. This improves overall dataset quality without manual annotation.

**Groq** is OpenAI-compatible (same SDK, different `base_url`) and runs on custom LPU hardware — making it 10-100x faster for the high-volume generation we need. **Gemini** is used for generation, but *also* as the judge (scorer) and embedding model, creating a natural check: Gemini evaluates whether Groq's outputs are good, and vice versa.

The dual-generator setup also enables **DPO preference pairs** natively — for every instruction, we have two independent responses that can become `chosen` / `rejected` without needing human labelers.

---

### Q2: How does the deduplication work and why is it important?

**A:** Deduplication uses **cosine similarity on Gemini `text-embedding-004` embeddings**.

When a paraphrase is generated, we compute its 768-dimensional embedding and compare it against all previously-accepted instruction embeddings using vectorized NumPy dot products. If similarity > 0.95, the paraphrase is discarded.

This matters because:
1. **Training data poisoning**: Near-duplicate instructions with different tool calls confuse the model during fine-tuning.
2. **Evaluation leakage**: If train and eval sets have near-duplicates, eval metrics are inflated.
3. **Diversity requirement**: LLMs generalize better when trained on diverse phrasings rather than slight variations of the same text.

The threshold (0.95) is intentionally high — we want to catch copies, not just similar topics. Two instructions can be in the same domain (both asking about weather) but semantically distinct enough to both be valuable.

---

### Q3: How do you select preference pairs? What makes a "good" pair?

**A:** A preference pair requires **both** examples to be valid *and* have a score delta ≥ 0.05.

The minimum delta threshold avoids building "trivial" pairs where the quality difference is noise. Both the `chosen` and `rejected` examples must have passed schema validation — DPO training works best when the `rejected` response is plausible-but-suboptimal, not completely broken.

The `reason` field documents *which axis* drove the difference (e.g., "chosen has better argument_completeness: filled in optional date parameter"). This makes the pairs interpretable and lets us filter pairs by failure mode for targeted training.

In practice, ~50% of instruction pairs produce a valid DPO pair (when both generators succeed and there's sufficient delta).

---

### Q4: Walk me through the validation pipeline. What does each stage catch?

**A:** Three stages, each catching different failure classes:

**Stage 1 — JSON Schema** (deterministic, fast):
Runs `jsonschema.Draft7Validator` against the tool's schema. Catches: hallucinated tool names, hallucinated parameter names, wrong types (`"5"` vs `5`), missing required parameters, and enum violations. This is the most common failure mode.

**Stage 2 — Semantic Similarity** (embedding-based):
Computes cosine similarity between the instruction embedding and the tool description embedding. If < 0.75, flags `LOW_SEMANTIC_SIMILARITY`. This catches cases where the model called a *valid* tool with *valid* args, but the tool doesn't match what the user asked for (e.g., calling `web_search` for a calendar instruction).

**Stage 3 — Chain Coherence** (rule-based, Type 2 only):
For multi-turn examples, checks that identifiers returned by a tool result (e.g., `contact_id: "cid_abc"`) appear in the next tool call's arguments. Catches "broken chains" where the model ignores intermediate results.

---

### Q5: Why Pydantic v2 specifically? What does it give you over v1 or plain dataclasses?

**A:** Three concrete benefits at pipeline scale:

1. **Speed**: Pydantic v2's Rust core is 5-10x faster than v1 for model instantiation and validation. When we're creating 6,000+ `Example` objects, this matters.

2. **Strict validation**: Field validators catch type mismatches at the boundary (API response → model), not deep inside pipeline logic. This means errors surface with a clear validation error rather than a cryptic `AttributeError` 50 lines later.

3. **JSON serialization**: `model.model_dump()` and `model.model_dump_json()` handle nested models, enums, and UUIDs correctly out of the box. We serialize to PostgreSQL's JSONB columns without any custom serializers.

`model_config = ConfigDict(extra="allow")` on `ToolParameter` lets us store non-standard JSON Schema fields (`$comment`, `examples`) without schema errors — important since our schemas use those for documentation.

---

### Q6: How would you scale this to 100k+ samples?

**A:** Four changes:

1. **Distributed generation**: Replace `ThreadPoolExecutor` with Celery + Redis. Each worker independently pulls instructions from a queue and pushes results back. Scale to 50+ workers across multiple machines.

2. **Batch embeddings**: Current code embeds one instruction at a time. Gemini's batch embedding API handles up to 100 texts per request — 100x fewer API calls for deduplication.

3. **Model diversity**: Add more generators (GPT-4o, Claude, Mistral) for richer preference pairs and better diversity. More generators = more DPO pairs per instruction without extra seed creation.

4. **Async pipeline**: Replace synchronous `ThreadPoolExecutor` with `asyncio` + `aiohttp` for the API calls. Groq and Gemini both support async clients, enabling much higher concurrency at the same cost.

5. **Streaming DB writes**: Use PostgreSQL `COPY` for bulk inserts instead of row-by-row `INSERT`. At 100k rows, this is a 10-50x write speedup.

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_registry.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run only fast tests (skip LLM-dependent ones)
python -m pytest tests/ -v -k "not scorer"
```

### Test Coverage

| Test File | What It Tests |
|-----------|---------------|
| `test_models.py` | Pydantic models, enums, serialization |
| `test_registry.py` | Schema loading, tool lookup, OpenAI/Gemini format |
| `test_validator.py` | All failure modes: hallucinated tool/param, missing required, enum violation |
| `test_scorer.py` | Score range, quality tiers, DPO pair delta threshold |

> **Note**: Validator and scorer tests that call Gemini APIs require `GOOGLE_API_KEY` to be set. Schema and model tests run fully offline.

---

## 📁 Project Structure

```
agent-function-call-generator/
├── run_pipeline.py          # CLI entry point (--samples, --workers, --domains)
├── export.py                # CLI exporter (JSONL / DPO / CSV / Parquet)
├── dashboard.py             # Streamlit visual dashboard
├── docker-compose.yml       # PostgreSQL 16 + pgAdmin
├── .env.example             # Config template (copy to .env)
├── requirements.txt         # Python dependencies
│
├── src/                     # Core pipeline modules
│   ├── __init__.py
│   ├── models.py            # Pydantic v2 data structures
│   ├── database.py          # SQLAlchemy Core CRUD
│   ├── registry.py          # SchemaRegistry (46 tools)
│   ├── generator.py         # GroqGenerator + GeminiGenerator + Paraphraser
│   ├── validator.py         # 3-stage validation engine
│   ├── scorer.py            # 5-axis rubric scorer
│   ├── preference.py        # DPO preference pair builder
│   └── pipeline.py          # ThreadPool orchestrator
│
├── schemas/                 # 46 JSON Schema Draft-07 tool definitions
│   ├── calendar/            # create_event, delete_event, reschedule_event, find_free_slot
│   ├── search/              # web_search, image_search, news_search
│   ├── code_exec/           # run_python, run_sql, lint_code, explain_code
│   ├── crm/                 # create_contact, update_contact, log_call, get_deal_stage, add_note
│   ├── weather/             # get_current_weather, get_forecast, get_historical_weather
│   ├── finance/             # get_stock_price, get_portfolio_summary, place_order, ...
│   ├── email/               # send_email, search_inbox, mark_read, get_thread
│   ├── files/               # read_file, write_file, list_directory, move_file
│   ├── notifications/       # send_push, send_sms, send_slack_message
│   ├── maps/                # get_directions, find_nearby, calculate_distance
│   ├── tasks/               # create_task, assign_task, update_status, get_overdue
│   └── database/            # query_records, insert_record, update_record, delete_record
│
├── instructions/            # 600 seed instructions (50 per domain)
│   ├── calendar.json
│   ├── search.json
│   ├── code_exec.json
│   ├── crm.json
│   ├── weather.json
│   ├── finance.json
│   ├── email.json
│   ├── files.json
│   ├── notifications.json
│   ├── maps.json
│   ├── tasks.json
│   └── database.json
│
├── exports/                 # Generated JSONL / CSV / Parquet output
│
└── tests/                   # pytest test suite
    ├── __init__.py
    ├── test_models.py
    ├── test_registry.py
    ├── test_validator.py
    └── test_scorer.py
```

---

## 📄 License

MIT — free to use, modify, and distribute.

---

*Built with ❤️ using Groq LPU + Google Gemini + LangChain + PostgreSQL + Pydantic v2*
