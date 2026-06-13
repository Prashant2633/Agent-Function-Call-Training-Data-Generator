"""
database.py — PostgreSQL connection, table creation, and all CRUD operations.

Uses SQLAlchemy 2.0 Core (not ORM) for explicit, performant SQL.
Connection pooling via QueuePool (default for psycopg2 engine).

Design decision: Using Core instead of ORM because:
  1. We need raw SQL control for the indexed queries
  2. Pydantic models are already our "ORM" layer
  3. Core is simpler to reason about when debugging generated data
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator, Optional

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

from src.models import (
    Example,
    ExportConfig,
    FailureRecord,
    Instruction,
    PreferencePair,
    QualityTier,
    ScoreResult,
    ToolSchema,
)

load_dotenv(override=True)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Engine Setup
# ─────────────────────────────────────────────────────────────────────────────

def get_engine() -> Engine:
    """
    Create SQLAlchemy engine from DATABASE_URL env variable.
    Pool size tuned for the pipeline's concurrent workers.
    """
    database_url = os.getenv("DATABASE_URL", "postgresql://agent:agentpass@localhost:54321/agent_training")
    return create_engine(
        database_url,
        poolclass=QueuePool,
        pool_size=10,          # Max persistent connections
        max_overflow=20,       # Extra connections during spikes
        pool_pre_ping=True,    # Verify connections are alive before using
        pool_recycle=3600,     # Recycle connections every hour
        echo=False,            # Set True for SQL debug logging
    )


# Singleton engine — created once at module import
_engine: Optional[Engine] = None

def get_db_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


@contextmanager
def get_connection() -> Generator:
    """Context manager for database connections with auto-commit/rollback."""
    engine = get_db_engine()
    conn = engine.connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Table Definitions (DDL)
# ─────────────────────────────────────────────────────────────────────────────

CREATE_TABLES_SQL = """
-- Tools: stores tool schema definitions loaded from schemas/
CREATE TABLE IF NOT EXISTS tools (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    domain      TEXT NOT NULL,
    schema_json TEXT NOT NULL,    -- full JSON schema as string
    created_at  TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Instructions: the 6,000 paraphrased instructions
CREATE TABLE IF NOT EXISTS instructions (
    id           TEXT PRIMARY KEY,
    text         TEXT NOT NULL,
    domain       TEXT NOT NULL,
    example_type INTEGER NOT NULL CHECK (example_type IN (1,2,3,4)),
    difficulty   TEXT NOT NULL CHECK (difficulty IN ('easy','medium','hard')),
    seed_id      TEXT NOT NULL,
    embedding    TEXT,            -- JSON-encoded float array (Gemini embedding)
    created_at   TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_instructions_text_hash 
    ON instructions (md5(text));

-- Examples: generated tool calls (one per generator per instruction)
CREATE TABLE IF NOT EXISTS examples (
    id                   TEXT PRIMARY KEY,
    instruction_id       TEXT NOT NULL REFERENCES instructions(id),
    domain               TEXT NOT NULL,
    example_type         INTEGER NOT NULL,
    difficulty           TEXT NOT NULL,
    generator            TEXT NOT NULL CHECK (generator IN ('groq','gemini')),
    tool_calls_json      TEXT NOT NULL,         -- JSON array of tool calls
    conversation_json    TEXT NOT NULL,         -- Full conversation JSON
    raw_response_json    TEXT,
    is_valid             BOOLEAN NOT NULL DEFAULT FALSE,
    validation_errors    TEXT,                  -- JSON array of error strings
    created_at           TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_examples_domain_score
    ON examples (domain);
CREATE INDEX IF NOT EXISTS idx_examples_type_valid
    ON examples (example_type, is_valid);
CREATE INDEX IF NOT EXISTS idx_examples_generator
    ON examples (generator);

-- Scores: 5-axis rubric scores per example
CREATE TABLE IF NOT EXISTS scores (
    id                     TEXT PRIMARY KEY,
    example_id             TEXT NOT NULL REFERENCES examples(id),
    schema_correctness     FLOAT NOT NULL,
    argument_completeness  FLOAT NOT NULL,
    intent_alignment       FLOAT NOT NULL,
    hallucination_score    FLOAT NOT NULL,
    chain_coherence        FLOAT NOT NULL,
    composite_score        FLOAT NOT NULL,
    quality_tier           TEXT NOT NULL,
    scorer_reasoning_json  TEXT NOT NULL,   -- Per-axis reasoning from Gemini judge
    created_at             TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_scores_composite
    ON scores (composite_score DESC);
CREATE INDEX IF NOT EXISTS idx_scores_quality_tier
    ON scores (quality_tier);

-- Preference pairs: DPO-ready chosen/rejected pairs
CREATE TABLE IF NOT EXISTS preference_pairs (
    id                   TEXT PRIMARY KEY,
    instruction_id       TEXT NOT NULL REFERENCES instructions(id),
    domain               TEXT NOT NULL,
    example_type         INTEGER NOT NULL,
    chosen_example_id    TEXT NOT NULL REFERENCES examples(id),
    rejected_example_id  TEXT NOT NULL REFERENCES examples(id),
    chosen_generator     TEXT NOT NULL,
    rejected_generator   TEXT NOT NULL,
    chosen_score         FLOAT NOT NULL,
    rejected_score       FLOAT NOT NULL,
    score_delta          FLOAT NOT NULL,
    prompt_text          TEXT NOT NULL,
    chosen_json          TEXT NOT NULL,
    rejected_json        TEXT NOT NULL,
    created_at           TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Failures: all failures logged (never discarded)
CREATE TABLE IF NOT EXISTS failures (
    id              TEXT PRIMARY KEY,
    instruction_id  TEXT,
    example_id      TEXT,
    failure_mode    TEXT NOT NULL,
    failure_detail  TEXT NOT NULL,
    generator       TEXT NOT NULL,
    raw_response    TEXT,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_failures_mode
    ON failures (failure_mode);

-- Export jobs: tracks export operations for dashboard
CREATE TABLE IF NOT EXISTS export_jobs (
    id            TEXT PRIMARY KEY,
    filter_json   TEXT NOT NULL,
    output_path   TEXT NOT NULL,
    sample_count  INTEGER,
    format        TEXT NOT NULL,
    created_at    TIMESTAMP NOT NULL DEFAULT NOW()
);
"""


def init_db() -> None:
    """
    Create all tables and indexes. Safe to run on existing DB (IF NOT EXISTS).
    Called once at pipeline startup.
    """
    engine = get_db_engine()
    with engine.connect() as conn:
        conn.execute(text(CREATE_TABLES_SQL))
        conn.commit()
    logger.info("Database initialized successfully")


# ─────────────────────────────────────────────────────────────────────────────
# Tool CRUD
# ─────────────────────────────────────────────────────────────────────────────

def upsert_tool(tool: ToolSchema) -> None:
    """Insert or update a tool schema. Called once per schema file at startup."""
    sql = text("""
        INSERT INTO tools (id, name, domain, schema_json, created_at)
        VALUES (:id, :name, :domain, :schema_json, :created_at)
        ON CONFLICT (name) DO UPDATE SET
            domain = EXCLUDED.domain,
            schema_json = EXCLUDED.schema_json
    """)
    with get_connection() as conn:
        conn.execute(sql, {
            "id": f"{tool.domain}_{tool.name}",
            "name": tool.name,
            "domain": tool.domain,
            "schema_json": tool.model_dump_json(),
            "created_at": datetime.utcnow(),
        })


# ─────────────────────────────────────────────────────────────────────────────
# Instruction CRUD
# ─────────────────────────────────────────────────────────────────────────────

def insert_instruction(instruction: Instruction) -> bool:
    """
    Insert an instruction. Returns False if duplicate (same text hash).
    Uses ON CONFLICT DO NOTHING to handle concurrent workers.
    """
    sql = text("""
        INSERT INTO instructions (id, text, domain, example_type, difficulty, seed_id, embedding, created_at)
        VALUES (:id, :text, :domain, :example_type, :difficulty, :seed_id, :embedding, :created_at)
        ON CONFLICT DO NOTHING
        RETURNING id
    """)
    with get_connection() as conn:
        result = conn.execute(sql, {
            "id": instruction.id,
            "text": instruction.text,
            "domain": instruction.domain,
            "example_type": instruction.example_type,
            "difficulty": instruction.difficulty,
            "seed_id": instruction.seed_id,
            "embedding": json.dumps(instruction.embedding) if instruction.embedding else None,
            "created_at": instruction.created_at,
        })
        return result.fetchone() is not None


def get_instruction_embeddings() -> list[tuple[str, list[float]]]:
    """Fetch all instruction IDs + embeddings for dedup checking."""
    with get_connection() as conn:
        rows = conn.execute(
            text("SELECT id, embedding FROM instructions WHERE embedding IS NOT NULL")
        ).fetchall()
    return [(row[0], json.loads(row[1])) for row in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Example CRUD
# ─────────────────────────────────────────────────────────────────────────────

def insert_example(example: Example) -> None:
    """Insert a generated example. Called after generation (before scoring)."""
    sql = text("""
        INSERT INTO examples (
            id, instruction_id, domain, example_type, difficulty, generator,
            tool_calls_json, conversation_json, raw_response_json,
            is_valid, validation_errors, created_at
        ) VALUES (
            :id, :instruction_id, :domain, :example_type, :difficulty, :generator,
            :tool_calls_json, :conversation_json, :raw_response_json,
            :is_valid, :validation_errors, :created_at
        )
        ON CONFLICT (id) DO NOTHING
    """)
    with get_connection() as conn:
        conn.execute(sql, {
            "id": example.id,
            "instruction_id": example.instruction_id,
            "domain": example.domain,
            "example_type": int(example.example_type) if not isinstance(example.example_type, int) else example.example_type,
            "difficulty": str(example.difficulty),
            "generator": str(example.generator),
            "tool_calls_json": json.dumps([tc.model_dump() for tc in example.tool_calls]),
            "conversation_json": json.dumps([t.model_dump() for t in example.conversation]),
            "raw_response_json": json.dumps(example.raw_response) if example.raw_response else None,
            "is_valid": example.is_valid,
            "validation_errors": json.dumps(example.validation_errors),
            "created_at": example.created_at,
        })


def get_examples_for_export(
    domain: Optional[str] = None,
    example_type: Optional[int] = None,
    min_score: float = 0.65,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Fetch examples with their scores for export."""
    conditions = ["e.is_valid = TRUE", "s.composite_score >= :min_score"]
    params: dict[str, Any] = {"min_score": min_score}

    if domain and domain != "all":
        conditions.append("e.domain = :domain")
        params["domain"] = domain
    if example_type:
        conditions.append("e.example_type = :example_type")
        params["example_type"] = example_type

    where_clause = " AND ".join(conditions)
    limit_clause = f"LIMIT {limit}" if limit else ""

    sql = text(f"""
        SELECT
            e.id, e.instruction_id, e.domain, e.example_type, e.difficulty,
            e.generator, e.tool_calls_json, e.conversation_json,
            s.composite_score, s.quality_tier,
            i.text as instruction_text
        FROM examples e
        JOIN scores s ON s.example_id = e.id
        JOIN instructions i ON i.id = e.instruction_id
        WHERE {where_clause}
        ORDER BY s.composite_score DESC
        {limit_clause}
    """)

    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()

    return [dict(row._mapping) for row in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Score CRUD
# ─────────────────────────────────────────────────────────────────────────────

def insert_score(score: ScoreResult) -> None:
    """Insert scoring results for an example."""
    sql = text("""
        INSERT INTO scores (
            id, example_id, schema_correctness, argument_completeness,
            intent_alignment, hallucination_score, chain_coherence,
            composite_score, quality_tier, scorer_reasoning_json, created_at
        ) VALUES (
            :id, :example_id, :schema_correctness, :argument_completeness,
            :intent_alignment, :hallucination_score, :chain_coherence,
            :composite_score, :quality_tier, :scorer_reasoning_json, :created_at
        )
        ON CONFLICT DO NOTHING
    """)
    reasoning = {
        "schema_correctness": score.schema_correctness.model_dump(),
        "argument_completeness": score.argument_completeness.model_dump(),
        "intent_alignment": score.intent_alignment.model_dump(),
        "hallucination_score": score.hallucination_score.model_dump(),
        "chain_coherence": score.chain_coherence.model_dump(),
    }
    with get_connection() as conn:
        conn.execute(sql, {
            "id": str(score.example_id) + "_score",
            "example_id": score.example_id,
            "schema_correctness": score.schema_correctness.score,
            "argument_completeness": score.argument_completeness.score,
            "intent_alignment": score.intent_alignment.score,
            "hallucination_score": score.hallucination_score.score,
            "chain_coherence": score.chain_coherence.score,
            "composite_score": score.composite_score,
            "quality_tier": str(score.quality_tier),
            "scorer_reasoning_json": json.dumps(reasoning),
            "created_at": score.created_at,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Preference Pair CRUD
# ─────────────────────────────────────────────────────────────────────────────

def insert_preference_pair(pair: PreferencePair) -> None:
    """Insert a DPO preference pair."""
    sql = text("""
        INSERT INTO preference_pairs (
            id, instruction_id, domain, example_type,
            chosen_example_id, rejected_example_id,
            chosen_generator, rejected_generator,
            chosen_score, rejected_score, score_delta,
            prompt_text, chosen_json, rejected_json, created_at
        ) VALUES (
            :id, :instruction_id, :domain, :example_type,
            :chosen_example_id, :rejected_example_id,
            :chosen_generator, :rejected_generator,
            :chosen_score, :rejected_score, :score_delta,
            :prompt_text, :chosen_json, :rejected_json, :created_at
        )
        ON CONFLICT DO NOTHING
    """)
    with get_connection() as conn:
        conn.execute(sql, {
            "id": pair.id,
            "instruction_id": pair.instruction_id,
            "domain": pair.domain,
            "example_type": int(pair.example_type) if not isinstance(pair.example_type, int) else pair.example_type,
            "chosen_example_id": pair.chosen_example_id,
            "rejected_example_id": pair.rejected_example_id,
            "chosen_generator": str(pair.chosen_generator),
            "rejected_generator": str(pair.rejected_generator),
            "chosen_score": pair.chosen_score,
            "rejected_score": pair.rejected_score,
            "score_delta": pair.score_delta,
            "prompt_text": pair.prompt,
            "chosen_json": json.dumps(pair.chosen),
            "rejected_json": json.dumps(pair.rejected),
            "created_at": pair.created_at,
        })


def get_preference_pairs_for_export(
    domain: Optional[str] = None,
    min_delta: float = 0.05,
) -> list[dict[str, Any]]:
    """Fetch preference pairs for JSONL export."""
    conditions = ["score_delta >= :min_delta"]
    params: dict[str, Any] = {"min_delta": min_delta}
    if domain and domain != "all":
        conditions.append("domain = :domain")
        params["domain"] = domain

    where = " AND ".join(conditions)
    sql = text(f"SELECT * FROM preference_pairs WHERE {where} ORDER BY score_delta DESC")
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row._mapping) for row in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Failure CRUD
# ─────────────────────────────────────────────────────────────────────────────

def insert_failure(failure: FailureRecord) -> None:
    """Log a failure. Called for every failed generation/validation."""
    sql = text("""
        INSERT INTO failures (id, instruction_id, example_id, failure_mode, failure_detail, generator, raw_response, created_at)
        VALUES (:id, :instruction_id, :example_id, :failure_mode, :failure_detail, :generator, :raw_response, :created_at)
    """)
    with get_connection() as conn:
        conn.execute(sql, {
            "id": failure.id,
            "instruction_id": failure.instruction_id,
            "example_id": failure.example_id,
            "failure_mode": str(failure.failure_mode),
            "failure_detail": failure.failure_detail,
            "generator": str(failure.generator),
            "raw_response": failure.raw_response,
            "created_at": failure.created_at,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard Stats
# ─────────────────────────────────────────────────────────────────────────────

def get_overview_stats() -> dict[str, Any]:
    """Returns aggregated stats for the dashboard Overview page."""
    sql = text("""
        SELECT
            (SELECT COUNT(*) FROM examples) as total_examples,
            (SELECT COUNT(*) FROM examples WHERE is_valid = TRUE) as valid_examples,
            (SELECT COUNT(*) FROM examples WHERE is_valid = FALSE) as invalid_examples,
            (SELECT COUNT(*) FROM scores WHERE quality_tier = 'high') as high_quality,
            (SELECT COUNT(*) FROM scores WHERE quality_tier = 'medium') as medium_quality,
            (SELECT COUNT(*) FROM scores WHERE quality_tier = 'low') as low_quality,
            (SELECT COUNT(*) FROM preference_pairs) as preference_pairs,
            (SELECT COUNT(*) FROM failures) as total_failures,
            (SELECT AVG(composite_score) FROM scores) as avg_score
    """)
    with get_connection() as conn:
        row = conn.execute(sql).fetchone()
    return dict(row._mapping) if row else {}


def get_domain_stats() -> list[dict[str, Any]]:
    """Pass rate and count by domain for dashboard bar chart."""
    sql = text("""
        SELECT
            e.domain,
            COUNT(*) as total,
            SUM(CASE WHEN e.is_valid THEN 1 ELSE 0 END) as valid,
            AVG(s.composite_score) as avg_score
        FROM examples e
        LEFT JOIN scores s ON s.example_id = e.id
        GROUP BY e.domain
        ORDER BY e.domain
    """)
    with get_connection() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(row._mapping) for row in rows]


def get_failure_mode_stats() -> list[dict[str, Any]]:
    """Failure mode breakdown for the pie chart."""
    sql = text("""
        SELECT failure_mode, COUNT(*) as count
        FROM failures
        GROUP BY failure_mode
        ORDER BY count DESC
    """)
    with get_connection() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(row._mapping) for row in rows]


def get_score_distribution() -> list[float]:
    """All composite scores for histogram."""
    sql = text("SELECT composite_score FROM scores ORDER BY composite_score")
    with get_connection() as conn:
        rows = conn.execute(sql).fetchall()
    return [row[0] for row in rows]


def get_throughput_by_hour() -> list[dict[str, Any]]:
    """Examples generated per hour for the throughput line chart."""
    sql = text("""
        SELECT
            DATE_TRUNC('hour', created_at) as hour,
            COUNT(*) as count
        FROM examples
        GROUP BY hour
        ORDER BY hour
    """)
    with get_connection() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(row._mapping) for row in rows]


def log_export_job(config: ExportConfig, count: int) -> None:
    """Record an export job in the DB."""
    sql = text("""
        INSERT INTO export_jobs (id, filter_json, output_path, sample_count, format, created_at)
        VALUES (:id, :filter_json, :output_path, :sample_count, :format, :created_at)
    """)
    import uuid as _uuid
    with get_connection() as conn:
        conn.execute(sql, {
            "id": str(_uuid.uuid4()),
            "filter_json": config.model_dump_json(),
            "output_path": config.output_dir,
            "sample_count": count,
            "format": config.format,
            "created_at": datetime.utcnow(),
        })
