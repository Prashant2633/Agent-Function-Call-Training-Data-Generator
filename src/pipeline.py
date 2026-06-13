"""
pipeline.py — Pipeline Orchestrator for data generation.

Orchestrates the end-to-end data generation process:
1. Initialize database (init_db)
2. Load seed instructions from instructions/ domain files
3. Paraphrase seeds (InstructionParaphraser)
4. Vector-based deduplication (Gemini embeddings cosine similarity > 0.95)
5. Generate examples in parallel (Groq and Gemini generators)
6. Validate (Validator)
7. Score (ScorerEngine)
8. Build preference pairs (PreferencePairBuilder)
9. Save results to PostgreSQL database
"""

from __future__ import annotations

import logging
import os
import json
import uuid
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from src.models import (
    Difficulty,
    Example,
    ExampleType,
    FailureMode,
    FailureRecord,
    Generator,
    Instruction,
    PipelineStats,
    SeedInstruction,
)
from src.database import (
    init_db,
    insert_instruction,
    get_instruction_embeddings,
    insert_example,
    insert_score,
    insert_preference_pair,
    insert_failure,
    get_overview_stats,
    upsert_tool,
)
from src.registry import load_registry, get_registry
from src.generator import GroqGenerator, GeminiGenerator, InstructionParaphraser
from src.validator import Validator
from src.scorer import ScorerEngine
from src.preference import PreferencePairBuilder

load_dotenv(override=True)
logger = logging.getLogger(__name__)
console = Console()

# Embedding config
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
SIMILARITY_THRESHOLD = float(os.getenv("EMBEDDING_SIMILARITY_THRESHOLD", "0.95"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


class Pipeline:
    """
    Main orchestration class for the data generation pipeline.
    """

    def __init__(
        self,
        samples: int = 50,
        workers: int = 4,
        domains: list[str] = ["all"],
        types: list[int] = [1, 2, 3, 4],
        resume: bool = True,
    ) -> None:
        self.samples = samples
        self.workers = workers
        self.domains = domains
        self.types = types
        self.resume = resume

        # Initialize registry & DB
        self.registry = load_registry()
        init_db()

        # Sync registry tools to database
        for tool in self.registry.list_all():
            upsert_tool(tool)

        # Initialize modules
        self.groq_gen = GroqGenerator()
        self.gemini_gen = GeminiGenerator()
        self.paraphraser = InstructionParaphraser()
        self.validator = Validator(self.registry)
        self.scorer = ScorerEngine(self.registry)
        self.pair_builder = PreferencePairBuilder()

        # Load existing instruction embeddings for dedup
        self.existing_embeddings = get_instruction_embeddings()
        self.stats = PipelineStats()

    def run(self) -> PipelineStats:
        """Runs the pipeline: paraphrasing, generation, scoring, and storing."""
        console.print("[bold green]Starting Data Generation Pipeline...[/bold green]")
        console.print(f"Target Samples: {self.samples} | Workers: {self.workers}")
        console.print(f"Domains: {self.domains} | Types: {self.types}")

        # ── Step 1: Pre-populate or load instructions ──
        instructions = self._prepare_instructions()
        if not instructions:
            console.print("[bold red]No instructions available for generation.[/bold red]")
            return self.stats

        # Limit to the requested sample size
        instructions = instructions[:self.samples]
        self.stats.total_instructions = len(instructions)

        console.print(f"[bold green]Processing {len(instructions)} instructions...[/bold green]")

        # ── Step 2: Generation Loop in ThreadPool ──
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(self._process_single_instruction, inst): inst
                for inst in instructions
            }

            # Progress bar
            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Generating data",
                unit="inst",
            )

            for future in pbar:
                inst = futures[future]
                try:
                    future.result()
                    self.stats.generated += 1
                    # Update progress bar description with live stats
                    pbar.set_postfix({
                        "Gen": self.stats.generated,
                        "Val": self.stats.validated,
                        "Rej": self.stats.rejected,
                        "Pairs": self.stats.preference_pairs,
                    })
                except Exception as e:
                    logger.error(f"Error processing instruction {inst.id}: {e}")
                    self.stats.rejected += 1

        # Print final report
        self._print_final_report()
        return self.stats

    def _prepare_instructions(self) -> list[Instruction]:
        """
        Loads seeds, generates paraphrases, checks for duplicates, and returns a list
        of ready-to-process Instructions.
        """
        # Load seeds from file system
        seeds = self._load_seed_instructions()
        if not seeds:
            return []

        # Filter by requested domains and types
        filtered_seeds = []
        for seed in seeds:
            if self.domains != ["all"] and seed.domain not in self.domains:
                continue
            if seed.type not in self.types:
                continue
            filtered_seeds.append(seed)

        console.print(f"Loaded {len(filtered_seeds)} filtered seed instructions.")

        instructions_to_process = []
        paraphrases_per_seed = 10  # Target count per seed

        # Load existing instructions from DB to resume if requested
        # We check if we already have instructions stored in the DB
        # If resume=True, we pull those instructions first.
        if self.resume:
            from sqlalchemy import text
            from src.database import get_connection
            with get_connection() as conn:
                rows = conn.execute(text("""
                    SELECT id, text, domain, example_type, difficulty, seed_id, embedding
                    FROM instructions
                    ORDER BY created_at
                """)).fetchall()

            for row in rows:
                # Reconstruct Instruction
                inst = Instruction(
                    id=row[0],
                    text=row[1],
                    domain=row[2],
                    example_type=row[3],
                    difficulty=row[4],
                    seed_id=row[5],
                    embedding=json.loads(row[6]) if row[6] else None,
                )
                # Filter by domain/type if needed
                if self.domains != ["all"] and inst.domain not in self.domains:
                    continue
                if inst.example_type not in self.types:
                    continue
                instructions_to_process.append(inst)

            if instructions_to_process:
                console.print(f"Resuming with {len(instructions_to_process)} existing instructions from database.")
                if len(instructions_to_process) >= self.samples:
                    return instructions_to_process

        # Generate new paraphrases if we need more
        needed_more = self.samples - len(instructions_to_process)
        if needed_more <= 0:
            return instructions_to_process

        console.print(f"Generating paraphrases to meet the sample goal of {self.samples}...")
        for seed in filtered_seeds:
            if len(instructions_to_process) >= self.samples:
                break

            console.print(f"Paraphrasing seed: '{seed.text}'")
            paraphrases = self.paraphraser.paraphrase(seed.text, n=paraphrases_per_seed)

            for p_text in paraphrases:
                if len(instructions_to_process) >= self.samples:
                    break

                # Get embedding
                embedding = self._get_embedding(p_text)
                if embedding is None:
                    continue

                # Dedup check
                is_dup, max_sim = self._is_duplicate(embedding)
                if is_dup:
                    self.stats.failures_by_mode["DUPLICATE"] = self.stats.failures_by_mode.get("DUPLICATE", 0) + 1
                    # Save duplicate failure
                    failure = FailureRecord(
                        instruction_id=seed.id,
                        failure_mode=FailureMode.DUPLICATE,
                        failure_detail=f"Paraphrase duplicate of existing instruction (similarity: {max_sim:.3f})",
                        generator=Generator.GEMINI,
                        raw_response=p_text,
                    )
                    insert_failure(failure)
                    continue

                # Create instruction
                inst = Instruction(
                    id=str(uuid.uuid4()),
                    text=p_text,
                    domain=seed.domain,
                    example_type=seed.type,
                    difficulty=seed.difficulty,
                    seed_id=seed.id,
                    embedding=embedding,
                )

                # Insert to DB
                inserted = insert_instruction(inst)
                if inserted:
                    instructions_to_process.append(inst)
                    self.existing_embeddings.append((inst.id, embedding))

        return instructions_to_process

    def _load_seed_instructions(self) -> list[SeedInstruction]:
        """Loads all seed instruction JSON files from the instructions directory."""
        seeds_dir = Path(__file__).parent.parent / "instructions"
        if not seeds_dir.exists():
            logger.warning(f"Instructions directory not found at {seeds_dir}")
            return []

        seeds = []
        for file in seeds_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    seeds.append(SeedInstruction(**item))
            except Exception as e:
                logger.error(f"Error loading seed file {file}: {e}")
        return seeds

    def _get_embedding(self, text: str) -> Optional[list[float]]:
        """Fetch embedding from Gemini."""
        if not GOOGLE_API_KEY:
            return [0.0] * 768  # Fallback for offline testing
        try:
            response = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document",
            )
            return response["embedding"]
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None

    def _is_duplicate(self, embedding: list[float]) -> tuple[bool, float]:
        """NumPy vectorized cosine similarity dedup check against existing database instructions."""
        if not self.existing_embeddings:
            return False, 0.0

        emb_arr = np.array(embedding)
        existing_arrs = np.array([emb for _, emb in self.existing_embeddings])

        dot_products = np.dot(existing_arrs, emb_arr)
        norms_existing = np.linalg.norm(existing_arrs, axis=1)
        norm_emb = np.linalg.norm(emb_arr)

        if norm_emb == 0 or len(norms_existing) == 0:
            return False, 0.0

        # Avoid divide by zero
        norms_existing[norms_existing == 0] = 1.0
        similarities = dot_products / (norms_existing * norm_emb)

        max_idx = np.argmax(similarities)
        max_sim = similarities[max_idx]

        return max_sim > SIMILARITY_THRESHOLD, float(max_sim)

    def _process_single_instruction(self, inst: Instruction) -> None:
        """Processes a single instruction: Groq + Gemini generation, validation, scoring, DPO mapping."""
        # 1. Run Generator A (Groq)
        if inst.example_type == ExampleType.CHAIN:
            ex_groq, fail_groq = self.groq_gen.generate_chain(inst, self.registry)
        else:
            ex_groq, fail_groq = self.groq_gen.generate(inst, self.registry)

        # 2. Run Generator B (Gemini)
        if inst.example_type == ExampleType.CHAIN:
            ex_gemini, fail_gemini = self.gemini_gen.generate_chain(inst, self.registry)
        else:
            ex_gemini, fail_gemini = self.gemini_gen.generate(inst, self.registry)

        # Log initial API failures if any
        if fail_groq:
            insert_failure(fail_groq)
            self.stats.failures_by_mode[fail_groq.failure_mode] = (
                self.stats.failures_by_mode.get(fail_groq.failure_mode, 0) + 1
            )
        if fail_gemini:
            insert_failure(fail_gemini)
            self.stats.failures_by_mode[fail_gemini.failure_mode] = (
                self.stats.failures_by_mode.get(fail_gemini.failure_mode, 0) + 1
            )

        # If both failed to generate, we stop here
        if not ex_groq and not ex_gemini:
            return

        # 3. Validation & Scoring Stage
        score_groq = None
        score_gemini = None

        if ex_groq:
            val_groq = self.validator.validate(ex_groq)
            ex_groq.is_valid = val_groq.is_valid
            ex_groq.validation_errors = val_groq.failure_details

            # Score example
            score_groq = self.scorer.score(ex_groq, val_groq)

            # Insert example + score into DB
            insert_example(ex_groq)
            insert_score(score_groq)

            if ex_groq.is_valid:
                self.stats.validated += 1
                if score_groq.quality_tier == "high":
                    self.stats.high_quality += 1
                elif score_groq.quality_tier == "medium":
                    self.stats.medium_quality += 1
            else:
                for mode in val_groq.failure_modes:
                    self.stats.failures_by_mode[mode] = self.stats.failures_by_mode.get(mode, 0) + 1
                    # Log failure in failures table
                    failure = FailureRecord(
                        instruction_id=inst.id,
                        example_id=ex_groq.id,
                        failure_mode=mode,
                        failure_detail="; ".join(val_groq.failure_details),
                        generator=Generator.GROQ,
                    )
                    insert_failure(failure)

        if ex_gemini:
            val_gemini = self.validator.validate(ex_gemini)
            ex_gemini.is_valid = val_gemini.is_valid
            ex_gemini.validation_errors = val_gemini.failure_details

            # Score example
            score_gemini = self.scorer.score(ex_gemini, val_gemini)

            # Insert example + score into DB
            insert_example(ex_gemini)
            insert_score(score_gemini)

            if ex_gemini.is_valid:
                self.stats.validated += 1
                if score_gemini.quality_tier == "high":
                    self.stats.high_quality += 1
                elif score_gemini.quality_tier == "medium":
                    self.stats.medium_quality += 1
            else:
                for mode in val_gemini.failure_modes:
                    self.stats.failures_by_mode[mode] = self.stats.failures_by_mode.get(mode, 0) + 1
                    # Log failure in failures table
                    failure = FailureRecord(
                        instruction_id=inst.id,
                        example_id=ex_gemini.id,
                        failure_mode=mode,
                        failure_detail="; ".join(val_gemini.failure_details),
                        generator=Generator.GEMINI,
                    )
                    insert_failure(failure)

        # 4. Form DPO Preference Pair if both are valid
        if ex_groq and ex_groq.is_valid and score_groq and ex_gemini and ex_gemini.is_valid and score_gemini:
            pair = self.pair_builder.build_pair(
                instruction_id=inst.id,
                instruction_text=inst.text,
                domain=inst.domain,
                example_type=inst.example_type,
                example_a=ex_groq,
                score_a=score_groq,
                example_b=ex_gemini,
                score_b=score_gemini,
            )
            if pair:
                insert_preference_pair(pair)
                self.stats.preference_pairs += 1

    def _print_final_report(self) -> None:
        """Prints a rich, formatted pipeline run summary to the console."""
        console.print("\n[bold green]Pipeline Run Completed successfully![/bold green]")
        
        # Load final aggregated stats from database
        db_stats = get_overview_stats()
        
        table = Table(title="Pipeline Execution Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim", width=25)
        table.add_column("Value", justify="right")

        table.add_row("Total Instructions", str(self.stats.total_instructions))
        table.add_row("Total Examples Generated", str(db_stats.get("total_examples", 0)))
        table.add_row("Valid Examples (Passed)", f"[green]{db_stats.get('valid_examples', 0)}[/green]")
        table.add_row("High Quality Tier (>=0.85)", f"[cyan]{db_stats.get('high_quality', 0)}[/cyan]")
        table.add_row("Medium Quality Tier (0.65-0.84)", f"[yellow]{db_stats.get('medium_quality', 0)}[/yellow]")
        table.add_row("Preference Pairs (DPO)", f"[bold green]{db_stats.get('preference_pairs', 0)}[/bold green]")
        table.add_row("Total Failure Records", f"[red]{db_stats.get('total_failures', 0)}[/red]")
        
        avg_score = db_stats.get("avg_score")
        avg_score_str = f"{avg_score:.4f}" if avg_score else "N/A"
        table.add_row("Average Composite Score", avg_score_str)

        console.print(table)
        
        # Breakdown of failures
        if self.stats.failures_by_mode:
            fail_table = Table(title="Failure Modes Breakdown", show_header=True, header_style="bold red")
            fail_table.add_column("Failure Mode", style="dim")
            fail_table.add_column("Count", justify="right")
            for mode, count in sorted(self.stats.failures_by_mode.items(), key=lambda x: x[1], reverse=True):
                fail_table.add_row(mode, str(count))
            console.print(fail_table)
