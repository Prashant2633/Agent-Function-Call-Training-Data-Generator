"""
export.py — CLI exporter for training data.

Exports generated, validated training data from PostgreSQL to:
  - JSONL (OpenAI fine-tune / Together AI / Axolotl format)
  - DPO JSONL (preference pairs in TRL / Axolotl DPO format)
  - CSV (for spreadsheet analysis)
  - Parquet (for data science / HuggingFace datasets)

Usage:
    python export.py --format jsonl --output exports/train.jsonl
    python export.py --format dpo --output exports/dpo_pairs.jsonl --min-quality high
    python export.py --format csv --output exports/analysis.csv
    python export.py --format parquet --output exports/dataset.parquet
    python export.py --all  # Export all formats at once
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv(override=True)
logger = logging.getLogger(__name__)
console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export training data from the database to JSONL/CSV/Parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "dpo", "csv", "parquet"],
        default="jsonl",
        help="Output format.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path. Defaults to exports/<format>_<timestamp>.<ext>.",
    )
    parser.add_argument(
        "--min-quality",
        choices=["high", "medium", "low"],
        default="medium",
        help="Minimum quality tier to include.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["all"],
        help="Filter by domain(s). Use 'all' for no filter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of records to export.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="export_all",
        help="Export all formats (jsonl, dpo, csv, parquet) in one shot.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def get_output_path(fmt: str, output: str | None) -> Path:
    """Resolve output file path."""
    if output:
        return Path(output)
    ext_map = {"jsonl": "jsonl", "dpo": "jsonl", "csv": "csv", "parquet": "parquet"}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exports_dir = Path("exports")
    exports_dir.mkdir(exist_ok=True)
    return exports_dir / f"{fmt}_{ts}.{ext_map[fmt]}"


def fetch_examples(min_quality: str, domains: list[str], limit: int | None) -> list[dict]:
    """Query validated examples from database."""
    from src.database import get_connection
    from sqlalchemy import text

    quality_map = {"high": ["high"], "medium": ["high", "medium"], "low": ["high", "medium", "low"]}
    tiers = quality_map[min_quality]
    tier_placeholders = ", ".join(f"'{t}'" for t in tiers)

    domain_filter = ""
    if domains != ["all"]:
        domain_list = ", ".join(f"'{d}'" for d in domains)
        domain_filter = f"AND e.domain IN ({domain_list})"

    limit_clause = f"LIMIT {limit}" if limit else ""

    query = text(f"""
        SELECT
            e.id, e.instruction_id, i.text AS instruction_text, e.domain,
            e.example_type, e.difficulty, e.generator,
            e.tool_calls_json, e.conversation_json, e.is_valid,
            s.composite_score, s.quality_tier,
            s.schema_correctness, s.argument_completeness,
            s.intent_alignment, s.hallucination_score, s.chain_coherence
        FROM examples e
        JOIN scores s ON s.example_id = e.id
        JOIN instructions i ON i.id = e.instruction_id
        WHERE e.is_valid = true
          AND s.quality_tier IN ({tier_placeholders})
          {domain_filter}
        ORDER BY s.composite_score DESC
        {limit_clause}
    """)

    with get_connection() as conn:
        rows = conn.execute(query).fetchall()

    return [
        {
            "id": row[0],
            "instruction_id": row[1],
            "instruction_text": row[2],
            "domain": row[3],
            "example_type": row[4],
            "difficulty": row[5],
            "generator": row[6],
            "tool_calls": json.loads(row[7]) if row[7] else [],
            "conversation": json.loads(row[8]) if row[8] else [],
            "is_valid": row[9],
            "composite_score": float(row[10]) if row[10] else 0.0,
            "quality_tier": row[11],
            "scores": {
                "schema_correctness": float(row[12]) if row[12] else 0.0,
                "argument_completeness": float(row[13]) if row[13] else 0.0,
                "intent_alignment": float(row[14]) if row[14] else 0.0,
                "hallucination_score": float(row[15]) if row[15] else 0.0,
                "chain_coherence": float(row[16]) if row[16] else 0.0,
            },
        }
        for row in rows
    ]


def fetch_preference_pairs(min_quality: str, domains: list[str], limit: int | None) -> list[dict]:
    """Query DPO preference pairs from database."""
    from src.database import get_connection
    from sqlalchemy import text

    quality_map = {"high": 0.85, "medium": 0.65, "low": 0.0}
    min_score = quality_map[min_quality]

    domain_filter = ""
    if domains != ["all"]:
        domain_list = ", ".join(f"'{d}'" for d in domains)
        domain_filter = f"AND domain IN ({domain_list})"

    limit_clause = f"LIMIT {limit}" if limit else ""

    query = text(f"""
        SELECT
            id, instruction_id, prompt_text, domain, example_type,
            chosen_example_id, rejected_example_id,
            chosen_json, rejected_json,
            score_delta, chosen_score, rejected_score
        FROM preference_pairs
        WHERE score_delta >= :min_delta
          {domain_filter}
        ORDER BY score_delta DESC
        {limit_clause}
    """).bindparams(min_delta=0.05)

    with get_connection() as conn:
        rows = conn.execute(query).fetchall()

    return [
        {
            "id": row[0],
            "instruction_id": row[1],
            "prompt": row[2],
            "domain": row[3],
            "example_type": row[4],
            "chosen_example_id": row[5],
            "rejected_example_id": row[6],
            "chosen": json.loads(row[7]) if row[7] else [],
            "rejected": json.loads(row[8]) if row[8] else [],
            "score_delta": float(row[9]) if row[9] else 0.0,
            "chosen_score": float(row[10]) if row[10] else 0.0,
            "rejected_score": float(row[11]) if row[11] else 0.0,
            "reason": "",
        }
        for row in rows
    ]


def export_jsonl(examples: list[dict], output_path: Path) -> int:
    """
    Export as JSONL in OpenAI fine-tune chat format:
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    """
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            # Build messages array from conversation
            messages = []
            for turn in ex["conversation"]:
                msg: dict = {"role": turn["role"]}
                if turn.get("content"):
                    msg["content"] = turn["content"]
                if turn.get("tool_calls"):
                    msg["tool_calls"] = [
                        {
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for tc in turn["tool_calls"]
                    ]
                messages.append(msg)

            record = {
                "messages": messages,
                "metadata": {
                    "id": ex["id"],
                    "domain": ex["domain"],
                    "example_type": ex["example_type"],
                    "difficulty": ex["difficulty"],
                    "generator": ex["generator"],
                    "composite_score": ex["composite_score"],
                    "quality_tier": ex["quality_tier"],
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def export_dpo_jsonl(pairs: list[dict], output_path: Path) -> int:
    """
    Export as DPO JSONL in TRL / Axolotl format:
    {"prompt": ..., "chosen": [...], "rejected": [...]}
    """
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            record = {
                "prompt": pair["prompt"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
                "metadata": {
                    "id": pair["id"],
                    "domain": pair["domain"],
                    "example_type": pair["example_type"],
                    "score_delta": pair["score_delta"],
                    "chosen_score": pair["chosen_score"],
                    "rejected_score": pair["rejected_score"],
                    "reason": pair["reason"],
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def export_csv(examples: list[dict], output_path: Path) -> int:
    """Export as CSV with flattened fields for spreadsheet analysis."""
    import csv
    fieldnames = [
        "id", "instruction_text", "domain", "example_type", "difficulty",
        "generator", "tool_names", "num_tool_calls",
        "composite_score", "quality_tier",
        "schema_correctness", "argument_completeness",
        "intent_alignment", "hallucination_score", "chain_coherence",
    ]
    count = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ex in examples:
            tool_names = ", ".join(tc["name"] for tc in ex["tool_calls"])
            writer.writerow({
                "id": ex["id"],
                "instruction_text": ex["instruction_text"],
                "domain": ex["domain"],
                "example_type": ex["example_type"],
                "difficulty": ex["difficulty"],
                "generator": ex["generator"],
                "tool_names": tool_names,
                "num_tool_calls": len(ex["tool_calls"]),
                "composite_score": ex["composite_score"],
                "quality_tier": ex["quality_tier"],
                **ex["scores"],
            })
            count += 1
    return count


def export_parquet(examples: list[dict], output_path: Path) -> int:
    """Export as Parquet using pandas + pyarrow."""
    try:
        import pandas as pd
    except ImportError:
        console.print("[red]pandas not installed. Run: pip install pandas pyarrow[/red]")
        return 0

    rows = []
    for ex in examples:
        rows.append({
            "id": ex["id"],
            "instruction_text": ex["instruction_text"],
            "domain": ex["domain"],
            "example_type": ex["example_type"],
            "difficulty": ex["difficulty"],
            "generator": ex["generator"],
            "tool_calls": json.dumps(ex["tool_calls"]),
            "conversation": json.dumps(ex["conversation"]),
            "composite_score": ex["composite_score"],
            "quality_tier": ex["quality_tier"],
            **ex["scores"],
        })
    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    return len(df)


def run_export(fmt: str, args: argparse.Namespace) -> None:
    """Run a single format export."""
    output_path = get_output_path(fmt, args.output if not args.export_all else None)

    console.print(f"Exporting [bold]{fmt.upper()}[/bold] to [cyan]{output_path}[/cyan]...")

    try:
        if fmt == "dpo":
            data = fetch_preference_pairs(args.min_quality, args.domains, args.limit)
            count = export_dpo_jsonl(data, output_path)
        else:
            data = fetch_examples(args.min_quality, args.domains, args.limit)
            if fmt == "jsonl":
                count = export_jsonl(data, output_path)
            elif fmt == "csv":
                count = export_csv(data, output_path)
            elif fmt == "parquet":
                count = export_parquet(data, output_path)
            else:
                count = 0

        console.print(f"  [green]OK[/green] Exported {count} records to {output_path}")
    except Exception as e:
        console.print(f"  [red]FAIL Export failed: {e}[/red]")
        logger.exception(f"Export error for {fmt}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    console.print("[bold cyan]Agent Training Data Exporter[/bold cyan]")

    if args.export_all:
        for fmt in ["jsonl", "dpo", "csv", "parquet"]:
            run_export(fmt, args)
    else:
        run_export(args.format, args)

    console.print("[bold green]Export complete.[/bold green]")


if __name__ == "__main__":
    main()
