"""
run_pipeline.py — CLI entry point for the Agent Function-Call Training Data Generator.

Usage examples:
    python run_pipeline.py --samples 100 --domains calendar weather --types 1 2
    python run_pipeline.py --samples 6000 --workers 8
    python run_pipeline.py --samples 50 --domains all --no-resume
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so 'src' package is importable
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

load_dotenv(override=True)
console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate function-call training data using Groq + Gemini APIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of training examples to generate.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker threads.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["all"],
        choices=[
            "all", "calendar", "search", "code_exec", "crm",
            "weather", "finance", "email", "files",
            "notifications", "maps", "tasks", "database",
        ],
        help="Domains to include in generation.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        choices=[1, 2, 3, 4],
        help="Example types: 1=single 2=chain 3=ambiguous 4=parallel.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh — ignore existing instructions in database.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Print banner
    banner = Text()
    banner.append("Agent Function-Call Training Data Generator\n", style="bold cyan")
    banner.append("Groq (Generator A) + Gemini (Generator B + Judge + Embeddings)\n", style="dim")
    banner.append(f"Samples: {args.samples} | Workers: {args.workers} | Domains: {args.domains}", style="green")
    console.print(Panel(banner, title="[bold]Pipeline Config[/bold]", border_style="cyan"))

    # Import here (after sys.path setup)
    try:
        from src.pipeline import Pipeline
    except ImportError as e:
        console.print(f"[bold red]Import error: {e}[/bold red]")
        console.print("Make sure you are running from the project root directory.")
        sys.exit(1)

    pipeline = Pipeline(
        samples=args.samples,
        workers=args.workers,
        domains=args.domains,
        types=args.types,
        resume=not args.no_resume,
    )

    stats = pipeline.run()

    # Exit with non-zero code if nothing was generated
    if stats.generated == 0:
        console.print("[bold red]Warning: No examples were generated. Check your API keys and database connection.[/bold red]")
        sys.exit(1)

    console.print("[bold green]\nPipeline complete! Run `python export.py` to export the data.[/bold green]")


if __name__ == "__main__":
    main()
