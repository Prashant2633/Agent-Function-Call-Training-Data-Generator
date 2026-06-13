"""
registry.py — Tool schema registry: loads all JSON schemas and provides fast lookup.

The registry is the single source of truth for what tools exist, what their
parameters are, and what valid values look like. It's used by:
  - Generator: to build the tools list for API calls
  - Validator: to check generated calls against schema
  - Scorer: to detect hallucinated tool names or parameters
  - Dashboard: to browse all available schemas

Design decision: In-memory dict for O(1) lookup by name. The 46 schemas are
small enough that loading them all at startup (< 1ms) is the right tradeoff.
We don't need a database round-trip on every validation call.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import jsonschema
from jsonschema import Draft7Validator

from src.models import ToolSchema, ValidationResult, FailureMode

logger = logging.getLogger(__name__)


class SchemaRegistry:
    """
    In-memory registry of all tool schemas.

    Usage:
        registry = SchemaRegistry()
        registry.load_all(Path("schemas/"))
        tool = registry.get_tool("create_event")
        domain_tools = registry.get_domain("calendar")
    """

    def __init__(self) -> None:
        # Primary index: tool_name → ToolSchema
        self._by_name: dict[str, ToolSchema] = {}
        # Domain index: domain_name → [ToolSchema, ...]
        self._by_domain: dict[str, list[ToolSchema]] = {}
        # Usage counters: tool_name → int (incremented by generator)
        self._usage_counts: dict[str, int] = {}

    def load_all(self, schemas_dir: Path) -> int:
        """
        Recursively load all .json files from schemas_dir.
        Returns the number of schemas loaded.

        Expected structure: schemas/<domain>/<tool_name>.json
        """
        if not schemas_dir.exists():
            raise FileNotFoundError(f"Schemas directory not found: {schemas_dir}")

        loaded = 0
        for json_file in sorted(schemas_dir.rglob("*.json")):
            try:
                self._load_file(json_file)
                loaded += 1
            except Exception as e:
                logger.error(f"Failed to load schema {json_file}: {e}")

        logger.info(f"Loaded {loaded} tool schemas from {schemas_dir}")
        return loaded

    def _load_file(self, path: Path) -> None:
        """Load a single schema JSON file into the registry."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Infer domain from directory name if not in JSON
        if "domain" not in data:
            data["domain"] = path.parent.name

        schema = ToolSchema(**data)

        # Validate the schema itself is valid JSON Schema draft-07
        # (catches authoring errors early)
        try:
            Draft7Validator.check_schema(schema.parameters)
        except jsonschema.SchemaError as e:
            logger.warning(f"Schema {schema.name} has JSON Schema error: {e.message}")

        self._by_name[schema.name] = schema
        self._by_domain.setdefault(schema.domain, []).append(schema)
        self._usage_counts[schema.name] = 0

    def get_tool(self, name: str) -> Optional[ToolSchema]:
        """
        O(1) lookup by tool name.
        Returns None if tool doesn't exist (caller should flag as hallucination).
        """
        return self._by_name.get(name)

    def get_domain(self, domain: str) -> list[ToolSchema]:
        """Returns all tools for a domain. Empty list if domain not found."""
        return self._by_domain.get(domain, [])

    def list_all(self) -> list[ToolSchema]:
        """Returns all registered tools sorted by domain then name."""
        return sorted(self._by_name.values(), key=lambda t: (t.domain, t.name))

    def list_domains(self) -> list[str]:
        """Returns list of all domain names."""
        return sorted(self._by_domain.keys())

    def tool_exists(self, name: str) -> bool:
        """Quick existence check used by hallucination detector."""
        return name in self._by_name

    def increment_usage(self, tool_name: str) -> None:
        """Track how often each tool appears in generated examples."""
        if tool_name in self._usage_counts:
            self._usage_counts[tool_name] += 1

    def get_usage_stats(self) -> dict[str, int]:
        """Returns usage counts per tool for the dashboard Schema Registry page."""
        return dict(sorted(self._usage_counts.items(), key=lambda x: x[1], reverse=True))

    def validate_call(
        self, tool_name: str, arguments: dict
    ) -> tuple[bool, list[str]]:
        """
        Validate a tool call against its registered schema.
        Returns (is_valid, list_of_error_messages).

        This is the core validation logic used by Validator. Uses jsonschema
        for full draft-07 validation including nested objects, enums, type checks.
        """
        errors: list[str] = []

        # Check 1: Tool exists (hallucination detection)
        tool = self._by_name.get(tool_name)
        if tool is None:
            return False, [f"Tool '{tool_name}' not found in registry (hallucinated API)"]

        # Check 2: Full JSON Schema validation
        validator = Draft7Validator(tool.parameters)
        for error in sorted(validator.iter_errors(arguments), key=lambda e: e.path):
            path = ".".join(str(p) for p in error.path) or "root"
            errors.append(f"{path}: {error.message}")

        # Check 3: Check for extra properties not in schema
        # (catches hallucinated parameters beyond what jsonschema reports)
        schema_params = set(tool.parameters.get("properties", {}).keys())
        extra_params = set(arguments.keys()) - schema_params
        if extra_params and not tool.parameters.get("additionalProperties", False):
            errors.extend([f"Hallucinated parameter: '{p}'" for p in extra_params])

        return len(errors) == 0, errors

    def get_tools_for_generation(
        self, domain: str, include_all: bool = False
    ) -> list[dict]:
        """
        Returns tool definitions in OpenAI/Grok format for use in API calls.

        Design decision: We pass ALL tools in the domain (not just the expected one)
        to test whether the model correctly selects the right tool.
        For cross-domain examples (Type 4), include_all=True passes all 46 tools.
        """
        if include_all:
            tools = self.list_all()
        else:
            tools = self.get_domain(domain)

        return [t.to_openai_tool() for t in tools]

    def get_gemini_tools_for_generation(
        self, domain: str, include_all: bool = False
    ) -> list[dict]:
        """Returns tool definitions in Gemini function declaration format."""
        if include_all:
            tools = self.list_all()
        else:
            tools = self.get_domain(domain)
        return [t.to_gemini_tool() for t in tools]

    def __len__(self) -> int:
        return len(self._by_name)

    def __repr__(self) -> str:
        return f"SchemaRegistry({len(self)} tools, {len(self._by_domain)} domains)"


# Module-level singleton — initialized once and shared across the pipeline
_registry: Optional[SchemaRegistry] = None


def get_registry() -> SchemaRegistry:
    """Returns the global registry singleton. Must call load_registry() first."""
    if _registry is None:
        raise RuntimeError("Registry not initialized. Call load_registry() first.")
    return _registry


def load_registry(schemas_dir: Optional[Path] = None) -> SchemaRegistry:
    """
    Initialize the global registry from schemas directory.
    Idempotent: calling multiple times returns the same registry.
    """
    global _registry
    if _registry is None:
        if schemas_dir is None:
            schemas_dir = Path(__file__).parent.parent / "schemas"
        _registry = SchemaRegistry()
        _registry.load_all(schemas_dir)
    return _registry
