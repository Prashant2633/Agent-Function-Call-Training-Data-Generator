"""
test_registry.py — Tests for the SchemaRegistry.

Tests that the registry can load all 46 schemas across 12 domains,
validate their structure, and serve them in the correct API formats.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.registry import SchemaRegistry, load_registry


@pytest.fixture(scope="module")
def registry() -> SchemaRegistry:
    """Load the registry once for all tests in this module."""
    return load_registry()


def test_registry_loads_all_domains(registry: SchemaRegistry):
    """All 12 expected domains should be present."""
    expected_domains = {
        "calendar", "search", "code_exec", "crm", "weather", "finance",
        "email", "files", "notifications", "maps", "tasks", "database"
    }
    loaded_domains = set(registry.list_domains())
    assert expected_domains == loaded_domains, f"Missing domains: {expected_domains - loaded_domains}"


def test_registry_total_tool_count(registry: SchemaRegistry):
    """Should have exactly 46 tools loaded."""
    all_tools = registry.list_all()
    assert len(all_tools) == 46, f"Expected 46 tools, got {len(all_tools)}"


def test_registry_schema_has_required_fields(registry: SchemaRegistry):
    """Every schema must have name, description, and parameters."""
    for tool in registry.list_all():
        assert tool.name, f"Tool is missing name"
        assert tool.description, f"Tool {tool.name} is missing description"
        assert tool.parameters is not None, f"Tool {tool.name} is missing parameters"


def test_registry_openai_tool_format(registry: SchemaRegistry):
    """Tools returned for OpenAI/Groq format must have correct structure."""
    tools = registry.get_tools_for_generation("calendar")
    assert len(tools) > 0, "Calendar domain returned no tools"
    for t in tools:
        assert "type" in t, "Missing 'type' key"
        assert t["type"] == "function"
        assert "function" in t
        assert "name" in t["function"]
        assert "description" in t["function"]
        assert "parameters" in t["function"]


def test_registry_get_tool_by_name(registry: SchemaRegistry):
    """Should be able to retrieve a specific tool by name."""
    tool = registry.get_tool("create_event")
    assert tool is not None
    assert tool.name == "create_event"
    assert tool.domain == "calendar"


def test_registry_missing_tool_returns_none(registry: SchemaRegistry):
    """Getting a non-existent tool should return None gracefully."""
    tool = registry.get_tool("nonexistent_tool_xyz")
    assert tool is None


def test_registry_calendar_tools(registry: SchemaRegistry):
    """Calendar domain should have all 4 expected tools."""
    tools = registry.get_domain("calendar")
    tool_names = {t.name for t in tools}
    assert "create_event" in tool_names
    assert "delete_event" in tool_names
    assert "reschedule_event" in tool_names
    assert "find_free_slot" in tool_names


def test_registry_finance_tools(registry: SchemaRegistry):
    """Finance domain should have all 5 expected tools."""
    tools = registry.get_domain("finance")
    tool_names = {t.name for t in tools}
    expected = {"get_stock_price", "get_portfolio_summary", "place_order", "get_exchange_rate", "calculate_roi"}
    assert expected == tool_names


def test_registry_gemini_tool_format(registry: SchemaRegistry):
    """Gemini tool dicts must not include unsupported JSON Schema fields."""
    tools = registry.get_gemini_tools_for_generation("weather")
    for tool in tools:
        # Gemini tools should not have $schema or $comment keys in nested props
        assert "$schema" not in str(tool)
        assert "$comment" not in str(tool)


def test_registry_include_all_flag(registry: SchemaRegistry):
    """include_all=True should return more tools than domain-only."""
    domain_tools = registry.get_tools_for_generation("calendar", include_all=False)
    all_tools = registry.get_tools_for_generation("calendar", include_all=True)
    assert len(all_tools) > len(domain_tools)
