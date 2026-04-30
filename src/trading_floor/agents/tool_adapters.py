"""Adapt LangChain MCP tools so CrewAI agents can consume them.

LangChain and CrewAI use different BaseTool base classes. MCP-loaded tools
also expose `args_schema` as a JSON-schema dict rather than a Pydantic class,
which CrewAI's tool description generator does not handle. This module wraps
each LangChain tool in a CrewAI BaseTool and synthesizes a Pydantic args
model from the JSON schema on the fly.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Type

from crewai.tools import BaseTool as CrewBaseTool
from langchain_core.tools import BaseTool as LCBaseTool
from pydantic import BaseModel, Field, create_model


def _run_async(coro) -> Any:
    """Run a coroutine to completion regardless of whether a loop is active."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


_JSON_TO_PY: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _py_type(spec: dict) -> Any:
    t = spec.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        t = non_null[0] if non_null else "string"
    if t == "array":
        item_type = _py_type(spec.get("items") or {})
        return list[item_type]  # type: ignore[valid-type]
    return _JSON_TO_PY.get(t, Any)


def _schema_to_model(name: str, schema: dict | type[BaseModel] | None) -> type[BaseModel]:
    """Coerce a LangChain tool's args_schema into a Pydantic model class."""
    if schema is None:
        return create_model(f"{name}Args")
    if inspect.isclass(schema) and issubclass(schema, BaseModel):
        return schema
    if not isinstance(schema, dict):
        return create_model(f"{name}Args")

    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    fields: dict[str, tuple[Any, Any]] = {}
    for fname, spec in properties.items():
        py_t = _py_type(spec) if isinstance(spec, dict) else Any
        description = spec.get("description") if isinstance(spec, dict) else None
        if fname in required:
            fields[fname] = (py_t, Field(..., description=description))
        else:
            default = spec.get("default") if isinstance(spec, dict) else None
            fields[fname] = (py_t | None, Field(default=default, description=description))
    if not fields:
        return create_model(f"{name}Args")
    return create_model(f"{name}Args", **fields)  # type: ignore[arg-type]


def adapt(lc_tool: LCBaseTool) -> CrewBaseTool:
    """Wrap a LangChain BaseTool as a CrewAI BaseTool."""
    args_model = _schema_to_model(lc_tool.name, getattr(lc_tool, "args_schema", None))

    class _Adapted(CrewBaseTool):
        name: str = lc_tool.name
        description: str = (lc_tool.description or lc_tool.name).strip()
        args_schema: Type[BaseModel] = args_model  # type: ignore[assignment]

        def _run(self, **kwargs: Any) -> Any:
            return _run_async(lc_tool.ainvoke(kwargs))

    _Adapted.__name__ = f"Adapted_{lc_tool.name}"
    return _Adapted()


def adapt_many(lc_tools: list[LCBaseTool]) -> list[CrewBaseTool]:
    return [adapt(t) for t in lc_tools]
