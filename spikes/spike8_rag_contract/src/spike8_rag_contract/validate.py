from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator, validate


def assert_valid(data: dict[str, Any], schema: dict[str, Any]) -> None:
    Draft202012Validator.check_schema(schema)
    validate(instance=data, schema=schema)
