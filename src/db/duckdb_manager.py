"""Safe, read-only DuckDB access for analytical agents."""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from pathlib import Path

import duckdb
import pandas as pd

_MUTATION = re.compile(
    r"\b(drop|delete|insert|update|alter|truncate|create|replace|copy|attach|detach|install|load)\b",
    re.IGNORECASE,
)


def validate_read_only_sql(query: str) -> str:
    """Validate a single SELECT/WITH query before execution."""

    normalized = query.strip().rstrip(";").strip()
    if not normalized:
        raise ValueError("query must not be empty")
    if ";" in normalized or _MUTATION.search(normalized):
        raise ValueError("only read-only SQL queries are allowed")
    if not re.match(r"^(select|with)\b", normalized, re.IGNORECASE):
        raise ValueError("query must start with SELECT or WITH")
    return normalized


class DuckDBManager:
    """Own a bounded DuckDB connection and expose only analytical operations."""

    def __init__(
        self, database: str | Path = ":memory:", max_rows: int = 500, timeout_seconds: float = 5.0
    ) -> None:
        if max_rows < 1 or timeout_seconds <= 0:
            raise ValueError("max_rows and timeout_seconds must be positive")
        self._database = str(database)
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds
        self._connection = duckdb.connect(self._database, read_only=self._database != ":memory:")

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> DuckDBManager:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def register_dataframe(self, name: str, frame: pd.DataFrame) -> None:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            raise ValueError("table name must be a simple identifier")
        self._connection.register(name, frame)

    def schema(self, table_name: str) -> list[dict[str, str]]:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table_name):
            raise ValueError("table name must be a simple identifier")
        rows = self._connection.execute(f"DESCRIBE {table_name}").fetchall()
        return [{"name": str(row[0]), "type": str(row[1])} for row in rows]

    def run_read_only(self, query: str) -> pd.DataFrame:
        safe_query = validate_read_only_sql(query)
        bounded_query = f"SELECT * FROM ({safe_query}) AS _rca_result LIMIT {self.max_rows}"
        started = time.monotonic()
        try:
            result = self._connection.execute(bounded_query).fetchdf()
        except duckdb.Error as exc:
            raise ValueError(f"analytical query failed: {exc}") from exc
        if time.monotonic() - started > self.timeout_seconds:
            raise TimeoutError("analytical query exceeded the configured timeout")
        return result

    def cardinalities(
        self, table_name: str, columns: Iterable[str], limit: int = 20
    ) -> dict[str, int]:
        if limit < 1:
            raise ValueError("limit must be positive")
        results: dict[str, int] = {}
        for column in columns:
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", column):
                raise ValueError("column names must be simple identifiers")
            result = self.run_read_only(
                f"SELECT COUNT(DISTINCT {column}) AS cardinality FROM {table_name}"
            )
            results[column] = min(int(result.iloc[0, 0]), limit)
        return results
