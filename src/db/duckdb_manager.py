"""Safe, read-only DuckDB access for analytical agents."""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import duckdb
import pandas as pd

_MUTATION = re.compile(
    r"\b(drop|delete|insert|update|alter|truncate|create|replace|copy|attach|detach|install|load)\b",
    re.IGNORECASE,
)
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_ALIAS = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_COMMENTS = re.compile(r"(--|/\*|\*/)")


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


def _validate_identifier(value: str, label: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be a valid SQL identifier")
    return value


def _validate_alias(alias: str) -> str:
    if not _ALIAS.fullmatch(alias):
        raise ValueError("database alias must be a valid SQL identifier")
    return alias


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _validate_where_clause(where_clause: str) -> str:
    if not where_clause.strip():
        raise ValueError("where_clause must not be empty")
    if ";" in where_clause or _COMMENTS.search(where_clause) or _MUTATION.search(where_clause):
        raise ValueError("where_clause contains unsafe SQL")
    return where_clause.strip()


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
        self._attached_aliases: set[str] = set()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> DuckDBManager:
        return self

    def __exit__(self, *_: object) -> None:
        try:
            for alias in tuple(self._attached_aliases):
                self.detach_database(alias)
        finally:
            self.close()

    def attach_database(
        self, db_type: str, connection_string: str, alias: str = "remote_db"
    ) -> None:
        """Attach a live database using a dedicated read-only path."""

        normalized_type = db_type.strip().lower()
        safe_alias = _validate_alias(alias)
        if not connection_string:
            raise ValueError("connection_string must not be empty")
        if safe_alias in self._attached_aliases:
            raise ValueError(f"database alias '{safe_alias}' is already attached")

        try:
            if normalized_type == "sqlite":
                self._connection.execute("LOAD sqlite")
                statement = (
                    f"ATTACH {_sql_literal(connection_string)} AS {safe_alias} "
                    "(TYPE SQLITE, READ_ONLY)"
                )
            elif normalized_type in {"postgres", "postgresql"}:
                self._connection.execute("INSTALL postgres")
                self._connection.execute("LOAD postgres")
                statement = (
                    f"ATTACH {_sql_literal(connection_string)} AS {safe_alias} "
                    "(TYPE POSTGRES, READ_ONLY)"
                )
            else:
                raise ValueError("db_type must be 'sqlite' or 'postgresql'")
            self._connection.execute(statement)
        except ValueError:
            raise
        except duckdb.Error as exc:
            raise ValueError(f"failed to attach {normalized_type} database") from exc
        self._attached_aliases.add(safe_alias)

    def list_tables(self, alias: str = "remote_db") -> list[str]:
        """Return attached base-table names without reading table rows."""

        safe_alias = _validate_alias(alias)
        try:
            rows = self._connection.execute(
                "SELECT table_name "
                "FROM information_schema.tables "
                f"WHERE table_catalog = {_sql_literal(safe_alias)} "
                "AND table_type = 'BASE TABLE' ORDER BY table_name"
            ).fetchall()
        except duckdb.Error as exc:
            raise ValueError("failed to list tables") from exc
        return [str(row[0]) for row in rows]

    def detach_database(self, alias: str = "remote_db") -> None:
        """Detach a previously attached database."""

        safe_alias = _validate_alias(alias)
        if safe_alias not in self._attached_aliases:
            return
        try:
            self._connection.execute(f"DETACH {safe_alias}")
        except duckdb.Error as exc:
            raise ValueError("failed to detach database") from exc
        self._attached_aliases.discard(safe_alias)

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

    def query_aggregated_metric(
        self,
        table: str,
        dimensions: list[str],
        numerator: str,
        denominator: str,
        where_clause: str | None = None,
    ) -> pd.DataFrame:
        """Push down a grouped metric aggregation and return only aggregate rows."""

        safe_table = _validate_identifier(table, "table")
        if not dimensions:
            raise ValueError("dimensions must not be empty")
        safe_dimensions = [_validate_identifier(item, "dimension") for item in dimensions]
        safe_numerator = _validate_identifier(numerator, "numerator")
        safe_denominator = _validate_identifier(denominator, "denominator")
        dimension_sql = ", ".join(safe_dimensions)
        where_sql = f" WHERE {_validate_where_clause(where_clause)}" if where_clause else ""
        query = (
            f"SELECT {dimension_sql}, SUM({safe_numerator}) AS {safe_numerator}, "
            f"SUM({safe_denominator}) AS {safe_denominator} "
            f"FROM {safe_table}{where_sql} GROUP BY {dimension_sql}"
        )
        bounded_query = f"SELECT * FROM ({query}) AS _rca_aggregate LIMIT {self.max_rows}"
        started = time.monotonic()
        try:
            result = self._connection.execute(bounded_query).fetchdf()
        except duckdb.Error as exc:
            raise ValueError("aggregated metric query failed") from exc
        if time.monotonic() - started > self.timeout_seconds:
            raise TimeoutError("aggregated metric query exceeded the configured timeout")
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
            cardinality = cast(Any, result.iloc[0, 0])
            results[column] = min(int(cardinality), limit)
        return results
