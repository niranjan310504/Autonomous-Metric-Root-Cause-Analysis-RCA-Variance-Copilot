import sqlite3
import time
from pathlib import Path
from typing import Any, cast

import pytest

from db.duckdb_manager import DuckDBManager


@pytest.fixture(scope="module")
def clickstream_database(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[tuple[str, str], tuple[int, int]]]:
    database_path = tmp_path_factory.mktemp("sqlite") / "clickstream.sqlite"
    expected: dict[tuple[str, str], tuple[int, int]] = {}
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE clicks (channel TEXT, device TEXT, conversions INTEGER, visits INTEGER)"
        )
        rows = []
        for index in range(50_000):
            channel = ("organic", "paid", "social")[index % 3]
            device = ("mobile", "desktop")[index % 2]
            conversions = index % 5
            visits = 10 + index % 7
            rows.append((channel, device, conversions, visits))
            key = (channel, device)
            previous = expected.get(key, (0, 0))
            expected[key] = (previous[0] + conversions, previous[1] + visits)
        connection.executemany("INSERT INTO clicks VALUES (?, ?, ?, ?)", rows)
        connection.commit()
    return database_path, expected


def test_attach_list_tables_and_aggregate_pushdown(
    clickstream_database: tuple[Path, dict[tuple[str, str], tuple[int, int]]],
) -> None:
    database_path, expected = clickstream_database
    with DuckDBManager(max_rows=20) as manager:
        manager.attach_database("sqlite", str(database_path))
        assert manager.list_tables() == ["clicks"]

        started = time.perf_counter()
        result = manager.query_aggregated_metric(
            "remote_db.clicks",
            ["channel", "device"],
            "conversions",
            "visits",
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        records = cast(list[dict[str, Any]], result.to_dict(orient="records"))
        actual = {
            (str(row["channel"]), str(row["device"])): (
                int(row["conversions"]),
                int(row["visits"]),
            )
            for row in records
        }
        assert actual == expected
        assert len(result) < 10
        assert elapsed_ms < 1_000


@pytest.fixture
def writable_probe_database(tmp_path: Path) -> Path:
    database_path = tmp_path / "probe.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE clicks (channel TEXT, device TEXT, conversions INTEGER, visits INTEGER)"
        )
        connection.execute("INSERT INTO clicks VALUES ('organic', 'mobile', 1, 10)")
        connection.commit()
    return database_path


def test_attached_database_rejects_writes(
    writable_probe_database: Path,
) -> None:
    database_path = writable_probe_database
    with DuckDBManager() as manager:
        manager.attach_database("sqlite", str(database_path))
        for statement in (
            "INSERT INTO remote_db.clicks VALUES ('x', 'y', 1, 1)",
            "UPDATE remote_db.clicks SET visits = 0",
            "DROP TABLE remote_db.clicks",
        ):
            with pytest.raises(ValueError, match="read-only"):
                manager.run_read_only(statement)

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM clicks").fetchone() == (1,)


def test_identifiers_and_connection_errors_do_not_leak_credentials(tmp_path: Path) -> None:
    with DuckDBManager() as manager:
        with pytest.raises(ValueError, match="identifier"):
            manager.attach_database("sqlite", "missing.sqlite", "bad;alias")
        with pytest.raises(ValueError, match="db_type") as error:
            manager.attach_database("unsupported", "password=secret")
        assert "secret" not in str(error.value)
        with pytest.raises(ValueError, match="unsafe"):
            manager.query_aggregated_metric(
                "remote_db.clicks",
                ["channel"],
                "conversions",
                "visits",
                "1=1; SELECT 'password=secret'",
            )
