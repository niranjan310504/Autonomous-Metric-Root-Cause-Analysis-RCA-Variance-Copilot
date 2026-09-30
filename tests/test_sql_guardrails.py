import pandas as pd
import pytest

from db.duckdb_manager import DuckDBManager


def test_query_results_are_bounded() -> None:
    with DuckDBManager(max_rows=2) as manager:
        manager.register_dataframe("metrics", pd.DataFrame({"value": [1, 2, 3]}))
        assert len(manager.run_read_only("SELECT * FROM metrics")) == 2


def test_multi_statement_and_mutation_are_rejected() -> None:
    with DuckDBManager() as manager:
        with pytest.raises(ValueError, match="read-only"):
            manager.run_read_only("SELECT 1; DROP TABLE metrics")
        with pytest.raises(ValueError, match="read-only"):
            manager.run_read_only("DELETE FROM metrics")
