"""Schema and cardinality inspection node."""

from agents.state import RCAState, SchemaProfile
from db.duckdb_manager import DuckDBManager


def inspect_schema(state: RCAState, manager: DuckDBManager) -> RCAState:
    """Discover dimensions from the current metric table."""

    profile = manager.schema(state.table_name)
    metric_columns = {state.numerator, state.denominator}
    dimensions = [item["name"] for item in profile if item["name"] not in metric_columns]
    dimensions = [name for name in dimensions if name != "period"]
    cardinalities = manager.cardinalities(state.table_name, dimensions)
    dimensions = [name for name in dimensions if cardinalities[name] > 1]
    state.schema_profile = SchemaProfile(
        columns=profile,
        cardinalities=cardinalities,
        dimensions=dimensions,
    )
    state.trace.append("schema_inspector")
    return state
