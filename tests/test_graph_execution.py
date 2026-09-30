import importlib.util

from graph import run_analysis


def test_graph_produces_trace_and_finding() -> None:
    spec = importlib.util.spec_from_file_location("generator", "data/generate_synthetic_data.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline, current = module.generate_dataset()
    result = run_analysis(baseline, current)
    assert result.trace == [
        "schema_inspector",
        "hypothesis_generator",
        "sql_investigator",
        "narrative_synthesizer",
    ]
    assert result.findings
    assert "driver" in result.narrative
