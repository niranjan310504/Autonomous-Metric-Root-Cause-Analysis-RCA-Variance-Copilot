# Autonomous Metric RCA & Variance Copilot

This project is a modular, deterministic-first agentic workflow for diagnosing metric drops. It
uses DuckDB for bounded read-only investigations, Python for variance attribution, and LangGraph
for an explicit trace. The default reasoning nodes do not require an LLM or API key.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
```

Run the demo:

```bash
streamlit run src/app.py
```

Generate the sample CSVs with `python data/generate_synthetic_data.py`.

## Architecture and safety

- `src/db/duckdb_manager.py` validates single-statement `SELECT`/`WITH` SQL, caps results at
  500 rows, and exposes schema/cardinality discovery.
- `src/engine/` contains deterministic conversion-rate waterfall and decision-tree subgroup math.
- `src/agents/` contains strict Pydantic state and small graph nodes.
- `src/graph.py` composes the nodes into a LangGraph workflow; `src/app.py` presents the trace.
- Arithmetic and attribution never depend on an LLM. An LLM adapter can be added around the
  hypothesis/narrative seams without changing the compute engine.

## Using Copilot and agentic workspaces

For the most reliable workflow, ask Copilot to inspect first, then make one bounded change at a
time. Keep the repository instructions in `.github/copilot-instructions.md`, review the proposed
plan before implementation, and run the narrowest relevant test after each change. Use a separate
workspace for experiments, keep generated data reproducible, and ask Copilot to explain the trace
and evidence rather than accepting an unsupported narrative. Never paste credentials or private
data into prompts; configure provider secrets through environment variables.
