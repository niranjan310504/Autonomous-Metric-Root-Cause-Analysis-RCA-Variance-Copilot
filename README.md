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

To enable Gemini-backed executive narratives, install the optional provider SDK:

```bash
pip install "google-genai>=1.0"
```

Set `GEMINI_API_KEY` in the environment or a local `.env` file. Without the key, without the
optional SDK, or when Gemini is unavailable, the narrative node returns a labeled deterministic
fallback. The Gemini prompt contains only precomputed findings and subgroup evidence; it does not
perform metric calculations or attribution.

Verify the real provider path before relying on Gemini-backed output:

```bash
python scripts/verify_gemini.py
```

This check calls the same public adapter used by the application and exits with an error if the
key is missing or Gemini falls back. It does not print the API key.

The default model can be changed without editing code by setting `GEMINI_MODEL` in `.env` if the
provider reports that the default model is unavailable or temporarily overloaded.

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
