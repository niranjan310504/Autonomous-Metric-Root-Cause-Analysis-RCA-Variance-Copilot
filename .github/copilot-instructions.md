# Autonomous Metric RCA & Variance Copilot - Architectural Rules & Enterprise Roadmap

## Tech Stack
- Runtime: Python 3.11+
- OLAP Engine: DuckDB (in-memory / local `.duckdb`), with connectors for BigQuery/Snowflake pushdown
- Agent Orchestration: LangGraph (`StateGraph`), Pydantic v2
- Statistical Engine: Scikit-learn (`DecisionTreeClassifier`), NumPy, SciPy (Mix-Shift / Rate-Shift decomposition & Tree Mining)
- Reasoning & Narrative: Gemini API (`gemini-3.8-flash`) / OpenAI (Hybrid LLM Seam for action recommendations)
- Presentation & Alerting: Streamlit, Plotly (`Waterfall`), Slack Webhooks

## Non-Negotiable Architectural Contracts

1. **Separation of Compute and Reasoning:** 
   - Never use the LLM to calculate arithmetic, aggregate metrics, or perform statistical attribution.
   - All calculations must happen deterministically in DuckDB SQL or dedicated Python statistical modules.
   - LLMs are strictly bounded to executive narrative synthesis and actionable business recommendations using pre-calculated evidence.

2. **Hermetic Testing & Zero Network Leakage:**
   - Unit tests (`tests/`) must NEVER make outbound network calls to external APIs (Gemini, OpenAI, Slack).
   - In unit tests, always mock provider entrypoints or unset provider keys (e.g., `monkeypatch.delenv("GEMINI_API_KEY", raising=False)`).
   - The entire unit test suite must execute in **< 2 seconds** and run cleanly in offline/CI environments.
   - Live provider smoke tests belong strictly in standalone verification scripts (`scripts/verify_gemini.py`).

3. **Algorithmic Scalability & Memory Safety:**
   - Never unroll or expand aggregated metric counts into virtual individual rows (`* conversions` or `* visits`).
   - Statistical tree mining (`src/engine/subgroup_miner.py`) must use Scikit-learn's native `sample_weight` parameter across aggregated segment matrices.
   - Memory complexity must scale with unique segment combinations $O(K)$, never with raw event volume $O(N_{\text{visits}})$.

4. **Schema Completeness & Variance Invariant:**
   - The 3 Kitsberger decomposition components must exist as explicit typed float fields on `Finding`: `rate_effect`, `mix_effect`, and `interaction_effect`.
   - Invariant contract: `rate_effect + mix_effect + interaction_effect == contribution`.

5. **Read-Only Database Safety:**
   - The DuckDB connection used by agents must be strictly read-only (`read_only=True` or PRAGMA-enforced).
   - SQL queries must be sanitized; DDL/DML statements (`DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`, `TRUNCATE`, `REPLACE`, `ATTACH`) must raise immediate exceptions.
   - Enforce a maximum query timeout and a strict limit on returned rows (`LIMIT 500`).

6. **Repository Single Source of Truth:**
   - No orphan, duplicate, or abandoned scaffold packages (do not maintain `src/agent/` alongside `src/agents/`, or dead stubs in `src/core/`, `src/ui/`, `src/data/`).
   - Canonical layout:
     - `src/agents/`: Pydantic state, schema inspector, hypothesis generator, SQL investigator, narrative synthesizer, LLM adapter.
     - `src/engine/`: Kitsberger variance waterfall, time-series baseline slicer, subgroup tree miner.
     - `src/db/`: Safe DuckDB manager and SQL sanitization.
     - `src/alerting/`: Slack webhook dispatcher.
     - `src/graph.py`: LangGraph orchestration entrypoint.
     - `src/app.py`: Streamlit executive dashboard.

7. **Typing & Validation:**
   - All agent states, tool arguments, and node outputs must use strict Pydantic v2 models (`extra="forbid"`).
   - Every module must maintain 100% type annotations (`mypy --strict` compliance).
   - Modular file structure: no single file exceeding 250 lines.

## Enterprise Architecture Roadmap

```mermaid
flowchart TD
    A[Raw Metric Observations] --> B1["1. Time-Series Baseline Slicing (WoW / YoY)"]
    B1 --> B2["2. Kitsberger Waterfall Decomposition (Rate vs Mix vs Interaction)"]
    B2 --> B3["3. Multi-Dimensional Tree Subgroup Mining (sample_weight)"]
    B3 --> B4["4. Hybrid LLM Narrative Synthesis (Gemini / Deterministic Fallback)"]
    B4 --> C1["5. Executive Plotly Waterfall & Stacked Bar UI"]
    B4 --> C2["6. Automated Slack / Webhook Push Alerts"]
## Security guidelines
   Do not read the env file or any secrets in the repo. All secrets must be injected via CI/CD or runtime environment variables.