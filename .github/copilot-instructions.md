# Autonomous Metric RCA & Variance Copilot - Architectural Rules

## Tech Stack
- Runtime: Python 3.11+
- OLAP Engine: DuckDB (in-memory or local `.duckdb` file)
- Agent Orchestration: LangGraph (`StateGraph`), Pydantic v2
- Statistical Engine: Scikit-learn (`DecisionTreeRegressor`/`DecisionTreeClassifier`), NumPy, SciPy
- Presentation: Streamlit, Plotly

## Non-Negotiable Architectural Contracts
1. **Separation of Compute and Reasoning:** 
   - Never use the LLM to calculate arithmetic, aggregate metrics, or perform statistical attribution.
   - All calculations must happen deterministically in DuckDB SQL or dedicated Python statistical modules.
2. **Read-Only Database Safety:**
   - The DuckDB connection used by agents must be strictly read-only (`read_only=True` or PRAGMA-enforced).
   - SQL queries must be sanitized; DDL/DML statements (`DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`) must raise immediate exceptions.
   - Enforce a maximum query timeout and a strict limit on returned rows (`LIMIT 500`).
3. **Typing & Validation:**
   - All agent states, tool arguments, and node outputs must use strict Pydantic v2 models.
   - Every module must have complete type annotations (`typing`).
4. **Code Quality:**
   - Modular structure: no single file exceeding 250 lines.
   - Self-contained, deterministic unit tests for every mathematical engine component.