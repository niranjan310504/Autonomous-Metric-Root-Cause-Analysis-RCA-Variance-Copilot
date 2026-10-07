"""Fail-fast smoke test for the public Gemini narrative adapter."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from agents.llm_adapter import (  # noqa: E402
    GeminiProviderError,
    generate_actionable_narrative,
)


def main() -> int:
    """Verify that the public adapter reaches Gemini instead of using fallback text."""

    try:
        narrative = generate_actionable_narrative(
            metric_name="conversion_rate",
            findings=[
                {
                    "dimension": "device",
                    "segment": "ios",
                    "contribution": -0.12,
                    "evidence": "4.00% current vs 16.00% baseline",
                }
            ],
            subgroups=[],
            require_gemini=True,
        )
    except GeminiProviderError as exc:
        print(f"Gemini verification failed: {exc}")
        return 1
    if not narrative.startswith("[Gemini]"):
        raise RuntimeError("Gemini verification returned a non-Gemini narrative")
    print("Gemini verification passed: public adapter returned a provider narrative.")
    print(narrative)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())