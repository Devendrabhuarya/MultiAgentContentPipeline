"""
Experiment script: compare baseline (Writer -> Final) vs. agentic
(full feedback-loop) content generation across multiple topics.

Usage:
    python experiments/run_evaluation.py
    python experiments/run_evaluation.py --mock
    python experiments/run_evaluation.py --topics "Topic A" "Topic B" "Topic C"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.evaluation.experiments import run_comparison
from app.llm.client import MockLLMClient, OpenAICompatibleLLMClient
from app.logging_config import configure_logging
from app.research import get_research_provider
from app.research.mock import MockResearchProvider

DEFAULT_TOPICS = [
    "Federated Learning in Healthcare",
    "The Future of Renewable Energy Storage",
    "Zero Trust Security Architecture",
]


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Baseline vs. feedback-loop evaluation experiment")
    parser.add_argument("--topics", nargs="+", default=DEFAULT_TOPICS)
    parser.add_argument("--mock", action="store_true", help="Run fully offline with mock LLM/research")
    parser.add_argument("--output", type=str, default="experiments/results.json")
    args = parser.parse_args()

    if args.mock or not settings.openai_api_key:
        if not args.mock:
            print("No OPENAI_API_KEY found; using --mock offline mode.\n")
        llm_client = MockLLMClient()
        research_provider = MockResearchProvider()
    else:
        llm_client = OpenAICompatibleLLMClient()
        research_provider = get_research_provider()

    report = run_comparison(args.topics, llm_client, research_provider)

    print("\n" + "=" * 70)
    print("BASELINE (Writer -> Final, no feedback loop)")
    print("=" * 70)
    print(f"  Runs: {report.baseline.n_runs}")
    print(f"  Avg overall quality: {report.baseline.avg_overall_score}")
    print(f"  Avg factuality: {report.baseline.avg_factuality}")
    print(f"  Avg SEO score: {report.baseline.avg_seo_score}")
    print(f"  Avg execution time: {report.baseline.avg_execution_time_seconds}s")

    print("\n" + "=" * 70)
    print("AGENTIC (full closed-loop feedback pipeline)")
    print("=" * 70)
    print(f"  Runs: {report.agentic.n_runs}")
    print(f"  Avg overall quality: {report.agentic.avg_overall_score}")
    print(f"  Avg factuality: {report.agentic.avg_factuality}")
    print(f"  Avg SEO score: {report.agentic.avg_seo_score}")
    print(f"  Avg iterations: {report.agentic.avg_iterations}")
    print(f"  Avg execution time: {report.agentic.avg_execution_time_seconds}s")

    print("\n" + "=" * 70)
    print(f"Baseline quality: {report.baseline.avg_overall_score}")
    print(f"Feedback pipeline: {report.agentic.avg_overall_score}")
    print(f"Improvement: {report.improvement_percentage}%")
    print("=" * 70)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": {
            "n_runs": report.baseline.n_runs,
            "avg_overall_score": report.baseline.avg_overall_score,
            "avg_factuality": report.baseline.avg_factuality,
            "avg_seo_score": report.baseline.avg_seo_score,
            "avg_execution_time_seconds": report.baseline.avg_execution_time_seconds,
        },
        "agentic": {
            "n_runs": report.agentic.n_runs,
            "avg_overall_score": report.agentic.avg_overall_score,
            "avg_factuality": report.agentic.avg_factuality,
            "avg_seo_score": report.agentic.avg_seo_score,
            "avg_iterations": report.agentic.avg_iterations,
            "avg_execution_time_seconds": report.agentic.avg_execution_time_seconds,
        },
        "improvement_percentage": report.improvement_percentage,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved detailed report to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
