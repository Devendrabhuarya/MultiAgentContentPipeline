"""
CLI entrypoint for the Multi-Agent Content Pipeline.

Usage:
    python main.py
    python main.py --topic "Federated Learning in Healthcare" --word-count 1500
    python main.py --mock          # run fully offline with MockLLMClient
"""
from __future__ import annotations

import argparse
import sys

from app.config import settings
from app.graph.workflow import run_pipeline
from app.llm.client import LLMError, MockLLMClient, OpenAICompatibleLLMClient
from app.logging_config import configure_logging
from app.models.schemas import ContentType, UserRequest
from app.research import get_research_provider
from app.research.mock import MockResearchProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-agent content pipeline")
    parser.add_argument("--topic", type=str, default=None)
    parser.add_argument("--content-type", type=str, default=ContentType.BLOG.value,
                         choices=[c.value for c in ContentType])
    parser.add_argument("--audience", type=str, default="general audience")
    parser.add_argument("--tone", type=str, default="professional")
    parser.add_argument("--word-count", type=int, default=1200)
    parser.add_argument("--keywords", type=str, default="")
    parser.add_argument("--max-iterations", type=int, default=settings.max_iterations)
    parser.add_argument("--quality-threshold", type=float, default=settings.quality_threshold)
    parser.add_argument("--mock", action="store_true", help="Run offline with a mock LLM + mock research")
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()

    topic = args.topic or input("Enter a topic: ").strip()
    if not topic:
        print("A topic is required.")
        return 1

    request = UserRequest(
        topic=topic,
        content_type=ContentType(args.content_type),
        target_audience=args.audience,
        tone=args.tone,
        word_count=args.word_count,
        keywords=[k.strip() for k in args.keywords.split(",") if k.strip()],
        max_iterations=args.max_iterations,
        quality_threshold=args.quality_threshold,
    )

    if args.mock or not settings.openai_api_key:
        if not args.mock:
            print("No OPENAI_API_KEY found; falling back to --mock offline mode.\n")
        llm_client = MockLLMClient()  # generic filler responses; good for smoke-testing the graph
        research_provider = MockResearchProvider()
    else:
        llm_client = OpenAICompatibleLLMClient()
        research_provider = get_research_provider()

    try:
        state = run_pipeline(request, llm_client, research_provider)
    except LLMError as exc:
        print(f"LLM error: {exc}")
        return 1

    final = state["final_content"]
    print("\n" + "=" * 70)
    print(f"TOPIC: {final.topic}")
    print(f"STATUS: {final.status.value}")
    print(f"ITERATIONS: {final.iterations_used}")
    print(f"EXECUTION TIME: {final.total_execution_time_seconds}s")
    print("=" * 70)
    print(f"\nQUALITY SCORE: {final.quality_score.overall_score:.1f}/100")
    print(f"  Factuality: {final.quality_score.factuality:.1f}")
    print(f"  Relevance: {final.quality_score.relevance:.1f}")
    print(f"  Structure: {final.quality_score.structure:.1f}")
    print(f"  Readability: {final.quality_score.readability:.1f}")
    print(f"  SEO: {final.quality_score.seo:.1f}")
    print(f"  Evidence: {final.quality_score.evidence:.1f}")

    print("\nREVISION HISTORY:")
    for r in state.get("revision_history", []):
        print(f"  Version {r.version}: score {r.critic_evaluation.overall_score:.1f}")

    if final.warnings:
        print("\nWARNINGS:")
        for w in final.warnings:
            print(f"  - {w}")

    print("\n" + "-" * 70)
    print(f"FINAL CONTENT: {final.final_content.title}")
    print("-" * 70)
    print(final.final_content.body_markdown)

    return 0


if __name__ == "__main__":
    sys.exit(main())
