"""
test_pipeline.py
CLI test runner for the Multi-Stage Content Generation Pipeline.
Verifies stage execution with Ollama (e.g. qwen2.5:3b), Cloud, or Mock mode.
"""

import sys
import os
import argparse
from pipeline.config import ContentBrief
from pipeline.models import build_llm, check_ollama_status, get_available_ollama_models
from pipeline.chains import ContentPipeline


def main():
    parser = argparse.ArgumentParser(description="Test Content Generation Pipeline")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openai", "mock"], help="LLM Provider")
    parser.add_argument("--model", default="qwen2.5:3b", help="Model name (e.g. qwen2.5:3b or gpt-4o-mini)")
    parser.add_argument("--format", default="Blog Post", choices=["Blog Post", "Social Media Thread", "Newsletter", "Executive Report"], help="Output format")
    parser.add_argument("--topic", default="Autonomous AI Agent Pipelines in 2026", help="Content topic")
    args = parser.parse_args()

    print("=" * 70)
    print("Project 8: Content Generation Pipeline — CLI Test Runner")
    print("=" * 70)
    print(f"Provider:    {args.provider}")
    print(f"Model:       {args.model}")
    print(f"Format:      {args.format}")
    print(f"Topic:       {args.topic}")
    print("-" * 70)

    if args.provider == "ollama":
        status = check_ollama_status()
        print(f"Ollama Status: {status['message']}")
        if not status["online"]:
            print("WARNING: Ollama server unreachable. Switching to mock provider for test verification.")
            args.provider = "mock"
        else:
            models = get_available_ollama_models()
            print(f"Detected Ollama Models: {', '.join(models)}")

    print("\nInstantiating Chat Model via LLM Factory...")
    llm = build_llm(provider=args.provider, model_name=args.model, temperature=0.7)
    pipeline = ContentPipeline(llm=llm, model_name=args.model)

    brief = ContentBrief(
        topic=args.topic,
        target_audience="Software Engineers & Tech Leaders",
        tone="Insightful & Practical",
        length="Short (~500 words)",
        content_format=args.format,
        keywords=["LangChain", "LCEL", "quality control", "orchestration"],
        additional_guidelines="Emphasize pipeline reliability and intermediate verification.",
    )

    def progress_tracker(step_text: str, pct: int):
        print(f"[{pct:3d}%] {step_text}")

    print("\nExecuting Pipeline with Automated Outline Approval...")
    start_time = os.times()
    artifacts = pipeline.run_full_pipeline(
        brief=brief,
        auto_approve_outline=True,
        force_refinement=True,
        progress_callback=progress_tracker,
    )

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Model Used:         {artifacts.model_used}")
    print(f"Research Length:    {len(artifacts.research)} chars")
    print(f"Approved Outline:   {len(artifacts.approved_outline)} chars")
    print(f"Raw Draft:          {len(artifacts.raw_draft)} chars")
    if artifacts.quality_report:
        scores = artifacts.quality_report.scores
        print(f"Quality Score:      {scores.overall_score}/100")
        print(f" - Grammar:         {scores.grammar_and_syntax}/20")
        print(f" - Clarity:         {scores.clarity_and_coherence}/20")
        print(f" - Tone:            {scores.audience_and_tone}/20")
        print(f" - Outline:         {scores.outline_adherence}/20")
        print(f" - Depth:           {scores.factuality_and_depth}/20")
        print(f"Refinement Needed:  {artifacts.quality_report.needs_refinement}")
    print(f"Final Output:       {len(artifacts.final_output)} chars")
    print("-" * 70)

    # Save artifact to disk
    os.makedirs("output", exist_ok=True)
    out_file = os.path.join("output", "test_run_output.md")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(artifacts.final_output)
    print(f"\nFinal publish-ready content saved to: {out_file}")
    print("Test run completed successfully!")


if __name__ == "__main__":
    main()
