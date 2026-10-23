"""
pipeline/chains.py
LangChain LCEL orchestration engine for the multi-stage content generation pipeline.
Includes stage-by-stage execution, human-in-the-loop gates, and quality self-refinement loops.
"""

import time
from datetime import datetime
from typing import Optional, Callable, Dict, Any, Tuple
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from pipeline.config import ContentBrief, PipelineArtifacts, QualityReport
from pipeline.prompts import (
    RESEARCH_PROMPT,
    OUTLINE_PROMPT,
    DRAFT_PROMPT,
    EVALUATOR_PROMPT,
    REFINEMENT_PROMPT,
    get_formatter_prompt,
)
from pipeline.evaluator import build_quality_report


class ContentPipeline:
    """Orchestrates chained LLM operations with quality control across 5 distinct stages."""

    def __init__(self, llm: BaseChatModel, model_name: str = "qwen2.5:3b"):
        self.llm = llm
        self.model_name = model_name
        self.output_parser = StrOutputParser()

        # Build individual LCEL chains
        self.research_chain = RESEARCH_PROMPT | self.llm | self.output_parser
        self.outline_chain = OUTLINE_PROMPT | self.llm | self.output_parser
        self.draft_chain = DRAFT_PROMPT | self.llm | self.output_parser
        self.eval_chain = EVALUATOR_PROMPT | self.llm | self.output_parser
        self.refinement_chain = REFINEMENT_PROMPT | self.llm | self.output_parser

    # --------------------------------------------------------------------------
    # STAGE 1: Research & Synthesis
    # --------------------------------------------------------------------------
    def run_research(self, brief: ContentBrief) -> str:
        """Execute Stage 1: Chain-of-Thought inquiry and factual synthesis."""
        inputs = {
            "topic": brief.topic,
            "target_audience": brief.target_audience,
            "tone": brief.tone,
            "content_format": brief.content_format,
            "length": brief.length,
            "keywords": ", ".join(brief.keywords) if brief.keywords else "None specified",
            "additional_guidelines": brief.additional_guidelines or "Follow standard professional best practices.",
        }
        return self.research_chain.invoke(inputs)

    # --------------------------------------------------------------------------
    # STAGE 2: Outline Blueprint Generation
    # --------------------------------------------------------------------------
    def run_outline(self, brief: ContentBrief, research: str) -> str:
        """Execute Stage 2: Structural Outline Blueprint."""
        inputs = {
            "topic": brief.topic,
            "target_audience": brief.target_audience,
            "tone": brief.tone,
            "content_format": brief.content_format,
            "length": brief.length,
            "keywords": ", ".join(brief.keywords) if brief.keywords else "None specified",
            "additional_guidelines": brief.additional_guidelines or "Follow standard professional best practices.",
            "research": research,
        }
        return self.outline_chain.invoke(inputs)

    # --------------------------------------------------------------------------
    # STAGE 3: Draft Generation (Strict Outline Adherence)
    # --------------------------------------------------------------------------
    def run_draft(self, brief: ContentBrief, research: str, approved_outline: str) -> str:
        """Execute Stage 3: Full first draft adhering to approved outline."""
        inputs = {
            "topic": brief.topic,
            "target_audience": brief.target_audience,
            "tone": brief.tone,
            "content_format": brief.content_format,
            "length": brief.length,
            "keywords": ", ".join(brief.keywords) if brief.keywords else "None specified",
            "additional_guidelines": brief.additional_guidelines or "Follow standard professional best practices.",
            "approved_outline": approved_outline,
            "research": research,
        }
        return self.draft_chain.invoke(inputs)

    # --------------------------------------------------------------------------
    # STAGE 4: Quality Control & Editorial Self-Evaluation
    # --------------------------------------------------------------------------
    def run_eval(self, brief: ContentBrief, draft: str, approved_outline: str) -> Tuple[QualityReport, str]:
        """Execute Stage 4: Senior Editorial Critique & Rubric Scoring."""
        inputs = {
            "topic": brief.topic,
            "target_audience": brief.target_audience,
            "tone": brief.tone,
            "content_format": brief.content_format,
            "length": brief.length,
            "approved_outline": approved_outline,
            "draft": draft,
        }
        raw_eval_text = self.eval_chain.invoke(inputs)
        report = build_quality_report(raw_eval_text, threshold=85)
        return report, raw_eval_text

    # --------------------------------------------------------------------------
    # STAGE 4B: Self-Refinement Loop
    # --------------------------------------------------------------------------
    def run_refinement(self, brief: ContentBrief, draft: str, report: QualityReport) -> str:
        """Execute Stage 4B: Apply editorial critique in an iterative refinement pass."""
        improvements_text = "\n".join(f"- {item}" for item in report.areas_for_improvement)
        inputs = {
            "topic": brief.topic,
            "target_audience": brief.target_audience,
            "tone": brief.tone,
            "content_format": brief.content_format,
            "overall_score": report.scores.overall_score,
            "areas_for_improvement": improvements_text,
            "editorial_critique": report.editorial_critique,
            "draft": draft,
        }
        return self.refinement_chain.invoke(inputs)

    # --------------------------------------------------------------------------
    # STAGE 5: Multi-Format Publish-Ready Adaptation
    # --------------------------------------------------------------------------
    def run_formatting(self, brief: ContentBrief, polished_content: str) -> str:
        """Execute Stage 5: Package content for publication (Blog, Social, Newsletter, Report)."""
        formatter_prompt = get_formatter_prompt(brief.content_format)
        formatter_chain = formatter_prompt | self.llm | self.output_parser
        inputs = {
            "topic": brief.topic,
            "target_audience": brief.target_audience,
            "tone": brief.tone,
            "content_format": brief.content_format,
            "polished_content": polished_content,
        }
        return formatter_chain.invoke(inputs)

    # --------------------------------------------------------------------------
    # END-TO-END AUTOMATED PIPELINE EXECUTION
    # --------------------------------------------------------------------------
    def run_full_pipeline(
        self,
        brief: ContentBrief,
        user_approved_outline: Optional[str] = None,
        auto_approve_outline: bool = False,
        force_refinement: bool = True,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> PipelineArtifacts:
        """
        Execute the end-to-end chained pipeline with intermediate quality gates.
        """
        artifacts = PipelineArtifacts(
            brief=brief,
            model_used=self.model_name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # 1. Research
        if progress_callback:
            progress_callback("Stage 1/5: Synthesizing domain research...", 15)
        artifacts.research = self.run_research(brief)

        # 2. Outline
        if progress_callback:
            progress_callback("Stage 2/5: Building structural outline...", 35)
        artifacts.raw_outline = self.run_outline(brief, artifacts.research)

        # Human-in-the-loop outline resolution
        if user_approved_outline:
            artifacts.approved_outline = user_approved_outline
        elif auto_approve_outline:
            artifacts.approved_outline = artifacts.raw_outline
        else:
            artifacts.approved_outline = artifacts.raw_outline

        # 3. Draft
        if progress_callback:
            progress_callback("Stage 3/5: Writing comprehensive first draft...", 55)
        artifacts.raw_draft = self.run_draft(brief, artifacts.research, artifacts.approved_outline)

        # 4. Quality Evaluation
        if progress_callback:
            progress_callback("Stage 4/5: Running editorial quality control & rubric scoring...", 75)
        report, _ = self.run_eval(brief, artifacts.raw_draft, artifacts.approved_outline)
        artifacts.quality_report = report

        # Refinement loop (if below threshold or forced for maximum polish)
        should_refine = report.needs_refinement or force_refinement
        if should_refine:
            if progress_callback:
                progress_callback("Stage 4B: Refining draft through self-correction loop...", 85)
            artifacts.refined_draft = self.run_refinement(brief, artifacts.raw_draft, report)
        else:
            artifacts.refined_draft = artifacts.raw_draft

        # 5. Final Formatting
        if progress_callback:
            progress_callback(f"Stage 5/5: Formatting final {brief.content_format} artifact...", 95)
        artifacts.final_output = self.run_formatting(brief, artifacts.refined_draft)

        if progress_callback:
            progress_callback("Content generation pipeline complete!", 100)

        return artifacts
