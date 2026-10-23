"""
pipeline/config.py
Data structures and configuration models for the Content Generation Pipeline.
"""

from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field


class ContentBrief(BaseModel):
    """Input specifications for content generation."""
    topic: str = Field(..., description="Main topic or headline of the content")
    target_audience: str = Field(
        default="Tech Professionals & Developers",
        description="Target audience demographic and expertise level"
    )
    tone: str = Field(
        default="Professional & Informative",
        description="Voice and tone of writing"
    )
    length: str = Field(
        default="Medium (~800 words)",
        description="Target output length"
    )
    content_format: Literal["Blog Post", "Social Media Thread", "Newsletter", "Executive Report"] = Field(
        default="Blog Post",
        description="Desired publication format"
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Key terms or phrases to naturally incorporate"
    )
    additional_guidelines: Optional[str] = Field(
        default="",
        description="Special instructions, citations, or formatting rules"
    )


class QualityScores(BaseModel):
    """Rubric evaluation scores across key editorial dimensions (0-20 each, total 0-100)."""
    grammar_and_syntax: int = Field(..., ge=0, le=20, description="Grammar, spelling, and sentence flow")
    clarity_and_coherence: int = Field(..., ge=0, le=20, description="Logical structure and smooth transitions")
    audience_and_tone: int = Field(..., ge=0, le=20, description="Alignment with tone and target audience")
    outline_adherence: int = Field(..., ge=0, le=20, description="Coverage of approved outline sections")
    factuality_and_depth: int = Field(..., ge=0, le=20, description="Substantive depth, logic, and factual rigor")
    overall_score: int = Field(..., ge=0, le=100, description="Calculated total score out of 100")


class QualityReport(BaseModel):
    """Comprehensive editorial quality report with self-evaluation and refinement instructions."""
    scores: QualityScores
    strengths: List[str] = Field(default_factory=list, description="Highlighted strengths of the draft")
    areas_for_improvement: List[str] = Field(default_factory=list, description="Weaknesses or ambiguities detected")
    editorial_critique: str = Field(default="", description="Qualitative feedback and synthesis")
    needs_refinement: bool = Field(default=False, description="Flag indicating if draft falls below threshold")


class PipelineArtifacts(BaseModel):
    """Stores intermediate and final outputs of each pipeline stage."""
    brief: ContentBrief
    research: str = ""
    raw_outline: str = ""
    approved_outline: str = ""
    raw_draft: str = ""
    quality_report: Optional[QualityReport] = None
    refined_draft: str = ""
    final_output: str = ""
    model_used: str = ""
    timestamp: str = ""
