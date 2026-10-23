"""
samples package initialization and loader utility.
"""

import os
import json
from typing import Dict, Any, List

SAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))


def load_sample(format_key: str) -> Dict[str, Any]:
    """Load pre-generated sample JSON by format key."""
    mapping = {
        "blog": "blog_sample.json",
        "thread": "thread_sample.json",
        "newsletter": "newsletter_sample.json",
        "report": "report_sample.json",
    }
    filename = mapping.get(format_key.lower(), "blog_sample.json")
    path = os.path.join(SAMPLES_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_all_samples() -> Dict[str, Dict[str, Any]]:
    """Return all pre-generated samples keyed by format name."""
    return {
        "Blog Post": load_sample("blog"),
        "Social Media Thread": load_sample("thread"),
        "Newsletter": load_sample("newsletter"),
        "Executive Report": load_sample("report"),
    }
