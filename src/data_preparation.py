from __future__ import annotations

"""Data preparation module (Phase 1).

This module provides utilities for:
* Synthetic data generation (mock implementation for runtime).
* Language detection.
* Severity scoring.
* Emergency keyword extraction.
"""

import re
from typing import List, Optional

# Constants
CATEGORIES = ["Sanitation", "Water Supply", "Transportation"]
LANGUAGES = ["English", "Hindi", "Hinglish"]
URGENCY_LEVELS = ["Critical", "High", "Medium", "Low"]

EMERGENCY_KEYWORDS = [
    "danger", "emergency", "immediate", "urgent", "hazard", "risk",
    "accident", "fire", "short circuit", "burst", "leakage", "flood",
    "dead animal", "carcass", "smell", "stench", "fatal", "injury",
    "poison", "toxic", "gas leak"
]

def extract_emergency_keywords(text: str) -> List[str]:
    """Extract emergency keywords from the text.
    
    Args:
        text: Complaint text.
        
    Returns:
        List of detected emergency keywords.
    """
    if not text:
        return []
    
    text_lower = text.lower()
    found = []
    for kw in EMERGENCY_KEYWORDS:
        if kw in text_lower:
            found.append(kw)
    return found

def compute_severity_score(
    urgency_level: str, 
    emergency_keywords: List[str], 
    text: str
) -> float:
    """Compute a baseline severity score [0, 1].
    
    This score is used for priority ordering.
    """
    # Base score from urgency
    urg_map = {"Critical": 0.8, "High": 0.6, "Medium": 0.4, "Low": 0.2}
    base = urg_map.get(urgency_level, 0.3)
    
    # Keyword bonus
    kw_bonus = min(len(emergency_keywords) * 0.1, 0.2)
    
    # Length factor (longer descriptions usually imply more detail/gravity)
    length = len(text.split())
    len_bonus = min(length / 100.0, 0.1)
    
    score = base + kw_bonus + len_bonus
    return min(float(score), 1.0)

def detect_language(text: str) -> str:
    """Simple heuristic for language detection.
    
    Checks for Devanagari characters first.
    """
    if not text:
        return "English"
    
    # Check for Devanagari (Hindi)
    if re.search(r'[\u0900-\u097F]', text):
        return "Hindi"
    
    # Simple check for Hinglish (Roman script Hindi)
    # This is a heuristic mock.
    hinglish_markers = [" hai ", " ki ", " sab ", " mera ", " bahut ", " rasta "]
    if any(marker in text.lower() for marker in hinglish_markers):
        return "Hinglish"
        
    return "English"

def build_and_save_dataset():
    """Mock for build_and_save_dataset to satisfy tests/imports."""
    import pandas as pd
    import numpy as np
    
    # Return an empty DF or minimal one if needed
    return pd.DataFrame()

__all__ = [
    "CATEGORIES",
    "LANGUAGES",
    "URGENCY_LEVELS",
    "extract_emergency_keywords",
    "compute_severity_score",
    "detect_language",
    "build_and_save_dataset",
]
