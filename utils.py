"""Utility functions for message parsing and text processing."""

import re
from typing import List, Dict


def is_timestamp(text: str) -> bool:
    """Detect if a line looks like a timestamp."""
    timestamp_patterns = [
        r'\d{1,2}:\d{2}\s*(AM|PM|am|pm)',
        r'\d{1,2}:\d{2}',
        r'(yesterday|today|just now|ago)',
        r'\d{1,2}/\d{1,2}/\d{2,4}',
    ]
    text_lower = text.lower().strip()
    return any(re.search(pattern, text_lower) for pattern in timestamp_patterns)


def extract_suggestions(text: str) -> List[str]:
    """Extract all text enclosed in curly brackets."""
    pattern = r'\{([^}]+)\}'
    matches = re.findall(pattern, text)
    suggestions = [match.strip() for match in matches if match.strip()]
    
    # If no curly brackets found, try alternative formats (for debugging)
    if not suggestions:
        # Try numbered list: "1. suggestion" or "1) suggestion"
        numbered = re.findall(r'\d+[\.\)]\s*([^\n]+)', text)
        if numbered:
            suggestions = [s.strip() for s in numbered[:9]]
            print(f"[DEBUG] Fallback: Found {len(suggestions)} numbered suggestions")
        else:
            # Try bullet points: "- suggestion" or "* suggestion"
            bullets = re.findall(r'[-*]\s*([^\n]+)', text)
            if bullets:
                suggestions = [s.strip() for s in bullets[:9]]
                print(f"[DEBUG] Fallback: Found {len(suggestions)} bullet suggestions")
    
    return suggestions


def remove_suggestions_from_text(text: str) -> str:
    """Remove curly-bracketed suggestions from text, return clean coaching."""
    pattern = r'\{[^}]+\}'
    cleaned = re.sub(pattern, '', text)
    return cleaned.strip()