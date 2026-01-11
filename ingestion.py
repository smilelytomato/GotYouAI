"""Message ingestion and normalization module."""

from typing import List, Dict, Optional, Union
from PIL import Image
from .ocr import extract_text_from_screenshot, extract_text_from_multiple_screenshots
from .utils import is_timestamp


def ingest_from_screenshots(image_files: Union[List, bytes, Image.Image], max_messages: int = 16) -> List[Dict]:
    """
    Main entry point: Process screenshots → OCR → parse messages.
    
    Args:
        image_files: Single image file, bytes, or list of image files
        max_messages: Maximum number of messages to retrieve
        
    Returns:
        List of normalized message dictionaries
    """
    # Extract text from screenshot(s)
    if isinstance(image_files, list):
        raw_text = extract_text_from_multiple_screenshots(image_files)
    else:
        raw_text = extract_text_from_screenshot(image_files)
    
    # Validate OCR output
    if not raw_text or len(raw_text.strip()) < 10:
        raise ValueError(
            "Could not extract text from screenshot. "
            "Please ensure image is clear and contains visible messages."
        )
    
    # Parse and normalize messages
    messages = ingest_conversation(raw_text, max_messages=max_messages)
    
    return messages


def ingest_conversation(raw_text: str, max_messages: int = 16) -> List[Dict]:
    """
    Parse and normalize conversation messages from OCR text.
    
    Args:
        raw_text: Raw text extracted from OCR
        max_messages: Maximum number of messages to process
        
    Returns:
        List of normalized message dictionaries
    """
    # Split by lines
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    
    # Detect message boundaries
    messages = detect_message_boundaries(lines)
    
    # Limit to last N messages
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
    
    # Normalize each message
    normalized = []
    client_count = 0
    target_count = 0
    
    for msg_text in recent_messages:
        content = clean_message_content(msg_text)
        if not content or len(content) < 1:
            continue
        
        # Determine sender using alternating pattern (simple heuristic for MVP)
        # In real implementation, this could use screenshot layout analysis
        if client_count <= target_count:
            label = "[client]"
            client_count += 1
        else:
            label = "[target]"
            target_count += 1
        
        normalized.append({
            "label": label,
            "content": content,
            "raw": msg_text
        })
    
    return normalized


def detect_message_boundaries(lines: List[str]) -> List[str]:
    """
    Parse OCR output to identify individual messages.
    Handles common Instagram screenshot patterns.
    """
    messages = []
    current_message = []
    
    for line in lines:
        # Skip timestamps
        if is_timestamp(line):
            # Save previous message if exists
            if current_message:
                messages.append(" ".join(current_message))
                current_message = []
            continue
        
        # Skip very short lines that look like metadata
        if len(line) < 3:
            continue
        
        # If line looks like start of new message (starts with common patterns)
        if current_message and looks_like_message_start(line):
            messages.append(" ".join(current_message))
            current_message = [line]
        else:
            current_message.append(line)
    
    # Don't forget last message
    if current_message:
        messages.append(" ".join(current_message))
    
    return messages


def looks_like_message_start(line: str) -> bool:
    """Heuristic to detect if a line looks like the start of a new message."""
    # Messages often start with capitalization or common starters
    if len(line) > 0 and line[0].isupper():
        return True
    # Or if it's a very short line after a longer one, might be a separator
    return False


def clean_message_content(message_text: str) -> str:
    """Clean and normalize message content."""
    # Remove extra whitespace
    cleaned = " ".join(message_text.split())
    # Remove common OCR artifacts
    cleaned = cleaned.replace("|", "I")  # Common OCR mistake
    return cleaned.strip()


def format_for_analysis(messages: List[Dict], proposition: Optional[str] = None) -> str:
    """
    Format messages into string format for LLM/analysis.
    
    Args:
        messages: List of normalized message dictionaries
        proposition: Optional draft message to include
        
    Returns:
        Formatted conversation string
    """
    formatted_lines = []
    for msg in messages:
        formatted_lines.append(f"{msg['label']} {msg['content']}")
    
    if proposition and proposition.strip():
        formatted_lines.append(f"[proposition] {proposition}")
    
    return "\n".join(formatted_lines)