"""OCR module for extracting text from screenshots using EasyOCR."""

import easyocr
from PIL import Image
from typing import List, Union
import io
import numpy as np

# Initialize EasyOCR reader (cached globally)
_reader = None


def get_easyocr_reader():
    """Get or create EasyOCR reader instance (cached for performance)."""
    global _reader
    if _reader is None:
        # Initialize EasyOCR reader for English
        # Try to use GPU if available for faster processing
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except:
            use_gpu = False
        # First initialization is slow (downloads models), then cached
        _reader = easyocr.Reader(['en'], gpu=use_gpu)
    return _reader


def extract_text_from_screenshot(image_file: Union[bytes, Image.Image]) -> str:
    """
    Extract text from uploaded screenshot using EasyOCR.
    
    Args:
        image_file: Image file bytes or PIL Image object
        
    Returns:
        Extracted text string
    """
    # Read image if it's bytes
    if isinstance(image_file, bytes):
        image = Image.open(io.BytesIO(image_file))
    elif isinstance(image_file, Image.Image):
        image = image_file
    else:
        raise ValueError("Invalid image file type")
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image if too large to speed up OCR (max width 1920px)
    # This significantly speeds up OCR without much quality loss
    max_width = 1920
    if image.width > max_width:
        ratio = max_width / image.width
        new_height = int(image.height * ratio)
        image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
    
    # Convert PIL Image to numpy array for EasyOCR
    img_array = np.array(image)
    
    # Get EasyOCR reader
    reader = get_easyocr_reader()
    
    try:
        # Perform OCR - EasyOCR returns list of (bbox, text, confidence)
        results = reader.readtext(img_array)
        
        # Extract text from results and combine
        text_lines = []
        for (bbox, text, confidence) in results:
            # Filter out low confidence results (optional, adjust threshold as needed)
            if confidence > 0.3:  # Only include results with >30% confidence
                text_lines.append(text.strip())
        
        # Combine all text lines
        extracted_text = '\n'.join(text_lines)
        
        return extracted_text.strip()
        
    except Exception as e:
        raise ValueError(f"EasyOCR extraction failed: {str(e)}")


def extract_text_from_multiple_screenshots(image_files: List[Union[bytes, Image.Image]]) -> str:
    """Combine text from multiple screenshot uploads."""
    all_text = []
    for image_file in image_files:
        try:
            text = extract_text_from_screenshot(image_file)
            if text.strip():
                all_text.append(text.strip())
        except Exception as e:
            # Log error but continue with other images
            print(f"Error processing image: {e}")
            continue
    
    # Combine with separator
    combined_text = "\n---\n".join(all_text)
    return combined_text