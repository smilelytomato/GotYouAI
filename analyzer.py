"""Conversation state analysis module using sentiment analysis only."""

from typing import List, Dict
from transformers import pipeline
import os


# Initialize sentiment analysis pipeline (cached)
_sentiment_pipeline = None


def get_sentiment_pipeline():
    """Get or create sentiment analysis pipeline using cardiffnlp/twitter-roberta-base-sentiment-latest."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        try:
            _sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1  # CPU
            )
        except Exception as e:
            print(f"Error: Could not load sentiment model: {e}")
            raise ValueError(
                f"Failed to load sentiment analysis model. Please ensure the model is available. Error: {e}"
            )
    return _sentiment_pipeline


def analyze_conversation_state(messages: List[Dict]) -> str:
    """
    Analyze conversation and return state: "cooking", "wait", or "cooked".
    Uses only sentiment analysis from the model.
    
    Args:
        messages: List of normalized message dictionaries
        
    Returns:
        State string: "cooking", "wait", or "cooked"
    """
    if not messages:
        return "wait"
    
    # Sentiment analysis using the model
    sentiment_pipeline = get_sentiment_pipeline()
    sentiment_scores = []
    
    for msg in messages:
        try:
            content = msg['content'][:512]  # Limit length for model
            if not content.strip():
                continue
                
            result = sentiment_pipeline(content)[0]
            label = result['label'].lower()
            score = result['score']
            
            # Map labels to scores
            # The model returns: LABEL_0 (negative), LABEL_1 (neutral), LABEL_2 (positive)
            if 'label_2' in label or 'positive' in label:
                sentiment_scores.append(score)  # Positive score
            elif 'label_0' in label or 'negative' in label:
                sentiment_scores.append(-score)  # Negative score
            else:  # label_1 or neutral
                sentiment_scores.append(0)  # Neutral
                
        except Exception as e:
            print(f"Sentiment analysis error for message: {e}")
            # Skip if analysis fails
            sentiment_scores.append(0)
    
    if not sentiment_scores:
        return "wait"
    
    # Calculate average sentiment
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    
    # Classification based purely on sentiment
    # "cooking" = positive sentiment
    if avg_sentiment > 0.2:
        return "cooking"
    
    # "cooked" = negative sentiment
    elif avg_sentiment < -0.2:
        return "cooked"
    
    # "wait" = neutral sentiment (default)
    else:
        return "wait"