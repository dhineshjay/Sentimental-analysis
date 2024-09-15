from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline

import re


def extract_topics(transcription):
    """
    Extract topics from a transcription using CountVectorizer.
    Ensures the transcription is meaningful before processing.
    """
    try:
        # Debugging: log the transcription
        print(f"Transcription received for topic extraction: {transcription}")

        # Check if transcription is empty or consists of gibberish
        if not transcription or not transcription.strip():
            raise ValueError("Transcription is empty or invalid.")

        # Optional: Clean up the transcription using regex (remove non-alphabetic characters)
        transcription_cleaned = re.sub(r'[^a-zA-Z\s]', '', transcription)

        # Further clean by removing repeated sequences (e.g., "HELLO HELLO" -> "HELLO")
        transcription_cleaned = re.sub(r'\b(\w+)\b(?:\s+\1\b)+', r'\1', transcription_cleaned)

        # Check if cleaned transcription still has meaningful content
        if len(transcription_cleaned.split()) < 3:  # Arbitrary threshold for minimal valid content
            raise ValueError("Transcription is too short or contains meaningless content.")

        # Initialize CountVectorizer
        vectorizer = CountVectorizer(stop_words='english')

        # Fit and transform the cleaned transcription into a document-term matrix
        doc_term_matrix = vectorizer.fit_transform([transcription_cleaned])

        # Check if the vocabulary is empty (i.e., no valid words remain after stop words removal)
        if doc_term_matrix.shape[1] == 0:
            raise ValueError("Transcription only contains stop words or is empty.")

        # Return the vocabulary as topics
        return vectorizer.get_feature_names_out()

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during topic extraction: {e}")
        return None


# Function for sentiment analysis
def analyze_sentiment(transcription):
    try:
        # Use a PyTorch-based model
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", framework="pt")
        sentiments = sentiment_analyzer(transcription)
        return sentiments
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return None
