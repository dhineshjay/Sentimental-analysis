# Conversational Insights Platform

## Overview
This project is a Conversational Insights Platform that processes audio and video files, transcribes the content, extracts topics, analyzes sentiment, and generates insights using NLP and deep learning models.

### Features:
- **File Upload**: Upload audio/video files (WAV, MP4, etc.).
- **Automatic Transcription**: Uses a pre-trained Automatic Speech Recognition (ASR) model.
- **Topic Extraction**: Uses Latent Dirichlet Allocation (LDA) to extract topics from the transcribed text.
- **Sentiment Analysis**: Analyzes sentiment using pre-trained NLP models.
- **Insight Generation**: Generates insights based on the topics and sentiment analysis.
- **Summary Generation**: Uses a pre-trained GPT-2 model to summarize transcriptions.

## Setup

### Prerequisites
- Python 3.7+
- Pip

### Installation
1. Clone the repository.
2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Run the application:
    ```
    python app.py
    ```
4. Access the app at `http://localhost:5000`.

## Future Enhancements
- Add support for more audio and video formats.
- Improve speaker identification and topic extraction.
- Enhance UI with better visualizations.
