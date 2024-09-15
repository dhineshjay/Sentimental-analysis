def generate_insights(transcription, topics, sentiments):
    # Simple insight generation based on sentiment and topics
    insights = {
        "transcription_summary": transcription[:200] + "...",  # Summarize transcription
        "positive_statements": sum(1 for s in sentiments if s['label'] == 'POSITIVE'),
        "negative_statements": sum(1 for s in sentiments if s['label'] == 'NEGATIVE'),
        "key_topics": topics,
        "suggestions": "Consider improving areas related to: " + ", ".join(topics[:2]) if len(topics) > 2 else "No significant suggestions."
    }
    return insights
