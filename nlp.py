from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer

def extract_topics(transcription):
    """
    Extract topics from the transcription using a text vectorizer.
    """
    if transcription is None:
        raise ValueError("Transcription is None. Cannot extract topics.")

    vectorizer = CountVectorizer()
    doc_term_matrix = vectorizer.fit_transform([transcription])

    # Your topic extraction logic here
    return doc_term_matrix

# Function to extract topics from transcription
def extract_topics(transcription, n_topics=5):
    vectorizer = CountVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform([transcription])

    lda = LatentDirichletAllocation(n_components=n_topics)
    lda.fit(doc_term_matrix)

    words = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        topic_words = [words[i] for i in topic.argsort()[:-n_topics - 1:-1]]
        topics.append(f"Topic {topic_idx}: {', '.join(topic_words)}")

    return topics

# Function for sentiment analysis
def analyze_sentiment(transcription):
    sentiment_analyzer = pipeline("sentiment-analysis")
    sentiments = sentiment_analyzer(transcription)
    return sentiments
