from flask import Flask, request, render_template, jsonify
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from gensim import corpora
from gensim.models import LdaModel
import librosa
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-960h")
sentiment_pipeline = pipeline("sentiment-analysis")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Process file
    transcription = transcribe_audio(file_path)
    topics = extract_topics(transcription)
    sentiment_summary = analyze_sentiment(transcription)
    insights = generate_insights(topics, sentiment_summary)
    summary = summarize_transcription(transcription)

    return render_template('results.html', transcription=transcription, topics=topics, sentiment=sentiment_summary, insights=insights, summary=summary)

def transcribe_audio(file_path):
    with open(file_path, 'rb') as audio_file:
        transcription = asr(audio_file)
    return transcription['text']

def extract_topics(transcription):
    words = transcription.lower().split()
    dictionary = corpora.Dictionary([words])
    corpus = [dictionary.doc2bow(words)]

    lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
    topics = lda_model.print_topics(num_words=3)

    return topics

def analyze_sentiment(transcription):
    sentiments = sentiment_pipeline(transcription.split('.'))
    sentiment_summary = {"positive": 0, "negative": 0, "neutral": 0}

    for result in sentiments:
        label = result['label'].lower()
        sentiment_summary[label] += 1

    return sentiment_summary

def generate_insights(topics, sentiment_summary):
    insights = {
        "key_topics": topics,
        "sentiment_summary": sentiment_summary,
        "suggestions": "Focus on improving areas with negative sentiment."
    }
    return insights

def summarize_transcription(transcription):
    inputs = gpt2_tokenizer.encode(transcription, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = gpt2_model.generate(inputs, max_length=100, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = gpt2_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

if __name__ == '__main__':
    app.run(debug=True)
