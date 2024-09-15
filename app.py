
from flask import Flask, render_template, request, redirect, url_for
from asr import transcribe_audio
from nlp import extract_topics, analyze_sentiment
from insights import generate_insights

app = Flask(__name__)

# Route for the dashboard
@app.route('/')
def index():
    return render_template('index.html')

# Route for uploading and processing files
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    # Save the file to local disk
    file_path = './uploads/' + file.filename
    file.save(file_path)

    # Step 1: Transcribe the file
    transcription = transcribe_audio(file_path)

    # Step 2: Extract topics
    topics = extract_topics(transcription)

    # Step 3: Analyze sentiment
    sentiments = analyze_sentiment(transcription)

    # Step 4: Generate insights
    insights = generate_insights(transcription, topics, sentiments)

    # Render results to the dashboard
    return render_template('results.html', transcription=transcription, topics=topics, sentiments=sentiments, insights=insights)

if __name__ == '__main__':
    app.run(debug=True)
