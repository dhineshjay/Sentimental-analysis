from flask import Flask, render_template, request, redirect, url_for
from asr import transcribe_audio
from nlp import extract_topics, analyze_sentiment
from insights import generate_insights
import os

app = Flask(__name__)

# Ensure the uploads directory exists
os.makedirs('./uploads', exist_ok=True)

# Route for the dashboard
@app.route('/')
def index():
    return render_template('index.html')

# Route for uploading and processing files
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(url_for('index'))

    file = request.files['file']

    # Save the file to local disk
    file_path = os.path.join('./uploads/', file.filename)
    file.save(file_path)

    try:
        # Step 1: Transcribe the file
        transcription = transcribe_audio(file_path)

        if not transcription:
            raise ValueError("Transcription failed or is empty.")

        # Step 2: Extract topics
        topics = extract_topics(transcription)

        if topics is None:
            raise ValueError("Topic extraction failed.")

        # Step 3: Analyze sentiment
        sentiments = analyze_sentiment(transcription)

        # Step 4: Generate insights
        insights = generate_insights(transcription, topics, sentiments)

        # Render results to the dashboard
        return render_template('results.html', transcription=transcription, topics=topics, sentiments=sentiments, insights=insights)

    except Exception as e:
        print(f"Error processing file: {e}")
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
