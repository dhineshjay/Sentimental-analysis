import os
import numpy as np
import torch
import soundfile as sf
from moviepy.editor import VideoFileClip
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC

# Initialize the tokenizer and model
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


def extract_audio_from_video(video_path, audio_path):
    """
    Extracts audio from a video file and saves it as a WAV file.
    """
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path, codec='pcm_s16le')  # Save audio as WAV

def load_audio(file_path):
    """
    Loads an audio file or extracts audio from a video file and returns the audio data and sample rate.
    """
    # Check if file is a video
    if file_path.lower().endswith('.mp4'):
        temp_audio_path = 'temp_audio.wav'
        extract_audio_from_video(file_path, temp_audio_path)
        file_path = temp_audio_path

    try:
        # Load audio file
        audio, sr = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono if stereo
    except Exception as e:
        raise RuntimeError(f"Error loading audio file: {e}")

    # Clean up temporary file if it was created
    if file_path.lower().endswith('temp_audio.wav'):
        os.remove(file_path)

    return audio, sr

def transcribe_audio(file_path):
    """
    Transcribes audio from a file using the Wav2Vec2 model.
    """
    audio, _ = load_audio(file_path)

    # Tokenize and transcribe
    input_values = tokenizer(audio, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode transcription
    transcription = tokenizer.decode(predicted_ids[0])
    return transcription
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import soundfile as sf
import librosa
import numpy as np

# Load the tokenizer and model
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Ensure the model is in evaluation mode
model.eval()

def load_audio(file_path, target_sr=16000):
    """
    Load an audio file and resample to the target sample rate.
    """
    # Use librosa to load and resample audio
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio

def process_chunk(audio_chunk):
    """
    Process a chunk of audio through the ASR model.
    """
    # Tokenize the audio
    input_values = tokenizer(audio_chunk, return_tensors="pt").input_values

    # Use CUDA if available
    if torch.cuda.is_available():
        model = model.cuda()
        input_values = input_values.cuda()

    # Get model predictions
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)

    return transcription[0]

def transcribe_audio(file_path, chunk_duration=30):
    """
    Transcribe an audio file to text using Wav2Vec2 model.
    Splits the audio into chunks to avoid memory issues.
    """
    try:
        # Load and preprocess the audio file
        audio = load_audio(file_path)

        # Define the chunk size in samples
        chunk_size = chunk_duration * 16000  # 16000 samples per second

        # Process the audio in chunks
        transcriptions = []
        for start in range(0, len(audio), chunk_size):
            end = min(start + chunk_size, len(audio))
            audio_chunk = audio[start:end]
            transcription = process_chunk(audio_chunk)
            transcriptions.append(transcription)

        return " ".join(transcriptions)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
