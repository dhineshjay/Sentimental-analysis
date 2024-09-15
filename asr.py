import os
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import librosa
import soundfile as sf
import torchaudio
from pydub import AudioSegment

# Load the tokenizer and model
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Ensure the model is in evaluation mode
model.eval()

  # Ensure soundfile is installed

def load_audio(file_path, target_sr=16000):
    """
    Load an audio file and resample it to the target sample rate if necessary.
    Attempts to use soundfile first, and falls back to audioread.
    """
    try:
        # Attempt to use soundfile first (better support for non-wav formats)
        audio, sr = sf.read(file_path, dtype='float32')

        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        return audio

    except RuntimeError as e:
        print(f"Soundfile failed: {e}, trying with librosa audioread fallback.")

        try:
            # Fallback to librosa's audioread-based loader
            audio, sr = librosa.load(file_path, sr=target_sr)  # Use target sample rate directly
            return audio

        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None

def process_chunk(audio_chunk):
    """
    Process a chunk of audio through the ASR model.
    """
    if len(audio_chunk) == 0:
        return ""

    try:
        # Tokenize the audio
        input_values = tokenizer(audio_chunk, return_tensors="pt", padding="longest").input_values

        # Move input values to device (CPU or GPU)
        input_values = input_values.to(device)

        # Get model predictions
        with torch.no_grad():
            logits = model(input_values).logits

        # Decode the logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)

        return transcription[0]
    except Exception as e:
        print(f"Error processing audio chunk: {e}")
        return ""

def transcribe_audio(file_path, chunk_duration=30):
    """
    Transcribe an audio file to text using Wav2Vec2 model.
    Splits the audio into chunks to avoid memory issues.
    """
    try:
        # Load and preprocess the audio file
        audio = load_audio(file_path)
        if audio is None:
            raise ValueError("Audio file could not be loaded.")

        # Define the chunk size in samples (16000 samples per second)
        chunk_size = chunk_duration * 16000

        # Process the audio in chunks
        transcriptions = []
        for start in range(0, len(audio), chunk_size):
            end = min(start + chunk_size, len(audio))
            audio_chunk = audio[start:end]
            transcription = process_chunk(audio_chunk)
            transcriptions.append(transcription)

        return " ".join(transcriptions).strip()
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
