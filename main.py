# main.py
import os
import warnings
import logging
from pydub import AudioSegment

# Suppress warnings and logs
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)

try:
    import torch
    torch.cuda.empty_cache()
except ImportError:
    pass

try:
    import transformers
    transformers.logging.set_verbosity_error()
except ImportError:
    pass

from analyzer import analyze_conversation, print_conversation_analysis


def convert_to_wav(input_file, output_file=None, target_sr=16000):
    """
    Convert any audio file to WAV format with single channel and target sample rate.
    """
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".wav"

    try:
        audio = AudioSegment.from_file(input_file)
        audio = audio.set_frame_rate(target_sr).set_channels(1)
        audio.export(output_file, format="wav")
        return output_file
    except Exception as e:
        print(f"Audio conversion error: {e}")
        return input_file  # Return original if conversion fails


def normalize_audio(input_wav, output_wav=None, target_dBFS=-20):
    """
    Normalize audio to a target loudness (dBFS).
    This helps low voices to be transcribed correctly.
    """
    if output_wav is None:
        base, ext = os.path.splitext(input_wav)
        output_wav = base + "_norm.wav"

    try:
        audio = AudioSegment.from_wav(input_wav)
        change_dBFS = target_dBFS - audio.dBFS
        normalized = audio.apply_gain(change_dBFS)
        normalized.export(output_wav, format="wav")
        return output_wav
    except Exception as e:
        print(f"Audio normalization error: {e}")
        return input_wav


def run_analysis(input_path):
    """
    End-to-end runner:
      - Convert to WAV
      - Normalize
      - Analyze with analyzer.py
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Audio file '{input_path}' not found.")

    # Convert to WAV if needed
    if not input_path.lower().endswith('.wav'):
        print("Converting audio to WAV format...")
        input_path = convert_to_wav(input_path)

    # Normalize volume
    print("Normalizing audio volume...")
    input_path = normalize_audio(input_path)

    # Run complete analysis
    print("Starting complete conversation analysis...")
    print("This may take several minutes depending on audio length...")
    analysis = analyze_conversation(input_path)

    return analysis


if __name__ == "__main__":
    # Input audio file (change path as needed)
    audio_file = "/content/drive/MyDrive/soniox/audio.aac"

    try:
        analysis = run_analysis(audio_file)
        print_conversation_analysis(analysis)
        print("\nAnalysis complete!")
    except Exception as e:
        print(f"Error during analysis: {e}")
