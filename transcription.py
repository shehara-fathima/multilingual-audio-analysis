 # #transcription.py
import os
import tempfile
import time
from dotenv import load_dotenv
from soniox import SpeechClient
from soniox.transcribe_async import transcribe_file_async

# Load environment variables
load_dotenv()

# Initialize Soniox client
client = SpeechClient()

def transcribe_audio_segment(audio_file_path, start_time, end_time):
    """
    Extract and transcribe a specific audio segment using Soniox.

    Parameters:
        audio_file_path (str): Path to the input audio file
        start_time (float): Segment start in seconds
        end_time (float): Segment end in seconds

    Returns:
        str: Transcribed text (Malayalam or other language)
    """
    try:
        # Load and extract the specific segment
        audio = AudioSegment.from_file(audio_file_path)
        start_ms, end_ms = int(start_time * 1000), int(end_time * 1000)
        segment_audio = audio[start_ms:end_ms]

        # Save segment to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            segment_audio.export(tmp.name, format="wav")
            temp_file_path = tmp.name

        # Transcribe using Soniox
        with open(temp_file_path, "rb") as audio_file:
            result = transcribe_file_async(
                audio_file,
                client,
                model="mal_v2",  # Specify the model for Malayalam
                language_code="ml-IN",  # Malayalam language code
            )

        # Wait for transcription to complete
        while result.status != "completed":
            time.sleep(1)
            result = client.get_transcription_result(result.id)

        # Clean up temporary file
        os.unlink(temp_file_path)

        return result.text.strip() if result.text else "[No speech detected]"

    except Exception as e:
        print(f"Transcription error for segment {start_time}-{end_time}: {e}")
        return "[Transcription failed]"
