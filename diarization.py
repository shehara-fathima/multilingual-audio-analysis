# diarization.py (updated)
import os
from pyannote.audio import Pipeline
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Load pipelines
vad_pipeline, scd_pipeline = None, None
try:
    vad_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=hf_token)
    scd_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    print("✓ VAD and SCD pipelines loaded successfully")
except Exception as e:
    print(f"Error loading VAD/SCD pipelines: {e}")

def diarize_audio(file_path):
    """Run VAD + SCD diarization with unique speaker IDs"""
    if vad_pipeline is None or scd_pipeline is None:
        print("Pipelines not available")
        return []

    try:
        # VAD: speech segments
        vad_result = vad_pipeline(file_path)
        speech_segments = [(seg.start, seg.end) for seg in vad_result.get_timeline()]

        # SCD: speaker-labeled segments
        scd_result = scd_pipeline(file_path)
        segments = []

        for turn, _, speaker in scd_result.itertracks(yield_label=True):
            seg_start, seg_end = turn.start, turn.end
            duration = seg_end - seg_start
            if duration >= 0.8:  # filter short segments
                segments.append({
                    "speaker": speaker,  # unique speaker ID from SCD
                    "start": seg_start,
                    "end": seg_end
                })

        # Merge consecutive segments of same speaker
        if segments:
            merged_segments = [segments[0]]
            for seg in segments[1:]:
                last = merged_segments[-1]
                if seg["speaker"] == last["speaker"] and seg["start"] - last["end"] < 0.5:
                    # Extend last segment
                    last["end"] = seg["end"]
                else:
                    merged_segments.append(seg)
            segments = merged_segments

        return segments

    except Exception as e:
        print(f"VAD+SCD diarization error: {e}")
        return []

