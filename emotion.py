# emotion.py
from transformers import pipeline

# Load Hugging Face audio emotion detection model once
classifier = pipeline(
    task="audio-classification",
    model="Hatman/audio-emotion-detection"
)

def analyze_emotion_from_audio(audio_file_path: str, speaker_id: str) -> str:
    """
    Analyze emotion directly from audio using Hugging Face's
    Hatman/audio-emotion-detection model.
    Returns one of: Angry, Happy, Sad, Fearful, Surprised, Neutral, Disgusted.
    """
    try:
        # Run classifier
        results = classifier(audio_file_path)

        # Pick the highest-scoring label
        top_result = max(results, key=lambda x: x["score"])
        emotion = top_result["label"]

        print(f"Speaker {speaker_id}: {emotion} ({top_result['score']:.2f})")
        return emotion

    except Exception as e:
        print(f"Error analyzing emotion: {e}")
        return "Unknown"
