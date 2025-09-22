# analyzer.py
from diarization import diarize_audio
from emotion import analyze_emotion_from_audio
from transcription import transcribe_audio_segment
from nlp_utils import translate_mal_to_eng, sentiment_score
from pydub import AudioSegment
import tempfile
import os
import torch

def extract_audio_segment(audio_file, start_time, end_time):
    """Extract a specific audio segment for emotion analysis"""
    try:
        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_frame_rate(16000).set_channels(1)
        start_ms, end_ms = start_time * 1000, end_time * 1000
        segment_audio = audio[start_ms:end_ms]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            segment_audio.export(tmp.name, format="wav")
            return tmp.name

    except Exception as e:
        print(f"Audio extraction error: {e}")
        return None

def analyze_conversation(audio_file):
    """Main analysis pipeline: Diarization → Transcription → Emotion → Translation → Sentiment"""
    print("Starting diarization...")
    segments = diarize_audio(audio_file)

    if not segments:
        print("No speech segments detected")
        return {"segment_results": [], "speaker_summary": {}, "conversation_duration": 0}

    print(f"Found {len(segments)} speech segments")
    results = []

    for i, seg in enumerate(segments):
        print(f"Processing segment {i+1}/{len(segments)}: {seg['speaker']} {seg['start']:.1f}-{seg['end']:.1f}s")

        try:
            # Extract audio segment for emotion analysis
            emotion_audio_file = extract_audio_segment(audio_file, seg['start'], seg['end'])
            emotion = analyze_emotion_from_audio(emotion_audio_file, seg['speaker']) if emotion_audio_file else "Unknown"
            if emotion_audio_file:
                os.unlink(emotion_audio_file)

            # Malayalam transcription from ElevenLabs
            mal_transcript = transcribe_audio_segment(audio_file, seg['start'], seg['end'])

            # English translation
            eng_translation = translate_mal_to_eng(mal_transcript)

            # Sentiment on English translation
            sentiment = sentiment_score(eng_translation)

            results.append({
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "duration": seg["end"] - seg["start"],
                "transcript": mal_transcript,       # Malayalam
                "translation": eng_translation,     # English
                "emotion": emotion,
                "sentiment": sentiment
            })

        except Exception as e:
            print(f"Error processing segment {i+1}: {e}")
            continue

        # Clear GPU memory periodically
        if i % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Build speaker summary
    speaker_summary = {}
    total_duration = sum(seg['end'] - seg['start'] for seg in segments)

    for result in results:
        spk = result['speaker']
        if spk not in speaker_summary:
            speaker_summary[spk] = {
                "total_segments": 0,
                "total_duration": 0,
                "emotions": [],
                "avg_sentiment": 0,
                "sentiments": []
            }

        speaker_summary[spk]["total_segments"] += 1
        speaker_summary[spk]["total_duration"] += result["duration"]
        speaker_summary[spk]["emotions"].append(result["emotion"])
        speaker_summary[spk]["sentiments"].append(result["sentiment"]["compound"])

    # Calculate averages and distributions
    for spk, data in speaker_summary.items():
        if data["total_duration"] > 0:
            data["speaking_time_percentage"] = 100 * data["total_duration"] / total_duration

        # Dominant emotion
        if data["emotions"]:
            data["dominant_emotion"] = max(set(data["emotions"]), key=data["emotions"].count)
            data["emotion_distribution"] = {e: data["emotions"].count(e) for e in set(data["emotions"])}
        else:
            data["dominant_emotion"] = "Unknown"
            data["emotion_distribution"] = {}

        # Average sentiment
        if data["sentiments"]:
            data["avg_sentiment"] = sum(data["sentiments"]) / len(data["sentiments"])
        else:
            data["avg_sentiment"] = 0

    return {
        "segment_results": results,
        "speaker_summary": speaker_summary,
        "conversation_duration": total_duration
    }

def print_conversation_analysis(analysis):
    """Print detailed analysis results"""
    print("\n" + "="*80)
    print("COMPLETE CONVERSATION ANALYSIS REPORT")
    print("="*80)

    print(f"\nTotal Duration: {analysis['conversation_duration']:.2f} seconds")

    # Speaker summary
    print("\n" + "-"*40)
    print("SPEAKER SUMMARY")
    print("-"*40)
    for spk, data in analysis['speaker_summary'].items():
        print(f"\n{spk}:")
        print(f"  Segments: {data['total_segments']}")
        print(f"  Duration: {data['total_duration']:.2f}s ({data.get('speaking_time_percentage', 0):.1f}%)")
        print(f"  Dominant Emotion: {data['dominant_emotion']}")
        print(f"  Avg Sentiment: {data['avg_sentiment']:.3f}")
        print("  Emotion Distribution:")
        for emotion, count in data.get("emotion_distribution", {}).items():
            print(f"    {emotion}: {count}")

    # Detailed segments
    print("\n" + "-"*40)
    print("DETAILED SEGMENT ANALYSIS")
    print("-"*40)

    for i, seg in enumerate(analysis['segment_results']):
        print(f"\nSegment {i+1}:")
        print(f"  Speaker: {seg['speaker']}")
        print(f"  Time: {seg['start']:.2f}s - {seg['end']:.2f}s")
        print(f"  Malayalam: {seg['transcript']}")
        print(f"  English: {seg['translation']}")
        print(f"  Emotion: {seg['emotion']}")
        print(f"  Sentiment: {seg['sentiment']['compound']:.3f} "
              f"(pos={seg['sentiment']['pos']:.2f}, neu={seg['sentiment']['neu']:.2f}, neg={seg['sentiment']['neg']:.2f})")
