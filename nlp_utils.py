# #nlp_utils.py

import torch
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load translation pipeline (Malayalam → English)
translator = pipeline(
    task="translation",
    model="facebook/nllb-200-distilled-600M",
    src_lang="mal_Mlym",   # Malayalam
    tgt_lang="eng_Latn",   # English
    torch_dtype=torch.float16,
    device=0 if torch.cuda.is_available() else -1
)

sentiment_analyzer = SentimentIntensityAnalyzer()

def translate_mal_to_eng(text: str) -> str:
    """Translate Malayalam text to English using NLLB translation pipeline"""
    if not text.strip():
        return ""
    try:
        result = translator(text, max_length=512)
        return result[0]['translation_text']
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # fallback: return original Malayalam

def sentiment_score(text: str) -> dict:
    """Get VADER sentiment scores for English text"""
    return sentiment_analyzer.polarity_scores(text)
