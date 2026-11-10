
"""Module for detecting emotions using Watson NLP API."""

import re

import requests
from spellchecker import SpellChecker


# Initialize English spell checker
spell = SpellChecker()

# Common spelling mistakes and corrections
CUSTOM_CORRECTIONS = {
    "happe": "happy",
    "happi": "happy",
    "happpy": "happy",
    "hapyy": "happy",
    "happpy": "happy",
    "hapy": "happy",
    "saddd": "sad",
    "sadd": "sad",
    "angrey": "angry",
    "angy": "angry",
    "lov": "love",
    "luv": "love",
}


def preprocess_text(text):
    """
    Normalize text and correct common spelling mistakes
    before sending it to Watson API.
    """

    text = str(text).strip()

    # Normalize common negative expressions.
    # These are simple business rules, not full semantic understanding.
    negative_patterns = {
        r"\bnot\s+happy\b": "sad",
        r"\bnot\s+feeling\s+happy\b": "sad",
        r"\bunhappy\b": "sad",
        r"\bnot\s+good\b": "bad",
        r"\bnot\s+enjoying\b": "dislike",
    }

    for pattern, replacement in negative_patterns.items():
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    # Correct spelling while preserving punctuation.
    def correct_word(match):
        word = match.group(0)
        lower_word = word.lower()

        if lower_word in CUSTOM_CORRECTIONS:
            corrected = CUSTOM_CORRECTIONS[lower_word]
        else:
            corrected = spell.correction(lower_word) or lower_word

        if word.isupper():
            return corrected.upper()

        if word[0].isupper():
            return corrected.capitalize()

        return corrected

    text = re.sub(r"[a-zA-Z]+", correct_word, text)

    return text


def emotion_detector(text):
    """Detect emotions in text using the Watson NLP API."""

    empty_result = {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None,
    }

    if not text or not str(text).strip():
        return empty_result

    # Preprocess input before sending it to Watson.
    cleaned_text = preprocess_text(text)

    url = (
        "https://sn-watson-emotion.labs.skills.network/"
        "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )

    headers = {
        "grpc-metadata-mm-model-id": (
            "emotion_aggregated-workflow_lang_en_stock"
        ),
        "Content-Type": "application/json",
    }

    payload = {"raw_document": {"text": cleaned_text}}

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=20,
        )

        response.raise_for_status()

        result = response.json()
        emotions = result["emotionPredictions"][0]["emotion"]

        detected_emotions = {
            "anger": round(emotions["anger"], 4),
            "disgust": round(emotions["disgust"], 4),
            "fear": round(emotions["fear"], 4),
            "joy": round(emotions["joy"], 4),
            "sadness": round(emotions["sadness"], 4),
        }

        detected_emotions["dominant_emotion"] = max(
            detected_emotions,
            key=detected_emotions.get,
        )

        return detected_emotions

    except (
        requests.exceptions.RequestException,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    ):
        return empty_result
