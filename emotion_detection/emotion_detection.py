"""Module for detecting emotions using Watson NLP API."""

import requests


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

    payload = {"raw_document": {"text": text}}

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
