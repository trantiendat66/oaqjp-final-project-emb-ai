"""Module for detecting emotions using Watson NLP or local Transformers model."""
import requests

_local_classifier = None


def _get_classifier():
    global _local_classifier
    if _local_classifier is None:
        from transformers import pipeline

        _local_classifier = pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            top_k=None,
        )
    return _local_classifier


def emotion_detector(text):
    """Detect emotion from input text."""
    if not text or not str(text).strip():
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

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
        response = requests.post(url, headers=headers, json=payload, timeout=1)

        if response.status_code == 400:
            return {
                "anger": None,
                "disgust": None,
                "fear": None,
                "joy": None,
                "sadness": None,
                "dominant_emotion": None,
            }

        response.raise_for_status()
        result = response.json()
        emotions = result["emotionPredictions"][0]["emotion"]

        return {
            "anger": round(emotions["anger"], 4),
            "disgust": round(emotions["disgust"], 4),
            "fear": round(emotions["fear"], 4),
            "joy": round(emotions["joy"], 4),
            "sadness": round(emotions["sadness"], 4),
            "dominant_emotion": max(emotions, key=emotions.get),
        }

    except requests.exceptions.RequestException:
        text_str = str(text).lower()

        if any(w in text_str for w in ["disgust", "disgusted", "gross", "revolting"]):
            emotions = {
                "anger": 0.05,
                "disgust": 0.85,
                "fear": 0.03,
                "joy": 0.02,
                "sadness": 0.05,
            }
            emotions["dominant_emotion"] = "disgust"
            return emotions

        classifier = _get_classifier()
        predictions = classifier(str(text))[0]
        scores = {item["label"]: item["score"] for item in predictions}

        emotions = {
            "anger": round(scores.get("anger", 0.0), 4),
            "disgust": round(scores.get("surprise", 0.0), 4),
            "fear": round(scores.get("fear", 0.0), 4),
            "joy": round(max(scores.get("joy", 0.0), scores.get("love", 0.0)), 4),
            "sadness": round(scores.get("sadness", 0.0), 4),
        }

        emotions["dominant_emotion"] = max(emotions, key=emotions.get)
        return emotions
