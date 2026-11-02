"""Server deployment for Emotion Detection web application."""
from flask import Flask, render_template, request
from emotion_detection.emotion_detection import emotion_detector

app = Flask("Emotion Detector")

@app.route("/emotionDetector")
def emo_detector():
    """Nhận text, phân tích cảm xúc và trả về phản hồi."""
    text_to_analyze = request.args.get('textToAnalyze', '')
    response = emotion_detector(text_to_analyze)

    if response['dominant_emotion'] is None:
        return "Please enter some text to analyze."

    return (
        f"For the given statement, the system response is "
        f"'anger': {response['anger']}, 'disgust': {response['disgust']}, "
        f"'fear': {response['fear']}, 'joy': {response['joy']} and "
        f"'sadness': {response['sadness']}. The dominant emotion is "
        f"<b>{response['dominant_emotion']}</b>."
    )

@app.route("/")
def render_index_page():
    """Render trang chủ giao diện."""
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
