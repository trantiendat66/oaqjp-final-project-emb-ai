"""
Server module for Emotion Detection application.
"""
from flask import Flask, render_template, request
from emotion_detection.emotion_detection import emotion_detector

app = Flask("Emotion Detector")

@app.route("/emotionDetector")
def emo_detector():
    """Nhận chuỗi đầu vào, phân tích cảm xúc và trả về kết quả hiển thị."""
    text_to_analyze = request.args.get('textToAnalyze')
    response = emotion_detector(text_to_analyze)

    # Kiểm tra trường hợp trả về None (input không hợp lệ)
    if response['dominant_emotion'] is None:
        return "Invalid text! Please try again!."

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
