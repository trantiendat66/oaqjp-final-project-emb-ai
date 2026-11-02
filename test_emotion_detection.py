import unittest
from emotion_detection.emotion_detection import emotion_detector

class TestEmotionDetection(unittest.TestCase):
    def test_emotion_detector(self):
        # 1. Test Joy
        res_joy = emotion_detector("I am glad this happened")
        self.assertEqual(res_joy['dominant_emotion'], 'joy')

        # 2. Test Anger
        res_anger = emotion_detector("I am really mad about this")
        self.assertEqual(res_anger['dominant_emotion'], 'anger')

        # 3. Test Disgust
        res_disgust = emotion_detector("I feel disgusted just hearing about this")
        self.assertEqual(res_disgust['dominant_emotion'], 'disgust')

        # 4. Test Sadness
        res_sadness = emotion_detector("I am so sad about this")
        self.assertEqual(res_sadness['dominant_emotion'], 'sadness')

        # 5. Test Fear
        res_fear = emotion_detector("I am really afraid that this will happen")
        self.assertEqual(res_fear['dominant_emotion'], 'fear')

if __name__ == '__main__':
    unittest.main()
