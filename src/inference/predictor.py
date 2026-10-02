import os
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.preprocessing.clean_text import clean_tweet
from src.preprocessing.tokenize import MAX_LENGTH


class SentimentPredictor:
    def __init__(
        self,
        model_path="models/lstm/lstm_model.keras",
        tokenizer_path="models/lstm/tokenizer.pkl",
        label_encoder_path="models/lstm/label_encoder.pkl"
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"Label encoder file not found: {label_encoder_path}")

        self.model = load_model(model_path)

        with open(tokenizer_path, "rb") as file:
            self.tokenizer = pickle.load(file)

        with open(label_encoder_path, "rb") as file:
            self.label_encoder = pickle.load(file)

        self.classes = list(self.label_encoder.classes_)

    def predict(self, text):
        cleaned_text = clean_tweet(text)
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequence, maxlen=MAX_LENGTH)

        probabilities = self.model.predict(padded, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))

        sentiment = str(self.label_encoder.inverse_transform([predicted_index])[0])
        confidence = float(probabilities[predicted_index])

        prob_dict = {
            self.classes[i]: float(probabilities[i])
            for i in range(len(self.classes))
        }

        return {
            "text": text,
            "cleaned_text": cleaned_text,
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in prob_dict.items()}
        }

    def predict_batch(self, texts):
        cleaned_texts = [clean_tweet(t) for t in texts]
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        padded = pad_sequences(sequences, maxlen=MAX_LENGTH)

        all_probs = self.model.predict(padded, verbose=0)
        results = []

        for orig_text, clean_txt, probs in zip(texts, cleaned_texts, all_probs):
            pred_idx = int(np.argmax(probs))
            sentiment = str(self.label_encoder.inverse_transform([pred_idx])[0])
            confidence = float(probs[pred_idx])
            prob_dict = {
                self.classes[i]: float(probs[i])
                for i in range(len(self.classes))
            }
            results.append({
                "text": orig_text,
                "cleaned_text": clean_txt,
                "sentiment": sentiment,
                "confidence": round(confidence, 4),
                "probabilities": {k: round(v, 4) for k, v in prob_dict.items()}
            })

        return results
