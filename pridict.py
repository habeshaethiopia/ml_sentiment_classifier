import joblib
import sys
import os

def predict(review_text):
    # Load model & vectorizer
    clf = joblib.load("model/model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")

    # Vectorize input
    X = vectorizer.transform([review_text])

    # Predict
    prob = clf.predict_proba(X)[0]
    label = "positive" if prob[1] >= 0.5 else "negative"
    confidence = prob[1] if prob[1] >= 0.5 else prob[0]

    print(f"Prediction: {label} (confidence: {confidence:.2f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"<your review here>\"")
        sys.exit(1)

    input_text = sys.argv[1]
    predict(input_text)
