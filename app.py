from flask import Flask, request, jsonify
import joblib
import os

# Load model and vectorizer once
model_path = "model/model.pkl"
vectorizer_path = "model/vectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    raise FileNotFoundError("Trained model or vectorizer not found. Run train.py first.")

clf = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request"}), 400

    review = data["text"]
    X = vectorizer.transform([review])
    prob = clf.predict_proba(X)[0]
    label = "positive" if prob[1] >= 0.5 else "negative"
    confidence = round(prob[1] if label == "positive" else prob[0], 3)

    return jsonify({
        "prediction": label,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)