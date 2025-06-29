# IMDb Review Sentiment Classifier

This project demonstrates a basic machine learning pipeline using scikit-learn to classify movie reviews as positive or negative.

## 🔧 Installation

```bash
git clone <repo-url>
cd ml_sentiment_classifier
pip install -r requirements.txt
````

## 📊 Training

```bash
python train.py
```

This will download a small IMDb review dataset, train a logistic regression classifier, and save the model in the `model/` folder.

## 🔍 Prediction

```bash
python predict.py "I loved this movie!"
```

**Output:**

```
Prediction: positive (confidence: 0.89)
```

---

## 📦 Files

* `train.py`: Trains and saves the model.
* `predict.py`: Loads the model and predicts from input text.
* `requirements.txt`: Python dependencies.

## 🌐 API 

You can run the Flask API with:

```bash
python app.py
````

### 🧪 Sample Request (POST)

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "An awesome experience with great acting!"}'
```

### 📤 Response

```json
{
  "prediction": "positive",
  "confidence": 0.91
}
```

