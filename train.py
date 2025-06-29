import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
def load_data():
    # Download small IMDb dataset
    df = pd.read_csv(
        "https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv"
    )
    df = df.sample(5000, random_state=42).reset_index(drop=True)
    return df

# Main training pipeline
def train():
    df = load_data()

    X = df['review']
    y = df['sentiment']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")

    # Save model
    os.makedirs("model", exist_ok=True)
    joblib.dump(clf, "model/model.pkl")
    joblib.dump(vectorizer, "model/vectorizer.pkl")

if __name__ == "__main__":
    train()