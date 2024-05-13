import os
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

def load_annotated_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def preprocess_data(data):
    X = []
    y_propaganda = []
    for article in data:
        if article['propaganda'] is not None:  # Ensure propaganda data is present
            X.append(article['content'])
            y_propaganda.append(1 if article['propaganda'] == 'Present' else 0)
    return X, y_propaganda

def evaluate_model(model, vectorizer, X_test, y_test, model_name):
    X_test_transformed = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_transformed)
    print(f"Evaluating {model_name} Model...")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='macro'):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred, average='macro'):.4f}")

def main():
    annotated_file = "data/test_annotated_articles.json"
    propaganda_model = joblib.load("models/propaganda_model.pkl")
    vectorizer = joblib.load("models/propaganda_vectorizer.pkl")

    data = load_annotated_data(annotated_file)
    X, y_propaganda = preprocess_data(data)

    if y_propaganda:  # Check if there is propaganda data to evaluate
        evaluate_model(propaganda_model, vectorizer, X, y_propaganda, "Propaganda")

if __name__ == "__main__":
    main()
