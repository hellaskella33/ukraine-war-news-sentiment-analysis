import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

def load_annotated_articles(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        articles = json.load(file)
    return articles

def preprocess_labels(labels):
    """Convert labels to integers, ensure all are categorical."""
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(labels)

def train_model(X, y, target):
    y = preprocess_labels(y)  # Ensure labels are in the correct format
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    model = SVC(kernel="linear", class_weight="balanced")
    model.fit(X_train_vectorized, y_train)

    y_pred = model.predict(X_test_vectorized)
    print_evaluation_scores(y_test, y_pred, target)

    return model, vectorizer

def print_evaluation_scores(y_test, y_pred, target):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Results for {target}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

def main():
    annotated_file = "data/annotated_articles.json"
    articles = load_annotated_articles(annotated_file)

    # Sentiment Ukraine
    sentiment_data = [(art["content"], art["sentiment_ukraine"]) for art in articles if "sentiment_ukraine" in art]
    X_sentiment, y_sentiment = zip(*sentiment_data)
    sentiment_model, sentiment_vectorizer = train_model(X_sentiment, y_sentiment, "Sentiment towards Ukraine")
    joblib.dump(sentiment_model, "models/sentiment_model.pkl")
    joblib.dump(sentiment_vectorizer, "models/sentiment_vectorizer.pkl")

    # Propaganda Detection
    propaganda_data = [(art["content"], art["propaganda"]) for art in articles if "propaganda" in art]
    X_propaganda, y_propaganda = zip(*propaganda_data)
    propaganda_model, propaganda_vectorizer = train_model(X_propaganda, y_propaganda, "Propaganda Detection")
    joblib.dump(propaganda_model, "models/propaganda_model.pkl")
    joblib.dump(propaganda_vectorizer, "models/propaganda_vectorizer.pkl")

    print("Models trained and saved successfully.")

if __name__ == "__main__":
    main()
