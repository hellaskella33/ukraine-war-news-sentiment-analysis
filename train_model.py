import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_annotated_articles(file_path):
    """Load annotated articles from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        articles = json.load(file)
    return articles

def preprocess_labels(label):
    """Preprocess labels to remove unwanted characters."""
    return label.strip("[]").replace('"', '').strip()

def train_model(articles, target):
    """Train a model based on the articles and the specified target sentiment or category."""
    # Preprocess labels and extract content
    X = [article["content"] for article in articles]
    y = [preprocess_labels(article[target]) for article in articles]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a TF-IDF vectorizer and transform the data
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Train a Support Vector Classifier
    model = SVC(kernel="linear")
    model.fit(X_train_vectorized, y_train)

    # Predict and evaluate the model
    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"{target.capitalize()} Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    return model, vectorizer

def save_model(model, vectorizer, model_file, vectorizer_file):
    """Save the trained model and vectorizer to files."""
    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vectorizer_file)

def main():
    annotated_file = "data/annotated_articles.json"
    sentiment_model_file = "models/sentiment_model.pkl"
    sentiment_vectorizer_file = "models/sentiment_vectorizer.pkl"
    
    articles = load_annotated_articles(annotated_file)
    
    # Check if the 'sentiment_ukraine' and 'propaganda' keys are in the articles
    if articles and 'sentiment_ukraine' in articles[0] and 'propaganda' in articles[0]:
        sentiment_model, sentiment_vectorizer = train_model(articles, "sentiment_ukraine")
        save_model(sentiment_model, sentiment_vectorizer, sentiment_model_file, sentiment_vectorizer_file)
        propaganda_model, propaganda_vectorizer = train_model(articles, "propaganda")
        save_model(propaganda_model, propaganda_vectorizer, "models/propaganda_model.pkl", "models/propaganda_vectorizer.pkl")
    else:
        print("Missing required annotation keys in the articles.")

if __name__ == "__main__":
    main()
