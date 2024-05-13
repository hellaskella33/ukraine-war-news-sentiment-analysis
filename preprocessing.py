import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib

def load_annotated_articles(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        articles = json.load(file)
    return articles

def preprocess_text_data(articles):
    """Convert text data to TF-IDF vectors."""
    texts = [article['content'] for article in articles]
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def preprocess_labels(articles, label_key):
    """Convert labels to integers, ensure all are categorical."""
    labels = [article[label_key] for article in articles if label_key in article]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    return y, label_encoder

def save_preprocessing_artifacts(vectorizer, label_encoder, vectorizer_path, label_encoder_path):
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(label_encoder, label_encoder_path)

def preprocess_and_save(annotated_file, vectorizer_path, label_encoder_paths):
    articles = load_annotated_articles(annotated_file)
    
    # Preprocess text data for TF-IDF vectors
    X, vectorizer = preprocess_text_data(articles)
    
    # Preprocess labels for sentiment and propaganda detection
    y_sentiment, sentiment_label_encoder = preprocess_labels(articles, "sentiment_ukraine")
    y_propaganda, propaganda_label_encoder = preprocess_labels(articles, "propaganda")
    
    # Save preprocessing artifacts
    save_preprocessing_artifacts(vectorizer, sentiment_label_encoder, vectorizer_path, label_encoder_paths[0])
    save_preprocessing_artifacts(vectorizer, propaganda_label_encoder, vectorizer_path, label_encoder_paths[1])
    
    return X, y_sentiment, y_propaganda, vectorizer, sentiment_label_encoder, propaganda_label_encoder

def main():
    annotated_file = "data/annotated_articles.json"
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    label_encoder_paths = ["models/sentiment_label_encoder.pkl", "models/propaganda_label_encoder.pkl"]
    
    X, y_sentiment, y_propaganda, vectorizer, sentiment_label_encoder, propaganda_label_encoder = preprocess_and_save(
        annotated_file, vectorizer_path, label_encoder_paths)
    
    print("Preprocessing complete and artifacts saved.")

if __name__ == "__main__":
    main()
