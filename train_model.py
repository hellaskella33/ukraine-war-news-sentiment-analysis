import json
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def load_annotated_articles(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        articles = json.load(file)
    return articles

def preprocess_data(articles, label_key):
    """Extracts texts and labels, ensuring both are present."""
    texts, labels = [], []
    for article in articles:
        if label_key in article and article[label_key] is not None and 'content' in article and article['content']:
            texts.append(article['content'])
            labels.append(article[label_key])
    return texts, labels

def preprocess_text_data(texts):
    """Convert text data to TF-IDF vectors."""
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def preprocess_labels(labels):
    """Convert labels to integers using the provided label encoder."""
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    return y, label_encoder

def balance_data(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def train_model(X, y, target):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hyperparameter tuning using GridSearchCV
    param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
    grid = GridSearchCV(SVC(class_weight="balanced"), param_grid, refit=True, verbose=2, cv=5)
    grid.fit(X_train, y_train)
    
    print(f"Best parameters for {target}: {grid.best_params_}")

    y_pred = grid.predict(X_test)
    print_evaluation_scores(y_test, y_pred, target)

    return grid.best_estimator_

def print_evaluation_scores(y_test, y_pred, target):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Results for {target}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

def main():
    annotated_file = "data/annotated_articles.json"
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    label_encoder_path = "models/propaganda_label_encoder.pkl"

    # Load and preprocess data
    articles = load_annotated_articles(annotated_file)
    texts, labels = preprocess_data(articles, "propaganda")
    X, vectorizer = preprocess_text_data(texts)
    y, label_encoder = preprocess_labels(labels)

    # Balance the dataset
    X_balanced, y_balanced = balance_data(X, y)

    # Train and save the propaganda detection model
    propaganda_model = train_model(X_balanced, y_balanced, "Propaganda Detection")
    joblib.dump(propaganda_model, "models/propaganda_model.pkl")
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(label_encoder, label_encoder_path)

    print("Propaganda detection model trained and saved successfully.")

if __name__ == "__main__":
    main()
