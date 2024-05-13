import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from src.preprocessing import preprocess_articles
from src.train_model import load_model

TEST_DATA_DIR = "data/test"

def load_test_data():
    test_articles = []
    for filename in os.listdir(TEST_DATA_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(TEST_DATA_DIR, filename), "r", encoding="utf-8") as file:
                content = file.read()
                test_articles.append(content)
    return test_articles

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"{model_name} Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Plot ROC curve
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} Model ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"results/{model_name.lower()}_roc_curve.png")
    plt.close()

    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Model Precision-Recall Curve')
    plt.savefig(f"results/{model_name.lower()}_pr_curve.png")
    plt.close()

def main():
    sentiment_model = load_model("models/sentiment/sentiment_model.pkl")
    propaganda_model = load_model("models/propaganda/propaganda_model.pkl")

    test_articles = load_test_data()
    preprocessed_articles = preprocess_articles(test_articles)

    # Sentiment Analysis Evaluation
    X_test_sentiment = preprocessed_articles
    y_test_sentiment = ...  # Load the true sentiment labels for the test articles
    evaluate_model(sentiment_model, X_test_sentiment, y_test_sentiment, "Sentiment")

    # Propaganda Detection Evaluation
    X_test_propaganda = preprocessed_articles
    y_test_propaganda = ...  # Load the true propaganda labels for the test articles
    evaluate_model(propaganda_model, X_test_propaganda, y_test_propaganda, "Propaganda")

if __name__ == "__main__":
    main()