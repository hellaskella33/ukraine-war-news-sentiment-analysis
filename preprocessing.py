import os
import json
import requests
from langdetect import detect, DetectorFactory
from dotenv import load_dotenv

# Fix for consistent language detection
DetectorFactory.seed = 0

# Load API key from .env
load_dotenv()
DEEPL_API_KEY = os.getenv('DEEPL_API_KEY')

def load_articles(directory):
    articles = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            articles.append({
                'filename': filename,
                'content': content
            })
    return articles

def translate_to_english(text, source_lang='auto'):
    """Translate text to English using DeepL API."""
    headers = {'Authorization': f'DeepL-Auth-Key {DEEPL_API_KEY}'}
    data = {
        'text': text,
        'target_lang': 'EN',
        'source_lang': source_lang.upper()
    }
    response = requests.post('https://api-free.deepl.com/v2/translate', headers=headers, data=data)
    if response.status_code == 200:
        return response.json()['translations'][0]['text']
    else:
        print(f"Failed to translate text: {response.status_code}, {response.text}")
        return text

def preprocess_text(text):
    """Basic text preprocessing."""
    return text.lower()

def preprocess_articles(articles):
    for article in articles:
        detected_lang = detect(article['content'])
        if detected_lang != 'en':  # Translate if not in English
            article['content'] = translate_to_english(article['content'], source_lang=detected_lang)
        article['content'] = preprocess_text(article['content'])
    return articles

def save_preprocessed_articles(articles, directory):
    os.makedirs(directory, exist_ok=True)
    for article in articles:
        preprocessed_path = os.path.join(directory, article['filename'])
        with open(preprocessed_path, 'w', encoding='utf-8') as file:
            file.write(article['content'])
        print(f"Saved preprocessed article to {preprocessed_path}")

def main():
    raw_articles_dir = 'data/articles'
    preprocessed_dir = 'data/preprocessed_articles'
    articles = load_articles(raw_articles_dir)
    preprocessed_articles = preprocess_articles(articles)
    save_preprocessed_articles(preprocessed_articles, preprocessed_dir)

if __name__ == "__main__":
    main()
