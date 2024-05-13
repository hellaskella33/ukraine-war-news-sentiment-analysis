import os
import openai
import json
import logging
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
openai.api_key = os.getenv('OpenAI_API_KEY')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_articles(directory):
    articles = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()
                if content:  # Ensure content is not empty
                    articles.append({"filename": filename, "content": content})
    return articles

def annotate_articles(articles):
    annotated = []
    for article in tqdm(articles, desc="Annotating articles"):
        try:
            prompt = f"..."  # Your existing prompt
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300,  # Increased max_tokens for more detailed responses
                stop=["\n\n"],  # Define a suitable stop sequence
            )
            annotation = response.choices[0].message['content'].strip()
            article['annotations'] = annotation
            annotated.append(article)
        except Exception as e:
            logging.error(f"Failed to annotate article {article['filename']}: {str(e)}")

    return annotated

def save_annotated_articles(articles, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(articles, file, indent=4)

def main():
    articles_dir = "data/test"
    annotated_file = "data/test_annotated_articles.json"

    articles = load_articles(articles_dir)
    if not articles:
        logging.info("No articles to process.")
        return

    annotated_articles = annotate_articles(articles)
    save_annotated_articles(annotated_articles, annotated_file)

    logging.info(f"Annotated {len(annotated_articles)} articles and saved to {annotated_file}")

if __name__ == "__main__":
    main()
