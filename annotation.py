import os
import openai
import json
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables and API keys
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def load_articles(directory):
    """Load articles from a directory."""
    articles = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                content = file.read()
                articles.append({"filename": filename, "content": content})
    return articles

def annotate_articles(articles):
    """Annotate articles using OpenAI's GPT-3.5 model."""
    for article in tqdm(articles, desc="Annotating articles"):
        prompt = f"""
        Analyze the following news article related to the Russia-Ukraine war. Provide detailed annotations based on the content presented, focusing on sentiment towards Ukraine and the presence of propaganda.

        Article Content:
        "{article['content']}"

        Instructions:
        1. Sentiment towards Ukraine:
           - Very Negative (1): The news is very bad for Ukraine, expressing strong negative criticism or portraying Ukraine in an extremely negative light.
           - Negative (2): The news is somewhat bad for Ukraine, showing some negative opinions or criticism towards Ukraine.
           - Neutral (3): The news is neutral or factual, without any positive or negative bias towards Ukraine.
           - Positive (4): The news is generally good for Ukraine, showing supportive or positive opinions towards Ukraine.
           - Very Positive (5): The news is very good for Ukraine, expressing strong support or portraying Ukraine in an extremely positive light.

        2. Propaganda Detection:
           - Present: Indicates propaganda is used, such as emotional manipulation, biased information, or one-sided reporting.
           - Not Present: Fair and balanced reporting without propaganda elements.

        3. Aspect-Based Sentiment:
           Identify key aspects discussed in the article and rate the sentiment for each aspect:
           - Military Support
           - Humanitarian Aid
           - Diplomacy
           - Political Stability
           - Economic Effects

        Provide your annotations in the following format:
        Overall Sentiment towards Ukraine: [1-5]
        Propaganda: [Present/Not Present]
        Aspects:
        - Military Support: [Sentiment Score]
        - Humanitarian Aid: [Sentiment Score]
        - Diplomacy: [Sentiment Score]
        - Political Stability: [Sentiment Score]
        - Economic Effects: [Sentiment Score]

        Example:
        Overall Sentiment towards Ukraine: 4
        Propaganda: Not Present
        Aspects:
        - Military Support: 3
        - Humanitarian Aid: 4
        - Diplomacy: 2
        - Political Stability: 3
        - Economic Effects: 4
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant for annotating news articles related to the Russia-Ukraine war."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300,
            stop=None
        )

        annotation = response.choices[0].message['content'].strip()
        lines = annotation.split("\n")
        article["sentiment_ukraine"] = None
        article["propaganda"] = None
        article["aspects"] = {}

        for line in lines:
            line = line.strip()
            if line.startswith("Overall Sentiment towards Ukraine:"):
                sentiment_str = line.split(":")[1].strip()
                if sentiment_str.isdigit():
                    article["sentiment_ukraine"] = int(sentiment_str)
            elif line.startswith("Propaganda:"):
                article["propaganda"] = line.split(":")[1].strip()
            elif line.startswith("-"):
                parts = line[1:].split(":")
                if len(parts) >= 2:
                    aspect = parts[0].strip()
                    score_str = parts[1].strip()
                    if score_str.isdigit():
                        article["aspects"][aspect] = int(score_str)
                    else:
                        article["aspects"][aspect] = None  # or set a default value

    return articles

def save_annotated_articles(articles, output_file):
    """Save the annotated articles to a JSON file."""
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(articles, file, indent=4)

def main():
    """Main function to load, annotate, and save articles."""
    articles_dir = "data/articles"
    annotated_file = "data/annotated_articles.json"
    articles = load_articles(articles_dir)
    annotated_articles = annotate_articles(articles)
    save_annotated_articles(annotated_articles, annotated_file)
    print(f"Annotated {len(annotated_articles)} articles and saved to {annotated_file}")

if __name__ == "__main__":
    main()
