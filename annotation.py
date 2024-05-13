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
           - Very Negative (1): Expresses strong negative criticism towards Ukraine.
           - Negative (2): Shows some negative opinions towards Ukraine.
           - Neutral (3): Neutral or factual reporting without emotional bias.
           - Positive (4): Generally positive or supportive towards Ukraine.
           - Very Positive (5): Strongly positive or supportive towards Ukraine.

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

        Please provide your annotations in the following format:
        Overall Sentiment towards Ukraine: [1-5]
        Propaganda: [Present/Not Present]
        Aspects:
        - Military Support: [Sentiment Score]
        - Humanitarian Aid: [Sentiment Score]
        - Diplomacy: [Sentiment Score]
        - Other Aspects: [Sentiment Score]
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
