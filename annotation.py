import os
import openai
import json
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
openai.api_key = os.getenv('OpenAI_API_KEY')

def load_articles(directory):
    articles = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                content = file.read()
                articles.append({"filename": filename, "content": content})
    return articles

def annotate_articles(articles):
    for article in tqdm(articles, desc="Annotating articles"):
        prompt = f"""
        Please analyze the following news article related to the Russia-Ukraine war and provide annotations for sentiment towards Ukraine and the presence of propaganda.

        Sentiment towards Ukraine:
        - Positive: The article expresses a positive or supportive tone towards Ukraine, its government, or its people.
        - Negative: The article expresses a negative or critical tone towards Ukraine, its government, or its people.
        - Neutral: The article presents information about Ukraine in an objective or impartial manner without a strong emotional tone.

        Propaganda:
        - Present: The article contains elements of propaganda, such as biased or misleading information, emotional manipulation, or persuasive techniques to promote a particular viewpoint.
        - Not Present: The article does not contain elements of propaganda and presents information in a factual and objective manner.

        Please provide the annotations in the following format:
        Sentiment towards Ukraine: [Positive/Negative/Neutral]
        Propaganda: [Present/Not Present]

        Article:
        {article['content']}
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant for annotating news articles related to the Russia-Ukraine war."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=100,
            n=1,
            stop=None,
        )

        annotation = response.choices[0].message['content'].strip()
        lines = annotation.split("\n")
        for line in lines:
            if line.startswith("Sentiment towards Ukraine:"):
                article["sentiment_ukraine"] = line.split(":")[1].strip()
            elif line.startswith("Propaganda:"):
                article["propaganda"] = line.split(":")[1].strip()

    return articles

def save_annotated_articles(articles, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(articles, file, indent=4)

def main():
    preprocessed_dir = "data/preprocessed_articles"
    annotated_file = "data/annotated_articles.json"

    articles = load_articles(preprocessed_dir)
    annotated_articles = annotate_articles(articles)
    save_annotated_articles(annotated_articles, annotated_file)

    print(f"Annotated {len(annotated_articles)} articles and saved to {annotated_file}")

if __name__ == "__main__":
    main()