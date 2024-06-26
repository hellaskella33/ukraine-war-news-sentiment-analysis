import os
import requests
import json
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
import hashlib
import time
from newspaper import Article
from googletrans import Translator
from langdetect import detect

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
COUNTRIES = ["US", "UA", "GB", "BY", "PL"]  # Added Poland (PL)
MEDIA_OUTLETS = {
    "US": ["washingtonpost.com", "cnn.com", "foxnews.com"],
    "UA": ["kyivpost.com", "ukrinform.net", "unian.info"],
    "GB": ["bbc.co.uk", "theguardian.com", "telegraph.co.uk"],
    "BY": ["belarusfeed.com", "eng.belta.by"],
    "PL": ["tvn24.pl", "gazeta.pl", "onet.pl"]  # Added Poland's media outlets
}
TOPICS = {
    "US": "(Russia OR Ukraine) AND (war OR conflict OR invasion OR aggression)",
    "UA": "(Russia OR Ukraine) AND (war OR conflict OR invasion OR aggression OR Crimea OR Donbas)",
    "GB": "(Russia OR Ukraine) AND (war OR conflict OR invasion OR aggression)",
    "BY": "(Russia OR Ukraine) AND (war OR conflict OR invasion OR aggression OR sanctions)",
    "PL": "(Russia OR Ukraine) AND (war OR conflict OR invasion OR aggression OR sanctions)"  # Added Poland's topic
}

# Define key events
key_events = {
    '2022-02-24': 'Start of Russian Invasion',
    '2022-04-02': 'Bucha Massacre Discovered',
    '2022-09-06': 'Ukrainian Counteroffensive in Kharkiv',
    '2023-02-20': 'Russian Offensive in Eastern Ukraine',
    '2023-07-01': 'Ukraine Joins NATO',
    '2022-03-15': 'Sanctions Law Against Russia',
    '2022-06-26': 'G7 Leaders Meeting on Ukraine',
    '2023-03-10': 'Military Aid Agreement Signed'
}

# Function to get date ranges for key events
def get_date_ranges(events, months_before=1, months_after=1):
    ranges = []
    for event_date in events.keys():
        start_date = datetime.strptime(event_date, '%Y-%m-%d') - timedelta(days=30*months_before)
        end_date = datetime.strptime(event_date, '%Y-%m-%d') + timedelta(days=30*months_after)
        ranges.append((start_date.strftime("%Y%m%d%H%M%S"), end_date.strftime("%Y%m%d%H%M%S")))
    return ranges

date_ranges = get_date_ranges(key_events)

os.makedirs("data/articles", exist_ok=True)

def fetch_article_content(url, retry_count=3, delay=1):
    for attempt in range(retry_count):
        try:
            article = Article(url)
            article.download()
            article.parse()
            content = article.text
            language = detect(content)
            if language != 'en':
                translator = Translator()
                translation = translator.translate(content, dest='en')
                content = translation.text
            return content
        except Exception as e:
            logging.warning(f"Exception fetching article content for {url}: {e}")
        time.sleep(delay)  # Delay before retrying
    return ""

def fetch_articles():
    articles = []
    for country in COUNTRIES:
        for media in tqdm(MEDIA_OUTLETS[country], desc=f"Fetching articles for {country}"):
            for start_date, end_date in date_ranges:
                params = {
                    "query": TOPICS[country] + f" domain:{media}",
                    "mode": "artlist",
                    "maxrecords": 5,
                    "format": "json",
                    "startdatetime": start_date,
                    "enddatetime": end_date
                }
                try:
                    response = requests.get(GDELT_API_URL, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if "articles" in data and data["articles"]:
                            for article in data["articles"]:
                                content = fetch_article_content(article['url'])
                                if content:
                                    articles.append({
                                        "title": article.get("title", ""),
                                        "url": article['url'],
                                        "published_date": article.get("seendate", ""),
                                        "language": "en",
                                        "country": country,
                                        "content": content
                                    })
                            logging.info(f"Fetched {len(data['articles'])} articles from {media} in {country}.")
                        else:
                            logging.warning(f"No articles found for {media} in {country}.")
                    else:
                        logging.error(f"API error from {media}: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    logging.error(f"Exception fetching articles from {media}: {e}")
                time.sleep(1)  # Add a delay between requests to avoid overwhelming the API
    return articles

def save_articles_to_files(articles):
    for article in articles:
        hash_id = hashlib.md5(article['title'].encode('utf-8')).hexdigest()[:10]
        filename = f"{article['published_date']}_{article['country']}_{article['language']}_{hash_id}.txt"
        file_path = os.path.join("data/articles", filename)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(article['content'])
        logging.info(f"Saved article to {file_path}")

def main():
    articles = fetch_articles()
    if articles:
        save_articles_to_files(articles)
        logging.info(f"Collected {len(articles)} articles and saved.")
    else:
        logging.warning("No articles collected.")

if __name__ == "__main__":
    main()
