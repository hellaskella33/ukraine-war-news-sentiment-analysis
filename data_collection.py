import os
import requests
import json
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
import hashlib
import time
from newspaper import Article

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
COUNTRIES = ["US", "UA", "GB", "RU", "BY"]
MEDIA_OUTLETS = {
    "US": ["washingtonpost.com", "cnn.com", "foxnews.com"],
    "UA": ["pravda.com.ua", "unian.info", "ukrinform.net", "kyivpost.com"],
    "GB": ["bbc.co.uk", "theguardian.com", "telegraph.co.uk"],
    "RU": ["rt.com","interfax.ru", "rbth.com"],
    "BY": ["belta.by", "naviny.by"]
}
TOPICS = {
    "US": "(Russia OR Ukraine) AND (war OR conflict OR invasion OR aggression)",
    "UA": "(Russia OR Ukraine) AND (war OR conflict OR invasion OR aggression)",
    "GB": "(Russia OR Ukraine) AND (war OR conflict OR invasion OR aggression)",
    "RU": "(Russia OR Ukraine) AND (war OR conflict OR invasion OR aggression)",
    "BY": "(Russia OR Ukraine) AND (war OR conflict OR invasion OR aggression)"
}
FROM_DATE = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d%H%M%S")
TO_DATE = datetime.now().strftime("%Y%m%d%H%M%S")
os.makedirs("data/articles", exist_ok=True)

def fetch_article_content(url, retry_count=3, delay=1):
    for attempt in range(retry_count):
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            logging.warning(f"Exception fetching article content for {url}: {e}")
        time.sleep(delay)  # Delay before retrying
    return ""

def fetch_articles():
    articles = []
    for country in COUNTRIES:
        for media in tqdm(MEDIA_OUTLETS[country], desc=f"Fetching articles for {country}"):
            params = {
                "query": TOPICS[country] + f" domain:{media}",
                "mode": "artlist",
                "maxrecords": 8,
                "format": "json",
                "startdatetime": FROM_DATE,
                "enddatetime": TO_DATE
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
                                    "language": article.get("language", ""),
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