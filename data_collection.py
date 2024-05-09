import os
import requests
import json
import logging
from datetime import datetime
from bs4 import BeautifulSoup
from tqdm import tqdm
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# GDELT API endpoint and parameters
GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
COUNTRIES = ["US", "UA", "RU"]
MEDIA_OUTLETS = {
    "US": ["washingtonpost.com", "cnn.com", "foxnews.com"],
    "UA": ["pravda.com.ua", "unian.info", "ukrinform.net", "kyivpost.com"],
    "UK": ["bbc.co.uk", "theguardian.com", "telegraph.co.uk"],
    "RU": ["rt.com", "sputniknews.com", "interfax.ru"]  
}
TOPIC = "(Russia OR Ukraine) AND (war OR conflict)"
FROM_DATE = "20220224000000"
TO_DATE = datetime.now().strftime("%Y%m%d%H%M%S")

# Create data directories if they don't exist
os.makedirs("data/raw/articles", exist_ok=True)

def extract_article_content(html):
    """Extract the main content of an article from the HTML using BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")
    # Adjusted to more generic content extraction
    article_content = soup.find_all('p')
    content = '\n'.join([p.get_text() for p in article_content])
    return content.strip()

def fetch_article_content(url):
    """Fetch the full content of an article given its URL."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return extract_article_content(response.text)
        else:
            logging.warning(f"Failed to fetch article content for {url}: {response.status_code} {response.reason}")
    except requests.exceptions.RequestException as e:
        logging.warning(f"Failed to fetch article content for {url}: {e}")
    return ""

def fetch_articles():
    articles = []
    for country in COUNTRIES:
        for media in tqdm(MEDIA_OUTLETS[country], desc=f"Fetching articles for {country}"):
            params = {
                "query": f"{TOPIC} domain:{media}",
                "mode": "artlist",
                "maxrecords": 2,  # Directly limiting the records here
                "format": "json",
                "startdatetime": FROM_DATE,
                "enddatetime": TO_DATE
            }
            response = requests.get(GDELT_API_URL, params=params)
            if response.status_code != 200:
                logging.error(f"Failed to fetch articles from {media}: {response.status_code}, {response.text}")
                continue

            data = response.json()
            if "articles" in data:
                for article in data["articles"]:
                    article_data = {
                        "title": article.get("title", ""),
                        "url": article.get("url", ""),
                        "published_date": article.get("seendate", ""),
                        "language": article.get("language", ""),
                        "country": country
                    }
                    article_data['content'] = fetch_article_content(article['url'])
                    if article_data['content']:  # Only append if content is successfully fetched
                        articles.append(article_data)
                logging.info(f"Fetched {len(data['articles'])} articles from {media}.")
            else:
                logging.warning(f"No articles found or invalid response format for {media} in {country}.")
    return articles

def save_articles_to_files(articles):
    for article in articles:
        # Using hash to avoid long filename issues
        hash_id = hashlib.md5(article['title'].encode('utf-8')).hexdigest()[:10]
        filename = f"{article['published_date']}_{article['country']}_{article['language']}_{hash_id}.txt"
        file_path = os.path.join("data/raw/articles", filename)
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(article['content'])
            logging.info(f"Saved article to {file_path}")
        except Exception as e:
            logging.error(f"Failed to save article {filename}: {e}")

def main():
    """Main function to fetch and save articles."""
    articles = fetch_articles()
    if articles:
        save_articles_to_files(articles)
        logging.info(f"Collected {len(articles)} articles and saved to individual files.")
    else:
        logging.warning("No articles collected.")

if __name__ == "__main__":
    main()
