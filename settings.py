import os
from pathlib import Path

BASE_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
DATA_DIR = BASE_DIR / 'data'
ANNOTATED_ARTICLES_FILE = DATA_DIR / 'annotated_articles.json'

KEY_ANNOTATED_ARTICLES_FILE = DATA_DIR / 'key_annotated_articles.json'