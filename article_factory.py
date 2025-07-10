import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from database import SessionLocal

from database import SessionLocal
from article import create_article, get_article_by_guid
from source import get_source_by_key

load_dotenv()
API_KEY = os.getenv("COINDESK_API_KEY")
BASE_URL = "https://data-api.coindesk.com/news/v1"
HEADERS = {"Content-type": "application/json; charset=UTF-8"}

db = SessionLocal()

def safe_fromtimestamp(ts):
    if ts is None:
        return None
    return datetime.fromtimestamp(ts)

def fetch_latest_articles(limit=50, lang="EN"):
    response = requests.get(
        f"{BASE_URL}/article/list",
        params={"lang": lang, "limit": limit, "api_key": API_KEY},
        headers=HEADERS
    )
    response.raise_for_status()
    return response.json()["Data"]


def store_articles(db, articles):
    for art in articles:
        if get_article_by_guid(db, art["GUID"]):
            continue  # Skip duplicates

        article_data = {
            "id": art["ID"],
            "guid": art["GUID"],
            "published_on": datetime.fromtimestamp(art["PUBLISHED_ON"]),
            "image_url": art.get("IMAGE_URL"),
            "title": art.get("TITLE"),
            "authors": art.get("AUTHORS"),
            "url": art["URL"],
            "source_id": art["SOURCE_ID"],
            "body": art.get("BODY"),
            "keywords": art.get("KEYWORDS"),
            "lang": art.get("LANG"),
            "upvotes": art.get("UPVOTES", 0),
            "downvotes": art.get("DOWNVOTES", 0),
            "score": art.get("SCORE", 0),
            "sentiment": art.get("SENTIMENT"),
            "status": art.get("STATUS"),
            "created_on": datetime.fromtimestamp(art["CREATED_ON"]),
            "updated_on": safe_fromtimestamp(art["UPDATED_ON"]),

        }

        create_article(db, article_data)

def search_articles(search_string, from_ts=None, to_ts=None, source_key=None, lang="EN", limit=100):
    params = {
        "search_string": search_string,
        "lang": lang,
        "limit": limit,
        "api_key": API_KEY,
        "source_key": source_key
    }
    if from_ts: params["from_ts"] = from_ts
    if to_ts: params["to_ts"] = to_ts
    if source_key: params["source_key"] = source_key

    response = requests.get(
        "https://data-api.coindesk.com/news/v1/search",
        params=params,
        headers={"Content-type": "application/json; charset=UTF-8"}
    )
    return response.json().get("Data", [])

def search_articles_from_sources(search_string, from_ts=None, to_ts=None, source_keys=None, lang="EN", limit=100):
    all_articles = []
    for key in source_keys or []:
        params = {
            "search_string": search_string,
            "lang": lang,
            "limit": limit,
            "api_key": API_KEY,
            "source_key": key  # looped here one at a time
        }
        if from_ts:
            params["from_ts"] = from_ts
        if to_ts:
            params["to_ts"] = to_ts

        response = requests.get(
            "https://data-api.coindesk.com/news/v1/search",
            params=params,
            headers={"Content-type": "application/json; charset=UTF-8"}
        )

        if response.status_code == 200:
            data = response.json().get("Data", [])
            all_articles.extend(data)
        else:
            print(f"[{key}] Request failed with {response.status_code}: {response.text}")
    
    return all_articles

def bitcoin_articles_30_days():
    to_ts = int(datetime.now().timestamp())
    from_ts = int((datetime.now() - timedelta(days=30)).timestamp())

    sources=["bitcoin.com", "coindesk", "forbes", "bloomberg_crypto_", "cryptointelligence", "cryptodaily", "crypto_news", "coinquora", "cryptocompare", "financialtimes_crypto_"]

    articles = search_articles_from_sources(
        search_string="BTC",
        from_ts=from_ts,
        to_ts=to_ts,
        source_keys=sources
    )

    store_articles(db=db, articles=articles)



        

