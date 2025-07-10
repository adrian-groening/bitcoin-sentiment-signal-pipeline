import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
from database import SessionLocal
from source import create_source, get_source_by_key, update_source

load_dotenv()

API_KEY = os.getenv("COINDESK_API_KEY")

def safe_timestamp(ts):
    if ts is not None:
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    return None

def fetch_news_sources():
    response = requests.get(
        "https://data-api.coindesk.com/news/v1/source/list",
        params={
            "lang": "EN",
            "status": "ACTIVE",
            "api_key": API_KEY
        },
        headers={"Content-Type": "application/json; charset=UTF-8"}
    )
    response.raise_for_status()
    return response.json().get("Data", [])

def store_news_sources(source_list):
    db = SessionLocal()
    for src in source_list:
        existing = get_source_by_key(db, src["SOURCE_KEY"])
        source_dict = {
            "name": src["NAME"],
            "image_url": src["IMAGE_URL"],
            "url": src["URL"],
            "lang": src["LANG"],
            "source_type": src["SOURCE_TYPE"],
            "launch_date": safe_timestamp(src.get("LAUNCH_DATE")),
            "sort_order": src["SORT_ORDER"],
            "benchmark_score": src["BENCHMARK_SCORE"],
            "status": src["STATUS"],
            "last_updated_ts": safe_timestamp(src.get("LAST_UPDATED_TS")),
            "created_on": safe_timestamp(src.get("CREATED_ON")),
            "updated_on": safe_timestamp(src.get("UPDATED_ON")),
        }

        if existing:
            update_source(db, src["SOURCE_KEY"], source_dict)
        else:
            source_dict["id"] = src["ID"]
            source_dict["source_key"] = src["SOURCE_KEY"]
            create_source(db, source_dict)
    db.close()

if __name__ == "__main__":
    sources = fetch_news_sources()
    store_news_sources(sources)
