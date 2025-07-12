import requests
import os
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from database import SessionLocal
from price import price

load_dotenv()
API_KEY = os.getenv("COINDESK_API_KEY")

def safe_fromtimestamp(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None

def fetch_hourly_price_data(market: str, instrument: str, to_ts: int, limit: int = 720):
    response = requests.get(
        'https://data-api.coindesk.com/index/cc/v1/historical/hours',
        params={
            "market": market,
            "instrument": instrument,
            "aggregate": 1,
            "fill": "true",
            "apply_mapping": "true",
            "response_format": "JSON",
            "to_ts": to_ts,
            "limit": limit,
            "api_key": API_KEY
        },
        headers={"Content-type": "application/json; charset=UTF-8"}
    )
    response.raise_for_status()
    return response.json().get("Data", [])

def store_prices(db: Session, prices: list):
    count = 0
    for p in prices:
        timestamp = safe_fromtimestamp(p.get("TIMESTAMP"))
        instrument = p.get("INSTRUMENT")
        if not timestamp or not instrument:
            continue

        exists = db.query(price).filter(price.timestamp == timestamp, price.instrument == instrument).first()
        if exists:
            continue

        db.add(price(
            unit=p.get("UNIT"),
            timestamp=timestamp,
            market=p.get("MARKET"),
            instrument=instrument,
            open=p.get("OPEN"),
            high=p.get("HIGH"),
            low=p.get("LOW"),
            close=p.get("CLOSE"),
            first_message_timestamp=safe_fromtimestamp(p.get("FIRST_MESSAGE_TIMESTAMP")),
            last_message_timestamp=safe_fromtimestamp(p.get("LAST_MESSAGE_TIMESTAMP")),
            first_message_value=p.get("FIRST_MESSAGE_VALUE"),
            high_message_value=p.get("HIGH_MESSAGE_VALUE"),
            high_message_timestamp=safe_fromtimestamp(p.get("HIGH_MESSAGE_TIMESTAMP")),
            low_message_value=p.get("LOW_MESSAGE_VALUE"),
            low_message_timestamp=safe_fromtimestamp(p.get("LOW_MESSAGE_TIMESTAMP")),
            last_message_value=p.get("LAST_MESSAGE_VALUE"),
            total_index_updates=p.get("TOTAL_INDEX_UPDATES"),
            volume=p.get("VOLUME"),
            quote_volume=p.get("QUOTE_VOLUME"),
            volume_top_tier=p.get("VOLUME_TOP_TIER"),
            quote_volume_top_tier=p.get("QUOTE_VOLUME_TOP_TIER"),
            volume_direct=p.get("VOLUME_DIRECT"),
            quote_volume_direct=p.get("QUOTE_VOLUME_DIRECT"),
            volume_top_tier_direct=p.get("VOLUME_TOP_TIER_DIRECT"),
            quote_volume_top_tier_direct=p.get("QUOTE_VOLUME_TOP_TIER_DIRECT"),
        ))
        count += 1
    db.commit()
    print(f"Stored {count} new hourly price entries.")

def get_latest_timestamp_in_db(db: Session, instrument: str):
    latest = db.query(price).filter(price.instrument == instrument).order_by(price.timestamp.desc()).first()
    if latest and latest.timestamp:
        return int(latest.timestamp.timestamp())
    return None

def fetch_and_store_backward(db: Session, market: str, instrument: str, from_ts: int):
    print("Starting backward fetch...")
    to_ts = int(datetime.now(timezone.utc).timestamp())

    while to_ts > from_ts:
        print(f"Fetching hourly prices up to timestamp: {to_ts}")
        prices = fetch_hourly_price_data(market, instrument, to_ts=to_ts)
        if not prices:
            print("No more data returned.")
            break
        store_prices(db, prices)
        oldest_ts = min(p["TIMESTAMP"] for p in prices)
        if oldest_ts <= from_ts:
            print("Reached desired start timestamp.")
            break
        to_ts = oldest_ts - 3600  # step back one hour

def fetch_and_store_forward(db: Session, market: str, instrument: str, from_ts: int):
    print("Starting forward fetch...")
    to_ts = int(datetime.now(timezone.utc).timestamp())
    print(f"Fetching hourly prices up to timestamp: {to_ts}")
    prices = fetch_hourly_price_data(market, instrument, to_ts=to_ts, limit=720)
    new_prices = [p for p in prices if p["TIMESTAMP"] > from_ts]
    store_prices(db, new_prices)
    print("Forward fetch complete.")

def update_prices(market: str, instrument: str):
    db = SessionLocal()
    latest_ts = get_latest_timestamp_in_db(db, instrument)
    one_month_ago_ts = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp())

    if latest_ts is None:
        print("No existing price data; fetching last 30 days.")
        fetch_and_store_backward(db, market, instrument, one_month_ago_ts)
    else:
        print(f"Latest timestamp in DB: {latest_ts}")
        fetch_and_store_forward(db, market, instrument, latest_ts)

def truncate_price_table():
    db = SessionLocal()
    try:
        db.execute("TRUNCATE TABLE price RESTART IDENTITY CASCADE;")
        db.commit()
    except Exception as e:
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    update_prices("cadli", "BTC-USD")
