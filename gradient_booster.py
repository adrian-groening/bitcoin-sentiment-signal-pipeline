import pandas as pd
from sqlalchemy.orm import Session
from database import SessionLocal
from price import price
from article import article
from datetime import datetime

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load price and article data from the database into dataframes."""

    db: Session = SessionLocal()

    # Load price data for BTC-USD, resample to hourly, take first price in each hour
    price_rows = db.query(price).filter(price.instrument == "BTC-USD").order_by(price.timestamp).all()
    price_df = pd.DataFrame([{
        "timestamp": p.timestamp,
        "open": p.open,
        "high": p.high,
        "low": p.low,
        "close": p.close,
        "volume": p.volume,
    } for p in price_rows])

    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
    price_df = price_df.set_index('timestamp')
    hourly_price = price_df.resample('h').first().dropna().reset_index()

    # Load article data with sentiment scores and published_on timestamp
    article_rows = db.query(article).filter(
        article.published_on.isnot(None),
        # Filter only articles with sentiment scores (positive, neutral, or negative) present
        (
            (article.title_positive_sentiment.isnot(None)) |
            (article.title_neutral_sentiment.isnot(None)) |
            (article.title_negative_sentiment.isnot(None)) |
            (article.body_positive_sentiment.isnot(None)) |
            (article.body_neutral_sentiment.isnot(None)) |
            (article.body_negative_sentiment.isnot(None))
        )
    ).all()

    article_df = pd.DataFrame([{
        "published_on": a.published_on,
        "title_positive_sentiment": a.title_positive_sentiment or 0,
        "title_neutral_sentiment": a.title_neutral_sentiment or 0,
        "title_negative_sentiment": a.title_negative_sentiment or 0,
        "body_positive_sentiment": a.body_positive_sentiment or 0,
        "body_neutral_sentiment": a.body_neutral_sentiment or 0,
        "body_negative_sentiment": a.body_negative_sentiment or 0,
    } for a in article_rows])

    article_df['published_on'] = pd.to_datetime(article_df['published_on'])
    article_df['published_on_hour'] = article_df['published_on'].dt.floor('h')

    # Aggregate sentiment scores by hour (mean)
    sentiment_cols = [
        'title_positive_sentiment', 'title_neutral_sentiment', 'title_negative_sentiment',
        'body_positive_sentiment', 'body_neutral_sentiment', 'body_negative_sentiment'
    ]
    sentiment_agg = article_df.groupby('published_on_hour')[sentiment_cols].mean().reset_index()

    # Merge hourly price and hourly sentiment data
    merged_df = pd.merge(
        hourly_price,
        sentiment_agg,
        how='left',
        left_on='timestamp',
        right_on='published_on_hour'
    ).drop(columns=['published_on_hour'])

    db.close()
    return hourly_price, merged_df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add extra features to the dataframe for modeling."""

    # Example feature: percent change close to open
    df['close_open_return'] = (df['close'] - df['open']) / df['open']

    # Example feature: rolling mean of body positive sentiment over last 3 hours
    if 'body_positive_sentiment' in df.columns:
        df['rolling_body_positive_3h'] = df['body_positive_sentiment'].rolling(window=3, min_periods=1).mean()
    else:
        df['rolling_body_positive_3h'] = 0

    # Add other feature computations here...

    return df

if __name__ == "__main__":
    hourly_price, synthesized_df = load_data()
    enriched_df = add_features(synthesized_df)
    enriched_df.to_csv("gradient_booster_hourly_data.csv", index=False)
    print("Saved hourly synthesized data with features to gradient_booster_hourly_data.csv")
