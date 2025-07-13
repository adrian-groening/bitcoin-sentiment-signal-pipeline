from sqlalchemy.orm import Session
from database import SessionLocal
from price import price
from article import article
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and align hourly price data and enriched article features."""
    db: Session = SessionLocal()

    # --- Load Price Data ---
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
    price_df['timestamp_hour'] = price_df['timestamp'].dt.floor('h')
    hourly_price = price_df.groupby('timestamp_hour').first().reset_index()

    # --- Load Article Data ---
    article_rows = db.query(article).filter(
        article.published_on.isnot(None),
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
        "keywords": a.keywords,
        "sentiment": a.sentiment,
        "source_id": a.source_id
    } for a in article_rows])

    article_df['published_on'] = pd.to_datetime(article_df['published_on'])
    article_df['published_on_hour'] = article_df['published_on'].dt.floor('h')

    # --- Enrich Metadata ---
    article_df['sentiment_encoded'] = article_df['sentiment'].map({"POSITIVE": 1, "NEGATIVE": 0})
    article_df['contains_bitcoin_keyword'] = article_df['keywords'].str.lower().fillna('').apply(
        lambda x: int(any(term in x for term in ['bitcoin', 'btc', 'btcusd']))
    )

    source_dummies = pd.get_dummies(article_df['source_id'], prefix='source').astype(int)
    source_dummies['published_on_hour'] = article_df['published_on_hour']
    source_encoded = source_dummies.groupby('published_on_hour').max().reset_index()

    sentiment_cols = [
        'title_positive_sentiment', 'title_neutral_sentiment', 'title_negative_sentiment',
        'body_positive_sentiment', 'body_neutral_sentiment', 'body_negative_sentiment'
    ]
    sentiment_agg = article_df.groupby('published_on_hour')[sentiment_cols].mean().reset_index()
    keyword_agg = article_df.groupby('published_on_hour')['contains_bitcoin_keyword'].max().reset_index()
    sentiment_flag_agg = article_df.groupby('published_on_hour')['sentiment_encoded'].mean().reset_index()

    article_agg = sentiment_agg \
        .merge(keyword_agg, on='published_on_hour', how='left') \
        .merge(sentiment_flag_agg, on='published_on_hour', how='left') \
        .merge(source_encoded, on='published_on_hour', how='left')

    merged_df = pd.merge(
        hourly_price,
        article_agg,
        how='left',
        left_on='timestamp_hour',
        right_on='published_on_hour'
    ).drop(columns=['published_on_hour'])

    # --- Fill NaNs ---
    merged_df[sentiment_cols] = merged_df[sentiment_cols].fillna(0)
    merged_df['contains_bitcoin_keyword'] = merged_df['contains_bitcoin_keyword'].fillna(0)
    merged_df['sentiment_encoded'] = merged_df['sentiment_encoded'].fillna(0)

    source_cols = [col for col in merged_df.columns if col.startswith("source_")]
    all_meta_cols = source_cols + ["contains_bitcoin_keyword", "sentiment_encoded"]
    merged_df[all_meta_cols] = merged_df[all_meta_cols].fillna(0).astype(int)

    #for col in all_meta_cols:
    #    merged_df[f"{col}_lag_1h"] = merged_df[col].shift(1)
    #    merged_df[f"{col}_rolling_3h"] = merged_df[col].rolling(window=3, min_periods=1).sum()

    #for lag in [2, 3]:
    #    merged_df[f"sentiment_encoded_lag_{lag}h"] = merged_df["sentiment_encoded"].shift(lag)

    rolling_6h_features = {}
    for col in ["contains_bitcoin_keyword", "sentiment_encoded"] + source_cols:
        rolling_6h_features[f"{col}_rolling_6h"] = merged_df[col].rolling(window=6, min_periods=1).mean()

    merged_df = pd.concat([merged_df, pd.DataFrame(rolling_6h_features)], axis=1)

    #lag_rolling_cols = [col for col in merged_df.columns if any(x in col for x in ['lag', 'rolling'])]
    #binary_cols = [col for col in lag_rolling_cols if 'lag' in col]
    #rolling_cols = [col for col in lag_rolling_cols if 'rolling' in col]

    #merged_df[binary_cols] = merged_df[binary_cols].fillna(0).astype(int)
    #merged_df[rolling_cols] = merged_df[rolling_cols].fillna(0).astype(float)

    # --- Sentiment Correct Labels ---
    merged_df['close_next_hour'] = merged_df['close'].shift(-1)
    merged_df['return_next_hour'] = (merged_df['close_next_hour'] - merged_df['close']) / merged_df['close']

    merged_df['title_total_sentiment'] = (merged_df['title_positive_sentiment'] - merged_df['title_negative_sentiment']) / (
        merged_df['title_positive_sentiment'] + merged_df['title_negative_sentiment'] + merged_df['title_neutral_sentiment'] + 1e-9
    )
    merged_df['body_total_sentiment'] = (merged_df['body_positive_sentiment'] - merged_df['body_negative_sentiment']) / (
        merged_df['body_positive_sentiment'] + merged_df['body_negative_sentiment'] + merged_df['body_neutral_sentiment'] + 1e-9
    )
    merged_df['total_sentiment'] = merged_df['title_total_sentiment'] + merged_df['body_total_sentiment']

    def sign_match(sentiment, ret):
        if pd.isna(ret) or ret == 0:
            return 0
        return int((sentiment > 0) == (ret > 0))

    # Batch creation of sentiment_correct columns to avoid fragmentation
    correct_cols = {}
    max_horizon = 5
    for h in range(1, max_horizon + 1):
        return_col = f'return_t_plus_{h}h'
        total_sentiment_col = f'sentiment_correct_total_sentiment_t+{h}h'
        sentiment_encoded_col = f'sentiment_correct_sentiment_encoded_t+{h}h'

        merged_df[return_col] = (merged_df['close'].shift(-h) - merged_df['close']) / merged_df['close']

        correct_cols[total_sentiment_col] = merged_df.apply(
            lambda row: sign_match(row['total_sentiment'], row[return_col]), axis=1
        )
        correct_cols[sentiment_encoded_col] = merged_df.apply(
            lambda row: sign_match(row['sentiment_encoded'], row[return_col]), axis=1
        )

    # Add the sentiment_correct columns all at once
    merged_df = pd.concat([merged_df, pd.DataFrame(correct_cols)], axis=1)

    # Add t+1 sentiment_correct columns as main targets for convenience
    merged_df['sentiment_correct_total_sentiment'] = correct_cols['sentiment_correct_total_sentiment_t+1h']
    merged_df['sentiment_correct_sentiment_encoded'] = correct_cols['sentiment_correct_sentiment_encoded_t+1h']

    # Cleanup temp columns
    merged_df.drop(columns=[f'return_t_plus_{h}h' for h in range(1, max_horizon + 1)] + ['close_next_hour'], inplace=True)

    # Debug prints to verify target column presence and null counts
    print("Columns in merged_df:", merged_df.columns.tolist())
    print("Nulls in sentiment_correct_total_sentiment:", merged_df['sentiment_correct_total_sentiment'].isnull().sum())

    db.close()
    return hourly_price, merged_df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df['close_open_return'] = (df['close'] - df['open']) / df['open']

    if 'body_positive_sentiment' in df.columns:
        df['rolling_body_positive_3h'] = df['body_positive_sentiment'].rolling(window=3, min_periods=1).mean()
    else:
        df['rolling_body_positive_3h'] = 0

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.time
    df = df.sort_values('timestamp')
    df['delta_close'] = df['close'].diff()

    sentiment_cols = [
        'title_positive_sentiment', 'title_neutral_sentiment', 'title_negative_sentiment',
        'body_positive_sentiment', 'body_neutral_sentiment', 'body_negative_sentiment'
    ]
    for col in sentiment_cols:
        df[col] = df[col].fillna(0)

    df['title_total_sentiment'] = (
        df['title_positive_sentiment'] - df['title_negative_sentiment']
    ) / (df['title_positive_sentiment'] + df['title_negative_sentiment'] + df['title_neutral_sentiment'] + 1e-9)

    df['body_total_sentiment'] = (
        df['body_positive_sentiment'] - df['body_negative_sentiment']
    ) / (df['body_positive_sentiment'] + df['body_negative_sentiment'] + df['body_neutral_sentiment'] + 1e-9)

    df['total_sentiment'] = df['title_total_sentiment'] + df['body_total_sentiment']
    return df

def train_model(df: pd.DataFrame):
    target_col = 'sentiment_correct_total_sentiment'

    # Exclude unwanted columns including datetime
    exclude_cols = [
        'timestamp', 'timestamp_hour', 'date', 'time',  # remove any datetime columns
        'sentiment_correct_total_sentiment',
        'sentiment_correct_sentiment_encoded',
        'sentiment_correct_total_sentiment_t+1h',
        'sentiment_correct_sentiment_encoded_t+1h',
        'sentiment_correct_total_sentiment_t+2h',
        'sentiment_correct_sentiment_encoded_t+2h',
        'sentiment_correct_total_sentiment_t+3h',
        'sentiment_correct_sentiment_encoded_t+3h',
        'sentiment_correct_total_sentiment_t+4h',
        'sentiment_correct_sentiment_encoded_t+4h',
        'sentiment_correct_total_sentiment_t+5h',
        'sentiment_correct_sentiment_encoded_t+5h',
        'open',
        'high',
        'low',
        'volume',
        'return_next_hour',
        'title_positive_sentiment',
        'title_neutral_sentiment',
        'title_negative_sentiment',
        'body_positive_sentiment',
        'body_neutral_sentiment',
        'body_negative_sentiment',
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    df_model = df.dropna(subset=feature_cols + [target_col])

    df.to_csv('data.csv')

    X = df_model[feature_cols]
    y = df_model[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nModel Performance:\n")
    print(classification_report(y_test, y_pred))

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\nTop 20 Features by Importance:\n")
    print(importance_df.head(20))

    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df["feature"].head(20)[::-1], importance_df["importance"].head(20)[::-1])
    plt.title("Top 20 Most Important Features")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
    print(df_model['sentiment_correct_total_sentiment'].value_counts())



if __name__ == "__main__":
    hourly_price, synthesized_df = load_data()
    enriched_df = add_features(synthesized_df)
    train_model(enriched_df)
