from database import SessionLocal
from article_factory import search_articles_from_sources, store_articles, bitcoin_articles_30_days
from datetime import datetime, timedelta
import os
from sentiment_factory import update_article_sentiments
from price_factory import update_prices, truncate_price_table
from gradient_booster import load_data, add_features, train_model

truncate_price_table()
update_prices("cadli", "BTC-USD")
bitcoin_articles_30_days()
update_article_sentiments()

hourly_price, synthesized_df = load_data()
enriched_df = add_features(synthesized_df)
enriched_df.to_csv('data.csv')
train_model(enriched_df)