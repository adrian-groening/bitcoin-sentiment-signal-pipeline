from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sqlalchemy.orm import Session
from database import SessionLocal
from article import article
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text: str) -> dict:
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text or "")

def update_article_sentiments():
    db: Session = SessionLocal()
    articles = db.query(article).filter(
        article.title_positive_sentiment == None,
        article.body_positive_sentiment == None
    ).all()

    for art in articles:
        title_scores = analyze_sentiment(art.title)
        body_scores = analyze_sentiment(art.body)

        art.title_positive_sentiment = title_scores['pos']
        art.title_neutral_sentiment = title_scores['neu']
        art.title_negative_sentiment = title_scores['neg']

        art.body_positive_sentiment = body_scores['pos']
        art.body_neutral_sentiment = body_scores['neu']
        art.body_negative_sentiment = body_scores['neg']

    db.commit()
    db.close()
