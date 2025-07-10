from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from datetime import datetime
from source import Base, source 
from sqlalchemy.orm import relationship




class article(Base):
    __tablename__ = 'article'

    id = Column(Integer, primary_key=True)
    guid = Column(String, unique=True, nullable=False)
    published_on = Column(DateTime)
    image_url = Column(String)
    title = Column(String)
    authors = Column(String)
    url = Column(String)
    source_id = Column(Integer, ForeignKey("source.id"), nullable=False)
    source = relationship("source")
    body = Column(Text)
    keywords = Column(String)
    lang = Column(String)
    upvotes = Column(Integer)
    downvotes = Column(Integer)
    score = Column(Integer)
    sentiment = Column(String)
    status = Column(String)
    created_on = Column(DateTime)
    updated_on = Column(DateTime)

    # Sentiment scores for title
    title_positive_sentiment = Column(Float)
    title_neutral_sentiment = Column(Float)
    title_negative_sentiment = Column(Float)

    # Sentiment scores for body
    body_positive_sentiment = Column(Float)
    body_neutral_sentiment = Column(Float)
    body_negative_sentiment = Column(Float)

def create_article(db: Session, article_data: dict):
    new_article = article(**article_data)
    db.add(new_article)
    db.commit()

def get_article_by_guid(db: Session, guid: str):
    return db.query(article).filter(article.guid == guid).first()

def update_article(db: Session, guid: str, updated_data: dict):
    existing = get_article_by_guid(db, guid)
    if existing:
        for key, value in updated_data.items():
            if hasattr(existing, key):
                setattr(existing, key, value)
        db.commit()