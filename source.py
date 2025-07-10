from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from sqlalchemy.orm import Session


Base = declarative_base()

class source(Base):
    __tablename__ = 'source'

    id = Column(Integer, primary_key=True)
    source_key = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    image_url = Column(String)
    url = Column(String, nullable=False)
    lang = Column(String)
    source_type = Column(String)
    launch_date = Column(DateTime)
    sort_order = Column(Integer)
    benchmark_score = Column(Integer)
    status = Column(String)
    last_updated_ts = Column(DateTime)
    created_on = Column(DateTime)
    updated_on = Column(DateTime)

def create_source(db: Session, source_data: dict):
    new_source = source(**source_data)
    db.add(new_source)
    db.commit()

def get_source_by_key(db: Session, source_key: str):
    return db.query(source).filter(source.source_key == source_key).first()

def update_source(db: Session, source_key: str, updated_data: dict):
    existing = get_source_by_key(db, source_key)
    if existing:
        for key, value in updated_data.items():
            if hasattr(existing, key):
                setattr(existing, key, value)
        db.commit()