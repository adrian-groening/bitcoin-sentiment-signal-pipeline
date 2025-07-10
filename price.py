from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime
from source import Base

class price(Base):
    __tablename__ = 'price'

    id = Column(Integer, primary_key=True)
    unit = Column(String)
    timestamp = Column(DateTime, index=True)
    market = Column(String)
    instrument = Column(String)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    first_message_timestamp = Column(DateTime)
    last_message_timestamp = Column(DateTime)
    first_message_value = Column(Float)
    high_message_value = Column(Float)
    high_message_timestamp = Column(DateTime)
    low_message_value = Column(Float)
    low_message_timestamp = Column(DateTime)
    last_message_value = Column(Float)
    total_index_updates = Column(Integer)
    volume = Column(Float)
    quote_volume = Column(Float)
    volume_top_tier = Column(Float)
    quote_volume_top_tier = Column(Float)
    volume_direct = Column(Float)
    quote_volume_direct = Column(Float)
    volume_top_tier_direct = Column(Float)
    quote_volume_top_tier_direct = Column(Float)
    created_on = Column(DateTime, default=datetime.utcnow)