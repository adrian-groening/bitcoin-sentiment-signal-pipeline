from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from source import Base
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)  # no connect_args here for Postgres

SessionLocal = sessionmaker(bind=engine)

# Create tables (run once or at startup)
Base.metadata.create_all(bind=engine)
