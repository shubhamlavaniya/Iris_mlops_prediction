# This script is developed to log predictions and errors in a database.
# It includes functions to initialize the database, log predictions, and log errors.
# The database is SQLite by default but can be configured via an environment variable.

from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from contextlib import contextmanager

Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    sepal_length = Column(Float, nullable=True)
    sepal_width = Column(Float, nullable=True)
    petal_length = Column(Float, nullable=True)
    petal_width = Column(Float, nullable=True)
    prediction = Column(String, nullable=True)
    model_version = Column(String, nullable=True)
    error = Column(String, nullable=True)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
default_db_path = os.path.join(project_root, 'logs', 'logs.db')

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{default_db_path}")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(engine)

@contextmanager
def get_db_session():
    """Provides a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

def log_prediction(data: dict, prediction: str, version: str):
    with get_db_session() as session:
        log = PredictionLog(
            sepal_length=data.get("sepal_length"),
            sepal_width=data.get("sepal_width"),
            petal_length=data.get("petal_length"),
            petal_width=data.get("petal_width"),
            prediction=prediction,
            model_version=version,
        )
        session.add(log)

def log_error(data: dict, error_message: str, version: str = None):
    with get_db_session() as session:
        log = PredictionLog(
            sepal_length=data.get("sepal_length"),
            sepal_width=data.get("sepal_width"),
            petal_length=data.get("petal_length"),
            petal_width=data.get("petal_width"),
            prediction=None,
            model_version=version,
            error=error_message
        )
        session.add(log)