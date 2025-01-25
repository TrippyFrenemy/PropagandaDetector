import os
from dotenv import load_dotenv

load_dotenv()
# XXX = os.getenv('XXX')
FOREST_PATH = os.getenv("FOREST_PATH")
TFIDF_PATH = os.getenv("TFIDF_PATH")

MODEL_PATH = os.getenv("MODEL_PATH")
CASCADE_PATH = os.getenv("CASCADE_PATH")
IMPROVED_CASCADE_PATH = os.getenv("IMPROVED_CASCADE_PATH")

REDIS_PORT = os.environ.get("REDIS_PORT")
REDIS_HOST = os.environ.get("REDIS_HOST")