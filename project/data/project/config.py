# ==========================================
# CONFIGURATION FILE - DIELLA AI
# ==========================================

import os

from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Data
DATA_PATH = "diella_speeches_clean.csv"

# Models
VECTOR_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


USE_GROQ = True
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# Q&A Settings
MAX_QA_DOCS = 8
MAX_CHARS_CONTEXT = 3500

# Sentiment Thresholds
SENTIMENT_POSITIVE_THRESHOLD = 0.05
SENTIMENT_NEGATIVE_THRESHOLD = -0.05

# Topic Modeling
NUM_TOPICS = 5
NUM_TOP_WORDS = 10
TFIDF_MAX_FEATURES = 5000
TFIDF_MIN_DF = 1

# Page Config
PAGE_TITLE = "DIELLA AI"
PAGE_LAYOUT = "wide"

# Colors
COLOR_POSITIVE = "green"
COLOR_NEGATIVE = "red"
COLOR_NEUTRAL = "blue"

# Header styling
HEADER_BG_COLOR = "#0f172a"
HEADER_GRADIENT_START = "#6366f1"
HEADER_GRADIENT_END = "#ec4899"
SUN_COLOR = "#facc15"