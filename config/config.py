"""
Configuration settings for the Civic Complaint System.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Database configuration
DATABASE_PATH = DATA_DIR / 'civic_complaints.db'

# Model configuration
MURIL_MODEL_NAME = 'google/muril-base-cased'
CATEGORY_MODEL_PATH = MODELS_DIR / 'muril_category_classifier'
URGENCY_MODEL_PATH = MODELS_DIR / 'xgboost_urgency_predictor.pkl'

# Feature dimensions
MURIL_EMBEDDING_DIM = 768
STRUCTURED_FEATURES_DIM = 8
TOTAL_FEATURE_DIM = MURIL_EMBEDDING_DIM + STRUCTURED_FEATURES_DIM  # 776

# Categories and urgency levels
CATEGORIES = ['Water Supply', 'Sanitation', 'Transportation']
URGENCY_LEVELS = ['Critical', 'High', 'Medium', 'Low']
LANGUAGES = ['English', 'Hindi', 'Hinglish']

# Dataset configuration
TOTAL_COMPLAINTS = 200
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Training configuration
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
RANDOM_SEED = 42

# Session configuration
SESSION_TIMEOUT_MINUTES = 30
PASSWORD_RESET_TOKEN_HOURS = 1
MAX_LOGIN_ATTEMPTS = 3

# Notification configuration (placeholders - configure in production)
EMAIL_HOST = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', '587'))
EMAIL_USER = os.getenv('EMAIL_USER', '')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')

# SMS configuration (optional - Twilio)
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', '')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', '')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER', '')

# System configuration
SYSTEM_NAME = "Explainable Multilingual Civic Complaint Resolution System"
SYSTEM_VERSION = "1.0.0"
HELPLINE_NUMBER = "1800-XXX-XXXX"
SUPPORT_EMAIL = "support@civiccomplaints.gov"

# Performance targets
TARGET_CATEGORY_ACCURACY = 0.94
TARGET_URGENCY_ACCURACY = 0.89
MAX_PROCESSING_TIME_SECONDS = 4
MAX_SHAP_TIME_SECONDS = 2

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
