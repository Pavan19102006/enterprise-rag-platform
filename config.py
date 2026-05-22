import os

# Base directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")

# Manually load environment variables from base .env file
env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

# Relational Database path
DB_PATH = os.path.join(DATA_DIR, "secure_db.sqlite")

# Security Configurations
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 2

# Access Control Roles
ROLES = ["Executive", "HR", "Finance", "Compliance", "Engineering", "Intern"]

# Department Definitions
DEPARTMENTS = ["Executive", "Human Resources", "Finance", "Compliance", "Engineering", "Operations"]

# Role access rights (what role can view which classification)
CLASSIFICATION_ACCESS = {
    "Intern": ["Public"],
    "Engineering": ["Public", "Engineering Confidential"],
    "HR": ["Public", "HR Confidential"],
    "Finance": ["Public", "Finance Confidential"],
    "Compliance": ["Public", "HR Confidential", "Finance Confidential", "Engineering Confidential", "Compliance Audit"],
    "Executive": ["Public", "HR Confidential", "Finance Confidential", "Engineering Confidential", "Compliance Audit", "Highly Restricted"]
}

# Embedding & Search settings
CHUNK_SIZE = 750
CHUNK_OVERLAP = 100

# Hallucination and generation parameters
LLM_CONFIDENCE_THRESHOLD = 0.70
MAX_RESPONSE_WORDS = 300
