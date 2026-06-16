import os

# Fix tokenizers parallelism deadlock on Python 3.9
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

# ── Multimodal Parsing Settings ──────────────────────────────────────
ENABLE_TABLE_EXTRACTION = True
SUPPORTED_FILE_TYPES = ["pdf", "csv", "json", "txt", "xlsx", "docx"]

# ── Hierarchical Chunking Settings ───────────────────────────────────
# Small child chunks for retrieval precision
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 80
# Large parent chunks for LLM context
PARENT_CHUNK_SIZE = 1500
PARENT_CHUNK_OVERLAP = 200
# Legacy flat chunking (kept for backward compat)
CHUNK_SIZE = 750
CHUNK_OVERLAP = 100

# ── Embedding Model Settings ────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Matches all-MiniLM-L6-v2 output dim

# Auto-detect if sentence-transformers is installed
# Set USE_DENSE_EMBEDDINGS=true/false in .env to override auto-detection
_use_dense_override = os.environ.get("USE_DENSE_EMBEDDINGS", "").lower()
if _use_dense_override == "true":
    USE_DENSE_EMBEDDINGS = True
elif _use_dense_override == "false":
    USE_DENSE_EMBEDDINGS = False
else:
    # Use importlib.util.find_spec to check without importing (avoids tokenizer deadlock)
    import importlib.util
    USE_DENSE_EMBEDDINGS = importlib.util.find_spec("sentence_transformers") is not None

# ── FAISS Vector Store Paths ────────────────────────────────────────
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index.bin")
FAISS_METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_metadata.json")
FAISS_HIERARCHY_PATH = os.path.join(VECTOR_STORE_DIR, "chunk_hierarchy.json")

# ── Reranker Settings ───────────────────────────────────────────────
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVAL_TOP_K = 20      # Candidates from FAISS
RERANKER_TOP_N = 5        # Final results after reranking
RERANKER_WEIGHT = 0.7     # Weight for reranker score in fusion
RETRIEVAL_WEIGHT = 0.3    # Weight for retrieval score in fusion

# ── Citation Enforcement Settings ───────────────────────────────────
CITATION_FORMAT = "[Source: Page {page}, Chunk {chunk}]"
MIN_CITATION_COVERAGE = 0.6   # Min ratio of cited claims / total claims
MIN_CITATION_ACCURACY = 0.8   # Min ratio of valid citations / total citations
MAX_CITATION_RETRIES = 1      # Max re-prompts if citation quality is poor

# ── Hallucination and Generation Parameters ─────────────────────────
LLM_CONFIDENCE_THRESHOLD = 0.70
MAX_RESPONSE_WORDS = 500

# ── Ragas Evaluation Settings ───────────────────────────────────────
EVAL_DATASET_PATH = os.path.join(DATA_DIR, "eval_dataset.json")
EVAL_RESULTS_PATH = os.path.join(DATA_DIR, "eval_results.json")
EVAL_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
EVAL_BATCH_SIZE = 5
