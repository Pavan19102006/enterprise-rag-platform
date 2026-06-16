"""
Ingestion Pipeline
==================
Advanced document ingestion with:
  - Multimodal parsing (PDF tables, structured text)
  - Hierarchical Small-to-Big chunking
  - Dense sentence-transformer embeddings
  - FAISS vector store for fast ANN search
  - Rich mock financial/legal document generation
"""

import os
import json
import csv
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple

from config import (
    RAW_DATA_DIR, DATA_DIR, VECTOR_STORE_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION,
    FAISS_INDEX_PATH, FAISS_METADATA_PATH, FAISS_HIERARCHY_PATH,
    USE_DENSE_EMBEDDINGS,
)
from core.database import get_db_connection
from core.multimodal_parser import parse_document, ParsedDocument
from core.hierarchical_chunker import HierarchicalChunker, HierarchicalChunk

logger = logging.getLogger(__name__)

# Ensure folders exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# ── Embedding model management ──────────────────────────────────────
_embedding_model = None
_embedding_available = None
_tfidf_vectorizer = None  # Persisted TF-IDF vectorizer for query-time transforms


def _load_embedding_model():
    """Load the sentence-transformer embedding model (only if USE_DENSE_EMBEDDINGS is True)."""
    global _embedding_model, _embedding_available
    if _embedding_available is not None:
        return _embedding_available

    if not USE_DENSE_EMBEDDINGS:
        logger.info("Dense embeddings disabled (USE_DENSE_EMBEDDINGS=false). Using TF-IDF+FAISS.")
        _embedding_available = False
        return False

    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        _embedding_available = True
        logger.info("Sentence-transformer embedding model loaded successfully.")
        return True
    except Exception as e:
        logger.warning(f"Failed to load embedding model: {e}. Using TF-IDF fallback.")
        _embedding_available = False
        return False


def compute_embeddings(texts: List[str]) -> np.ndarray:
    """Compute embeddings for a list of texts. Uses sentence-transformers if available, else TF-IDF."""
    if _load_embedding_model() and _embedding_model is not None:
        embeddings = _embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)
    else:
        return _compute_tfidf_embeddings(texts)


def _compute_tfidf_embeddings(texts: List[str]) -> np.ndarray:
    """TF-IDF embeddings with FAISS — works on all platforms without ML model downloads."""
    global _tfidf_vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Fit a new vectorizer on the corpus
    _tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=EMBEDDING_DIMENSION,
        ngram_range=(1, 2),  # Unigrams + bigrams for better semantic matching
        sublinear_tf=True,   # Apply sublinear TF scaling (1 + log(tf))
    )
    matrix = _tfidf_vectorizer.fit_transform(texts).toarray().astype(np.float32)

    # Pad to match expected dimension if needed
    if matrix.shape[1] < EMBEDDING_DIMENSION:
        padding = np.zeros((matrix.shape[0], EMBEDDING_DIMENSION - matrix.shape[1]), dtype=np.float32)
        matrix = np.hstack([matrix, padding])

    # L2 normalize for cosine similarity via inner product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    return matrix


def compute_query_embedding(query: str) -> np.ndarray:
    """Compute embedding for a single query at search time."""
    if _embedding_available and _embedding_model is not None:
        emb = _embedding_model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        return np.array(emb, dtype=np.float32)
    elif _tfidf_vectorizer is not None:
        matrix = _tfidf_vectorizer.transform([query]).toarray().astype(np.float32)
        if matrix.shape[1] < EMBEDDING_DIMENSION:
            padding = np.zeros((1, EMBEDDING_DIMENSION - matrix.shape[1]), dtype=np.float32)
            matrix = np.hstack([matrix, padding])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        return matrix
    else:
        # Emergency fallback: return zero vector (will be re-indexed on search)
        return np.zeros((1, EMBEDDING_DIMENSION), dtype=np.float32)


# ── FAISS-based Vector Database ─────────────────────────────────────

class FAISSVectorDB:
    """
    Production vector database using FAISS for approximate nearest neighbor search
    with sentence-transformer dense embeddings.
    """

    def __init__(self):
        self.chunks: List[Dict] = []         # [{id, text, metadata}, ...]
        self.parent_chunks: List[Dict] = []  # Parent chunks for Small-to-Big
        self.index = None                     # FAISS index
        self._faiss_available = None
        self._chunker = HierarchicalChunker()

    def _ensure_faiss(self):
        """Check if FAISS is available."""
        if self._faiss_available is not None:
            return self._faiss_available
        try:
            import faiss
            self._faiss_available = True
            return True
        except ImportError:
            logger.warning("faiss-cpu not installed. Using numpy cosine similarity fallback.")
            self._faiss_available = False
            return False

    def add_chunks(self, new_child_chunks: List[Dict], new_parent_chunks: List[Dict] = None):
        """Add chunks to the database."""
        self.chunks.extend(new_child_chunks)
        if new_parent_chunks:
            self.parent_chunks.extend(new_parent_chunks)

    def rebuild_index(self):
        """Build/rebuild the FAISS index from all chunks."""
        if not self.chunks:
            logger.warning("No chunks to index.")
            return

        texts = [c["text"] for c in self.chunks]
        logger.info(f"Computing embeddings for {len(texts)} chunks...")
        embeddings = compute_embeddings(texts)

        if self._ensure_faiss():
            import faiss
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # Inner product (works with normalized vectors)
            self.index.add(embeddings)
            logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")
        else:
            # Store embeddings for numpy fallback
            self._embeddings = embeddings
            logger.info(f"Numpy fallback index built: {len(embeddings)} vectors")

    def similarity_search(
        self,
        query_text: str,
        allowed_classifications: List[str],
        top_k: int = 20,
    ) -> List[Dict]:
        """
        Search for similar chunks with RBAC filtering.
        Returns top-k results filtered by allowed classifications.
        """
        if not self.chunks:
            return []

        query_embedding = compute_query_embedding(query_text)

        if self._ensure_faiss() and self.index is not None:
            # Search more candidates than top_k to account for RBAC filtering
            search_k = min(top_k * 4, self.index.ntotal)
            scores, indices = self.index.search(query_embedding, search_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # Numpy fallback
            if not hasattr(self, '_embeddings') or len(self._embeddings) == 0:
                return []
            scores = np.dot(self._embeddings, query_embedding.T).flatten()
            indices = np.argsort(scores)[::-1][:top_k * 4]
            scores = scores[indices]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]
            classification = chunk["metadata"].get("data_classification", "Public")

            # RBAC filter
            if classification not in allowed_classifications:
                continue

            results.append({
                "score": float(score),
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "chunk_id": chunk.get("id", ""),
                "parent_id": chunk["metadata"].get("parent_chunk_id", ""),
            })

            if len(results) >= top_k:
                break

        return results

    def get_parent_text(self, parent_id: str) -> Optional[str]:
        """Get parent chunk text by ID for Small-to-Big context expansion."""
        for p in self.parent_chunks:
            if p.get("id") == parent_id:
                return p["text"]
        return None

    def save(self):
        """Persist FAISS index and metadata to disk."""
        # Save metadata
        metadata = {
            "chunks": self.chunks,
            "parent_chunks": self.parent_chunks,
        }
        with open(FAISS_METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save FAISS index
        if self._ensure_faiss() and self.index is not None:
            import faiss
            faiss.write_index(self.index, FAISS_INDEX_PATH)
            logger.info(f"FAISS index saved to {FAISS_INDEX_PATH}")
        elif hasattr(self, '_embeddings'):
            # Save numpy embeddings as fallback
            np.save(FAISS_INDEX_PATH.replace('.bin', '.npy'), self._embeddings)

        # Save hierarchy
        hierarchy_data = self._chunker.export_hierarchy()
        with open(FAISS_HIERARCHY_PATH, "w") as f:
            json.dump(hierarchy_data, f, indent=2)

        logger.info("Vector database saved successfully.")

    def load(self) -> bool:
        """Load FAISS index and metadata from disk."""
        if not os.path.exists(FAISS_METADATA_PATH):
            # Try legacy format
            legacy_path = os.path.join(VECTOR_STORE_DIR, "vector_db.json")
            if os.path.exists(legacy_path):
                return self._load_legacy(legacy_path)
            return False

        try:
            with open(FAISS_METADATA_PATH, "r") as f:
                metadata = json.load(f)
            self.chunks = metadata.get("chunks", [])
            self.parent_chunks = metadata.get("parent_chunks", [])

            if self._ensure_faiss() and os.path.exists(FAISS_INDEX_PATH):
                import faiss
                self.index = faiss.read_index(FAISS_INDEX_PATH)
                logger.info(f"FAISS index loaded: {self.index.ntotal} vectors")
            else:
                npy_path = FAISS_INDEX_PATH.replace('.bin', '.npy')
                if os.path.exists(npy_path):
                    self._embeddings = np.load(npy_path)
                else:
                    # Rebuild from chunks
                    self.rebuild_index()

            # Load hierarchy
            if os.path.exists(FAISS_HIERARCHY_PATH):
                with open(FAISS_HIERARCHY_PATH, "r") as f:
                    hierarchy_data = json.load(f)
                self._chunker.import_hierarchy(hierarchy_data)

            return True
        except Exception as e:
            logger.error(f"Failed to load vector database: {e}")
            return False

    def _load_legacy(self, legacy_path: str) -> bool:
        """Load from legacy TF-IDF vector_db.json format and migrate."""
        try:
            with open(legacy_path, "r") as f:
                data = json.load(f)
            self.chunks = data.get("chunks", [])
            # Add missing fields to legacy chunks
            for chunk in self.chunks:
                if "metadata" not in chunk:
                    chunk["metadata"] = {}
                chunk["metadata"].setdefault("page_number", 1)
                chunk["metadata"].setdefault("section_title", "")
                chunk["metadata"].setdefault("parent_chunk_id", "")
            self.rebuild_index()
            # Save in new format
            self.save()
            logger.info("Migrated legacy vector database to FAISS format.")
            return True
        except Exception as e:
            logger.error(f"Failed to load legacy vector DB: {e}")
            return False


# ── Global Vector DB Instance ───────────────────────────────────────
vector_db = FAISSVectorDB()
chunker = HierarchicalChunker()


# ── Document Parsing Helpers (kept for backward compat) ─────────────

def parse_pdf(file_path: str) -> str:
    """Extract text from PDF (backward compatible wrapper)."""
    doc = parse_document(file_path)
    return doc.get_full_text()


def parse_csv(file_path: str) -> str:
    """Extract text from CSV (backward compatible wrapper)."""
    doc = parse_document(file_path)
    return doc.get_full_text()


def parse_json(file_path: str) -> str:
    """Extract text from JSON (backward compatible wrapper)."""
    doc = parse_document(file_path)
    return doc.get_full_text()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Legacy flat chunking (backward compatible)."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += (chunk_size - overlap)
        if start >= text_len:
            break
    return chunks


def register_document(doc_id, filename, file_type, allowed_roles, classification, checksum):
    """Register document into SQLite Document Metadata database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM document_metadata WHERE id = ?;", (doc_id,))
    cursor.execute("""
    INSERT INTO document_metadata (id, filename, file_type, allowed_roles, data_classification, checksum)
    VALUES (?, ?, ?, ?, ?, ?);
    """, (doc_id, filename, file_type, allowed_roles, classification, checksum))
    conn.commit()
    conn.close()


# ── Mock Document Generators ────────────────────────────────────────

def generate_mock_pdf_hr():
    """Generate a simulated HR Policy PDF using reportlab."""
    pdf_path = os.path.join(RAW_DATA_DIR, "hr_policy.pdf")
    if os.path.exists(pdf_path):
        return pdf_path

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        # Write as text fallback
        with open(pdf_path.replace('.pdf', '.txt'), 'w') as f:
            f.write(_get_hr_policy_text())
        return pdf_path.replace('.pdf', '.txt')

    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "VERTEX CORP - CONFIDENTIAL HUMAN RESOURCES POLICY")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, "Classification: HR Confidential | Target Audience: HR Department, Executives")
    c.line(50, height - 80, width - 50, height - 80)

    y = height - 110
    for line in _get_hr_policy_lines():
        if y < 80:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 10)
        if line.startswith("SECTION"):
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, line)
            c.setFont("Helvetica", 10)
            y -= 20
        else:
            c.drawString(50, y, line)
            y -= 15

    c.save()
    return pdf_path


def generate_mock_pdf_tech():
    """Generate a simulated Technical Specification PDF."""
    pdf_path = os.path.join(RAW_DATA_DIR, "tech_spec.pdf")
    if os.path.exists(pdf_path):
        return pdf_path

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        with open(pdf_path.replace('.pdf', '.txt'), 'w') as f:
            f.write(_get_tech_spec_text())
        return pdf_path.replace('.pdf', '.txt')

    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "PROJECT ALPHA: CLOUD ARCHITECTURE SPECIFICATION")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, "Classification: Engineering Confidential | Target Audience: Engineers, Executives")
    c.line(50, height - 80, width - 50, height - 80)

    y = height - 110
    for line in _get_tech_spec_lines():
        if y < 80:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 10)
        if not line:
            y -= 15
            continue
        if line.isupper() or (len(line) > 1 and line[0].isdigit() and line[1] == "."):
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, line)
            c.setFont("Helvetica", 10)
            y -= 20
        else:
            c.drawString(50, y, line)
            y -= 15

    c.save()
    return pdf_path


def generate_mock_pdf_financial_report():
    """Generate a simulated 10-K style financial report with tables."""
    pdf_path = os.path.join(RAW_DATA_DIR, "financial_report_2025.pdf")
    if os.path.exists(pdf_path):
        return pdf_path

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors
    except ImportError:
        with open(pdf_path.replace('.pdf', '.txt'), 'w') as f:
            f.write(_get_financial_report_text())
        return pdf_path.replace('.pdf', '.txt')

    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    # Page 1: Cover
    c.setFont("Helvetica-Bold", 22)
    c.drawString(50, height - 80, "VERTEX CORPORATION")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 110, "Annual Financial Report — Fiscal Year 2025")
    c.setFont("Helvetica", 11)
    c.drawString(50, height - 140, "Classification: Finance Confidential | For Authorized Personnel Only")
    c.drawString(50, height - 160, "Prepared by: Office of the CFO | Date: March 15, 2026")
    c.line(50, height - 175, width - 50, height - 175)

    y = height - 210
    report_lines = [
        "SECTION 1: EXECUTIVE SUMMARY",
        "",
        "Vertex Corporation achieved record revenue of $487.3 million in FY2025, representing",
        "a 23.4% year-over-year increase from $394.9 million in FY2024. Net income grew by",
        "31.2% to $72.8 million, driven by strong performance in Cloud Services and AI Solutions.",
        "",
        "Key Financial Highlights:",
        "- Total Revenue: $487.3M (up 23.4% YoY)",
        "- Gross Profit: $298.2M (gross margin: 61.2%)",
        "- Operating Income: $98.4M (operating margin: 20.2%)",
        "- Net Income: $72.8M (net margin: 14.9%)",
        "- Free Cash Flow: $84.1M",
        "- Total Assets: $1.23B",
        "- Return on Equity: 18.7%",
        "",
        "SECTION 2: REVENUE BREAKDOWN BY SEGMENT",
        "",
        "Revenue by Business Segment (in millions USD):",
        "",
        "| Segment                | FY2025    | FY2024    | Growth   |",
        "|------------------------|-----------|-----------|----------|",
        "| Cloud Services         | $198.4    | $152.1    | 30.4%    |",
        "| AI & Analytics         | $124.7    | $89.3     | 39.6%    |",
        "| Enterprise Software    | $98.2     | $93.5     | 5.0%     |",
        "| Professional Services  | $42.8     | $38.7     | 10.6%    |",
        "| Data Infrastructure    | $23.2     | $21.3     | 8.9%     |",
        "| Total                  | $487.3    | $394.9    | 23.4%    |",
        "",
        "SECTION 3: QUARTERLY PERFORMANCE",
        "",
        "Quarterly Revenue & Profitability (FY2025):",
        "",
        "| Quarter | Revenue ($M) | Net Profit ($M) | Margin (%) |",
        "|---------|-------------|-----------------|------------|",
        "| Q1 2025 | $108.2      | $15.4           | 14.2%      |",
        "| Q2 2025 | $115.8      | $16.9           | 14.6%      |",
        "| Q3 2025 | $127.4      | $19.8           | 15.5%      |",
        "| Q4 2025 | $135.9      | $20.7           | 15.2%      |",
        "",
        "SECTION 4: BALANCE SHEET SUMMARY",
        "",
        "Consolidated Balance Sheet as of December 31, 2025:",
        "",
        "| Item                        | Amount ($M) |",
        "|-----------------------------|-------------|",
        "| Cash & Equivalents          | $234.7      |",
        "| Accounts Receivable         | $89.3       |",
        "| Total Current Assets        | $412.8      |",
        "| Property & Equipment (net)  | $187.4      |",
        "| Goodwill & Intangibles      | $398.2      |",
        "| Total Assets                | $1,231.6    |",
        "| Total Current Liabilities   | $198.4      |",
        "| Long-Term Debt              | $245.0      |",
        "| Total Stockholders Equity   | $623.8      |",
        "",
        "SECTION 5: RISK FACTORS",
        "",
        "Material risk factors identified for FY2026 include:",
        "1. Concentration Risk: Cloud Services represents 40.7% of total revenue.",
        "2. Regulatory Risk: Pending EU AI Act compliance requirements may increase costs by $12-15M.",
        "3. Competition Risk: Three new market entrants in AI Analytics vertical.",
        "4. Foreign Exchange Risk: 28% of revenue is denominated in non-USD currencies.",
        "5. Talent Risk: Attrition rate in engineering increased to 14.2% (from 11.8% in FY2024).",
        "",
        "SECTION 6: LEGAL PROCEEDINGS",
        "",
        "Vertex Corporation is currently party to the following material legal proceedings:",
        "- DataShield LLC v. Vertex Corp (Case No. 2025-CV-8834): Patent infringement claim",
        "  related to data encryption algorithms. Potential exposure: $15-25M. Trial date: Sept 2026.",
        "- SEC Investigation (File No. HQ-2025-00442): Routine examination of insider trading",
        "  compliance. No charges filed. Expected resolution: Q2 2026.",
        "- Employment Class Action (Docket 2025-LA-1192): Wage & hour claims in California.",
        "  Settlement reserve established: $4.2M.",
    ]

    for line in report_lines:
        if y < 80:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 10)
        if not line:
            y -= 10
            continue
        if line.startswith("SECTION"):
            c.setFont("Helvetica-Bold", 13)
            c.drawString(50, y, line)
            c.setFont("Helvetica", 10)
            y -= 22
        elif line.startswith("|"):
            c.setFont("Courier", 8)
            c.drawString(50, y, line)
            c.setFont("Helvetica", 10)
            y -= 13
        elif line.startswith("- ") or line.startswith("  "):
            c.drawString(65, y, line)
            y -= 14
        else:
            c.drawString(50, y, line)
            y -= 14

    c.save()
    return pdf_path


def generate_mock_legal_contract():
    """Generate a simulated legal services contract."""
    pdf_path = os.path.join(RAW_DATA_DIR, "legal_services_agreement.pdf")
    if os.path.exists(pdf_path):
        return pdf_path

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        with open(pdf_path.replace('.pdf', '.txt'), 'w') as f:
            f.write(_get_legal_contract_text())
        return pdf_path.replace('.pdf', '.txt')

    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 60, "MASTER SERVICES AGREEMENT")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 80, "Classification: Highly Restricted | Legal Department Only")
    c.drawString(50, height - 95, "Effective Date: January 1, 2026 | Contract ID: MSA-2026-001")
    c.line(50, height - 108, width - 50, height - 108)

    y = height - 135
    for line in _get_legal_contract_lines():
        if y < 80:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 10)
        if not line:
            y -= 10
            continue
        if line.startswith("Article") or line.startswith("ARTICLE"):
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, line)
            c.setFont("Helvetica", 10)
            y -= 20
        elif line.startswith("  "):
            c.drawString(65, y, line)
            y -= 14
        else:
            c.drawString(50, y, line)
            y -= 14

    c.save()
    return pdf_path


def generate_mock_csv_finance():
    """Generate a simulated CSV ledger for Finance."""
    csv_path = os.path.join(RAW_DATA_DIR, "finance.csv")
    if os.path.exists(csv_path):
        return csv_path

    headers = ["TransactionID", "Date", "Department", "Category", "ExpenseUSD", "ApprovedBy", "Description"]
    rows = [
        ["TXN-1001", "2026-01-15", "Engineering", "Cloud Infrastructure", "45230.50", "fred", "AWS January cloud hosting ledger costs"],
        ["TXN-1002", "2026-01-20", "Human Resources", "Recruitment Platforms", "12500.00", "helen", "LinkedIn Recruiter annual seat renewal"],
        ["TXN-1003", "2026-02-05", "Finance", "Auditing Services", "35000.00", "elena", "Q1 External accounting audit downpayment"],
        ["TXN-1004", "2026-02-12", "Operations", "Office Lease", "82000.00", "elena", "Chicago central headquarters monthly rental"],
        ["TXN-1005", "2026-02-28", "Engineering", "Hardware Upgrades", "24100.80", "fred", "High-performance developer laptop batches"],
        ["TXN-1006", "2026-03-05", "Operations", "Utility Services", "9800.00", "fred", "Corporate high-speed fiber internet backup lines"],
    ]

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    return csv_path


def generate_mock_json_audits():
    """Generate a simulated JSON file for Compliance logs."""
    json_path = os.path.join(RAW_DATA_DIR, "audit_logs.json")
    if os.path.exists(json_path):
        return json_path

    records = [
        {
            "audit_id": "AUD-2026-001",
            "scope": "SOC2 Type II Readiness Audit",
            "findings": "All core access control policies verified. 9 out of 10 controls active. Identified minor logging gaps in testing databases.",
            "status": "In Progress",
            "lead_auditor": "charlie",
            "remediation_date": "2026-08-30"
        },
        {
            "audit_id": "AUD-2026-002",
            "scope": "GDPR Compliance Review",
            "findings": "Personal data mappings documented. User deletion API is operational. Encryption keys meet standard criteria. No leakage detected.",
            "status": "Passed",
            "lead_auditor": "charlie",
            "remediation_date": "N/A"
        },
        {
            "audit_id": "AUD-2026-003",
            "scope": "HIPAA Privacy Audit",
            "findings": "Determined that Vertex Corp core cloud servers do not store Protected Health Information (PHI). System scope declared non-HIPAA.",
            "status": "Completed",
            "lead_auditor": "elena",
            "remediation_date": "N/A"
        },
    ]

    with open(json_path, "w") as f:
        json.dump(records, f, indent=4)

    return json_path


def _ingest_document(
    file_path: str,
    doc_id: str,
    classification: str,
    allowed_roles: List[str],
    all_child_chunks: List[Dict],
    all_parent_chunks: List[Dict],
):
    """Ingest a single document with multimodal parsing and hierarchical chunking."""
    filename = os.path.basename(file_path)
    file_ext = filename.split(".")[-1].upper()
    roles_str = ",".join(allowed_roles)

    # Parse with multimodal parser
    parsed_doc = parse_document(file_path)

    # Hierarchical chunking
    child_chunks, parent_chunks = chunker.chunk_document(parsed_doc)

    # Register in SQLite
    register_document(doc_id, filename, file_ext, roles_str, classification, f"MD5-{doc_id}")

    # Convert HierarchicalChunks to vector DB format
    for chunk in child_chunks:
        all_child_chunks.append({
            "id": chunk.chunk_id,
            "text": chunk.text,
            "metadata": {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": 0,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "data_classification": classification,
                "allowed_roles": allowed_roles,
                "parent_chunk_id": chunk.parent_id or "",
                "level": chunk.level,
                "chunk_id": chunk.chunk_id,
            },
        })

    for chunk in parent_chunks:
        all_parent_chunks.append({
            "id": chunk.chunk_id,
            "text": chunk.text,
            "metadata": {
                "doc_id": doc_id,
                "filename": filename,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "data_classification": classification,
                "level": chunk.level,
            },
        })

    logger.info(f"Ingested {filename}: {len(child_chunks)} child chunks, {len(parent_chunks)} parent chunks")


def run_ingestion_pipeline():
    """Run the full document ingestion pipeline with multimodal parsing and hierarchical chunking."""
    # Clear existing data
    vector_db.chunks = []
    vector_db.parent_chunks = []
    vector_db.index = None

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM document_metadata;")
    conn.commit()
    conn.close()

    # Generate mock documents
    pdf_hr = generate_mock_pdf_hr()
    pdf_tech = generate_mock_pdf_tech()
    pdf_financial = generate_mock_pdf_financial_report()
    pdf_legal = generate_mock_legal_contract()
    csv_fin = generate_mock_csv_finance()
    json_aud = generate_mock_json_audits()

    all_child_chunks = []
    all_parent_chunks = []

    # Ingest each document
    print("📄 Ingesting HR Policy PDF...")
    _ingest_document(pdf_hr, "doc-hr", "HR Confidential",
                     ["HR", "Compliance", "Executive"], all_child_chunks, all_parent_chunks)

    print("📄 Ingesting Tech Spec PDF...")
    _ingest_document(pdf_tech, "doc-tech", "Engineering Confidential",
                     ["Engineering", "Compliance", "Executive"], all_child_chunks, all_parent_chunks)

    print("📊 Ingesting Financial Report PDF...")
    _ingest_document(pdf_financial, "doc-financial", "Finance Confidential",
                     ["Finance", "Compliance", "Executive"], all_child_chunks, all_parent_chunks)

    print("📜 Ingesting Legal Services Agreement...")
    _ingest_document(pdf_legal, "doc-legal", "Highly Restricted",
                     ["Executive"], all_child_chunks, all_parent_chunks)

    print("📊 Ingesting Financial CSV...")
    _ingest_document(csv_fin, "doc-fin", "Finance Confidential",
                     ["Finance", "Executive"], all_child_chunks, all_parent_chunks)

    print("🔍 Ingesting Compliance Audit JSON...")
    _ingest_document(json_aud, "doc-aud", "Compliance Audit",
                     ["Compliance", "Executive"], all_child_chunks, all_parent_chunks)

    # Public document for all roles
    pub_file = os.path.join(RAW_DATA_DIR, "public_compliance.txt")
    if not os.path.exists(pub_file):
        with open(pub_file, "w") as f:
            f.write(
                "VERTEX CORP - PUBLIC COMPLIANCE CHARTER\n"
                "This is a public document outlining our adherence to general regulatory laws.\n"
                "Vertex Corp is compliant with equal opportunity employment guidelines.\n"
                "All public complaints or queries can be sent directly to feedback@vertexcorp.com.\n"
                "We maintain standard SOC2 certification scopes across primary operational platforms.\n"
                "Our privacy policy governs data collection, processing, and retention practices.\n"
                "Third-party auditors verify our compliance annually through independent assessments.\n"
            )

    _ingest_document(pub_file, "doc-pub", "Public",
                     ["Intern", "Engineering", "HR", "Finance", "Compliance", "Executive"],
                     all_child_chunks, all_parent_chunks)

    # Dynamic folder scanning
    import glob
    subfolders = {
        "finance": ("Finance Confidential", ["Finance", "Executive"]),
        "hr": ("HR Confidential", ["HR", "Executive", "Compliance"]),
        "engineering": ("Engineering Confidential", ["Engineering", "Executive", "Compliance"]),
        "compliance": ("Compliance Audit", ["Compliance", "Executive"]),
        "logs": ("Highly Restricted", ["Compliance", "Executive"]),
        "policies": ("Public", ["Intern", "Engineering", "HR", "Finance", "Compliance", "Executive"]),
        "executive": ("Highly Restricted", ["Executive"]),
    }

    data_dir = os.path.dirname(RAW_DATA_DIR)
    for folder, (classification, allowed_roles) in subfolders.items():
        folder_path = os.path.join(data_dir, folder)
        if os.path.exists(folder_path):
            for filepath in glob.glob(os.path.join(folder_path, "*")):
                if os.path.isdir(filepath):
                    continue
                filename = os.path.basename(filepath)
                if filename.startswith(".") or filename == "secure_db.sqlite":
                    continue

                doc_id = f"doc-{folder}-{filename.replace('.', '_')}"
                print(f"📂 Dynamic Ingestion: {filename} from data/{folder}...")

                try:
                    _ingest_document(filepath, doc_id, classification, allowed_roles,
                                     all_child_chunks, all_parent_chunks)
                except Exception as e:
                    print(f"  ⚠️ Error ingesting {filename}: {e}")

    # Build vector index
    print(f"\n🔨 Building vector index for {len(all_child_chunks)} child chunks...")
    vector_db.add_chunks(all_child_chunks, all_parent_chunks)
    vector_db.rebuild_index()
    vector_db.save()

    print(f"✅ Ingestion complete! {len(all_child_chunks)} chunks indexed, "
          f"{len(all_parent_chunks)} parent contexts stored.")


# ── Text Content Helpers ────────────────────────────────────────────

def _get_hr_policy_lines():
    return [
        "SECTION 1: OVERVIEW AND VALUES",
        "Vertex Corp values confidentiality, inclusion, and operational integrity above all.",
        "This document governs internal HR behaviors, salary structure adjustments, and offboarding.",
        "",
        "SECTION 2: PROBATION AND ONBOARDING",
        "All new hires undergo a mandatory 90-day probationary review window.",
        "Managers must submit a performance evaluation at least 15 days before the end of the window.",
        "Full dental and medical benefits commence on Day 1 of full-time standard employment.",
        "",
        "SECTION 3: ANNUAL LEAVE AND COMPENSATORY TIME",
        "Full-time standard staff receive 22 paid annual leave days per fiscal year, accrued monthly.",
        "Sick leave accrues at 1.25 days per month up to a maximum accumulation of 30 days total.",
        "Maternity leave provides 16 fully paid weeks. Paternity leave provides 4 fully paid weeks.",
        "All bereavement leaves allow up to 5 consecutive paid working days for immediate family members.",
        "",
        "SECTION 4: SALARY STAGES AND ADJUSTMENTS",
        "Base salaries are calculated against standardized role bands: Band 4 (Intern), Band 5 (Junior),",
        "Band 6 (Senior Specialist), Band 7 (Director), and Band 8 (Vice President / Executive).",
        "Salary revision proposals are initiated during Q4 performance appraisals by Department Heads.",
        "Out-of-cycle salary increases exceeding 10% require double signatory sign-off from the CHRO and CEO.",
        "",
        "SECTION 5: DISCIPLINARY ACTIONS AND TERMINATIONS",
        "Vertex Corp follows a strict progressive discipline model: verbal, written, suspension, exit.",
        "Immediate termination occurs in events of severe compliance violation or intellectual property theft.",
        "Standard severance packages start at 2 weeks of pay per year of service, capped at 24 weeks max.",
    ]


def _get_hr_policy_text():
    return "\n".join(_get_hr_policy_lines())


def _get_tech_spec_lines():
    return [
        "PROJECT WORKSTREAM OVERVIEW",
        "This specification documents the enterprise deployment of Project Alpha's cloud tier.",
        "All details are classified under Engineering Confidential. Unauthorized export is strictly prohibited.",
        "",
        "1. CENTRAL SYSTEM WORKFLOW AND TOPOLOGY",
        "Project Alpha relies on a high-availability microservices model deployed in AWS region us-east-1.",
        "Primary application routes traffic through a centralized Amazon API Gateway using Cognito OAuth2.",
        "The core processing nodes run in Amazon ECS Fargate on private VPC subnets with NAT Gateways.",
        "Primary database relies on Amazon Aurora PostgreSQL serverless cluster with cross-region read replicas.",
        "",
        "2. SECURITY PROTOCOLS AND ENCRYPTION KEYS",
        "All data at rest is encrypted using AWS KMS with customer-managed keys (CMK) rotated every 90 days.",
        "In-transit network encryption requires TLS 1.3 with Perfect Forward Secrecy (PFS).",
        "Internal service keys are managed via AWS Secrets Manager with automatic role-based rotation.",
        "Access tokens generated have a strict validity window of 15 minutes before refreshing is mandatory.",
        "",
        "3. HIGH FREQUENCY VECTOR INTEGRATION",
        "Vector search integration is built using PGVector extensions running directly on Amazon Aurora.",
        "Vector representations use 1536-dimension embeddings generated by secure text-embedding models.",
        "Reindexing routines run incrementally every evening at 02:00 UTC using batch updates.",
        "",
        "4. CI/CD INTEGRATION PIPELINE",
        "Code repositories are strictly isolated in private GitHub enterprise repositories under VertexCorp-Alpha.",
        "Deployment relies on GitHub Actions workflows validating linting, unit tests, and security scans.",
        "Static Application Security Testing (SAST) is handled using SonarQube. Critical vulnerabilities block deployments.",
    ]


def _get_tech_spec_text():
    return "\n".join(_get_tech_spec_lines())


def _get_financial_report_text():
    return """VERTEX CORPORATION - Annual Financial Report FY2025

SECTION 1: EXECUTIVE SUMMARY
Vertex Corporation achieved record revenue of $487.3 million in FY2025, representing a 23.4% YoY increase.
Net income grew by 31.2% to $72.8 million. Free Cash Flow: $84.1M. Total Assets: $1.23B.

SECTION 2: REVENUE BREAKDOWN BY SEGMENT
Cloud Services: $198.4M (30.4% growth), AI & Analytics: $124.7M (39.6% growth),
Enterprise Software: $98.2M (5.0% growth), Professional Services: $42.8M, Data Infrastructure: $23.2M.

SECTION 3: QUARTERLY PERFORMANCE
Q1 2025: Revenue $108.2M, Net Profit $15.4M (14.2% margin)
Q2 2025: Revenue $115.8M, Net Profit $16.9M (14.6% margin)
Q3 2025: Revenue $127.4M, Net Profit $19.8M (15.5% margin)
Q4 2025: Revenue $135.9M, Net Profit $20.7M (15.2% margin)

SECTION 5: RISK FACTORS
1. Concentration Risk: Cloud Services represents 40.7% of total revenue.
2. Regulatory Risk: EU AI Act compliance may increase costs by $12-15M.
3. Competition Risk: Three new market entrants in AI Analytics vertical.
"""


def _get_legal_contract_lines():
    return [
        "Article 1: DEFINITIONS AND INTERPRETATION",
        "  1.1 'Agreement' means this Master Services Agreement including all schedules and amendments.",
        "  1.2 'Service Provider' means TechLegal Solutions Inc., registered in Delaware (EIN: 84-2947163).",
        "  1.3 'Client' means Vertex Corporation and its authorized subsidiaries.",
        "  1.4 'Confidential Information' includes trade secrets, financial data, customer lists, and algorithms.",
        "  1.5 'Effective Date' means January 1, 2026.",
        "",
        "Article 2: SCOPE OF SERVICES",
        "  2.1 The Service Provider shall deliver enterprise legal technology consulting services.",
        "  2.2 Services include: (a) Contract lifecycle management, (b) Regulatory compliance automation,",
        "      (c) IP portfolio analysis, (d) Data privacy impact assessments.",
        "  2.3 All deliverables must comply with ABA Model Rules of Professional Conduct.",
        "",
        "Article 3: COMPENSATION AND PAYMENT TERMS",
        "  3.1 Base annual retainer: $2,400,000 payable in monthly installments of $200,000.",
        "  3.2 Variable fees for ad-hoc engagements: $450/hour for Senior Partners, $275/hour for Associates.",
        "  3.3 Payment terms: Net 30 days from invoice date. Late payments incur 1.5% monthly interest.",
        "  3.4 Annual fee escalation: CPI + 2%, capped at 5% per annum.",
        "",
        "Article 4: INTELLECTUAL PROPERTY RIGHTS",
        "  4.1 All pre-existing IP remains with the originating party.",
        "  4.2 Work product IP created under this Agreement transfers to Client upon final payment.",
        "  4.3 Service Provider retains a non-exclusive license to use anonymized methodologies.",
        "",
        "Article 5: CONFIDENTIALITY AND DATA PROTECTION",
        "  5.1 Both parties shall maintain strict confidentiality for 5 years post-termination.",
        "  5.2 Confidential Information excludes publicly available data and independently developed materials.",
        "  5.3 Data processing shall comply with GDPR, CCPA, and applicable local data protection laws.",
        "  5.4 Breach notification required within 72 hours of discovery.",
        "",
        "Article 6: LIMITATION OF LIABILITY",
        "  6.1 Neither party's aggregate liability shall exceed the total fees paid in the prior 12 months.",
        "  6.2 Exclusions: Neither party excludes liability for fraud, gross negligence, or willful misconduct.",
        "  6.3 Consequential damages are excluded except for breaches of confidentiality obligations.",
        "",
        "Article 7: TERMINATION",
        "  7.1 Either party may terminate with 90 days written notice.",
        "  7.2 Immediate termination permitted upon material breach not cured within 30 days.",
        "  7.3 Upon termination, all Confidential Information must be returned or destroyed within 15 days.",
        "",
        "Article 8: GOVERNING LAW AND DISPUTE RESOLUTION",
        "  8.1 This Agreement is governed by the laws of the State of Delaware.",
        "  8.2 Disputes shall first be submitted to mediation under JAMS Commercial Mediation Rules.",
        "  8.3 If mediation fails within 60 days, disputes proceed to binding arbitration under AAA rules.",
        "  8.4 The prevailing party in any dispute shall be entitled to reasonable attorney's fees.",
    ]


def _get_legal_contract_text():
    return "\n".join(_get_legal_contract_lines())


if __name__ == "__main__":
    run_ingestion_pipeline()
