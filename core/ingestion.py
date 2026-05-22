import os
import json
import csv
import sqlite3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from config import RAW_DATA_DIR, DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_STORE_DIR
from core.database import get_db_connection

# Ensure folders exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

def generate_mock_pdf_hr():
    """Generate a simulated HR Policy PDF using reportlab."""
    pdf_path = os.path.join(RAW_DATA_DIR, "hr_policy.pdf")
    if os.path.exists(pdf_path):
        return pdf_path

    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    
    # Page 1
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "VERTEX CORP - CONFIDENTIAL HUMAN RESOURCES POLICY")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, "Classification: HR Confidential | Target Audience: HR Department, Executives")
    c.line(50, height - 80, width - 50, height - 80)
    
    y = height - 110
    text_lines = [
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
        "Standard severance packages start at 2 weeks of pay per year of service, capped at 24 weeks max."
    ]
    
    for line in text_lines:
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
    """Generate a simulated Technical Specification PDF using reportlab."""
    pdf_path = os.path.join(RAW_DATA_DIR, "tech_spec.pdf")
    if os.path.exists(pdf_path):
        return pdf_path

    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    
    # Page 1
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "PROJECT ALPHA: CLOUD ARCHITECTURE SPECIFICATION")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, "Classification: Engineering Confidential | Target Audience: Engineers, Executives")
    c.line(50, height - 80, width - 50, height - 80)
    
    y = height - 110
    text_lines = [
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
        "Static Application Security Testing (SAST) is handled using SonarQube. Critical vulnerabilities block deployments."
    ]
    
    for line in text_lines:
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
        ["TXN-1006", "2026-03-05", "Operations", "Utility Services", "9800.00", "fred", "Corporate high-speed fiber internet backup lines"]
    ]
    
    with open(csv_path, mode='w', newline='') as f:
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
        }
    ]
    
    with open(json_path, 'w') as f:
        json.dump(records, f, indent=4)
        
    return json_path

# Vector Database Simulation using TF-IDF + Cosine Similarity
class LocalVectorDB:
    def __init__(self):
        self.chunks = []      # list of dicts: {"id", "text", "metadata"}
        self.embeddings = []  # list of numpy arrays
        self.vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
        
    def add_chunks(self, new_chunks):
        """Add chunks to the database list."""
        self.chunks.extend(new_chunks)
        
    def rebuild_index(self):
        """Fit TF-IDF on all chunk text to represent our vector space."""
        if not self.chunks:
            return
        corpus = [c["text"] for c in self.chunks]
        tfidf_matrix = self.vectorizer.fit_transform(corpus).toarray()
        self.embeddings = [row for row in tfidf_matrix]
        
    def similarity_search(self, query_text, allowed_classifications, top_k=4):
        """Search relevant chunks filtering by role permissions."""
        if not self.chunks or not self.embeddings:
            return []
            
        # Fit vectorizer on query
        try:
            query_vector = self.vectorizer.transform([query_text]).toarray()[0]
        except Exception:
            return []
            
        results = []
        for idx, chunk in enumerate(self.chunks):
            # Enforce metadata-based security boundary
            classification = chunk["metadata"]["data_classification"]
            if classification not in allowed_classifications:
                continue
                
            chunk_vector = self.embeddings[idx]
            
            # Cosine similarity
            dot_product = np.dot(query_vector, chunk_vector)
            norm_q = np.linalg.norm(query_vector)
            norm_c = np.linalg.norm(chunk_vector)
            
            if norm_q > 0 and norm_c > 0:
                score = float(dot_product / (norm_q * norm_c))
            else:
                score = 0.0
                
            results.append({
                "score": score,
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            })
            
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def save(self):
        """Save vector DB model index to files."""
        model_path = os.path.join(VECTOR_STORE_DIR, "vector_db.json")
        data = {
            "chunks": self.chunks
        }
        with open(model_path, 'w') as f:
            json.dump(data, f, indent=4)
            
    def load(self):
        """Load vector DB model index and rebuild."""
        model_path = os.path.join(VECTOR_STORE_DIR, "vector_db.json")
        if not os.path.exists(model_path):
            return False
        with open(model_path, 'r') as f:
            data = json.load(f)
            self.chunks = data["chunks"]
        self.rebuild_index()
        return True

# Initialize Global Vector DB instance
vector_db = LocalVectorDB()

def parse_pdf(file_path):
    """Extract text from a PDF document page by page."""
    reader = PdfReader(file_path)
    text = ""
    for idx, page in enumerate(reader.pages):
        text += f"--- Page {idx+1} ---\n"
        text += page.extract_text() + "\n"
    return text

def parse_csv(file_path):
    """Convert CSV rows into descriptive semantic sentences."""
    text = ""
    filename = os.path.basename(file_path)
    with open(file_path, mode='r') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            serialized_row = f"Row {idx+1} in {filename}: " + ", ".join([f"{k} is {v}" for k, v in row.items()])
            text += serialized_row + "\n\n"
    return text

def parse_json(file_path):
    """Convert JSON array elements into detailed text summaries."""
    text = ""
    filename = os.path.basename(file_path)
    with open(file_path, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            for idx, item in enumerate(data):
                serialized_item = f"Record {idx+1} in {filename}: "
                serialized_item += "; ".join([f"{k}: {v}" for k, v in item.items()])
                text += serialized_item + "\n\n"
        else:
            text += json.dumps(data, indent=2)
    return text

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chunk clean text intelligently using recursive character sizing."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
        if start >= text_len:
            break
            
    return chunks

def register_document(doc_id, filename, file_type, allowed_roles, classification, checksum):
    """Register document into SQLite Document Metadata database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if doc exists, if so delete to enable re-indexing
    cursor.execute("DELETE FROM document_metadata WHERE id = ?;", (doc_id,))
    
    cursor.execute("""
    INSERT INTO document_metadata (id, filename, file_type, allowed_roles, data_classification, checksum)
    VALUES (?, ?, ?, ?, ?, ?);
    """, (doc_id, filename, file_type, allowed_roles, classification, checksum))
    conn.commit()
    conn.close()

def run_ingestion_pipeline():
    """Run simulated PDF, CSV, JSON and SQL database generation & ingestion pipeline."""
    # Clear existing vector database and relational metadata to prevent duplicate entries on reindexing
    vector_db.chunks = []
    vector_db.embeddings = []
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM document_metadata;")
    conn.commit()
    conn.close()

    # Ensure raw files exist
    pdf_hr = generate_mock_pdf_hr()
    pdf_tech = generate_mock_pdf_tech()
    csv_fin = generate_mock_csv_finance()
    json_aud = generate_mock_json_audits()
    
    all_chunks = []
    
    # 1. Ingest PDF HR
    print("Ingesting HR Policy PDF...")
    hr_text = parse_pdf(pdf_hr)
    hr_chunks = chunk_text(hr_text)
    register_document("doc-hr", "hr_policy.pdf", "PDF", "HR,Compliance,Executive", "HR Confidential", "MD5-HR-MOCK")
    for i, c_text in enumerate(hr_chunks):
        all_chunks.append({
            "id": f"chunk-hr-{i}",
            "text": c_text,
            "metadata": {
                "doc_id": "doc-hr",
                "filename": "hr_policy.pdf",
                "chunk_index": i,
                "data_classification": "HR Confidential",
                "allowed_roles": ["HR", "Compliance", "Executive"]
            }
        })
        
    # 2. Ingest PDF Tech
    print("Ingesting Tech Spec PDF...")
    tech_text = parse_pdf(pdf_tech)
    tech_chunks = chunk_text(tech_text)
    register_document("doc-tech", "tech_spec.pdf", "PDF", "Engineering,Compliance,Executive", "Engineering Confidential", "MD5-TECH-MOCK")
    for i, c_text in enumerate(tech_chunks):
        all_chunks.append({
            "id": f"chunk-tech-{i}",
            "text": c_text,
            "metadata": {
                "doc_id": "doc-tech",
                "filename": "tech_spec.pdf",
                "chunk_index": i,
                "data_classification": "Engineering Confidential",
                "allowed_roles": ["Engineering", "Compliance", "Executive"]
            }
        })
        
    # 3. Ingest CSV Finance
    print("Ingesting Financial Expenses CSV...")
    fin_text = parse_csv(csv_fin)
    fin_chunks = chunk_text(fin_text)
    register_document("doc-fin", "finance.csv", "CSV", "Finance,Executive", "Finance Confidential", "MD5-FIN-MOCK")
    for i, c_text in enumerate(fin_chunks):
        all_chunks.append({
            "id": f"chunk-fin-{i}",
            "text": c_text,
            "metadata": {
                "doc_id": "doc-fin",
                "filename": "finance.csv",
                "chunk_index": i,
                "data_classification": "Finance Confidential",
                "allowed_roles": ["Finance", "Executive"]
            }
        })
        
    # 4. Ingest JSON Audits
    print("Ingesting Compliance Audits JSON...")
    aud_text = parse_json(json_aud)
    aud_chunks = chunk_text(aud_text)
    register_document("doc-aud", "audit_logs.json", "JSON", "Compliance,Executive", "Compliance Audit", "MD5-AUD-MOCK")
    for i, c_text in enumerate(aud_chunks):
        all_chunks.append({
            "id": f"chunk-aud-{i}",
            "text": c_text,
            "metadata": {
                "doc_id": "doc-aud",
                "filename": "audit_logs.json",
                "chunk_index": i,
                "data_classification": "Compliance Audit",
                "allowed_roles": ["Compliance", "Executive"]
            }
        })

    # Load public data for Intern level
    intern_file = os.path.join(RAW_DATA_DIR, "public_compliance.txt")
    if not os.path.exists(intern_file):
        with open(intern_file, 'w') as f:
            f.write("VERTEX CORP - PUBLIC COMPLIANCE CHARTER\n"
                    "This is a public document outlining our adherence to general regulatory laws.\n"
                    "Vertex Corp is compliant with equal opportunity employment guidelines.\n"
                    "All public complaints or queries can be sent directly to feedback@vertexcorp.com.\n"
                    "We maintain standard SOC2 certification scopes across primary operational platforms.")
    
    pub_text = open(intern_file).read()
    pub_chunks = chunk_text(pub_text)
    register_document("doc-pub", "public_compliance.txt", "TXT", "Intern,Engineering,HR,Finance,Compliance,Executive", "Public", "MD5-PUB-MOCK")
    for i, c_text in enumerate(pub_chunks):
        all_chunks.append({
            "id": f"chunk-pub-{i}",
            "text": c_text,
            "metadata": {
                "doc_id": "doc-pub",
                "filename": "public_compliance.txt",
                "chunk_index": i,
                "data_classification": "Public",
                "allowed_roles": ["Intern", "Engineering", "HR", "Finance", "Compliance", "Executive"]
            }
        })

    # Dynamic Ingestion Scanner for Newly Copied Folder Datasets
    import glob
    subfolders = {
        "finance": ("Finance Confidential", ["Finance", "Executive"]),
        "hr": ("HR Confidential", ["HR", "Executive", "Compliance"]),
        "engineering": ("Engineering Confidential", ["Engineering", "Executive", "Compliance"]),
        "compliance": ("Compliance Audit", ["Compliance", "Executive"]),
        "logs": ("Highly Restricted", ["Compliance", "Executive"]),
        "policies": ("Public", ["Intern", "Engineering", "HR", "Finance", "Compliance", "Executive"]),
        "executive": ("Highly Restricted", ["Executive"])
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
                    
                file_ext = filename.split(".")[-1].upper()
                doc_id = f"doc-{folder}-{filename.replace('.', '_')}"
                
                print(f"Dynamic Ingestion: Processing {filename} from data/{folder}...")
                
                try:
                    if file_ext == "PDF":
                        text = parse_pdf(filepath)
                    elif file_ext == "CSV":
                        text = parse_csv(filepath)
                    elif file_ext == "JSON":
                        text = parse_json(filepath)
                    else:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read()
                            
                    chunks = chunk_text(text)
                    roles_str = ",".join(allowed_roles)
                    register_document(doc_id, filename, file_ext, roles_str, classification, f"MD5-DYNAMIC-{filename}")
                    
                    for i, c_text in enumerate(chunks):
                        all_chunks.append({
                            "id": f"chunk-{folder}-{filename.replace('.', '_')}-{i}",
                            "text": c_text,
                            "metadata": {
                                "doc_id": doc_id,
                                "filename": filename,
                                "chunk_index": i,
                                "data_classification": classification,
                                "allowed_roles": allowed_roles
                            }
                        })
                except Exception as e:
                    print(f"Error dynamically ingesting {filename}: {str(e)}")

    # Add all gathered chunks into global vector db simulation
    vector_db.add_chunks(all_chunks)
    vector_db.rebuild_index()
    vector_db.save()
    print("Ingestion completed successfully. Index rebuilt and saved.")

if __name__ == "__main__":
    run_ingestion_pipeline()
