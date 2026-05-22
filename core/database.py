import sqlite3
import os
import hashlib
import binascii
import uuid
from config import DB_PATH, DATA_DIR

def hash_password(password: str) -> str:
    """Hash a password using PBKDF2 with a secure salt."""
    salt = hashlib.sha256(os.urandom(60)).hexdigest().encode('utf-8')
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    pwdhash = binascii.hexlify(pwdhash)
    return (salt + pwdhash).decode('utf-8')

def verify_password(stored_password: str, provided_password: str) -> bool:
    """Verify a stored password hash against a provided password."""
    salt = stored_password[:64].encode('utf-8')
    stored_hash = stored_password[64:]
    pwdhash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
    pwdhash = binascii.hexlify(pwdhash).decode('utf-8')
    return pwdhash == stored_hash

def get_db_connection():
    """Return an active connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    """Create schemas and seed default enterprise users and data."""
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Create Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username VARCHAR(50) PRIMARY KEY,
        password_hash VARCHAR(128) NOT NULL,
        role VARCHAR(20) NOT NULL,
        department VARCHAR(50) NOT NULL,
        jwt_secret VARCHAR(64) NOT NULL
    );
    """)
    
    # 2. Create Document Metadata Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS document_metadata (
        id VARCHAR(100) PRIMARY KEY,
        filename VARCHAR(255) NOT NULL,
        file_type VARCHAR(10) NOT NULL,
        allowed_roles VARCHAR(255) NOT NULL,
        allowed_departments VARCHAR(255),
        owner_department VARCHAR(50),
        data_classification VARCHAR(50) NOT NULL,
        checksum VARCHAR(64) NOT NULL,
        ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 3. Create Audit Logs Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        username VARCHAR(50) NOT NULL,
        user_role VARCHAR(20) NOT NULL,
        query_text TEXT NOT NULL,
        intent_classification VARCHAR(50),
        security_verdict VARCHAR(50) NOT NULL,
        retrieved_documents TEXT,
        llm_confidence FLOAT,
        execution_time_ms INTEGER
    );
    """)
    
    # 4. Create Corporate Revenue Table (Structured Database for direct SQL Query Routing)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS corporate_revenue (
        quarter VARCHAR(10) PRIMARY KEY,
        revenue_usd REAL NOT NULL,
        net_profit_usd REAL NOT NULL,
        status VARCHAR(20) NOT NULL
    );
    """)
    
    conn.commit()
    
    # Seed default users if table is empty
    cursor.execute("SELECT COUNT(*) FROM users;")
    if cursor.fetchone()[0] == 0:
        default_users = [
            ("bob", "bob123", "Intern", "Operations"),
            ("alice", "alice123", "Engineering", "Engineering"),
            ("helen", "helen123", "HR", "Human Resources"),
            ("fred", "fred123", "Finance", "Finance"),
            ("charlie", "charlie123", "Compliance", "Compliance"),
            ("elena", "elena123", "Executive", "Executive")
        ]
        for uname, pwd, role, dept in default_users:
            p_hash = hash_password(pwd)
            jwt_sec = uuid.uuid4().hex
            cursor.execute(
                "INSERT OR IGNORE INTO users (username, password_hash, role, department, jwt_secret) VALUES (?, ?, ?, ?, ?);",
                (uname, p_hash, role, dept, jwt_sec)
            )
        conn.commit()
        print("Seeded default users.")

    # Seed corporate revenue if table is empty
    cursor.execute("SELECT COUNT(*) FROM corporate_revenue;")
    if cursor.fetchone()[0] == 0:
        financial_rows = [
            ("2025-Q1", 12500000.0, 3100000.0, "Audited"),
            ("2025-Q2", 14200000.0, 3800000.0, "Audited"),
            ("2025-Q3", 13900000.0, 3500000.0, "Audited"),
            ("2025-Q4", 16800000.0, 4800000.0, "Audited"),
            ("2026-Q1", 15400000.0, 4200000.0, "Unaudited")
        ]
        for qtr, rev, prof, stat in financial_rows:
            cursor.execute(
                "INSERT INTO corporate_revenue (quarter, revenue_usd, net_profit_usd, status) VALUES (?, ?, ?, ?);",
                (qtr, rev, prof, stat)
            )
        conn.commit()
        print("Seeded corporate revenue rows.")
        
    conn.close()

if __name__ == "__main__":
    initialize_database()
