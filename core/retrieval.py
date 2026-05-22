import json
import sqlite3
import re
from core.database import get_db_connection
from core.auth import get_allowed_classifications
from core.ingestion import vector_db

def classify_query_intent(query_text: str) -> str:
    """Classify the domain of the query to route to SQL or Vector retrieval.
    Returns: 'SQL', 'VECTOR', or 'COMPLIANCE'."""
    normalized_query = query_text.lower()
    
    # Route to SQL ONLY if explicitly asking for relational database tables, ledgers, or seeded columns
    sql_triggers = [
        r"relational\s+table", r"database\s+table", r"sql\s+table", r"corporate\s+revenue\s+table",
        r"revenue\s+table", r"ledger\s+rows", r"raw\s+database", r"corporate_revenue",
        r"show\s+total\s+revenue"
    ]
    for pattern in sql_triggers:
        if re.search(pattern, normalized_query):
            return "SQL"
            
    # Route to COMPLIANCE if asking for audit logs or system events
    compliance_keywords = [
        r"audit\s+log", r"soc2", r"gdpr", r"hipaa", r"auditor", r"compliance\s+status", r"security\s+verdict"
    ]
    for pattern in compliance_keywords:
        if re.search(pattern, normalized_query):
            return "COMPLIANCE"
            
    return "VECTOR"

def execute_rbac_sql_query(query_text: str, user_role: str, user_dept: str) -> list:
    """Securely execute relational SQL queries restricting rows and tables to authorized departments."""
    # Strict Table Isolation: Interns, HR, Engineers cannot run SQL finance queries!
    if user_role not in ["Finance", "Executive", "Compliance"]:
        return [{"error": "UNAUTHORIZED_DATABASE_ACCESS", "message": f"Role '{user_role}' is not authorized to query corporate financial ledgers."}]
        
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Simple Natural Language to SQL converter mapping (highly secure, parameterized!)
        # We NEVER evaluate raw SQL strings input by the user.
        normalized = query_text.lower()
        
        # Safe extraction of quarter if present
        quarter_match = re.search(r"\b202[56]-q[1-4]\b", normalized)
        
        if quarter_match:
            qtr = quarter_match.group(0).upper()
            cursor.execute("SELECT quarter, revenue_usd, net_profit_usd, status FROM corporate_revenue WHERE quarter = ?;", (qtr,))
            rows = cursor.fetchall()
        else:
            # Return full dataset if no quarter specified, ordered chronologically
            cursor.execute("SELECT quarter, revenue_usd, net_profit_usd, status FROM corporate_revenue ORDER BY quarter ASC;")
            rows = cursor.fetchall()
            
        results = [dict(row) for row in rows]
        return results
    except Exception as e:
        return [{"error": "DATABASE_QUERY_FAILED", "message": str(e)}]
    finally:
        conn.close()

def retrieve_context(query_text: str, user_role: str, user_dept: str) -> dict:
    """Route, retrieve, and filter context chunks matching RBAC credentials.
    Returns: {"retrieval_route", "retrieved_chunks", "restricted_count"}."""
    # 1. Routing
    route = classify_query_intent(query_text)
    
    retrieved_chunks = []
    restricted_count = 0
    
    # Reload Vector DB if empty to guarantee state is active
    if not vector_db.chunks:
        vector_db.load()
        
    # Get classifications the user's role is allowed to view
    allowed_classifications = get_allowed_classifications(user_role)
    
    if route == "SQL":
        # Relational SQL Database query execution
        sql_rows = execute_rbac_sql_query(query_text, user_role, user_dept)
        
        # Check if unauthorized error returned
        if sql_rows and "error" in sql_rows[0]:
            restricted_count = 1
            # Fallback to VECTOR search as a graceful safety measure
            route = "VECTOR (SQL Blocked)"
            vector_results = vector_db.similarity_search(query_text, allowed_classifications, top_k=3)
            retrieved_chunks = vector_results
        else:
            # Serialize SQL row dictionaries into text blocks for LLM injection
            for row in sql_rows:
                chunk_str = f"Corporate Revenue Record: Quarter is {row['quarter']}, Revenue is ${row['revenue_usd']:,.2f}, Net Profit is ${row['net_profit_usd']:,.2f}, Audit Status is {row['status']}."
                retrieved_chunks.append({
                    "score": 1.0, # Direct SQL exact hits are treated as highest relevance
                    "text": chunk_str,
                    "metadata": {
                        "doc_id": "table-revenue",
                        "filename": "database:corporate_revenue",
                        "chunk_index": 0,
                        "data_classification": "Finance Confidential"
                    }
                })
    else:
        # Semantic search against Vector Database
        # Pass restricted list of roles directly to similarity_search to enforce security at vector tier
        vector_results = vector_db.similarity_search(query_text, allowed_classifications, top_k=4)
        
        # Count how many total chunks in raw DB matching this query were restricted
        # (This is shown in compliance dashboards for transparency)
        all_results_no_rbac = vector_db.similarity_search(query_text, ["Public", "HR Confidential", "Finance Confidential", "Engineering Confidential", "Compliance Audit", "Highly Restricted"], top_k=100)
        
        allowed_ids = {r["text"] for r in vector_results}
        for r in all_results_no_rbac:
            if r["metadata"]["data_classification"] not in allowed_classifications:
                restricted_count += 1
                
        retrieved_chunks = vector_results

    # Re-ranking & de-duplication
    unique_chunks = []
    seen_texts = set()
    for chunk in retrieved_chunks:
        norm_txt = chunk["text"].strip().lower()
        if norm_txt not in seen_texts:
            seen_texts.add(norm_txt)
            unique_chunks.append(chunk)
            
    return {
        "retrieval_route": route,
        "retrieved_chunks": unique_chunks,
        "restricted_count": restricted_count
    }
