import json
from core.database import get_db_connection

def write_audit_log(username: str, role: str, query_text: str, intent: str, 
                    security_verdict: str, retrieved_documents: list, 
                    llm_confidence: float, execution_time_ms: int):
    """Insert a detailed, immutable record of user transactions into the relational audit logs table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Format retrieved document metadata as stringified JSON array for secure auditing
    docs_json = json.dumps([
        {
            "doc_id": c.get("metadata", {}).get("doc_id", "unknown"),
            "filename": c.get("metadata", {}).get("filename", "unknown"),
            "chunk_index": c.get("metadata", {}).get("chunk_index", -1),
            "classification": c.get("metadata", {}).get("data_classification", "Public"),
            "retrieval_relevance": c.get("score", 1.0)
        } for c in retrieved_documents
    ]) if retrieved_documents else "[]"
    
    try:
        cursor.execute("""
        INSERT INTO audit_logs (username, user_role, query_text, intent_classification, security_verdict, retrieved_documents, llm_confidence, execution_time_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """, (username, role, query_text, intent, security_verdict, docs_json, llm_confidence, execution_time_ms))
        conn.commit()
    except Exception as e:
        print(f"Failed to write audit log to database: {e}")
    finally:
        conn.close()

def get_audit_logs(limit=50) -> list:
    """Fetch recent compliance audit records for dashboard monitoring."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, timestamp, username, user_role, query_text, intent_classification, security_verdict, retrieved_documents, llm_confidence, execution_time_ms
    FROM audit_logs
    ORDER BY timestamp DESC
    LIMIT ?;
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    logs = []
    for r in rows:
        # Load stringified document json safely
        try:
            docs = json.loads(r["retrieved_documents"]) if r["retrieved_documents"] else []
        except Exception:
            docs = []
            
        logs.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "username": r["username"],
            "role": r["user_role"],
            "query": r["query_text"],
            "intent": r["intent_classification"],
            "verdict": r["security_verdict"],
            "retrieved_docs": docs,
            "confidence": r["llm_confidence"],
            "latency_ms": r["execution_time_ms"]
        })
    return logs
