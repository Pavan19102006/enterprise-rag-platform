"""
Retrieval Pipeline
==================
Routes queries to SQL or Vector retrieval, applies RBAC filtering,
cross-encoder reranking, and Small-to-Big context expansion.
"""

import json
import re
import logging
from typing import List, Dict

from core.database import get_db_connection
from core.auth import get_allowed_classifications
from core.ingestion import vector_db
from core.reranker import rerank, is_reranker_available
from config import RETRIEVAL_TOP_K, RERANKER_TOP_N

logger = logging.getLogger(__name__)


def classify_query_intent(query_text: str) -> str:
    """Classify the domain of the query to route to SQL or Vector retrieval.
    Returns: 'SQL', 'VECTOR', or 'COMPLIANCE'."""
    normalized_query = query_text.lower()

    # Route to SQL ONLY if explicitly asking for relational database tables
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
    if user_role not in ["Finance", "Executive", "Compliance"]:
        return [{"error": "UNAUTHORIZED_DATABASE_ACCESS",
                 "message": f"Role '{user_role}' is not authorized to query corporate financial ledgers."}]

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        normalized = query_text.lower()
        quarter_match = re.search(r"\b202[56]-q[1-4]\b", normalized)

        if quarter_match:
            qtr = quarter_match.group(0).upper()
            cursor.execute("SELECT quarter, revenue_usd, net_profit_usd, status FROM corporate_revenue WHERE quarter = ?;", (qtr,))
            rows = cursor.fetchall()
        else:
            cursor.execute("SELECT quarter, revenue_usd, net_profit_usd, status FROM corporate_revenue ORDER BY quarter ASC;")
            rows = cursor.fetchall()

        results = [dict(row) for row in rows]
        return results
    except Exception as e:
        return [{"error": "DATABASE_QUERY_FAILED", "message": str(e)}]
    finally:
        conn.close()


def _expand_to_parent_context(chunks: List[Dict]) -> List[Dict]:
    """
    Small-to-Big expansion: for each retrieved child chunk, fetch
    its parent section chunk to provide richer context to the LLM.
    """
    expanded = []
    seen_parents = set()

    for chunk in chunks:
        parent_id = chunk.get("parent_id") or chunk.get("metadata", {}).get("parent_chunk_id", "")

        if parent_id and parent_id not in seen_parents:
            parent_text = vector_db.get_parent_text(parent_id)
            if parent_text:
                # Use parent text for LLM context but keep child metadata for citation
                expanded_chunk = dict(chunk)
                expanded_chunk["child_text"] = chunk["text"]  # Preserve original child text
                expanded_chunk["text"] = parent_text  # Expand to parent context
                expanded_chunk["metadata"] = dict(chunk.get("metadata", {}))
                expanded_chunk["metadata"]["expanded_from_child"] = True
                expanded_chunk["metadata"]["child_chunk_id"] = chunk.get("chunk_id", "")
                expanded.append(expanded_chunk)
                seen_parents.add(parent_id)
            else:
                expanded.append(chunk)
        else:
            # No parent or already seen — use child chunk as-is
            if parent_id not in seen_parents:
                expanded.append(chunk)

    return expanded


def retrieve_context(query_text: str, user_role: str, user_dept: str) -> dict:
    """
    Route, retrieve, rerank, and expand context chunks matching RBAC credentials.
    
    Pipeline:
    1. Route query (SQL / Vector / Compliance)
    2. Retrieve top-K candidates from FAISS
    3. Apply cross-encoder reranking → top-N
    4. Small-to-Big: expand child chunks to parent context
    5. Return final context with provenance metadata
    """
    route = classify_query_intent(query_text)
    retrieved_chunks = []
    restricted_count = 0

    # Reload Vector DB if empty
    if not vector_db.chunks:
        vector_db.load()

    allowed_classifications = get_allowed_classifications(user_role)

    if route == "SQL":
        sql_rows = execute_rbac_sql_query(query_text, user_role, user_dept)

        if sql_rows and "error" in sql_rows[0]:
            restricted_count = 1
            route = "VECTOR (SQL Blocked)"
            raw_results = vector_db.similarity_search(query_text, allowed_classifications, top_k=RETRIEVAL_TOP_K)
            ranked = rerank(query_text, raw_results, top_n=RERANKER_TOP_N)
            retrieved_chunks = [r.to_dict() for r in ranked]
        else:
            for row in sql_rows:
                chunk_str = (f"Corporate Revenue Record: Quarter is {row['quarter']}, "
                             f"Revenue is ${row['revenue_usd']:,.2f}, "
                             f"Net Profit is ${row['net_profit_usd']:,.2f}, "
                             f"Audit Status is {row['status']}.")
                retrieved_chunks.append({
                    "score": 1.0,
                    "text": chunk_str,
                    "metadata": {
                        "doc_id": "table-revenue",
                        "filename": "database:corporate_revenue",
                        "chunk_index": 0,
                        "page_number": 0,
                        "section_title": "Corporate Revenue Table",
                        "data_classification": "Finance Confidential",
                    },
                    "retrieval_score": 1.0,
                    "reranker_score": 1.0,
                })
    else:
        # Step 1: Retrieve candidates from FAISS (larger pool for reranking)
        raw_results = vector_db.similarity_search(
            query_text, allowed_classifications, top_k=RETRIEVAL_TOP_K
        )

        # Count restricted chunks for transparency
        all_results_no_rbac = vector_db.similarity_search(
            query_text,
            ["Public", "HR Confidential", "Finance Confidential",
             "Engineering Confidential", "Compliance Audit", "Highly Restricted"],
            top_k=50
        )
        for r in all_results_no_rbac:
            if r["metadata"]["data_classification"] not in allowed_classifications:
                restricted_count += 1

        # Step 2: Rerank with cross-encoder
        ranked = rerank(query_text, raw_results, top_n=RERANKER_TOP_N)
        retrieved_chunks = [r.to_dict() for r in ranked]

        # Step 3: Apply Highly Restricted RBAC filter
        filtered_chunks = []
        for doc in retrieved_chunks:
            clearance = doc["metadata"].get("data_classification", "Public")
            if clearance == "Highly Restricted" and user_role != "Executive":
                restricted_count += 1
                continue
            filtered_chunks.append(doc)
        retrieved_chunks = filtered_chunks

    # Step 4: Small-to-Big context expansion
    expanded_chunks = _expand_to_parent_context(retrieved_chunks)

    # De-duplication
    unique_chunks = []
    seen_texts = set()
    for chunk in expanded_chunks:
        norm_txt = chunk["text"].strip().lower()[:200]
        if norm_txt not in seen_texts:
            seen_texts.add(norm_txt)
            unique_chunks.append(chunk)

    return {
        "retrieval_route": route,
        "retrieved_chunks": unique_chunks,
        "restricted_count": restricted_count,
        "reranker_active": is_reranker_available(),
        "small_to_big_expanded": any(
            c.get("metadata", {}).get("expanded_from_child", False) for c in unique_chunks
        ),
    }
