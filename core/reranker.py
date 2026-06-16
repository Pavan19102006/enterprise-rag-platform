"""
Cross-Encoder Reranker
======================
Reranks retrieved candidate chunks using a cross-encoder model for 
higher precision. Uses score fusion between retrieval and reranking scores.

Falls back to retrieval-only ranking if the cross-encoder model is unavailable.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from config import (
    RERANKER_MODEL_NAME,
    RERANKER_TOP_N,
    RERANKER_WEIGHT,
    RETRIEVAL_WEIGHT,
    USE_DENSE_EMBEDDINGS,
)

logger = logging.getLogger(__name__)

# Lazy-load cross-encoder model
_cross_encoder = None
_cross_encoder_available = None


def _load_cross_encoder():
    """Load the cross-encoder model (only if USE_DENSE_EMBEDDINGS is True)."""
    global _cross_encoder, _cross_encoder_available
    if _cross_encoder_available is not None:
        return _cross_encoder_available

    if not USE_DENSE_EMBEDDINGS:
        logger.info("Cross-encoder disabled (USE_DENSE_EMBEDDINGS=false). Using retrieval scores only.")
        _cross_encoder_available = False
        return False

    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading cross-encoder reranker: {RERANKER_MODEL_NAME}")
        _cross_encoder = CrossEncoder(RERANKER_MODEL_NAME)
        _cross_encoder_available = True
        logger.info("Cross-encoder reranker loaded successfully.")
        return True
    except Exception as e:
        logger.warning(f"Cross-encoder unavailable: {e}. Reranker disabled.")
        _cross_encoder_available = False
        return False


@dataclass
class RankedChunk:
    """A chunk with both retrieval and reranker scores."""
    text: str
    metadata: Dict
    retrieval_score: float
    reranker_score: float
    fused_score: float
    chunk_id: str = ""
    parent_id: str = ""

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "metadata": self.metadata,
            "score": self.fused_score,
            "retrieval_score": self.retrieval_score,
            "reranker_score": self.reranker_score,
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
        }


def rerank(
    query: str,
    chunks: List[Dict],
    top_n: int = RERANKER_TOP_N,
    retrieval_weight: float = RETRIEVAL_WEIGHT,
    reranker_weight: float = RERANKER_WEIGHT,
) -> List[RankedChunk]:
    """
    Rerank retrieved chunks using cross-encoder score fusion.
    
    Args:
        query: The user's search query
        chunks: List of retrieved chunks with 'text', 'metadata', 'score' keys
        top_n: Number of top results to return after reranking
        retrieval_weight: Weight for the original retrieval score
        reranker_weight: Weight for the cross-encoder reranking score
    
    Returns:
        List of RankedChunk sorted by fused score (descending)
    """
    if not chunks:
        return []

    has_reranker = _load_cross_encoder()

    if has_reranker and _cross_encoder is not None:
        return _rerank_with_cross_encoder(
            query, chunks, top_n, retrieval_weight, reranker_weight
        )
    else:
        return _rerank_fallback(query, chunks, top_n)


def _rerank_with_cross_encoder(
    query: str,
    chunks: List[Dict],
    top_n: int,
    retrieval_weight: float,
    reranker_weight: float,
) -> List[RankedChunk]:
    """Rerank using cross-encoder model with score fusion."""
    # Prepare query-document pairs for cross-encoder
    pairs = [(query, chunk["text"]) for chunk in chunks]

    try:
        # Get cross-encoder scores
        ce_scores = _cross_encoder.predict(pairs)

        # Normalize cross-encoder scores to [0, 1] range
        min_score = min(ce_scores) if len(ce_scores) > 0 else 0
        max_score = max(ce_scores) if len(ce_scores) > 0 else 1
        score_range = max_score - min_score if max_score != min_score else 1.0

        ranked_chunks = []
        for idx, chunk in enumerate(chunks):
            # Normalize CE score to [0, 1]
            normalized_ce = (ce_scores[idx] - min_score) / score_range

            retrieval_score = chunk.get("score", 0.0)
            reranker_score = float(normalized_ce)

            # Score fusion
            fused = retrieval_weight * retrieval_score + reranker_weight * reranker_score

            ranked_chunks.append(RankedChunk(
                text=chunk["text"],
                metadata=chunk.get("metadata", {}),
                retrieval_score=retrieval_score,
                reranker_score=reranker_score,
                fused_score=fused,
                chunk_id=chunk.get("chunk_id", chunk.get("metadata", {}).get("chunk_id", "")),
                parent_id=chunk.get("parent_id", chunk.get("metadata", {}).get("parent_chunk_id", "")),
            ))

        # Sort by fused score descending
        ranked_chunks.sort(key=lambda x: x.fused_score, reverse=True)

        logger.info(
            f"Reranked {len(chunks)} candidates → top {top_n}. "
            f"Score range: {ranked_chunks[0].fused_score:.3f} - "
            f"{ranked_chunks[-1].fused_score:.3f}"
        )

        return ranked_chunks[:top_n]

    except Exception as e:
        logger.error(f"Cross-encoder reranking failed: {e}. Falling back to retrieval scores.")
        return _rerank_fallback(query, chunks, top_n)


def _rerank_fallback(
    query: str,
    chunks: List[Dict],
    top_n: int,
) -> List[RankedChunk]:
    """Fallback reranking using only retrieval scores (no cross-encoder)."""
    ranked_chunks = []
    for chunk in chunks:
        retrieval_score = chunk.get("score", 0.0)
        ranked_chunks.append(RankedChunk(
            text=chunk["text"],
            metadata=chunk.get("metadata", {}),
            retrieval_score=retrieval_score,
            reranker_score=0.0,
            fused_score=retrieval_score,
            chunk_id=chunk.get("chunk_id", chunk.get("metadata", {}).get("chunk_id", "")),
            parent_id=chunk.get("parent_id", chunk.get("metadata", {}).get("parent_chunk_id", "")),
        ))

    ranked_chunks.sort(key=lambda x: x.fused_score, reverse=True)
    return ranked_chunks[:top_n]


def is_reranker_available() -> bool:
    """Check if the cross-encoder reranker model is available."""
    return _load_cross_encoder()
