"""
RAG Evaluation Framework
=========================
Quantifiable evaluation metrics for RAG quality:
  - Faithfulness: Are LLM claims grounded in context?
  - Answer Relevancy: Does the answer address the question?
  - Context Precision: Are retrieved chunks relevant?
  - Context Recall: Did we retrieve all necessary chunks?
  - Citation Accuracy: Custom metric for citation validation

Supports both Ragas-based evaluation (when available) and custom
built-in metrics for zero-dependency operation.
"""

import json
import re
import os
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from config import EVAL_RESULTS_PATH, EVAL_METRICS

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of evaluating a single query-answer pair."""
    query: str
    answer: str
    ground_truth: str
    contexts: List[str]
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    citation_accuracy: float = 0.0
    overall_score: float = 0.0
    details: Dict = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class BatchEvalResult:
    """Aggregated result from evaluating a batch of queries."""
    total_queries: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: float
    avg_citation_accuracy: float
    avg_overall_score: float
    per_query_results: List[Dict] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self):
        return asdict(self)


class RAGEvaluator:
    """
    Evaluates RAG pipeline quality using multiple metrics.
    Uses custom built-in metrics that work without external dependencies.
    """

    def __init__(self):
        self.results_history: List[EvalResult] = []
        self._load_history()

    def evaluate_single(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = "",
        citations: List[Dict] = None,
    ) -> EvalResult:
        """
        Evaluate a single query-answer pair against retrieved contexts.
        
        Args:
            query: The user's original query
            answer: The LLM's response
            contexts: List of retrieved context texts
            ground_truth: Expected correct answer (optional)
            citations: List of citation dicts from the orchestrator
        """
        # 1. Faithfulness: Is every claim in the answer supported by context?
        faithfulness = self._compute_faithfulness(answer, contexts)

        # 2. Answer Relevancy: Does the answer address the question?
        answer_relevancy = self._compute_answer_relevancy(query, answer)

        # 3. Context Precision: Are the retrieved chunks relevant to the query?
        context_precision = self._compute_context_precision(query, contexts)

        # 4. Context Recall: Did we retrieve chunks covering the ground truth?
        context_recall = self._compute_context_recall(ground_truth, contexts) if ground_truth else 0.5

        # 5. Citation Accuracy: Are citations valid?
        citation_accuracy = self._compute_citation_accuracy(answer, citations or [])

        # Overall score (weighted average)
        overall = (
            0.30 * faithfulness +
            0.20 * answer_relevancy +
            0.15 * context_precision +
            0.15 * context_recall +
            0.20 * citation_accuracy
        )

        result = EvalResult(
            query=query,
            answer=answer[:500],  # Truncate for storage
            ground_truth=ground_truth[:300],
            contexts=[c[:200] for c in contexts[:5]],
            faithfulness=round(faithfulness, 3),
            answer_relevancy=round(answer_relevancy, 3),
            context_precision=round(context_precision, 3),
            context_recall=round(context_recall, 3),
            citation_accuracy=round(citation_accuracy, 3),
            overall_score=round(overall, 3),
            timestamp=datetime.now().isoformat(),
            details={
                "answer_length": len(answer),
                "context_count": len(contexts),
                "citation_count": len(citations or []),
                "has_ground_truth": bool(ground_truth),
            },
        )

        self.results_history.append(result)
        return result

    def evaluate_batch(self, eval_dataset: List[Dict]) -> BatchEvalResult:
        """
        Evaluate a batch of query-answer pairs.
        
        Args:
            eval_dataset: List of dicts with keys: query, answer, contexts, ground_truth, citations
        """
        results = []
        for item in eval_dataset:
            result = self.evaluate_single(
                query=item["query"],
                answer=item["answer"],
                contexts=item.get("contexts", []),
                ground_truth=item.get("ground_truth", ""),
                citations=item.get("citations", []),
            )
            results.append(result)

        if not results:
            return BatchEvalResult(
                total_queries=0,
                avg_faithfulness=0, avg_answer_relevancy=0,
                avg_context_precision=0, avg_context_recall=0,
                avg_citation_accuracy=0, avg_overall_score=0,
                timestamp=datetime.now().isoformat(),
            )

        batch = BatchEvalResult(
            total_queries=len(results),
            avg_faithfulness=round(sum(r.faithfulness for r in results) / len(results), 3),
            avg_answer_relevancy=round(sum(r.answer_relevancy for r in results) / len(results), 3),
            avg_context_precision=round(sum(r.context_precision for r in results) / len(results), 3),
            avg_context_recall=round(sum(r.context_recall for r in results) / len(results), 3),
            avg_citation_accuracy=round(sum(r.citation_accuracy for r in results) / len(results), 3),
            avg_overall_score=round(sum(r.overall_score for r in results) / len(results), 3),
            per_query_results=[r.to_dict() for r in results],
            timestamp=datetime.now().isoformat(),
        )

        return batch

    def get_history(self, limit: int = 50) -> List[Dict]:
        """Get recent evaluation history."""
        return [r.to_dict() for r in self.results_history[-limit:]]

    def save_results(self):
        """Save evaluation results to disk."""
        try:
            data = {
                "results": [r.to_dict() for r in self.results_history],
                "last_updated": datetime.now().isoformat(),
            }
            with open(EVAL_RESULTS_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")

    def _load_history(self):
        """Load evaluation history from disk."""
        if os.path.exists(EVAL_RESULTS_PATH):
            try:
                with open(EVAL_RESULTS_PATH, "r") as f:
                    data = json.load(f)
                for item in data.get("results", []):
                    self.results_history.append(EvalResult(**{
                        k: v for k, v in item.items() if k in EvalResult.__dataclass_fields__
                    }))
            except Exception as e:
                logger.warning(f"Failed to load evaluation history: {e}")

    # ── Metric Implementations ──────────────────────────────────────

    @staticmethod
    def _compute_faithfulness(answer: str, contexts: List[str]) -> float:
        """
        Compute faithfulness: % of answer claims that are supported by context.
        Uses keyword/phrase overlap as a proxy for grounding.
        """
        if not answer or not contexts:
            return 0.0

        # Combine all context
        full_context = " ".join(contexts).lower()
        context_words = set(re.findall(r"\b\w{4,}\b", full_context))

        # Extract claims from answer (sentences with factual content)
        sentences = [s.strip() for s in re.split(r"[.!?\n]+", answer) if len(s.strip()) > 15]
        if not sentences:
            return 1.0

        grounded_count = 0
        for sentence in sentences:
            sent_lower = sentence.lower()
            # Skip meta-sentences
            if any(t in sent_lower for t in ["based on", "retrieved documents", "insufficient", "the documents do not"]):
                grounded_count += 1
                continue

            # Extract key terms from the sentence
            sent_words = set(re.findall(r"\b\w{4,}\b", sent_lower))
            if not sent_words:
                grounded_count += 1
                continue

            # Measure overlap with context
            overlap = len(sent_words & context_words) / len(sent_words)
            if overlap >= 0.3:  # At least 30% of key terms found in context
                grounded_count += 1

        return grounded_count / len(sentences)

    @staticmethod
    def _compute_answer_relevancy(query: str, answer: str) -> float:
        """
        Compute answer relevancy: how well does the answer address the query?
        Uses keyword overlap between query and answer.
        """
        if not answer or not query:
            return 0.0

        query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        answer_words = set(re.findall(r"\b\w{3,}\b", answer.lower()))

        if not query_words:
            return 1.0

        # How many query terms appear in the answer
        overlap = len(query_words & answer_words) / len(query_words)

        # Penalize very short or very long answers
        ideal_length = len(query) * 5  # ~5x query length is reasonable
        length_ratio = min(len(answer) / max(ideal_length, 1), 2.0)
        length_penalty = min(1.0, length_ratio) if length_ratio < 0.2 else 1.0

        # Check for refusal responses
        if "insufficient" in answer.lower() or "do not contain" in answer.lower():
            return 0.3  # Partial credit for honest refusal

        return min(1.0, overlap * 1.5) * length_penalty

    @staticmethod
    def _compute_context_precision(query: str, contexts: List[str]) -> float:
        """
        Compute context precision: % of retrieved contexts that are relevant to the query.
        """
        if not contexts or not query:
            return 0.0

        query_words = set(re.findall(r"\b\w{4,}\b", query.lower()))
        if not query_words:
            return 1.0

        relevant_count = 0
        for ctx in contexts:
            ctx_words = set(re.findall(r"\b\w{4,}\b", ctx.lower()))
            overlap = len(query_words & ctx_words) / len(query_words) if query_words else 0
            if overlap >= 0.15:  # At least 15% keyword overlap = relevant
                relevant_count += 1

        return relevant_count / len(contexts)

    @staticmethod
    def _compute_context_recall(ground_truth: str, contexts: List[str]) -> float:
        """
        Compute context recall: how much of the ground truth is covered by retrieved contexts.
        """
        if not ground_truth or not contexts:
            return 0.0

        gt_words = set(re.findall(r"\b\w{4,}\b", ground_truth.lower()))
        if not gt_words:
            return 1.0

        full_context = " ".join(contexts).lower()
        context_words = set(re.findall(r"\b\w{4,}\b", full_context))

        recall = len(gt_words & context_words) / len(gt_words)
        return min(1.0, recall)

    @staticmethod
    def _compute_citation_accuracy(answer: str, citations: List[Dict]) -> float:
        """
        Compute citation accuracy: % of citations in the answer that are valid.
        Also checks if factual sentences have citations.
        """
        # Count citation markers in the answer
        citation_pattern = r"\[Source:\s*Page\s+\d+,\s*Chunk\s+[^\]]+\]"
        citation_matches = re.findall(citation_pattern, answer)

        if not citation_matches:
            # No citations at all — check if answer has factual content
            sentences = [s.strip() for s in re.split(r"[.!?\n]+", answer) if len(s.strip()) > 20]
            factual = [s for s in sentences if not any(t in s.lower() for t in
                       ["based on", "retrieved", "insufficient", "documents do not"])]
            if not factual:
                return 1.0  # No factual claims = no citations needed
            return 0.2  # Has factual claims but no citations

        # If we have validated citations from the orchestrator
        if citations:
            return min(1.0, len(citations) / len(citation_matches))

        # Otherwise give credit for having citations at all
        return 0.8


# ── Evaluation Dataset ──────────────────────────────────────────────

def get_eval_dataset() -> List[Dict]:
    """
    Get the curated evaluation dataset with ground-truth answers.
    Covers financial queries, legal lookups, table questions, and multi-hop reasoning.
    """
    return [
        {
            "question": "What was Vertex Corporation's total revenue in FY2025?",
            "ground_truth": "Vertex Corporation achieved total revenue of $487.3 million in FY2025, representing a 23.4% year-over-year increase.",
            "source_pages": [1],
            "category": "financial",
        },
        {
            "question": "What is the Q3 2025 net profit margin?",
            "ground_truth": "Q3 2025 had revenue of $127.4M and net profit of $19.8M, resulting in a 15.5% margin.",
            "source_pages": [2],
            "category": "financial_table",
        },
        {
            "question": "Which business segment had the highest growth rate?",
            "ground_truth": "AI & Analytics had the highest growth rate at 39.6%, growing from $89.3M to $124.7M.",
            "source_pages": [1, 2],
            "category": "financial_table",
        },
        {
            "question": "What are the key risk factors for FY2026?",
            "ground_truth": "Key risks include: concentration risk (Cloud 40.7% of revenue), regulatory risk (EU AI Act $12-15M costs), competition risk (3 new entrants), FX risk (28% non-USD), and talent risk (14.2% attrition).",
            "source_pages": [3],
            "category": "financial",
        },
        {
            "question": "What is the base annual retainer in the legal services agreement?",
            "ground_truth": "The base annual retainer is $2,400,000 payable in monthly installments of $200,000.",
            "source_pages": [1],
            "category": "legal",
        },
        {
            "question": "What are the confidentiality terms in the Master Services Agreement?",
            "ground_truth": "Both parties must maintain strict confidentiality for 5 years post-termination. Data processing must comply with GDPR, CCPA, and local laws. Breach notification required within 72 hours.",
            "source_pages": [2],
            "category": "legal",
        },
        {
            "question": "How many paid annual leave days do full-time staff receive?",
            "ground_truth": "Full-time standard staff receive 22 paid annual leave days per fiscal year, accrued monthly.",
            "source_pages": [1],
            "category": "hr",
        },
        {
            "question": "What encryption is used for data at rest in Project Alpha?",
            "ground_truth": "All data at rest is encrypted using AWS KMS with customer-managed keys (CMK) rotated every 90 days.",
            "source_pages": [1],
            "category": "engineering",
        },
        {
            "question": "What is the limitation of liability in the legal agreement?",
            "ground_truth": "Neither party's aggregate liability shall exceed the total fees paid in the prior 12 months. Consequential damages are excluded except for confidentiality breaches.",
            "source_pages": [2],
            "category": "legal",
        },
        {
            "question": "What is the total stockholders equity and total assets?",
            "ground_truth": "Total Stockholders Equity is $623.8M and Total Assets are $1,231.6M as of December 31, 2025.",
            "source_pages": [2],
            "category": "financial_table",
        },
        {
            "question": "What are the current legal proceedings against Vertex Corp?",
            "ground_truth": "Three proceedings: DataShield patent infringement ($15-25M exposure), SEC insider trading investigation, and employment class action ($4.2M settlement reserve).",
            "source_pages": [3],
            "category": "legal",
        },
        {
            "question": "What is the SOC2 audit status?",
            "ground_truth": "SOC2 Type II Readiness Audit is in progress. 9 out of 10 controls are active. Minor logging gaps identified in testing databases. Remediation date: August 30, 2026.",
            "source_pages": [1],
            "category": "compliance",
        },
        {
            "question": "What database does Project Alpha use?",
            "ground_truth": "Project Alpha uses Amazon Aurora PostgreSQL serverless cluster with cross-region read replicas.",
            "source_pages": [1],
            "category": "engineering",
        },
        {
            "question": "What is the maternity and paternity leave policy?",
            "ground_truth": "Maternity leave provides 16 fully paid weeks. Paternity leave provides 4 fully paid weeks.",
            "source_pages": [1],
            "category": "hr",
        },
        {
            "question": "What is the governing law for the legal services agreement?",
            "ground_truth": "The Agreement is governed by the laws of the State of Delaware. Disputes go to JAMS mediation first, then AAA binding arbitration.",
            "source_pages": [2, 3],
            "category": "legal",
        },
    ]


# Global evaluator instance
evaluator = RAGEvaluator()
