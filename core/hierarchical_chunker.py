"""
Hierarchical Small-to-Big Chunker
==================================
Three-tier chunking strategy:
  - Document-level: full document metadata
  - Section-level (parent): ~1500 chars, passed to LLM for context
  - Paragraph-level (child): ~400 chars, used for retrieval precision

Tables are kept as atomic units and never split mid-table.
Each chunk maintains parent-child relationships for context expansion.
"""

import re
import uuid
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

from config import CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalChunk:
    """A chunk with hierarchy metadata for Small-to-Big retrieval."""
    chunk_id: str
    text: str
    level: str  # "document", "section", "paragraph"
    page_number: int
    section_title: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "HierarchicalChunk":
        return cls(**data)


class HierarchicalChunker:
    """
    Implements Small-to-Big chunking:
    - Child chunks (small, ~400 chars) are embedded and retrieved
    - Parent chunks (large, ~1500 chars) provide context to the LLM
    - Document-level metadata ties everything together
    """

    def __init__(
        self,
        child_chunk_size: int = CHILD_CHUNK_SIZE,
        child_overlap: int = CHILD_CHUNK_OVERLAP,
        parent_chunk_size: int = PARENT_CHUNK_SIZE,
        parent_overlap: int = PARENT_CHUNK_OVERLAP,
    ):
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap
        self.parent_chunk_size = parent_chunk_size
        self.parent_overlap = parent_overlap
        # Hierarchy storage: chunk_id → HierarchicalChunk
        self._hierarchy: Dict[str, HierarchicalChunk] = {}

    def chunk_document(self, parsed_doc) -> Tuple[List[HierarchicalChunk], List[HierarchicalChunk]]:
        """
        Chunk a ParsedDocument into hierarchical chunks.
        
        Returns:
            Tuple of (child_chunks, parent_chunks)
            - child_chunks: small chunks for embedding/retrieval
            - parent_chunks: large chunks for LLM context expansion
        """
        self._hierarchy.clear()
        
        # Step 1: Create document-level chunk
        doc_chunk = HierarchicalChunk(
            chunk_id=f"doc-{_short_id()}",
            text=f"Document: {parsed_doc.filename} ({parsed_doc.total_pages} pages)",
            level="document",
            page_number=0,
            section_title="",
            metadata={
                "filename": parsed_doc.filename,
                "file_type": parsed_doc.file_type,
                "total_pages": parsed_doc.total_pages,
            },
        )
        self._hierarchy[doc_chunk.chunk_id] = doc_chunk

        parent_chunks = []
        child_chunks = []

        # Step 2: Process each page and build section-level parent chunks
        for page in parsed_doc.pages:
            page_num = page.page_number

            # Group text blocks by section
            sections = self._group_by_section(page.text_blocks, page_num)

            for section_title, blocks in sections.items():
                section_text = "\n".join([b.text for b in blocks])
                
                if not section_text.strip():
                    continue

                # Create parent chunks from section text
                parent_texts = self._split_text(section_text, self.parent_chunk_size, self.parent_overlap)

                for p_idx, p_text in enumerate(parent_texts):
                    parent_id = f"parent-p{page_num}-s{_short_id()}"
                    parent_chunk = HierarchicalChunk(
                        chunk_id=parent_id,
                        text=p_text,
                        level="section",
                        page_number=page_num,
                        section_title=section_title,
                        parent_id=doc_chunk.chunk_id,
                        metadata={
                            "filename": parsed_doc.filename,
                            "page_number": page_num,
                            "section_title": section_title,
                            "parent_index": p_idx,
                        },
                    )
                    self._hierarchy[parent_id] = parent_chunk
                    doc_chunk.children_ids.append(parent_id)

                    # Step 3: Create child chunks from each parent
                    child_texts = self._split_text(p_text, self.child_chunk_size, self.child_overlap)

                    for c_idx, c_text in enumerate(child_texts):
                        child_id = f"child-p{page_num}-{_short_id()}"
                        child_chunk = HierarchicalChunk(
                            chunk_id=child_id,
                            text=c_text,
                            level="paragraph",
                            page_number=page_num,
                            section_title=section_title,
                            parent_id=parent_id,
                            metadata={
                                "filename": parsed_doc.filename,
                                "page_number": page_num,
                                "section_title": section_title,
                                "child_index": c_idx,
                                "parent_chunk_id": parent_id,
                            },
                        )
                        self._hierarchy[child_id] = child_chunk
                        parent_chunk.children_ids.append(child_id)
                        child_chunks.append(child_chunk)

                    parent_chunks.append(parent_chunk)

            # Step 4: Handle tables as atomic chunks (never split)
            for table in page.tables:
                table_text = f"[Table on Page {page_num}]\n{table.markdown}"
                
                # Table as a parent chunk
                table_parent_id = f"table-parent-p{page_num}-t{table.table_index}-{_short_id()}"
                table_parent = HierarchicalChunk(
                    chunk_id=table_parent_id,
                    text=table_text,
                    level="section",
                    page_number=page_num,
                    section_title=f"Table {table.table_index + 1}",
                    parent_id=doc_chunk.chunk_id,
                    metadata={
                        "filename": parsed_doc.filename,
                        "page_number": page_num,
                        "is_table": True,
                        "table_index": table.table_index,
                        "headers": table.headers,
                    },
                )
                self._hierarchy[table_parent_id] = table_parent
                doc_chunk.children_ids.append(table_parent_id)
                parent_chunks.append(table_parent)

                # Table as a single child chunk (atomic, not split)
                table_child_id = f"table-child-p{page_num}-t{table.table_index}-{_short_id()}"
                table_child = HierarchicalChunk(
                    chunk_id=table_child_id,
                    text=table_text,
                    level="paragraph",
                    page_number=page_num,
                    section_title=f"Table {table.table_index + 1}",
                    parent_id=table_parent_id,
                    metadata={
                        "filename": parsed_doc.filename,
                        "page_number": page_num,
                        "is_table": True,
                        "table_index": table.table_index,
                        "parent_chunk_id": table_parent_id,
                    },
                )
                self._hierarchy[table_child_id] = table_child
                table_parent.children_ids.append(table_child_id)
                child_chunks.append(table_child)

        logger.info(
            f"Chunked '{parsed_doc.filename}': "
            f"{len(parent_chunks)} parent chunks, {len(child_chunks)} child chunks"
        )

        return child_chunks, parent_chunks

    def get_parent_context(self, chunk_id: str) -> Optional[str]:
        """Get the parent section text for a child chunk (Small-to-Big expansion)."""
        chunk = self._hierarchy.get(chunk_id)
        if not chunk or not chunk.parent_id:
            return None
        parent = self._hierarchy.get(chunk.parent_id)
        return parent.text if parent else None

    def get_parent_chunk(self, chunk_id: str) -> Optional[HierarchicalChunk]:
        """Get the parent HierarchicalChunk for a given chunk_id."""
        chunk = self._hierarchy.get(chunk_id)
        if not chunk or not chunk.parent_id:
            return None
        return self._hierarchy.get(chunk.parent_id)

    def get_sibling_chunks(self, chunk_id: str) -> List[HierarchicalChunk]:
        """Get sibling chunks (same parent) for expanded context."""
        chunk = self._hierarchy.get(chunk_id)
        if not chunk or not chunk.parent_id:
            return []
        parent = self._hierarchy.get(chunk.parent_id)
        if not parent:
            return []
        siblings = []
        for sid in parent.children_ids:
            if sid != chunk_id:
                sibling = self._hierarchy.get(sid)
                if sibling:
                    siblings.append(sibling)
        return siblings

    def register_chunks(self, child_chunks: List[HierarchicalChunk], parent_chunks: List[HierarchicalChunk]):
        """Register externally created chunks into the hierarchy for lookup."""
        for chunk in parent_chunks:
            self._hierarchy[chunk.chunk_id] = chunk
        for chunk in child_chunks:
            self._hierarchy[chunk.chunk_id] = chunk

    def get_hierarchy_stats(self) -> Dict:
        """Get statistics about the current chunk hierarchy."""
        levels = {"document": 0, "section": 0, "paragraph": 0}
        for chunk in self._hierarchy.values():
            levels[chunk.level] = levels.get(chunk.level, 0) + 1
        return {
            "total_chunks": len(self._hierarchy),
            "by_level": levels,
        }

    def export_hierarchy(self) -> Dict:
        """Export the full hierarchy as a serializable dict."""
        return {
            chunk_id: chunk.to_dict()
            for chunk_id, chunk in self._hierarchy.items()
        }

    def import_hierarchy(self, data: Dict):
        """Import hierarchy from a serialized dict."""
        self._hierarchy.clear()
        for chunk_id, chunk_data in data.items():
            self._hierarchy[chunk_id] = HierarchicalChunk.from_dict(chunk_data)

    @staticmethod
    def _group_by_section(text_blocks, page_number: int) -> Dict[str, list]:
        """Group text blocks by their section title."""
        sections = {}
        current_section = f"Page {page_number} Content"

        for block in text_blocks:
            if block.block_type == "heading" and block.text.strip():
                current_section = block.text.strip()

            if current_section not in sections:
                sections[current_section] = []
            sections[current_section].append(block)

        return sections

    @staticmethod
    def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks, respecting sentence boundaries."""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        # Try splitting by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += (" " + sentence if current_chunk else sentence)
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                # Start new chunk with overlap from previous
                if overlap > 0 and current_chunk:
                    # Take last `overlap` characters from previous chunk
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    # Find sentence boundary in overlap
                    boundary = overlap_text.rfind(". ")
                    if boundary > 0:
                        overlap_text = overlap_text[boundary + 2:]
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Fallback: if sentence splitting produced nothing useful, do character-level
        if not chunks:
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunks.append(text[start:end].strip())
                start += chunk_size - overlap
                if start >= len(text):
                    break

        return [c for c in chunks if c.strip()]


def _short_id() -> str:
    """Generate a short unique ID."""
    return uuid.uuid4().hex[:8]
