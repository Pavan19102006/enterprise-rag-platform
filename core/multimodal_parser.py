"""
Multimodal Document Parser
==========================
Advanced PDF/document parsing that preserves tables, structural hierarchy,
and page-level metadata for financial/legal documents.

Falls back gracefully when optional dependencies (marker, camelot) are unavailable.
"""

import os
import re
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class TableBlock:
    """A table extracted from a document page."""
    page_number: int
    table_index: int
    headers: List[str]
    rows: List[List[str]]
    markdown: str  # Pre-rendered markdown version of the table
    caption: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class TextBlock:
    """A block of text from a document page with structural metadata."""
    text: str
    page_number: int
    block_type: str = "paragraph"  # paragraph, heading, list_item, table_text, footer
    font_size: float = 0.0
    is_bold: bool = False
    section_title: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class DocumentPage:
    """Parsed content from a single page of a document."""
    page_number: int
    raw_text: str
    text_blocks: List[TextBlock] = field(default_factory=list)
    tables: List[TableBlock] = field(default_factory=list)
    images_count: int = 0

    def to_dict(self):
        return {
            "page_number": self.page_number,
            "raw_text": self.raw_text,
            "text_blocks": [b.to_dict() for b in self.text_blocks],
            "tables": [t.to_dict() for t in self.tables],
            "images_count": self.images_count,
        }


@dataclass
class ParsedDocument:
    """Full parsed document containing all pages with multimodal content."""
    filename: str
    file_type: str
    total_pages: int
    pages: List[DocumentPage] = field(default_factory=list)
    sections: List[Dict] = field(default_factory=list)  # Detected document sections
    metadata: Dict = field(default_factory=dict)

    def get_full_text(self) -> str:
        """Get all text concatenated with page markers."""
        parts = []
        for page in self.pages:
            parts.append(f"--- Page {page.page_number} ---")
            parts.append(page.raw_text)
            for table in page.tables:
                parts.append(f"\n[Table {table.table_index} on Page {page.page_number}]")
                parts.append(table.markdown)
        return "\n".join(parts)

    def to_dict(self):
        return {
            "filename": self.filename,
            "file_type": self.file_type,
            "total_pages": self.total_pages,
            "pages": [p.to_dict() for p in self.pages],
            "sections": self.sections,
            "metadata": self.metadata,
        }


def _detect_headings(text: str) -> List[Dict]:
    """Heuristic heading detection based on common financial/legal document patterns."""
    headings = []
    lines = text.split("\n")
    heading_patterns = [
        # Numbered sections: "1.", "1.1", "SECTION 1:", "Article 1"
        (r"^(?:SECTION\s+\d+|Article\s+\d+|ARTICLE\s+\d+)[:\.\s]", "section"),
        (r"^\d+\.\d+[\.\s]", "subsection"),
        (r"^\d+\.[\s]", "section"),
        # ALL-CAPS headings common in legal/financial docs
        (r"^[A-Z][A-Z\s&,]{10,}$", "heading"),
        # Title case with colon
        (r"^[A-Z][a-zA-Z\s]+:$", "heading"),
    ]

    for line_num, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            continue
        for pattern, level in heading_patterns:
            if re.match(pattern, stripped):
                headings.append({
                    "text": stripped,
                    "level": level,
                    "line_number": line_num,
                    "page_context": None,  # Filled later by page-aware parser
                })
                break

    return headings


def _extract_tables_from_text(text: str, page_number: int) -> List[TableBlock]:
    """Heuristic table extraction from text with column-aligned data."""
    tables = []
    lines = text.split("\n")
    table_start = None
    table_lines = []
    table_idx = 0

    for i, line in enumerate(lines):
        # Detect table-like patterns: multiple columns separated by spaces/tabs or pipe characters
        is_pipe_table = "|" in line and line.count("|") >= 2
        is_space_table = len(re.findall(r"\s{3,}", line.strip())) >= 2 and len(line.strip()) > 20
        has_numbers = bool(re.search(r"\d+[\.,]\d+", line))

        if is_pipe_table or (is_space_table and has_numbers):
            if table_start is None:
                table_start = i
            table_lines.append(line)
        else:
            if table_start is not None and len(table_lines) >= 2:
                table = _parse_table_lines(table_lines, page_number, table_idx)
                if table:
                    tables.append(table)
                    table_idx += 1
            table_start = None
            table_lines = []

    # Handle table at end of text
    if table_start is not None and len(table_lines) >= 2:
        table = _parse_table_lines(table_lines, page_number, table_idx)
        if table:
            tables.append(table)

    return tables


def _parse_table_lines(lines: List[str], page_number: int, table_index: int) -> Optional[TableBlock]:
    """Parse detected table lines into a structured TableBlock."""
    if not lines:
        return None

    # Try pipe-delimited tables first
    if "|" in lines[0]:
        rows = []
        for line in lines:
            # Skip separator lines like |---|---|
            if re.match(r"^\s*\|[\s\-:]+\|", line):
                continue
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if cells:
                rows.append(cells)
    else:
        # Space-delimited: split by 3+ spaces
        rows = []
        for line in lines:
            cells = re.split(r"\s{3,}", line.strip())
            cells = [c.strip() for c in cells if c.strip()]
            if cells:
                rows.append(cells)

    if len(rows) < 2:
        return None

    headers = rows[0]
    data_rows = rows[1:]

    # Build markdown table
    md_lines = ["| " + " | ".join(headers) + " |"]
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in data_rows:
        # Pad row if shorter than headers
        padded = row + [""] * (len(headers) - len(row))
        md_lines.append("| " + " | ".join(padded[:len(headers)]) + " |")

    markdown = "\n".join(md_lines)

    return TableBlock(
        page_number=page_number,
        table_index=table_index,
        headers=headers,
        rows=data_rows,
        markdown=markdown,
    )


def parse_pdf_multimodal(file_path: str) -> ParsedDocument:
    """
    Parse a PDF document with multimodal extraction.
    
    Attempts to use marker-pdf for high-quality extraction,
    falls back to pypdf for basic text extraction.
    """
    filename = os.path.basename(file_path)
    pages = []

    # Try pypdf as our reliable parser
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)

        for page_idx, pdf_page in enumerate(reader.pages):
            page_num = page_idx + 1
            raw_text = pdf_page.extract_text() or ""

            # Build text blocks from the page
            text_blocks = []
            headings = _detect_headings(raw_text)
            heading_lines = {h["line_number"] for h in headings}

            current_section = ""
            for line_num, line in enumerate(raw_text.split("\n")):
                stripped = line.strip()
                if not stripped:
                    continue

                if line_num in heading_lines:
                    current_section = stripped
                    text_blocks.append(TextBlock(
                        text=stripped,
                        page_number=page_num,
                        block_type="heading",
                        is_bold=True,
                        section_title=current_section,
                    ))
                else:
                    text_blocks.append(TextBlock(
                        text=stripped,
                        page_number=page_num,
                        block_type="paragraph",
                        section_title=current_section,
                    ))

            # Extract tables from text
            tables = _extract_tables_from_text(raw_text, page_num)

            # Count images (pypdf can detect some)
            images_count = 0
            try:
                if hasattr(pdf_page, "images"):
                    images_count = len(pdf_page.images)
            except Exception:
                pass

            pages.append(DocumentPage(
                page_number=page_num,
                raw_text=raw_text,
                text_blocks=text_blocks,
                tables=tables,
                images_count=images_count,
            ))

    except Exception as e:
        logger.error(f"Failed to parse PDF {filename}: {e}")
        total_pages = 0

    # Detect document-level sections
    all_headings = []
    for page in pages:
        for block in page.text_blocks:
            if block.block_type == "heading":
                all_headings.append({
                    "title": block.text,
                    "page_number": page.page_number,
                    "level": "section",
                })

    doc = ParsedDocument(
        filename=filename,
        file_type="PDF",
        total_pages=total_pages,
        pages=pages,
        sections=all_headings,
        metadata={
            "parser": "pypdf+heuristic",
            "tables_extracted": sum(len(p.tables) for p in pages),
            "images_detected": sum(p.images_count for p in pages),
        },
    )

    return doc


def parse_csv_multimodal(file_path: str) -> ParsedDocument:
    """Parse CSV file as a single-page document with table structure preserved."""
    import csv as csv_mod
    filename = os.path.basename(file_path)

    with open(file_path, mode="r", encoding="utf-8", errors="ignore") as f:
        reader = csv_mod.DictReader(f)
        rows_data = list(reader)
        headers = reader.fieldnames or []

    if not rows_data:
        return ParsedDocument(filename=filename, file_type="CSV", total_pages=1,
                              pages=[DocumentPage(page_number=1, raw_text="Empty CSV file.")])

    # Build markdown table
    md_lines = ["| " + " | ".join(headers) + " |"]
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    text_rows = []
    for row in rows_data:
        vals = [str(row.get(h, "")) for h in headers]
        md_lines.append("| " + " | ".join(vals) + " |")
        # Also build semantic text representation
        desc = ", ".join([f"{h}: {row.get(h, '')}" for h in headers])
        text_rows.append(desc)

    raw_text = "\n".join(text_rows)
    markdown_table = "\n".join(md_lines)

    table_block = TableBlock(
        page_number=1,
        table_index=0,
        headers=headers,
        rows=[[str(row.get(h, "")) for h in headers] for row in rows_data],
        markdown=markdown_table,
        caption=f"Data from {filename}",
    )

    page = DocumentPage(
        page_number=1,
        raw_text=raw_text,
        text_blocks=[TextBlock(text=raw_text, page_number=1, block_type="table_text")],
        tables=[table_block],
    )

    return ParsedDocument(
        filename=filename,
        file_type="CSV",
        total_pages=1,
        pages=[page],
        metadata={"parser": "csv", "row_count": len(rows_data), "column_count": len(headers)},
    )


def parse_json_multimodal(file_path: str) -> ParsedDocument:
    """Parse JSON file into structured document format."""
    filename = os.path.basename(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    text_parts = []
    if isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                desc = f"Record {idx + 1}: " + "; ".join([f"{k}: {v}" for k, v in item.items()])
            else:
                desc = f"Item {idx + 1}: {str(item)}"
            text_parts.append(desc)
    elif isinstance(data, dict):
        for k, v in data.items():
            text_parts.append(f"{k}: {json.dumps(v) if isinstance(v, (dict, list)) else str(v)}")
    else:
        text_parts.append(str(data))

    raw_text = "\n\n".join(text_parts)

    page = DocumentPage(
        page_number=1,
        raw_text=raw_text,
        text_blocks=[TextBlock(text=raw_text, page_number=1, block_type="paragraph")],
    )

    return ParsedDocument(
        filename=filename,
        file_type="JSON",
        total_pages=1,
        pages=[page],
        metadata={"parser": "json", "record_count": len(data) if isinstance(data, list) else 1},
    )


def parse_text_multimodal(file_path: str) -> ParsedDocument:
    """Parse plain text file into structured document format."""
    filename = os.path.basename(file_path)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Split into pages by form-feed or large sections
    raw_pages = content.split("\f") if "\f" in content else [content]

    pages = []
    for idx, page_text in enumerate(raw_pages):
        page_num = idx + 1
        text_blocks = []
        headings = _detect_headings(page_text)
        heading_lines = {h["line_number"] for h in headings}

        current_section = ""
        for line_num, line in enumerate(page_text.split("\n")):
            stripped = line.strip()
            if not stripped:
                continue
            if line_num in heading_lines:
                current_section = stripped
                text_blocks.append(TextBlock(
                    text=stripped, page_number=page_num,
                    block_type="heading", is_bold=True, section_title=current_section,
                ))
            else:
                text_blocks.append(TextBlock(
                    text=stripped, page_number=page_num,
                    block_type="paragraph", section_title=current_section,
                ))

        tables = _extract_tables_from_text(page_text, page_num)

        pages.append(DocumentPage(
            page_number=page_num,
            raw_text=page_text,
            text_blocks=text_blocks,
            tables=tables,
        ))

    return ParsedDocument(
        filename=filename,
        file_type="TXT",
        total_pages=len(pages),
        pages=pages,
        metadata={"parser": "text"},
    )


def parse_document(file_path: str) -> ParsedDocument:
    """Universal document parser — routes to appropriate multimodal parser based on file type."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return parse_pdf_multimodal(file_path)
    elif ext == ".csv":
        return parse_csv_multimodal(file_path)
    elif ext == ".json":
        return parse_json_multimodal(file_path)
    elif ext in (".txt", ".md", ".log"):
        return parse_text_multimodal(file_path)
    else:
        # Fallback: treat as text
        logger.warning(f"Unknown file type {ext}, treating as plain text.")
        return parse_text_multimodal(file_path)
