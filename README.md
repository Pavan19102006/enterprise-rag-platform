# 🛡️ Vertex Corp — Enterprise AI RAG Platform

A production-grade **Retrieval-Augmented Generation (RAG)** platform with enterprise security guardrails, RBAC-based access control, real-time LLM integration, and citation-grounded responses.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

## ✨ Features

- **Real RAG Pipeline**: User Query → TF-IDF Embedding Search → Vector DB → Retrieved Chunks → LLM
- **Live LLM Integration**: Groq Cloud (Llama-3.3-70B) and Google Gemini support
- **RBAC Security**: Role-based access control (Intern, Engineer, HR, Finance, Auditor, Executive)
- **Prompt Injection Shield**: Detects and blocks SQL injection and prompt manipulation attacks
- **DLP Redactor**: Automatically masks sensitive PII (SSN, credit cards, emails)
- **Citation Validation**: Every fact in responses is verified against source chunks
- **Anti-Hallucination Engine**: Confidence scoring with automatic rejection of ungrounded responses
- **Multi-format Ingestion**: PDF, CSV, JSON, TXT files with automatic chunking

## 🏗️ Architecture

```
User Query
    ↓
Prompt Injection Shield
    ↓
DLP Redactor
    ↓
Intent Classifier (SQL / Vector / Compliance)
    ↓
TF-IDF Embedding Search → Vector Database
    ↓
RBAC Filter (role-based chunk access)
    ↓
Retrieved Chunks → LLM (Groq/Gemini/Offline)
    ↓
Citation Validator + Confidence Scorer
    ↓
Grounded Response
```

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/Pavan19102006/enterprise-rag-platform.git
cd enterprise-rag-platform

# Install dependencies
pip install -r requirements.txt

# Run ingestion pipeline
python core/ingestion.py

# Start the app
streamlit run app.py
```

## 🔑 API Configuration

Set your LLM API key in `.env`:
```
GROQ_API_KEY=your_groq_api_key_here
```

Or configure it live in the sidebar LLM settings panel.

## 📁 Data Structure

```
data/
├── finance/       # Financial reports (Finance Confidential)
├── hr/            # HR policies (HR Confidential)
├── engineering/   # Tech specs (Engineering Confidential)
├── compliance/    # Audit logs (Compliance Audit)
├── logs/          # System logs (Highly Restricted)
├── policies/      # Access policies (Public)
├── executive/     # Executive reports (Highly Restricted)
└── raw/           # Auto-generated mock documents
```

## 👥 Role Access Matrix

| Role | Public | Engineering | HR | Finance | Compliance | Restricted |
|------|--------|-------------|----|---------|-----------:|------------|
| Intern | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Engineer | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| HR Officer | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Finance | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Auditor | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Executive | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

## 🧪 Tests

```bash
python test_platform.py
```

## 🛠️ Tech Stack

- **Frontend**: Streamlit with custom CSS
- **Vector DB**: TF-IDF + Cosine Similarity (scikit-learn)
- **Database**: SQLite (RBAC, metadata, audit logs)
- **LLM**: Groq Cloud (Llama-3.3-70B) / Google Gemini / Offline Simulation
- **Security**: JWT authentication, prompt injection detection, DLP redaction
