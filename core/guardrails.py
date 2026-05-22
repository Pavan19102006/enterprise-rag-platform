import re

# Standard suspicious injection signatures
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous\s+)?instructions",
    r"bypass\s+security",
    r"system\s+override",
    r"you\s+are\s+now\s+an?\s+unrestricted",
    r"dan\s+mode",
    r"jailbreak",
    r"forget\s+what\s+you\s+were\s+told",
    r"reveal\s+your\s+system\s+prompt",
    r"output\s+the\s+full\s+prompt",
    r"override\s+role\s+restrictions",
    r"do\s+not\s+enforce\s+rbac",
    r"show\s+secret\s+keys",
    r"act\s+as\s+a\s+developer\s+mode"
]

# Sensitive Data Redaction Regex
PII_PATTERNS = {
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "PRIVATE_KEY": r"-----BEGIN\s+PRIVATE\s+KEY-----[\s\S]+?-----END\s+PRIVATE\s+KEY-----"
}

def check_prompt_injection(query_text: str) -> tuple:
    """Scan query for potential jailbreak / prompt injection attacks.
    Returns (is_flagged, threat_reason)."""
    normalized_query = query_text.lower()
    
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, normalized_query):
            return True, f"Triggered injection rule: '{pattern}'"
            
    # Check for direct SQL injection patterns
    sql_patterns = [
        r"union\s+select",
        r"drop\s+table",
        r"delete\s+from",
        r"insert\s+into",
        r"or\s+1\s*=\s*1"
    ]
    for pattern in sql_patterns:
        if re.search(pattern, normalized_query):
            return True, f"Potential SQL injection signature: '{pattern}'"
            
    return False, None

def redact_sensitive_data(text: str) -> tuple:
    """Mask PII or sensitive patterns to prevent leakage to external APIs.
    Returns (redacted_text, redaction_occurred)."""
    redacted_text = text
    occurred = False
    
    for name, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, redacted_text)
        if matches:
            occurred = True
            redacted_text = re.sub(pattern, f"[REDACTED_{name}]", redacted_text)
            
    return redacted_text, occurred
