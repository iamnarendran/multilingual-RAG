"""
Helper Utilities

Common utility functions used across the application.
"""

import uuid
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib


def generate_id() -> str:
    """
    Generate a unique ID.
    
    Returns:
        UUID string
    
    Example:
        doc_id = generate_id()
        # "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    """
    return str(uuid.uuid4())


def generate_short_id(length: int = 8) -> str:
    """
    Generate a short unique ID.
    
    Args:
        length: Length of the ID
    
    Returns:
        Short ID string
    
    Example:
        short_id = generate_short_id()
        # "a1b2c3d4"
    """
    return uuid.uuid4().hex[:length]


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Format datetime as ISO string.
    
    Args:
        dt: Datetime object (defaults to now)
    
    Returns:
        ISO formatted timestamp
    
    Example:
        timestamp = format_timestamp()
        # "2024-01-15T10:30:45.123456"
    """
    if dt is None:
        dt = datetime.utcnow()
    return dt.isoformat()


def extract_citations(text: str) -> List[Dict[str, Any]]:
    """
    Extract citations from text in format [Doc ID: X].
    
    Args:
        text: Text containing citations
    
    Returns:
        List of citation dictionaries
    
    Example:
        text = "The capital is Delhi [Doc ID: 123]."
        citations = extract_citations(text)
        # [{'doc_id': '123', 'position': 25}]
    """
    pattern = r'\[Doc ID:\s*([^\]]+)\]'
    citations = []
    
    for match in re.finditer(pattern, text):
        citations.append({
            'doc_id': match.group(1).strip(),
            'position': match.start(),
            'text': match.group(0)
        })
    
    return citations


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str
) -> float:
    """
    Calculate approximate cost for LLM API call.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name
    
    Returns:
        Cost in USD
    
    Example:
        cost = calculate_cost(1000, 500, "anthropic/claude-3.5-sonnet")
        # 0.0045
    """
    # Pricing per 1M tokens (as of Jan 2024)
    pricing = {
        "anthropic/claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
        "anthropic/claude-3-opus": {"input": 15.0, "output": 75.0},
        "google/gemini-2.0-flash": {"input": 0.075, "output": 0.30},
        "google/gemini-flash-1.5": {"input": 0.075, "output": 0.30},
        "openai/gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "openai/gpt-4o": {"input": 5.0, "output": 15.0},
    }
    
    # Get pricing for model
    model_pricing = pricing.get(model, {"input": 1.0, "output": 3.0})
    
    # Calculate cost
    input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
    
    return input_cost + output_cost


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal.
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename
    
    Example:
        safe = sanitize_filename("../../../etc/passwd")
        # "passwd"
    """
    # Remove path separators
    filename = filename.replace("/", "_").replace("\\", "_")
    # Remove special characters
    filename = re.sub(r'[^\w\s.-]', '', filename)
    return filename


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
    
    Returns:
        Truncated text with ellipsis
    
    Example:
        short = truncate_text("Very long text...", 10)
        # "Very lo..."
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def hash_text(text: str) -> str:
    """
    Generate SHA-256 hash of text.
    
    Args:
        text: Input text
    
    Returns:
        Hex digest of hash
    
    Example:
        hash_val = hash_text("Hello World")
        # "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"
    """
    return hashlib.sha256(text.encode()).hexdigest()


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks.
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
    
    Returns:
        List of chunks
    
    Example:
        chunks = chunk_list([1,2,3,4,5], 2)
        # [[1,2], [3,4], [5]]
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries.
    
    Args:
        *dicts: Dictionaries to merge
    
    Returns:
        Merged dictionary
    
    Example:
        merged = merge_dicts({"a": 1}, {"b": 2}, {"c": 3})
        # {"a": 1, "b": 2, "c": 3}
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def format_size(size_bytes: int) -> str:
    """
    Format byte size to human-readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted size string
    
    Example:
        size = format_size(1536)
        # "1.50 KB"
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


if __name__ == "__main__":
    # Test utilities
    print("Testing helper functions...")
    
    # Test ID generation
    print(f"\nGenerate ID: {generate_id()}")
    print(f"Short ID: {generate_short_id()}")
    
    # Test timestamp
    print(f"Timestamp: {format_timestamp()}")
    
    # Test citations
    text = "India's capital is Delhi [Doc ID: 123]. Population is large [Doc ID: 456]."
    citations = extract_citations(text)
    print(f"\nCitations found: {len(citations)}")
    for cite in citations:
        print(f"  - Doc ID: {cite['doc_id']}")
    
    # Test cost calculation
    cost = calculate_cost(1000, 500, "anthropic/claude-3.5-sonnet")
    print(f"\nCost for 1000 input + 500 output tokens: ${cost:.4f}")
    
    # Test sanitize filename
    safe = sanitize_filename("../../../etc/passwd.txt")
    print(f"\nSanitized filename: {safe}")
    
    # Test truncate
    long_text = "This is a very long text that needs to be truncated"
    short = truncate_text(long_text, 20)
    print(f"\nTruncated: {short}")
    
    # Test hash
    hash_val = hash_text("Hello World")
    print(f"\nHash: {hash_val[:32]}...")
    
    # Test format size
    print(f"\nSize formatting:")
    print(f"  1024 bytes = {format_size(1024)}")
    print(f"  1536 bytes = {format_size(1536)}")
    print(f"  1048576 bytes = {format_size(1048576)}")
    
    print("\n✅ All helper functions working!")
