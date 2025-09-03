"""
Configuration constants and default values for smart_cloud_tag.
"""

# Default LLM prompt template
DEFAULT_PROMPT_TEMPLATE = """Analyze the following content and generate exactly {num_tags} tag values.

Tag keys and constraints:
{constraints_text}{filename_context}
Content preview:
{content_preview}

Instructions:
1. Generate exactly {num_tags} values, one for each tag key
2. Return only the values in order, separated by commas
3. Keep values concise (1-3 words when possible)
4. Make values relevant and descriptive for the content
5. For tags with allowed values, use only those values
6. For tags without allowed values, deduce appropriate values based on content and key name
7. If you see any abbreviations, interpret them according to context, following these examples:
    Example: "BOL#: 7782-CA-TOR-2025" → "bill_of_lading" (a shipping document)
    Example: "PO# 5567-AB" → "purchase_order" (a procurement document)

File Context Guidelines:
- Consider the filename as additional context for tagging decisions
- Use filename context to inform your understanding of the document type and content
- However, if the filename is not relevant to the content, ignore it

Example output format:
value1, value2, value3

Generated tags:"""

# Default model configurations
DEFAULT_MODELS = {
    "openai": "gpt-5",
    "anthropic": "claude-3-opus-4.1",
    "gemini": "gemini-1.5-pro",
}

# Default configuration values
DEFAULT_MAX_BYTES = 5000
DEFAULT_LLM_PROVIDER = "openai"
