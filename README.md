# Smart Cloud Tag

AI-powered cloud storage object tagging using Large Language Models.

## Features

- **Multi-Cloud Support**: AWS S3, Azure Blob Storage, Google Cloud Storage
- **AI-Powered**: Uses OpenAI, Anthropic Claude, or Google Gemini for intelligent tagging
- **Auto-Detection**: Automatically detects storage provider from URI prefix
- **Batch Processing**: Process multiple files with one command
- **Preview Mode**: Preview tags before applying them
- **Custom Prompts**: Use your own LLM prompt templates

## Quick Start

### Installation

#### Basic Installation
```bash
pip install smart_cloud_tag
```

> **Note:** Basic installation includes AWS S3 and OpenAI support. For other cloud providers or LLM providers, use the optional dependencies below.

#### Installation with Optional Dependencies

You can install additional dependencies based on your needs:

```bash
# Install with all optional dependencies (recommended)
pip install smart_cloud_tag[all]

# Install with specific cloud providers
pip install smart_cloud_tag[aws]      # AWS S3 (included by default)
pip install smart_cloud_tag[azure]    # Azure Blob Storage
pip install smart_cloud_tag[gcp]      # Google Cloud Storage

# Install with specific LLM providers
pip install smart_cloud_tag[openai]   # OpenAI (included by default)
pip install smart_cloud_tag[anthropic] # Anthropic Claude
pip install smart_cloud_tag[gemini]   # Google Gemini

# Combine multiple options
pip install smart_cloud_tag[azure,anthropic]  # Azure + Anthropic
pip install smart_cloud_tag[gcp,gemini]       # GCP + Gemini
```

**Installation Options:**
- `[all]` - Installs all optional dependencies (all cloud providers + LLM providers)
- `[aws]` - AWS S3 support (included by default)
- `[azure]` - Azure Blob Storage support
- `[gcp]` - Google Cloud Storage support  
- `[openai]` - OpenAI LLM support (included by default)
- `[anthropic]` - Anthropic Claude LLM support
- `[gemini]` - Google Gemini LLM support
- `[dev]` - Development dependencies (testing, linting, formatting)

### Basic Usage

```python
from smart_cloud_tag import SmartCloudTagger

# Define your tag schema
tags = {
    "document_type": ["invoice", "contract", "report"],
    "department": ["finance", "legal", "hr"],
    "confidential": ["true", "false"]
}

# Initialize tagger (provider auto-detected from URI)
tagger = SmartCloudTagger(
    storage_uri="s3://my-bucket",  # or "az://container" or "gs://bucket"
    tags=tags
)

# Preview tags before applying
preview_result = tagger.preview_tags()
print(f"Preview: {preview_result.summary}")

# Apply tags
result = tagger.apply_tags()
print(f"Applied tags to {result.summary['applied']} objects")
```

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# LLM Provider API Key (used for all providers)
API_KEY=your_api_key_here

# AWS (if using S3)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1

# Azure (if using Blob Storage)
AZURE_STORAGE_CONNECTION_STRING=your_connection_string

# Google Cloud (if using GCS)
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
```

### Supported Storage Providers

| Provider | URI Format | Example |
|----------|------------|---------|
| AWS S3 | `s3://bucket` | `s3://my-documents` |
| Azure Blob | `az://container` | `az://documents` |
| Google Cloud | `gs://bucket` | `gs://my-files` |

### Supported LLM Providers

| Provider | Default Model | Environment Variable |
|----------|---------------|---------------------|
| OpenAI | `gpt-4.1` | `API_KEY` |
| Anthropic | `claude-3-5-sonnet-20241022` | `API_KEY` |
| Google Gemini | `gemini-1.5-pro` | `API_KEY` |

## Advanced Usage

### Custom Prompt Templates

```python
custom_prompt = """
Analyze this document and assign tags based on content.

Filename: {filename}
Content: {content}
Tags to assign: {tags}

Focus on document classification and confidentiality.
Return only tag values separated by commas.
"""

tagger = SmartCloudTagger(
    storage_uri="s3://my-bucket",
    tags=tags,
    custom_prompt_template=custom_prompt
)
```

### Different LLM Providers

```python
# Using Anthropic Claude
tagger = SmartCloudTagger(
    storage_uri="s3://my-bucket",
    tags=tags,
    llm_provider="anthropic"
)

# Using Google Gemini
tagger = SmartCloudTagger(
    storage_uri="s3://my-bucket", 
    tags=tags,
    llm_provider="gemini"
)
```

## Architecture

![Architecture Diagram](assets/Architecture.png)

## Example

![Example](assets/Code.png)

## Development

### Installation from Source

```bash
git clone https://github.com/yourusername/smart_cloud_tag.git
cd smart_cloud_tag
pip install -e ".[all]"
```

**Note**: Use quotes around `".[all]"` to prevent shell expansion issues in zsh and other shells.

### Running Tests

```bash
python -m pytest tests/
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## Support

- 📧 Email: dawarwaqar71@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/DawarWaqar/smart_cloud_tag/issues)