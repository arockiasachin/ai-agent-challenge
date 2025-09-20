# Bank Statement Parser Agent

An intelligent LLM-powered agent that automatically generates custom parsers for bank statement PDFs using LangGraph workflow orchestration.

## Quick Start (5 Steps)

### 1. Clone and Setup with uv
```bash
git clone https://github.com/apurv-korefi/ai-agent-challenge.git
cd ai-agent-challenge

# Install uv (ultra-fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
```


Create and activate the virtual environment:
```bash
uv venv
source .venv/bin/activate   # On Windows, use: .venv\Scripts\activate
```
#### Install dependencies:
```bash
uv pip install -r requirements.txt
```
#### Sync environment (optional, if dependencies are already installed or to ensure versions match):
```bash
uv sync
```

### 3. Configure API Key
```bash
echo "DEEPSEEK_API_KEY=your_api_key_here" > .env
```

### 4. Run Agent for ICICI
```bash
uv run agent.py --target icici
```
#### To Run on New Bank (e.g., SBI)
```bash
# Place your PDF in data/sbi/sbi_sample.pdf
uv run agent.py --target sbi --max-retries 3
```
#### To run custom_parsers separately
```bash
uv run custom_parsers/icici_parser.py
```

### 5. Test Generated Parser
```bash
uv run pytest test_parser.py::test_icici_parser -v
```

## Agent Architecture

The agent operates as a **self-correcting LangGraph workflow** that intelligently processes bank statement PDFs through specialized nodes: **Analysis** extracts PDF structure using LLM reasoning, **Code Generation** synthesizes custom parsers, **Execution** runs and validates the code, while **Comparison** checks output against expected results. When execution fails or validation doesn't match, the **Refinement** node automatically analyzes errors with context-aware prompting and regenerates improved code, creating an autonomous retry loop (up to 3 cycles) that ensures robust parser generation across different bank statement formats without manual intervention.

## Additional Documentation

- 📋 **[Project Structure & Features](docs/structure_docs.md)** - File organization, parser contract, and feature overview
- 🔧 **[Advanced Usage & Configuration](docs/usage_docs.md)** - CLI options, API providers, custom settings
- 🏗️ **[Technical Architecture](docs/architecture_docs.md)** - Detailed workflow states, node functions, routing logic
- 🧪 **[Development & Testing](docs/development_docs.md)** - Parser validation, extending to new banks, contribution guidelines

---
