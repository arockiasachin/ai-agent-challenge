# Advanced Usage & Configuration

## CLI Options

```bash
# Basic usage
uv run python agent.py --target icici

# With custom retry limit
uv run python agent.py --target hdfc --max-retries 5

# Test specific parser
uv run pytest test_parser.py::test_icici_parser -v -s
```

## API Providers

The agent uses DeepSeek's chat model for code generation. Get free credits at:

### Primary Provider
- **DeepSeek API** - Fast, reliable LLM service
  - Sign up: [deepseek.com](https://deepseek.com)
  - Model: `deepseek-chat`
  - Rate limits: Generous free tier

### Alternative Providers
- **Groq** - Ultra-fast inference
  - Sign up: [groq.com](https://groq.com)
  - Models: Llama, Mixtral variants
  
- **Google Gemini API** - Reliable performance  
  - Sign up: [Google AI Studio](https://makersuite.google.com)
  - Models: Gemini Pro, Flash

## Configuration

### Environment Variables
```bash
# Required
DEEPSEEK_API_KEY=your_deepseek_key

# Optional alternatives
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
```

### Custom Settings
```python
# In agent.py, modify these defaults:
llm = ChatDeepSeek(
    model="deepseek-chat", 
    temperature=0.1
)

# Retry configuration
initial_state = {
    "max_retries": 3,  # Increase for complex PDFs
    # ... other settings
}
```

## uv Package Manager

**Why uv?**
- **10-100x faster** than pip for dependency resolution
- **Built-in virtual environment** management
- **Project runner** with automatic environment handling
- **Drop-in pip replacement** with better caching

**Installation:**
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

**Common Commands:**
```bash
uv venv                    # Create virtual environment
uv pip install package    # Install package
uv run python script.py   # Run with auto-venv
uv add package            # Add to project dependencies
```