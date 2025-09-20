# Project Structure & Features

## Features

- **Autonomous Code Generation**: LLM analyzes PDF structure and writes custom `pdfplumber`-based parsers
- **Self-Debugging Loops**: Automatically refines code based on execution errors and comparison failures  
- **DataFrame-First Architecture**: Direct pandas DataFrame processing without intermediate CSV files
- **Multi-Bank Support**: Works with any bank statement format through intelligent structure detection
- **Robust Error Handling**: Comprehensive logging and graceful failure management

## Project Structure

```
├── agent.py              # Main agent workflow
├── test_parser.py              # Parser validation tests
├── custom_parsers/             # Generated parser modules
│   └── icici_parser.py        # Auto-generated ICICI parser
├── data/                      # Sample PDFs and expected outputs
│   ├── icici/
│   │   ├── icici_sample.pdf   # Sample ICICI statement
│   │   └── result.csv         # Expected parsing output
│   └── [bank_name]/           # Additional bank data
├── output/                    # Agent-generated CSV outputs
├── docs/                      # Documentation files
└── .env                       # API keys (create this)
```

## Parser Contract

Generated parsers follow a strict interface:

```python
def parse(pdf_path: str) -> pd.DataFrame:
    """
    Parse bank statement PDF and return structured DataFrame
    
    Args:
        pdf_path: Path to bank statement PDF
        
    Returns:
        pandas.DataFrame with columns matching expected schema
    """
```

## Success Criteria

✅ **Autonomous Operation**: Agent completes full cycle without manual intervention  
✅ **Code Quality**: Generated parsers include proper typing, imports, and error handling  
✅ **Test Validation**: `DataFrame.equals()` assertion passes for expected output  
✅ **Multi-Bank Support**: Same agent works for different bank statement formats