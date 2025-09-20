# Development & Testing

## Parser Validation

### Test Structure
```python
def test_icici_parser():
    """Test ICICI parser returns correct DataFrame"""
    from custom_parsers.icici_parser import parse
    
    result_df = parse("data/icici/icici_sample.pdf")
    expected_df = pd.read_csv("data/icici/result.csv")
    
    assert result_df.equals(expected_df)
```

### Running Tests
```bash
# Run all tests
uv run pytest test_parser.py -v

# Test specific parser
uv run pytest test_parser.py::test_icici_parser -v -s

# Run with detailed output
uv run pytest test_parser.py -v -s --tb=short
```

## Adding New Banks

### 1. Prepare Data Structure
```bash
mkdir data/[bank_name]
# Place sample PDF: data/[bank_name]/[bank_name]_sample.pdf
# Create expected CSV: data/[bank_name]/result.csv
```

### 2. Run Agent
```bash
uv run python agent.py --target [bank_name] --max-retries 5
```

### 3. Validate Results
```bash
# Check generated parser
cat custom_parsers/[bank_name]_parser.py

# Test manually
uv run python -c "
from custom_parsers.[bank_name]_parser import parse
df = parse('data/[bank_name]/[bank_name]_sample.pdf')
print(df.head())
print(df.columns.tolist())
"
```

### 4. Add Test Case
```python
def test_[bank_name]_parser():
    try:
        from custom_parsers.[bank_name]_parser import parse
        
        if os.path.exists("data/[bank_name]/[bank_name]_sample.pdf"):
            result_df = parse("data/[bank_name]/[bank_name]_sample.pdf")
            expected_df = pd.read_csv("data/[bank_name]/result.csv")
            
            assert result_df.equals(expected_df)
    except ImportError:
        pass  # Skip if parser doesn't exist yet
```

## Expected CSV Format

The `result.csv` should contain the exact structure you expect from the parser:

```csv
Date,Description,Debit,Credit,Balance
2024-01-01,Opening Balance,,,10000.00
2024-01-02,ATM Withdrawal,500.00,,9500.00
2024-01-03,Salary Credit,,5000.00,14500.00
```

**Key Requirements**:
- Headers match PDF column structure
- Date format consistent with PDF
- Numeric values as strings with proper formatting
- Empty cells represented as empty strings
- All transactions in chronological order

## Code Quality Guidelines

### Generated Parser Standards
```python
import pandas as pd
import pdfplumber

def parse(pdf_path: str) -> pd.DataFrame:
    """
    Parse [Bank] bank statement PDF
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        pandas.DataFrame with transaction data
    """
    try:
        # Implementation with proper error handling
        pass
    except Exception as e:
        # Return empty DataFrame on error
        return pd.DataFrame()
```

### Best Practices
- **Type Hints**: All functions should include type annotations
- **Error Handling**: Graceful failure with empty DataFrame return
- **Documentation**: Clear docstrings explaining purpose and return format
- **Data Cleaning**: Remove commas from numbers, handle negatives properly
- **Column Consistency**: Maintain exact column names and order

## Debugging Generated Parsers

### Manual Code Review
```python
# Check generated parser logic
import ast
import inspect

from custom_parsers.icici_parser import parse
print(inspect.getsource(parse))

# Test parsing steps manually
import pdfplumber
with pdfplumber.open('data/icici/icici_sample.pdf') as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
        print(f"Tables found: {len(tables)}")
        if tables:
            print("Sample rows:", tables[0][:3])
```

### Common Parser Issues
1. **Column Mapping**: Wrong index mapping to DataFrame columns
2. **Data Types**: Strings vs numeric conversion issues
3. **Date Parsing**: Inconsistent date format handling  
4. **Row Filtering**: Including header/summary rows in data
5. **Empty Values**: Not handling missing data properly

### Performance Optimization
```python
# Limit pages processed for large PDFs
def parse(pdf_path: str) -> pd.DataFrame:
    with pdfplumber.open(pdf_path) as pdf:
        # Process only transaction pages (skip summaries)
        pages_to_process = pdf.pages[1:5]  # Adjust as needed
        # ... rest of implementation
```

## Contributing Guidelines

### Code Style
- Follow PEP 8 style guidelines
- Use meaningful variable names
- Add comments for complex logic
- Keep functions focused and small

### Testing Requirements
- All parsers must pass DataFrame.equals() validation
- Include error handling test cases
- Test with various PDF formats and sizes
- Validate column structure and data types

### Documentation
- Update README for new features
- Add troubleshooting entries for new issues  
- Document any special PDF format requirements
- Include example usage for new CLI options