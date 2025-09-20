import pandas as pd
import os

def test_icici_parser():
    """Test ICICI parser returns correct DataFrame"""
    from custom_parsers.icici_parser import parse
    
    result_df = parse("data/icici/icici_sample.pdf")
    expected_df = pd.read_csv("data/icici/result.csv")
    
    assert result_df.equals(expected_df)


"""
def test_hdfc_parser():
    try:
        from custom_parsers.hdfc_parser import parse
        
        if os.path.exists("data/hdfc/hdfc_sample.pdf"):
            result_df = parse("data/hdfc/hdfc_sample.pdf")
            expected_df = pd.read_csv("data/hdfc/result.csv")
            
            assert result_df.equals(expected_df)
    except ImportError:
        # Skip if parser doesn't exist yet
        pass

def test_sbi_parser():
    try:
        from custom_parsers.sbi_parser import parse
        
        if os.path.exists("data/sbi/sbi_sample.pdf"):
            result_df = parse("data/sbi/sbi_sample.pdf")
            expected_df = pd.read_csv("data/sbi/result.csv")
            
            assert result_df.equals(expected_df)
    except ImportError:
        # Skip if parser doesn't exist yet
        pass

def test_parser_returns_dataframe():
    try:
        from custom_parsers.icici_parser import parse
        
        if os.path.exists("data/icici/icici_sample.pdf"):
            result = parse("data/icici/icici_sample.pdf")
            assert isinstance(result, pd.DataFrame), f"Expected DataFrame, got {type(result)}"
    except ImportError:
        # Skip if parser doesn't exist yet
        pass

"""