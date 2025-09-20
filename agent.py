import os
import re
import pandas as pd
from typing import TypedDict, Literal, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
import importlib.util
import logging
import json
#from langchain_google_genai import ChatGoogleGenerativeAI

# Configuration
load_dotenv()
os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatDeepSeek(model="deepseek-chat", temperature=0.1)

#llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0)


# ===== STATE DEFINITION =====
class AgentState(TypedDict):
    """Enhanced state with DataFrame-based processing"""
    pdf_path: str
    target_bank: str
    pdf_content: str
    column_headers: List[str]
    sample_rows: List[Dict[str, str]]
    raw_sample_text: List[str]
    generated_code: str
    parser_file_path: str
    output_csv_path: str  # For saving output
    expected_csv_path: str
    execution_success: bool
    generated_df: pd.DataFrame  # Direct DataFrame storage
    comparison_result: Dict[str, Any]
    retry_count: int
    max_retries: int
    errors: List[str]
    final_status: str
    failure_type: str

# ===== CORE FUNCTIONS =====
def extract_pdf_content(pdf_path: str) -> str:
    """Extract raw text content from PDF"""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages[:3])
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return ""

def extract_headers_and_rows_with_llm(pdf_content: str) -> tuple[List[str], List[Dict[str, str]], List[str]]:
    """Use LLM to intelligently extract headers and top 5 transaction rows"""
    
    prompt = f"""Analyze this bank statement PDF content and extract:
1. Column headers for the transaction table
2. The first 5 transaction rows with data

PDF CONTENT:
{pdf_content[:4000]}

INSTRUCTIONS:
1. Identify the transaction table structure (look for patterns with dates, amounts, descriptions)
2. Extract exact column headers as they appear
3. Extract first 5 complete transaction rows after headers
4. Focus on actual transaction data, not summary information

Return response in exact JSON format:
{{
    "headers": [],
    "rows": [],
    "raw_lines": []
}}

Important:
- Include ALL columns (Reference, Cheque No, etc.)
- Keep empty fields as empty strings ""
- Preserve exact date format from PDF
- Keep amounts as strings with original formatting
- For raw_lines, provide actual text lines as they appear

Return ONLY the JSON object, no other text."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        result_text = response.content
        
        # Clean response to get pure JSON
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        
        result = json.loads(result_text.strip())
        headers = result.get("headers", ["Date", "Description", "Debit", "Credit", "Balance"])
        rows = result.get("rows", [])
        raw_lines = result.get("raw_lines", [])
        
        return headers, rows, raw_lines
        
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"LLM extraction failed: {e}")
        return ["Date", "Description", "Debit", "Credit", "Balance"], [], []

# ===== WORKFLOW NODES =====
def analyze_node(state: AgentState) -> Dict[str, Any]:
    """Analyze PDF using LLM to extract structure and sample rows"""
    
    pdf_content = extract_pdf_content(state["pdf_path"])
    if not pdf_content:
        logger.error("Failed to extract PDF content")
        print("STEP 1: PDF Analysis - FAILED")
        return {
            "pdf_content": "",
            "column_headers": ["Date", "Description", "Debit", "Credit", "Balance"],
            "sample_rows": [],
            "raw_sample_text": [],
            "generated_df": pd.DataFrame()
        }
    
    headers, rows, raw_lines = extract_headers_and_rows_with_llm(pdf_content)
    print(f"STEP 1: PDF Analysis - COMPLETED ({len(headers)} columns, {len(rows)} sample rows)")
    
    return {
        "pdf_content": pdf_content[:3000],
        "column_headers": headers,
        "sample_rows": rows,
        "raw_sample_text": raw_lines,
        "generated_df": pd.DataFrame()
    }

def generate_node(state: AgentState) -> Dict[str, Any]:
    """Generate parser code that returns DataFrame directly"""
    
    # Build examples section from LLM-extracted rows
    examples_section = ""
    if state.get("sample_rows"):
        examples_section = f"""

LLM-extracted transaction structure:
Columns: {state['column_headers']}

Sample transactions (as parsed by LLM):
""" + "\n".join(f"Row {i}: {json.dumps(row, indent=2)}" for i, row in enumerate(state["sample_rows"][:3], 1))
        
        if state.get("raw_sample_text"):
            examples_section += "\n\nRaw text lines from PDF:\n" + "\n".join(
                f"Line {i}: {line}" for i, line in enumerate(state["raw_sample_text"][:3], 1)
            )
    
    prompt = f"""Generate a Python parser for bank statement PDF that returns a pandas DataFrame.

LLM-EXTRACTED STRUCTURE:
Column Headers: {', '.join(state['column_headers'])}
{examples_section}

CRITICAL REQUIREMENTS:

Function Signature (MUST MATCH EXACTLY):
    def parse(pdf_path: str) -> pd.DataFrame
        - Uses pdfplumber to extract tables
        - Returns pandas DataFrame directly (NOT CSV path)
        - DataFrame columns: {', '.join(state['column_headers'])}

Import Requirements:
    import pandas as pd
    import pdfplumber

Data Formatting:
    - Dates: keep as-is (same format as examples)
    - Amounts: remove commas (e.g. 1,234 → 1234)
    - Negatives: convert (123) or Dr/Cr to -123
    - Empty fields: use NaN
    - Strip whitespace
    - Preserve row order

Parsing Strategy:
    - Extract tables from each page
    - Map rows directly to target columns
    - Clean and normalize values
    - Skip headers, summaries, and malformed lines
    - Return DataFrame with exact column structure

Error Handling:
    - Ignore non-table lines safely
    - Return empty DataFrame if parsing fails
    - Ensure final DataFrame matches example structure exactly

Example return:
    return pd.DataFrame(processed_rows, columns={state['column_headers']})

Provide ONLY the complete Python code with proper imports."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    code = response.content
    
    # Clean code blocks
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]
    
    print("STEP 2: Code Generation - COMPLETED")
    return {"generated_code": code.strip()}

def save_node(state: AgentState) -> Dict[str, Any]:
    """Save parser to file with correct naming convention"""
    parser_dir = "custom_parsers"
    os.makedirs(parser_dir, exist_ok=True)
    
    # Use consistent naming (icici -> icici_parser.py)
    parser_file = os.path.join(parser_dir, f"{state['target_bank']}_parser.py")
    
    with open(parser_file, "w", encoding="utf-8") as f:
        f.write(state["generated_code"])
    
    print("STEP 3: Parser Save - COMPLETED")
    return {"parser_file_path": parser_file}

def execute_node(state: AgentState) -> Dict[str, Any]:    
    try:
        # Load and execute the parser module
        spec = importlib.util.spec_from_file_location("parser", state["parser_file_path"])
        parser_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parser_module)
        
        if not hasattr(parser_module, 'parse'):
            raise AttributeError("parse function not found")
        
        # Get DataFrame directly from parser (no CSV intermediate step)
        generated_df = parser_module.parse(state["pdf_path"])
        
        if generated_df is None or generated_df.empty:
            raise ValueError("Parser returned empty or None DataFrame")
        
        # Verify columns match what LLM extracted
        expected_cols = state["column_headers"]
        actual_cols = generated_df.columns.tolist()
        if set(expected_cols) != set(actual_cols):
            logger.warning(f"Column mismatch - Expected: {expected_cols}, Got: {actual_cols}")
        
        # Save DataFrame to CSV for compatibility (but don't use for processing)
        os.makedirs("output", exist_ok=True)
        output_path = state["output_csv_path"]
        generated_df.to_csv(output_path, index=False)
        
        print(f"STEP 4: Parser Execution - COMPLETED ({len(generated_df)} rows generated)")
        return {
            "execution_success": True,
            "generated_df": generated_df,
            "output_csv_path": output_path,
            "failure_type": ""
        }
        
    except Exception as e:
        error_msg = f"Execution failed: {str(e)}"
        logger.error(error_msg)
        
        errors = state.get("errors", [])
        errors.append(error_msg)
        
        print(f"STEP 4: Parser Execution - FAILED (attempt {state.get('retry_count', 0) + 1})")
        return {
            "execution_success": False,
            "generated_df": pd.DataFrame(),
            "errors": errors,
            "failure_type": "execution"
        }

def refine_node(state: AgentState) -> Dict[str, Any]:
    """Refine code based on errors and failure type"""
    retry_count = state.get("retry_count", 0) + 1
    failure_type = state.get("failure_type", "execution")
    
    last_error = state.get("errors", ["Unknown error"])[-1]
    
    # Build detailed context from LLM-extracted data
    llm_context = ""
    if state.get("sample_rows"):
        llm_context = f"""

LLM successfully extracted this structure:
Headers: {state['column_headers']}
Sample data:
""" + "\n".join(f"Row {i}: {json.dumps(row, indent=2)}" for i, row in enumerate(state["sample_rows"], 1))
        llm_context += "\n\nYour parser should produce exactly this structure!\n"
    
    # Build specific instructions based on failure type
    if failure_type == "comparison":
        comparison_details = state.get("comparison_result", {}).get("details", {})
        failure_specific_context = f"""
COMPARISON FAILURE ANALYSIS:
- Generated shape: {comparison_details.get('generated_shape', 'unknown')}
- Expected shape: {comparison_details.get('expected_shape', 'unknown')}
- Column match: {comparison_details.get('column_match', 'unknown')}

The parser executed successfully but the output doesn't match expected results.
Focus on data extraction accuracy and formatting consistency.
"""
    else:
        failure_specific_context = """
EXECUTION FAILURE:
The parser code failed to run properly. Focus on:
1. Syntax errors
2. Import issues
3. Function structure (must return pd.DataFrame)
4. Exception handling
"""
    
    prompt = f"""Fix this parser code that failed with: {last_error}

FAILURE TYPE: {failure_type.upper()}
{failure_specific_context}

Current code:
{state['generated_code']}

{llm_context}

CRITICAL FIXES NEEDED:

1. Function Signature (MANDATORY):
   def parse(pdf_path: str) -> pd.DataFrame
   - MUST return pd.DataFrame directly
   - NOT a string path to CSV file

2. Match LLM-extracted structure EXACTLY:
   - Columns must be: {', '.join(state['column_headers'])}
   - Data format must match the sample rows shown above

3. Required Imports:
   import pandas as pd
   import pdfplumber

4. Transaction Detection:
   - Look at the raw text lines and sample rows
   - Identify the pattern that distinguishes transaction lines

5. Value Extraction:
   - Parse each line to extract values for ALL columns
   - Empty values should be empty strings ""
   - Preserve the exact format from LLM examples

6. Common Issues:
   - Column count mismatch
   - Wrong value mapping to columns
   - Missing or extra columns in output
   - Date format not matching examples
   - Return type must be pd.DataFrame, not string

The LLM has already successfully parsed the structure - your code just needs to replicate it for the entire PDF!

Provide ONLY the corrected Python code."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    code = response.content
    
    # Clean code
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]
    
    print(f"STEP 5: Code Refinement - COMPLETED (attempt {retry_count})")
    return {
        "generated_code": code.strip(),
        "retry_count": retry_count
    }

def compare_node(state: AgentState) -> Dict[str, Any]:
    """Simplified comparison using direct DataFrame comparison"""
    
    if not os.path.exists(state["expected_csv_path"]):
        print("STEP 6: CSV Comparison - SKIPPED (no expected file)")
        return {
            "comparison_result": {"status": "skipped"},
            "final_status": "success"
        }
    
    try:
        # Use the generated DataFrame directly (no CSV loading)
        
        generated_df = state.get("generated_df", pd.DataFrame())
        expected_df = pd.read_csv(state["expected_csv_path"])
        if generated_df.empty:
            raise ValueError("Generated DataFrame is empty")
        
        # Simple equals comparison
        if generated_df.equals(expected_df):
            print("STEP 6: CSV Comparison - PASSED")
            return {
                "comparison_result": {"status": "passed"},
                "final_status": "success",
                "failure_type": ""
            }
        else:
            print(f"STEP 6: CSV Comparison - FAILED (Generated: {generated_df.shape}, Expected: {expected_df.shape})")
            comparison_error = f"DataFrame comparison failed - Generated: {generated_df.shape}, Expected: {expected_df.shape}"
            errors = state.get("errors", [])
            errors.append(comparison_error)
            
            return {
                "comparison_result": {"status": "failed"},
                "errors": errors,
                "failure_type": "comparison"
            }
            
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        comparison_error = f"Comparison error: {str(e)}"
        errors = state.get("errors", [])
        errors.append(comparison_error)
        
        print("STEP 6: CSV Comparison - ERROR")
        return {
            "comparison_result": {"status": "error", "error": str(e)},
            "errors": errors,
            "failure_type": "comparison"
        }

def finalize_node(state: AgentState) -> Dict[str, Any]:
    """Set final status based on execution results"""
    if state.get("execution_success") and state.get("comparison_result", {}).get("status") in ["passed", "skipped"]:
        return {"final_status": "success"}
    else:
        return {"final_status": "failed"}

# ===== ROUTING LOGIC =====
def route_after_execution(state: AgentState) -> Literal["compare", "refine", "finalize"]:
    """Enhanced routing after execution"""
    if state.get("execution_success"):
        return "compare"
    elif state.get("retry_count", 0) < state.get("max_retries", 3):
        return "refine"
    else:
        return "finalize"

def route_after_comparison(state: AgentState) -> Literal["refine", "finalize"]:
    """Enhanced routing after comparison"""
    comparison_status = state.get("comparison_result", {}).get("status")
    
    if comparison_status in ["passed", "skipped"]:
        return "finalize"
    elif state.get("retry_count", 0) < state.get("max_retries", 3):
        return "refine"
    else:
        return "finalize"

def route_after_refine(state: AgentState) -> Literal["save"]:
    """After refinement, always go back to save (which leads to execute)"""
    return "save"

# ===== BUILD WORKFLOW =====
def build_workflow():
    """Build enhanced workflow with proper retry loops"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    for node_name, node_func in [
        ("analyze", analyze_node),
        ("generate", generate_node),
        ("save", save_node),
        ("execute", execute_node),
        ("refine", refine_node),
        ("compare", compare_node),
        ("finalize", finalize_node)
    ]:
        workflow.add_node(node_name, node_func)
    
    # Linear flow for initial attempt
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", "save")
    workflow.add_edge("save", "execute")
    
    # Enhanced routing
    workflow.add_conditional_edges("execute", route_after_execution, {
        "compare": "compare", "refine": "refine", "finalize": "finalize"
    })
    
    workflow.add_conditional_edges("compare", route_after_comparison, {
        "refine": "refine", "finalize": "finalize"
    })
    
    workflow.add_conditional_edges("refine", route_after_refine, {"save": "save"})
    
    workflow.add_edge("finalize", END)
    
    return workflow.compile()

# ===== MAIN FUNCTION =====
def run_agent(pdf_path: str, target_bank: str = "icici", max_retries: int = 3):
    """Run the enhanced agent with DataFrame-based processing"""
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found: {pdf_path}")
        return {"error": "PDF not found"}
    
    # Initialize enhanced state
    initial_state = {
        "pdf_path": pdf_path,
        "target_bank": target_bank,
        "output_csv_path": f"output/{target_bank}_Output.csv",
        "expected_csv_path": f"data/{target_bank}/result.csv",
        "pdf_content": "",
        "column_headers": [],
        "sample_rows": [],
        "raw_sample_text": [],
        "generated_code": "",
        "parser_file_path": "",
        "execution_success": False,
        "generated_df": pd.DataFrame(),
        "comparison_result": {},
        "retry_count": 0,
        "max_retries": max_retries,
        "errors": [],
        "final_status": "processing",
        "failure_type": ""
    }
    
    os.makedirs("output", exist_ok=True)
    app = build_workflow()
    
    try:
        final_state = app.invoke(initial_state)
        
        # Final result assertion
        print("\n" + "="*60)
        print("FINAL RESULT:")
        if final_state["final_status"] == "success":
            print("ASSERTION: SUCCESS")
            print(f"Parser saved to: {final_state.get('parser_file_path', 'N/A')}")
            print(f"Output CSV: {final_state.get('output_csv_path', 'N/A')}")
            
            generated_df = final_state.get('generated_df', pd.DataFrame())
            if not generated_df.empty:
                print(f"Generated DataFrame shape: {generated_df.shape}")
            
            comparison_status = final_state.get("comparison_result", {}).get("status")
            if comparison_status == "passed":
                print("CSV comparison: PASSED")
            elif comparison_status == "skipped":
                print("CSV comparison: SKIPPED (no expected file)")
        else:
            print("ASSERTION: FAILED")
            if final_state.get("errors"):
                print(f"Last error: {final_state['errors'][-1]}")
            print(f"Total retry attempts: {final_state.get('retry_count', 0)}")
        print("="*60)
        
        return final_state
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        print("ASSERTION: WORKFLOW ERROR")
        return {"error": str(e)}

# ===== CLI INTERFACE =====
if __name__ == "__main__":
    import sys
    import argparse
    
    if len(sys.argv) == 1:
        result = run_agent("data/icici/icici_sample.pdf", "icici", max_retries=3)
    else:
        parser = argparse.ArgumentParser(description="Enhanced LLM-powered Bank Statement Parser Agent")
        parser.add_argument("--target", required=True, help="Bank name")
        parser.add_argument("--max-retries", type=int, default=3, help="Maximum retry attempts (default: 3)")
        
        args = parser.parse_args()
        pdf_path = f"data/{args.target}/{args.target}_sample.pdf"
        
        result = run_agent(pdf_path, args.target, args.max_retries)
        
        sys.exit(0 if result.get("final_status") == "success" else 1)