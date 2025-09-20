# Technical Architecture

## LangGraph Workflow Design

The agent implements a stateful workflow using LangGraph's StateGraph pattern with conditional routing and retry mechanisms.

### State Schema

```python
class AgentState(TypedDict):
    # Input
    pdf_path: str
    target_bank: str
    
    # Analysis results
    pdf_content: str
    column_headers: List[str]
    sample_rows: List[Dict[str, str]]
    raw_sample_text: List[str]
    
    # Code generation
    generated_code: str
    parser_file_path: str
    
    # Execution results
    execution_success: bool
    generated_df: pd.DataFrame
    output_csv_path: str
    
    # Validation
    expected_csv_path: str
    comparison_result: Dict[str, Any]
    
    # Retry logic
    retry_count: int
    max_retries: int
    errors: List[str]
    failure_type: str
    final_status: str
```

## Node Functions

### 1. Analysis Node
```python
def analyze_node(state: AgentState) -> Dict[str, Any]:
```
- Extracts PDF text using `pdfplumber`
- Uses LLM to identify table structure and headers
- Extracts sample transaction rows for code generation context
- Returns structured data for downstream processing

**Key Features**:
- Intelligent table detection via LLM reasoning
- Fallback to default schema if extraction fails
- Limited to first 3 pages for performance

### 2. Generation Node  
```python
def generate_node(state: AgentState) -> Dict[str, Any]:
```
- Constructs detailed prompt with PDF structure context
- Generates Python parser code using LLM
- Ensures proper function signature and imports
- Returns clean code without markdown formatting

**Prompt Strategy**:
- Includes sample rows and column headers
- Specifies exact function interface requirements
- Provides error handling and data cleaning instructions

### 3. Save Node
```python
def save_node(state: AgentState) -> Dict[str, Any]:
```
- Creates `custom_parsers/` directory
- Saves generated code to `{bank}_parser.py`
- Uses consistent naming convention
- Enables dynamic import for execution

### 4. Execute Node
```python
def execute_node(state: AgentState) -> Dict[str, Any]:
```
- Dynamically loads parser module using `importlib`
- Executes `parse()` function with PDF path
- Validates returned DataFrame structure
- Saves output CSV for comparison

**Error Handling**:
- Captures execution exceptions
- Validates function existence and signature
- Checks DataFrame validity and column structure

### 5. Compare Node
```python
def compare_node(state: AgentState) -> Dict[str, Any]:
```
- Loads expected CSV from `data/{bank}/result.csv`
- Performs DataFrame.equals() comparison
- Handles missing expected files gracefully
- Provides detailed mismatch information

### 6. Refine Node
```python
def refine_node(state: AgentState) -> Dict[str, Any]:
```
- Analyzes failure type (execution vs comparison)
- Constructs context-aware refinement prompt
- Includes previous errors and LLM-extracted structure
- Generates improved code with specific fixes

**Refinement Strategy**:
- Different prompts for execution vs comparison failures
- Includes original LLM analysis for consistency
- Focuses on common parsing issues (column mapping, data types)

### 7. Finalize Node
```python
def finalize_node(state: AgentState) -> Dict[str, Any]:
```
- Sets final status based on execution and comparison results
- Provides summary of success/failure
- Enables clean workflow termination

## Routing Logic

### After Execution
```python
def route_after_execution(state: AgentState) -> Literal["compare", "refine", "finalize"]:
    if state.get("execution_success"):
        return "compare"
    elif state.get("retry_count", 0) < state.get("max_retries", 3):
        return "refine" 
    else:
        return "finalize"
```

### After Comparison
```python
def route_after_comparison(state: AgentState) -> Literal["refine", "finalize"]:
    comparison_status = state.get("comparison_result", {}).get("status")
    
    if comparison_status in ["passed", "skipped"]:
        return "finalize"
    elif state.get("retry_count", 0) < state.get("max_retries", 3):
        return "refine"
    else:
        return "finalize"
```

### After Refinement
```python
def route_after_refine(state: AgentState) -> Literal["save"]:
    return "save"  # Always retry through save -> execute cycle
```

## Workflow Compilation

```python
def build_workflow():
    workflow = StateGraph(AgentState)
    
    # Add all nodes
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
    
    # Linear initial flow
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", "save")
    workflow.add_edge("save", "execute")
    
    # Conditional routing for retry loops
    workflow.add_conditional_edges("execute", route_after_execution)
    workflow.add_conditional_edges("compare", route_after_comparison)
    workflow.add_conditional_edges("refine", route_after_refine)
    
    workflow.add_edge("finalize", END)
    
    return workflow.compile()
```

## Key Design Principles

1. **Stateful Processing**: Rich state maintains context across retry cycles
2. **Self-Correction**: Automatic error analysis and code refinement  
3. **Graceful Degradation**: Continues with defaults when components fail
4. **DataFrame-First**: Direct pandas processing without CSV intermediates
5. **Modular Design**: Clear separation of concerns across workflow nodes
6. **Robust Routing**: Intelligent decision-making for retry vs termination