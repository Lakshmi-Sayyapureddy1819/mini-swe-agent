#!/usr/bin/env python3
"""
Karbon AI Challenge - Agent-as-Coder Implementation
====================================================

This agent autonomously writes custom parsers for bank statement PDFs using LangGraph
and Google Gemini API. It follows a plan → code → test → fix loop with up to 3 attempts.

Author: AI Agent
Date: October 2025
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, TypedDict
import traceback

# Core dependencies
import pandas as pd
import google.generativeai as genai
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# PDF processing
import pdfplumber
import PyPDF2
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State definition for the coding agent"""
    target_bank: str
    pdf_path: str
    csv_path: str
    pdf_text: str
    csv_data: pd.DataFrame
    parser_code: str
    test_result: Dict[str, Any]
    attempt_count: int
    max_attempts: int
    error_message: str
    success: bool
    plan: str

class BankStatementAgent:
    """
    Autonomous coding agent that writes custom parsers for bank statement PDFs.
    
    The agent follows a structured workflow:
    1. Plan - Analyze PDF and CSV to understand structure
    2. Code - Generate parser implementation
    3. Test - Execute and validate parser
    4. Fix - Debug and improve on failures (up to 3 attempts)
    """
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        """Initialize the agent with Gemini API configuration"""
        # Configure Gemini API
        api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Setup paths
        self.project_root = Path(__file__).parent
        self.custom_parsers_dir = self.project_root / "custom_parsers"
        self.data_dir = self.project_root / "data"
        
        # Ensure directories exist
        self.custom_parsers_dir.mkdir(exist_ok=True)
        (self.custom_parsers_dir / "__init__.py").touch(exist_ok=True)
        
        # Build workflow graph
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for the coding agent"""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("load_data", self._load_data_node)
        graph.add_node("plan", self._plan_node)
        graph.add_node("code", self._code_node)
        graph.add_node("test", self._test_node)
        graph.add_node("fix", self._fix_node)
        graph.add_node("success", self._success_node)
        graph.add_node("failure", self._failure_node)
        
        # Define workflow edges
        graph.add_edge(START, "load_data")
        graph.add_edge("load_data", "plan")
        graph.add_edge("plan", "code")
        graph.add_edge("code", "test")
        
        # Conditional edges for test results
        graph.add_conditional_edges(
            "test",
            self._should_retry,
            {
                "success": "success",
                "retry": "fix",
                "failure": "failure"
            }
        )
        
        graph.add_edge("fix", "code")
        graph.add_edge("success", END)
        graph.add_edge("failure", END)
        
        return graph.compile(checkpointer=MemorySaver())
    
    def _load_data_node(self, state: AgentState) -> AgentState:
        """Load and preprocess PDF and CSV data"""
        logger.info(f"Loading data for {state['target_bank']} bank")
        
        try:
            # Load PDF text
            pdf_text = self._extract_pdf_text(state['pdf_path'])
            state['pdf_text'] = pdf_text
            
            # Load expected CSV
            csv_data = pd.read_csv(state['csv_path'])
            state['csv_data'] = csv_data
            
            logger.info(f"Successfully loaded PDF ({len(pdf_text)} chars) and CSV ({len(csv_data)} rows)")
            
        except Exception as e:
            state['error_message'] = f"Data loading failed: {str(e)}"
            logger.error(state['error_message'])
            
        return state
    
    def _plan_node(self, state: AgentState) -> AgentState:
        """Analyze data structure and create implementation plan"""
        logger.info("Creating implementation plan...")
        
        prompt = f"""
        Analyze this bank statement data and create a detailed plan for parsing {state['target_bank']} statements.

        PDF TEXT SAMPLE (first 2000 chars):
        {state['pdf_text'][:2000]}
        
        EXPECTED CSV COLUMNS:
        {list(state['csv_data'].columns)}
        
        CSV SAMPLE DATA:
        {state['csv_data'].head(3).to_string()}
        
        Create a detailed plan that identifies:
        1. Transaction line patterns in the PDF
        2. Date format used
        3. Amount extraction logic
        4. Description parsing approach
        5. Any special handling needed for this bank's format
        
        Provide a concrete implementation strategy.
        """
        
        try:
            response = self.model.generate_content(prompt)
            state['plan'] = response.text
            logger.info("Plan created successfully")
            
        except Exception as e:
            state['error_message'] = f"Planning failed: {str(e)}"
            logger.error(state['error_message'])
            
        return state
    
    def _code_node(self, state: AgentState) -> AgentState:
        """Generate parser code based on the plan"""
        logger.info(f"Generating parser code (attempt {state['attempt_count'] + 1})")
        
        error_context = ""
        if state['attempt_count'] > 0 and state['error_message']:
            error_context = f"""
            PREVIOUS ERROR TO FIX:
            {state['error_message']}
            
            PREVIOUS CODE:
            {state.get('parser_code', 'No previous code')}
            """
        
        prompt = f"""
        Generate a complete Python parser for {state['target_bank']} bank statements.
        
        IMPLEMENTATION PLAN:
        {state['plan']}
        
        {error_context}
        
        PDF TEXT SAMPLE:
        {state['pdf_text'][:1500]}
        
        TARGET CSV FORMAT:
        Columns: {list(state['csv_data'].columns)}
        Sample: {state['csv_data'].head(2).to_string()}
        
        Generate COMPLETE, WORKING Python code with this exact structure:
        
        ```python
        import pandas as pd
        import re
        from datetime import datetime
        from typing import List, Dict, Any
        import pdfplumber
        
        def parse(pdf_path: str) -> pd.DataFrame:
            \"\"\"
            Parse {state['target_bank']} bank statement PDF and return DataFrame.
            
            Args:
                pdf_path: Path to the PDF file
                
            Returns:
                pd.DataFrame with columns matching expected CSV format
            \"\"\"
            # Your implementation here
            transactions = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        # Parse transactions from text
                        page_transactions = extract_transactions_from_page(text)
                        transactions.extend(page_transactions)
            
            # Convert to DataFrame with proper column names
            df = pd.DataFrame(transactions)
            return df
        
        def extract_transactions_from_page(page_text: str) -> List[Dict[str, Any]]:
            \"\"\"Extract transactions from a single page of text\"\"\"
            # Implementation based on the analysis
            pass
        ```
        
        Requirements:
        1. Use pdfplumber for PDF reading
        2. Return DataFrame with EXACT column names from target CSV
        3. Handle date parsing correctly
        4. Extract amounts as float values
        5. Include robust error handling
        6. Make the code production-ready with proper typing
        
        Return ONLY the complete Python code, no explanations.
        """
        
        try:
            response = self.model.generate_content(prompt)
            code = self._extract_code_from_response(response.text)
            state['parser_code'] = code
            state['attempt_count'] += 1
            logger.info("Parser code generated successfully")
            
        except Exception as e:
            state['error_message'] = f"Code generation failed: {str(e)}"
            logger.error(state['error_message'])
            
        return state
    
    def _test_node(self, state: AgentState) -> AgentState:
        """Test the generated parser code"""
        logger.info("Testing generated parser...")
        
        try:
            # Write parser code to file
            parser_file = self.custom_parsers_dir / f"{state['target_bank']}_parser.py"
            with open(parser_file, 'w') as f:
                f.write(state['parser_code'])
            
            # Import and test the parser
            sys.path.insert(0, str(self.custom_parsers_dir))
            
            try:
                # Dynamic import of the generated parser
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    f"{state['target_bank']}_parser", 
                    parser_file
                )
                parser_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(parser_module)
                
                # Execute parser
                result_df = parser_module.parse(state['pdf_path'])
                
                # Compare with expected CSV
                expected_df = state['csv_data']
                
                # Validate structure
                if list(result_df.columns) != list(expected_df.columns):
                    raise ValueError(f"Column mismatch. Expected: {list(expected_df.columns)}, Got: {list(result_df.columns)}")
                
                # Check if DataFrames are equal (allowing some tolerance for float comparison)
                try:
                    pd.testing.assert_frame_equal(result_df, expected_df, check_exact=False, atol=0.01)
                    state['test_result'] = {
                        'success': True,
                        'message': 'Parser test passed successfully',
                        'result_shape': result_df.shape,
                        'expected_shape': expected_df.shape
                    }
                    state['success'] = True
                    logger.info("Parser test PASSED!")
                    
                except AssertionError as e:
                    state['test_result'] = {
                        'success': False,
                        'message': f'DataFrame comparison failed: {str(e)}',
                        'result_shape': result_df.shape,
                        'expected_shape': expected_df.shape,
                        'sample_result': result_df.head(2).to_dict(),
                        'sample_expected': expected_df.head(2).to_dict()
                    }
                    logger.warning("Parser test failed - DataFrame mismatch")
                
            except Exception as import_error:
                state['test_result'] = {
                    'success': False,
                    'message': f'Parser execution failed: {str(import_error)}',
                    'traceback': traceback.format_exc()
                }
                logger.error(f"Parser execution failed: {import_error}")
            
        except Exception as e:
            state['test_result'] = {
                'success': False,
                'message': f'Testing setup failed: {str(e)}',
                'traceback': traceback.format_exc()
            }
            logger.error(f"Testing failed: {e}")
        
        return state
    
    def _fix_node(self, state: AgentState) -> AgentState:
        """Analyze test failures and prepare for retry"""
        logger.info("Analyzing failures and preparing fix...")
        
        test_result = state['test_result']
        error_message = f"""
        Test failed with: {test_result['message']}
        
        Additional details:
        {json.dumps(test_result, indent=2, default=str)}
        """
        
        state['error_message'] = error_message
        logger.info("Prepared error context for next attempt")
        
        return state
    
    def _success_node(self, state: AgentState) -> AgentState:
        """Handle successful completion"""
        logger.info("🎉 Parser generation completed successfully!")
        
        parser_file = self.custom_parsers_dir / f"{state['target_bank']}_parser.py"
        logger.info(f"Parser saved to: {parser_file}")
        
        return state
    
    def _failure_node(self, state: AgentState) -> AgentState:
        """Handle final failure after max attempts"""
        logger.error(f"❌ Failed to generate working parser after {state['max_attempts']} attempts")
        logger.error(f"Final error: {state.get('error_message', 'Unknown error')}")
        
        return state
    
    def _should_retry(self, state: AgentState) -> str:
        """Determine if agent should retry or stop"""
        if state.get('success', False):
            return "success"
        elif state['attempt_count'] >= state['max_attempts']:
            return "failure"
        else:
            return "retry"
    
    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF using multiple methods"""
        text = ""
        
        try:
            # Try pdfplumber first (better for tables)
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception:
            logger.warning("pdfplumber failed, trying PyPDF2...")
            
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e:
                raise Exception(f"Failed to extract PDF text: {str(e)}")
        
        return text.strip()
    
    def _extract_code_from_response(self, response_text: str) -> str:
        """Extract Python code from LLM response"""
        # Look for code blocks
        code_pattern = r'```python(.*?)```'
        matches = re.findall(code_pattern, response_text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks, try to find imports and function definitions
        lines = response_text.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.strip().startswith(('import ', 'from ', 'def ', 'class ')):
                in_code = True
            
            if in_code:
                code_lines.append(line)
        
        return '\n'.join(code_lines) if code_lines else response_text
    
    def run(self, target_bank: str, max_attempts: int = 3) -> bool:
        """
        Run the agent to generate a parser for the specified bank.
        
        Args:
            target_bank: Bank identifier (e.g., 'icici')
            max_attempts: Maximum retry attempts
            
        Returns:
            bool: Success status
        """
        logger.info(f"Starting agent for {target_bank} bank parser generation")
        
        # Setup paths
        pdf_path = self.data_dir / target_bank / f"{target_bank}_sample.pdf"
        csv_path = self.data_dir / target_bank / f"{target_bank}_expected.csv"
        
        # Validate input files exist
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return False
            
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            return False
        
        # Initialize state
        initial_state = AgentState(
            target_bank=target_bank,
            pdf_path=str(pdf_path),
            csv_path=str(csv_path),
            pdf_text="",
            csv_data=pd.DataFrame(),
            parser_code="",
            test_result={},
            attempt_count=0,
            max_attempts=max_attempts,
            error_message="",
            success=False,
            plan=""
        )
        
        try:
            # Execute workflow
            final_state = self.workflow.invoke(initial_state)
            return final_state.get('success', False)
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            logger.error(traceback.format_exc())
            return False

def main():
    """CLI entry point for the agent"""
    parser = argparse.ArgumentParser(
        description="Autonomous coding agent for bank statement parsers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py --target icici
  python agent.py --target sbi --attempts 5
        """
    )
    
    parser.add_argument(
        '--target',
        required=True,
        help='Target bank identifier (e.g., icici, sbi)'
    )
    
    parser.add_argument(
        '--attempts',
        type=int,
        default=3,
        help='Maximum retry attempts (default: 3)'
    )
    
    parser.add_argument(
        '--api-key',
        help='Gemini API key (or set GEMINI_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize and run agent
        agent = BankStatementAgent(gemini_api_key=args.api_key)
        success = agent.run(args.target, max_attempts=args.attempts)
        
        if success:
            print(f"\n✅ Successfully generated parser for {args.target} bank!")
            print(f"Parser saved to: custom_parsers/{args.target}_parser.py")
            sys.exit(0)
        else:
            print(f"\n❌ Failed to generate parser for {args.target} bank")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏸️ Agent execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Agent execution failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()