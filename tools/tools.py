#!/usr/bin/env python3
"""
Code Snippet Extractor for CULI Scoring Tools

This tool extracts useful code snippets from the tools folder for quick reference.
"""

import argparse
import ast
import inspect
from pathlib import Path
from typing import Dict, List, Any
import sys


class SnippetExtractor:
    """Extract useful code snippets from Python files"""
    
    def __init__(self, tools_dir: str = None):
        self.tools_dir = Path(tools_dir or Path(__file__).parent)
        self.snippets = {}
        
    def extract_functions(self, file_path: Path) -> Dict[str, str]:
        """Extract all function definitions from a Python file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            functions = {}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Get function source code
                    lines = content.split('\n')
                    start_line = node.lineno - 1
                    
                    # Find end of function by looking for next function or class
                    end_line = len(lines)
                    for other_node in ast.walk(tree):
                        if (isinstance(other_node, (ast.FunctionDef, ast.ClassDef)) and 
                            other_node.lineno > node.lineno):
                            end_line = min(end_line, other_node.lineno - 1)
                    
                    func_source = '\n'.join(lines[start_line:end_line]).rstrip()
                    functions[node.name] = func_source
                    
            return functions
        except Exception as e:
            print(f"Error extracting from {file_path}: {e}")
            return {}
    
    def extract_classes(self, file_path: Path) -> Dict[str, str]:
        """Extract all class definitions from a Python file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            classes = {}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    lines = content.split('\n')
                    start_line = node.lineno - 1
                    
                    # Find end of class
                    end_line = len(lines)
                    for other_node in ast.walk(tree):
                        if (isinstance(other_node, ast.ClassDef) and 
                            other_node.lineno > node.lineno):
                            end_line = min(end_line, other_node.lineno - 1)
                    
                    class_source = '\n'.join(lines[start_line:end_line]).rstrip()
                    classes[node.name] = class_source
                    
            return classes
        except Exception as e:
            print(f"Error extracting classes from {file_path}: {e}")
            return {}
    
    def extract_imports(self, file_path: Path) -> List[str]:
        """Extract all import statements from a Python file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(f"import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    names = [alias.name for alias in node.names]
                    imports.append(f"from {module} import {', '.join(names)}")
                    
            return imports
        except Exception as e:
            print(f"Error extracting imports from {file_path}: {e}")
            return []
    
    def get_file_summary(self, file_path: Path) -> Dict[str, Any]:
        """Get a complete summary of a Python file"""
        return {
            'file': str(file_path.name),
            'imports': self.extract_imports(file_path),
            'functions': self.extract_functions(file_path),
            'classes': self.extract_classes(file_path)
        }
    
    def list_available_tools(self) -> List[str]:
        """List all available Python tools"""
        tools = []
        for file_path in self.tools_dir.glob("*.py"):
            if file_path.name != "__init__.py" and file_path.name != "tools.py":
                tools.append(file_path.stem)
        return tools
    
    def extract_tool_snippets(self, tool_name: str) -> Dict[str, Any]:
        """Extract snippets from a specific tool"""
        file_path = self.tools_dir / f"{tool_name}.py"
        if not file_path.exists():
            raise FileNotFoundError(f"Tool '{tool_name}' not found")
        
        return self.get_file_summary(file_path)
    
    def extract_all_snippets(self) -> Dict[str, Dict[str, Any]]:
        """Extract snippets from all tools"""
        all_snippets = {}
        for tool in self.list_available_tools():
            try:
                all_snippets[tool] = self.extract_tool_snippets(tool)
            except Exception as e:
                print(f"Failed to extract from {tool}: {e}")
        return all_snippets


def format_output(snippets: Dict[str, Any], output_type: str = "summary"):
    """Format the output based on the requested type"""
    
    if output_type == "summary":
        print(f"\n{'='*60}")
        print(f"TOOL: {snippets['file']}")
        print(f"{'='*60}")
        
        if snippets['imports']:
            print(f"\n📦 IMPORTS ({len(snippets['imports'])}):")
            for imp in snippets['imports'][:5]:  # Show first 5
                print(f"  {imp}")
            if len(snippets['imports']) > 5:
                print(f"  ... and {len(snippets['imports']) - 5} more")
        
        if snippets['classes']:
            print(f"\n🏗️  CLASSES ({len(snippets['classes'])}):")
            for class_name in snippets['classes']:
                print(f"  - {class_name}")
        
        if snippets['functions']:
            print(f"\n⚙️  FUNCTIONS ({len(snippets['functions'])}):")
            for func_name in snippets['functions']:
                print(f"  - {func_name}")
    
    elif output_type == "functions":
        for func_name, func_code in snippets['functions'].items():
            print(f"\n{'='*40}")
            print(f"FUNCTION: {func_name}")
            print(f"{'='*40}")
            print(func_code)
    
    elif output_type == "classes":
        for class_name, class_code in snippets['classes'].items():
            print(f"\n{'='*40}")
            print(f"CLASS: {class_name}")
            print(f"{'='*40}")
            print(class_code)
    
    elif output_type == "imports":
        print(f"\n📦 IMPORTS from {snippets['file']}:")
        for imp in snippets['imports']:
            print(f"  {imp}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract useful code snippets from CULI Scoring tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools.py --list                    # List available tools
  python tools.py --tool eda               # Show summary of eda.py
  python tools.py --tool monitor --type functions  # Show all functions from monitor.py
  python tools.py --all --type summary     # Show summary of all tools
  python tools.py --tool move --type classes      # Show classes from move.py
        """
    )
    
    parser.add_argument("--list", "-l", action="store_true",
                       help="List all available tools")
    
    parser.add_argument("--tool", "-t", type=str,
                       help="Extract snippets from specific tool")
    
    parser.add_argument("--all", "-a", action="store_true",
                       help="Extract snippets from all tools")
    
    parser.add_argument("--type", choices=["summary", "functions", "classes", "imports"],
                       default="summary",
                       help="Type of output to display (default: summary)")
    
    parser.add_argument("--tools-dir", type=str,
                       help="Path to tools directory (default: current directory)")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = SnippetExtractor(args.tools_dir)
    
    try:
        if args.list:
            tools = extractor.list_available_tools()
            print("\n📁 Available Tools:")
            for tool in sorted(tools):
                print(f"  - {tool}")
            print(f"\nTotal: {len(tools)} tools found")
            
        elif args.tool:
            snippets = extractor.extract_tool_snippets(args.tool)
            format_output(snippets, args.type)
            
        elif args.all:
            all_snippets = extractor.extract_all_snippets()
            for tool_name, snippets in all_snippets.items():
                format_output(snippets, args.type)
                
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()