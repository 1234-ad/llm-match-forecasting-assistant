#!/usr/bin/env python3
"""
LLM-Powered Match Forecasting Assistant
Main entry point for the application
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='LLM Match Forecasting Assistant')
    parser.add_argument('--mode', choices=['cli', 'web'], default='web',
                       help='Run mode: CLI or Web interface')
    parser.add_argument('--model', choices=['gpt-4', 'claude-3'], default='gpt-4',
                       help='LLM model to use')
    
    args = parser.parse_args()
    
    if args.mode == 'cli':
        from src.interfaces.cli import run_cli
        run_cli(model=args.model)
    else:
        from src.interfaces.web import run_web
        run_web(model=args.model)

if __name__ == "__main__":
    main()