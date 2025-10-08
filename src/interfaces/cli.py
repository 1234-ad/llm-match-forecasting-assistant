"""
Command Line Interface
Interactive CLI for the match forecasting assistant
"""

import sys
import os
from typing import Dict, Any
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agent.core import MatchForecastingAgent


class MatchForecastingCLI:
    """Command line interface for the match forecasting assistant"""
    
    def __init__(self, model: str = "gpt-4"):
        self.agent = MatchForecastingAgent(model_name=model)
        self.model = model
        self.session_history = []
    
    def run(self):
        """Run the interactive CLI"""
        self.print_welcome()
        
        while True:
            try:
                query = input("\n🤖 Ask me anything (or 'quit' to exit): ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    self.print_goodbye()
                    break
                
                if query.lower() in ['help', 'h']:
                    self.print_help()
                    continue
                
                if query.lower() in ['history', 'hist']:
                    self.print_history()
                    continue
                
                if query.lower() in ['clear', 'cls']:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    self.print_welcome()
                    continue
                
                if query.lower().startswith('model '):
                    new_model = query.split(' ', 1)[1]
                    self.switch_model(new_model)
                    continue
                
                if not query:
                    continue
                
                self.process_query(query)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
    
    def print_welcome(self):
        """Print welcome message"""
        print("=" * 80)
        print("🏆 LLM MATCH FORECASTING ASSISTANT")
        print("=" * 80)
        print(f"🤖 Model: {self.model}")
        print("💡 Type 'help' for commands, 'quit' to exit")
        print("=" * 80)
    
    def print_help(self):
        """Print help information"""
        help_text = """
📚 AVAILABLE COMMANDS:

🔮 PREDICTIONS:
  • "Predict Manchester United vs Arsenal"
  • "What are the chances of Liverpool winning against Chelsea?"
  • "Analyze Real Madrid vs Barcelona El Clasico"

💰 STRATEGY SIMULATION:
  • "Simulate a Kelly criterion betting strategy"
  • "Test a fixed betting strategy with $50 bets"
  • "Compare different portfolio strategies"

🌤️ WHAT-IF ANALYSIS:
  • "How would rain affect the match outcome?"
  • "What if Messi was injured for PSG vs Bayern?"
  • "Analyze home vs away advantage"

🛠️ SYSTEM COMMANDS:
  • help, h          - Show this help
  • history, hist    - Show session history
  • clear, cls       - Clear screen
  • model <name>     - Switch LLM model
  • quit, exit, q    - Exit application

💡 EXAMPLES:
  • "Predict the outcome of Manchester City vs Tottenham with weather analysis"
  • "Simulate a betting strategy for the Premier League season"
  • "What if it rains during Liverpool vs Arsenal at Anfield?"
        """
        print(help_text)
    
    def print_history(self):
        """Print session history"""
        if not self.session_history:
            print("\n📝 No queries in session history")
            return
        
        print(f"\n📝 SESSION HISTORY ({len(self.session_history)} queries):")
        print("-" * 60)
        
        for i, entry in enumerate(self.session_history, 1):
            print(f"\n{i}. Query: {entry['query']}")
            print(f"   Response: {entry['response'][:100]}...")
            if 'confidence' in entry:
                print(f"   Confidence: {entry['confidence']:.1%}")
    
    def switch_model(self, model_name: str):
        """Switch LLM model"""
        try:
            self.agent = MatchForecastingAgent(model_name=model_name)
            self.model = model_name
            print(f"✅ Switched to model: {model_name}")
        except Exception as e:
            print(f"❌ Failed to switch model: {str(e)}")
    
    def process_query(self, query: str):
        """Process user query"""
        print(f"\n🤔 Processing: {query}")
        print("⏳ Please wait...")
        
        try:
            # Determine query type and route to appropriate method
            if any(word in query.lower() for word in ['simulate', 'strategy', 'betting', 'trading']):
                result = self.agent.simulate_strategy(query)
                response_key = 'simulation'
            elif any(word in query.lower() for word in ['what if', 'scenario', 'weather', 'injury']):
                result = self.agent.what_if_analysis(query)
                response_key = 'analysis'
            else:
                result = self.agent.predict_match(query)
                response_key = 'prediction'
            
            if result['success']:
                response = result[response_key]
                
                # Print response
                print("\n" + "=" * 80)
                print("🎯 ANALYSIS RESULT")
                print("=" * 80)
                print(response)
                
                # Print reasoning steps if available
                steps_key = 'reasoning_steps' if 'reasoning_steps' in result else \
                           'steps' if 'steps' in result else 'scenarios'
                
                if steps_key in result and result[steps_key]:
                    print("\n" + "-" * 40)
                    print("🧠 REASONING PROCESS")
                    print("-" * 40)
                    for i, step in enumerate(result[steps_key], 1):
                        print(f"{i}. {step}")
                
                # Print confidence if available
                if 'confidence' in result:
                    print(f"\n📊 Confidence Level: {result['confidence']:.1%}")
                
                print("=" * 80)
                
                # Add to history
                self.session_history.append({
                    'query': query,
                    'response': response,
                    'confidence': result.get('confidence'),
                    'reasoning_steps': result.get(steps_key, [])
                })
                
            else:
                print(f"\n❌ Error: {result.get('error', 'Unknown error occurred')}")
        
        except Exception as e:
            print(f"\n❌ Error processing query: {str(e)}")
    
    def print_goodbye(self):
        """Print goodbye message"""
        print("\n" + "=" * 80)
        print("👋 Thank you for using the LLM Match Forecasting Assistant!")
        print(f"📊 Session Summary: {len(self.session_history)} queries processed")
        print("🚀 Built for the AI Engineer Intern Hackathon")
        print("=" * 80)


def run_cli(model: str = "gpt-4"):
    """Run the CLI interface"""
    cli = MatchForecastingCLI(model=model)
    cli.run()


if __name__ == "__main__":
    run_cli()