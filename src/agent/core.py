"""
Core LLM Agent Implementation
Handles the main reasoning and orchestration logic
"""

from typing import Dict, List, Any, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage

from ..rag.retriever import MatchDataRetriever
from ..tools.calculator import ProbabilityCalculator
from ..tools.simulator import StrategySimulator
from ..tools.data_fetcher import DataFetcher


class MatchForecastingAgent:
    """Main LLM agent for match forecasting and strategy simulation"""
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize LLM
        if model_name.startswith("gpt"):
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                streaming=True
            )
        elif model_name.startswith("claude"):
            self.llm = ChatAnthropic(
                model=model_name,
                temperature=temperature
            )
        
        # Initialize components
        self.retriever = MatchDataRetriever()
        self.calculator = ProbabilityCalculator()
        self.simulator = StrategySimulator()
        self.data_fetcher = DataFetcher()
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=10,
            memory_key="chat_history",
            return_messages=True
        )
        
        # Setup agent
        self._setup_agent()
    
    def _setup_agent(self):
        """Setup the LangChain agent with tools and prompts"""
        
        # Define tools
        tools = [
            self.retriever.as_tool(),
            self.calculator.as_tool(),
            self.simulator.as_tool(),
            self.data_fetcher.as_tool()
        ]
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        # Create executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=5
        )
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent"""
        return """You are an expert match forecasting assistant that combines data analysis with strategic reasoning.

Your capabilities:
1. **Match Prediction**: Analyze team statistics, player form, historical data
2. **Strategy Simulation**: Model betting/trading strategies with risk analysis
3. **Explainable Reasoning**: Provide clear, step-by-step explanations
4. **What-if Analysis**: Simulate different scenarios and conditions

Guidelines:
- Always retrieve relevant historical data before making predictions
- Show your reasoning process step-by-step
- Calculate probabilities and confidence intervals
- Consider multiple factors: form, injuries, weather, venue
- Provide actionable insights and risk assessments
- Be honest about uncertainty and limitations

Tools available:
- retrieve_match_data: Get historical match and team statistics
- calculate_probabilities: Compute win/draw/loss probabilities
- simulate_strategy: Model betting or trading strategies
- fetch_live_data: Get real-time match and market data

Remember: Always explain your reasoning and show confidence levels."""
    
    def predict_match(self, query: str) -> Dict[str, Any]:
        """Make a match prediction with full reasoning"""
        try:
            result = self.agent_executor.invoke({
                "input": f"Predict the outcome for: {query}. Provide detailed analysis with probabilities and reasoning."
            })
            
            return {
                "prediction": result["output"],
                "reasoning_steps": result.get("intermediate_steps", []),
                "confidence": self._extract_confidence(result["output"]),
                "success": True
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    def simulate_strategy(self, strategy_query: str) -> Dict[str, Any]:
        """Simulate a betting or trading strategy"""
        try:
            result = self.agent_executor.invoke({
                "input": f"Simulate this strategy: {strategy_query}. Include risk analysis and expected returns."
            })
            
            return {
                "simulation": result["output"],
                "steps": result.get("intermediate_steps", []),
                "success": True
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    def what_if_analysis(self, scenario: str) -> Dict[str, Any]:
        """Perform what-if scenario analysis"""
        try:
            result = self.agent_executor.invoke({
                "input": f"Analyze this what-if scenario: {scenario}. Compare outcomes under different conditions."
            })
            
            return {
                "analysis": result["output"],
                "scenarios": result.get("intermediate_steps", []),
                "success": True
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    def _extract_confidence(self, output: str) -> float:
        """Extract confidence level from agent output"""
        # Simple regex-based extraction
        import re
        confidence_match = re.search(r'confidence[:\s]*(\d+(?:\.\d+)?)', output.lower())
        if confidence_match:
            return float(confidence_match.group(1))
        return 0.5  # Default confidence
    
    def get_explanation(self) -> List[str]:
        """Get explanation of the last prediction"""
        if hasattr(self, '_last_steps'):
            return [str(step) for step in self._last_steps]
        return ["No recent predictions to explain"]