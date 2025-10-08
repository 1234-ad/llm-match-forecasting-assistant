"""
Unit tests for the LLM Match Forecasting Agent
"""

import unittest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agent.core import MatchForecastingAgent
from src.rag.retriever import MatchDataRetriever
from src.tools.calculator import ProbabilityCalculator
from src.tools.simulator import StrategySimulator
from src.tools.data_fetcher import DataFetcher


class TestMatchForecastingAgent(unittest.TestCase):
    """Test cases for the main agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Use a mock model for testing to avoid API calls
        self.agent = MatchForecastingAgent(model_name="gpt-3.5-turbo", temperature=0.0)
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        self.assertIsNotNone(self.agent.llm)
        self.assertIsNotNone(self.agent.retriever)
        self.assertIsNotNone(self.agent.calculator)
        self.assertIsNotNone(self.agent.simulator)
        self.assertIsNotNone(self.agent.data_fetcher)
    
    def test_confidence_extraction(self):
        """Test confidence level extraction"""
        test_output = "Based on the analysis, I have a confidence level of 0.75 in this prediction."
        confidence = self.agent._extract_confidence(test_output)
        self.assertEqual(confidence, 0.75)
        
        # Test with percentage
        test_output2 = "My confidence is 85% for this match outcome."
        confidence2 = self.agent._extract_confidence(test_output2)
        self.assertEqual(confidence2, 85.0)


class TestMatchDataRetriever(unittest.TestCase):
    """Test cases for the RAG retriever"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.retriever = MatchDataRetriever(persist_directory="./test_chroma_db")
    
    def test_retriever_initialization(self):
        """Test retriever initialization"""
        self.assertIsNotNone(self.retriever.vectorstore)
        self.assertIsNotNone(self.retriever.embeddings)
    
    def test_format_match_data(self):
        """Test match data formatting"""
        sample_match = {
            "team": "Test Team",
            "opponent": "Test Opponent",
            "date": "2024-01-01",
            "result": "2-1",
            "venue": "Test Stadium",
            "form": "WWWWW",
            "stats": {"possession": 60, "shots": 15},
            "injuries": ["Player A"],
            "weather": "Clear"
        }
        
        formatted = self.retriever._format_match_data(sample_match)
        self.assertIn("Test Team", formatted)
        self.assertIn("Test Opponent", formatted)
        self.assertIn("2-1", formatted)
    
    def test_retrieve_matches(self):
        """Test match retrieval"""
        docs = self.retriever.retrieve_relevant_matches("Manchester United", k=3)
        self.assertIsInstance(docs, list)
    
    def tearDown(self):
        """Clean up test database"""
        import shutil
        if os.path.exists("./test_chroma_db"):
            shutil.rmtree("./test_chroma_db")


class TestProbabilityCalculator(unittest.TestCase):
    """Test cases for the probability calculator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = ProbabilityCalculator()
    
    def test_elo_probability(self):
        """Test Elo probability calculation"""
        prob = self.calculator.calculate_elo_probability(1500, 1400, is_home=True)
        
        self.assertIn('team1_win', prob)
        self.assertIn('team2_win', prob)
        self.assertIn('draw', prob)
        
        # Team1 should have higher probability (higher Elo + home advantage)
        self.assertGreater(prob['team1_win'], prob['team2_win'])
    
    def test_poisson_probabilities(self):
        """Test Poisson probability calculation"""
        prob = self.calculator.calculate_poisson_probabilities(1.5, 1.2)
        
        self.assertIn('team1_win', prob)
        self.assertIn('team2_win', prob)
        self.assertIn('draw', prob)
        
        # Probabilities should sum to approximately 1
        total = prob['team1_win'] + prob['team2_win'] + prob['draw']
        self.assertAlmostEqual(total, 1.0, places=2)
    
    def test_form_adjustment(self):
        """Test form-based probability adjustment"""
        base_prob = {'team1_win': 0.5, 'team2_win': 0.3, 'draw': 0.2}
        adjusted = self.calculator.calculate_form_adjusted_probability(
            base_prob, "WWWWW", "LLLLL"
        )
        
        # Team1 with better form should have higher probability
        self.assertGreater(adjusted['team1_win'], base_prob['team1_win'])
    
    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation"""
        result = self.calculator.monte_carlo_simulation(1.5, 1.2, num_simulations=1000)
        
        self.assertIn('probabilities', result)
        self.assertIn('expected_goals', result)
        self.assertIn('most_likely_score', result)
        self.assertIn('confidence_interval', result)
        
        # Check probability sum
        probs = result['probabilities']
        total = probs['team1_win'] + probs['team2_win'] + probs['draw']
        self.assertAlmostEqual(total, 1.0, places=2)


class TestStrategySimulator(unittest.TestCase):
    """Test cases for the strategy simulator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.simulator = StrategySimulator()
    
    def test_betting_strategy_simulation(self):
        """Test betting strategy simulation"""
        config = {'type': 'fixed', 'base_bet': 50}
        result = self.simulator.simulate_betting_strategy(config, num_matches=10)
        
        self.assertIn('summary', result)
        self.assertIn('risk_metrics', result)
        self.assertIn('betting_stats', result)
        
        # Check required fields
        self.assertIn('total_bets', result['summary'])
        self.assertIn('win_rate', result['summary'])
        self.assertIn('roi_percent', result['summary'])
    
    def test_portfolio_strategy_simulation(self):
        """Test portfolio strategy simulation"""
        assets = [
            {'name': 'Asset A', 'volatility': 0.2, 'expected_return': 0.08},
            {'name': 'Asset B', 'volatility': 0.15, 'expected_return': 0.06}
        ]
        
        result = self.simulator.simulate_portfolio_strategy(assets, 'equal_weight')
        
        self.assertIn('performance', result)
        self.assertIn('risk_metrics', result)
        self.assertIn('portfolio_value', result)
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation"""
        values = [1000, 1100, 900, 950, 800, 1200]
        max_dd = self.simulator._calculate_max_drawdown(values)
        
        # Should be around 27.3% (from 1100 to 800)
        self.assertGreater(max_dd, 0.25)
        self.assertLess(max_dd, 0.30)


class TestDataFetcher(unittest.TestCase):
    """Test cases for the data fetcher"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.fetcher = DataFetcher()
    
    def test_team_stats_fetch(self):
        """Test team statistics fetching"""
        stats = self.fetcher.fetch_team_stats("Manchester United")
        
        # Should return mock data
        self.assertIn('league_position', stats)
        self.assertIn('form', stats)
        self.assertIn('recent_matches', stats)
    
    def test_match_odds_fetch(self):
        """Test match odds fetching"""
        odds = self.fetcher.fetch_match_odds("Manchester United", "Arsenal")
        
        self.assertIn('bookmakers', odds)
        self.assertIn('average_odds', odds)
        self.assertIn('implied_probabilities', odds)
    
    def test_weather_data_fetch(self):
        """Test weather data fetching"""
        weather = self.fetcher.fetch_weather_data("Old Trafford", "2024-01-01")
        
        self.assertIn('condition', weather)
        self.assertIn('temperature_celsius', weather)
        self.assertIn('impact_assessment', weather)
    
    def test_weather_impact_assessment(self):
        """Test weather impact assessment"""
        weather = {"condition": "Heavy Rain", "temperature": 5, "wind_speed": 25}
        impact = self.fetcher._assess_weather_impact(weather)
        
        self.assertIn("Rain", impact)
        self.assertIn("Cold", impact)
        self.assertIn("wind", impact)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)