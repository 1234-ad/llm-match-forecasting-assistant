"""
Probability Calculator Tool
Handles mathematical calculations for match outcome probabilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from langchain.tools import Tool
import math


class ProbabilityCalculator:
    """Calculator for match outcome probabilities and betting odds"""
    
    def __init__(self):
        self.elo_k_factor = 32
        self.home_advantage = 100  # Elo points
    
    def calculate_elo_probability(self, team1_elo: float, team2_elo: float, 
                                 is_home: bool = True) -> Dict[str, float]:
        """Calculate win probability using Elo ratings"""
        if is_home:
            team1_elo += self.home_advantage
        
        # Expected score for team1
        expected_score = 1 / (1 + 10**((team2_elo - team1_elo) / 400))
        
        return {
            "team1_win": expected_score,
            "team2_win": 1 - expected_score,
            "draw": 0.25  # Simplified draw probability
        }
    
    def calculate_poisson_probabilities(self, team1_goals: float, team2_goals: float) -> Dict[str, float]:
        """Calculate match outcome probabilities using Poisson distribution"""
        max_goals = 10
        probabilities = np.zeros((max_goals + 1, max_goals + 1))
        
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                probabilities[i][j] = (
                    (team1_goals**i * math.exp(-team1_goals) / math.factorial(i)) *
                    (team2_goals**j * math.exp(-team2_goals) / math.factorial(j))
                )
        
        # Calculate outcome probabilities
        team1_win = np.sum(probabilities[1:, 0]) + np.sum([
            np.sum(probabilities[i+1:, i]) for i in range(1, max_goals)
        ])
        team2_win = np.sum(probabilities[0, 1:]) + np.sum([
            np.sum(probabilities[i, i+1:]) for i in range(1, max_goals)
        ])
        draw = np.sum([probabilities[i, i] for i in range(max_goals + 1)])
        
        return {
            "team1_win": float(team1_win),
            "team2_win": float(team2_win),
            "draw": float(draw)
        }
    
    def calculate_form_adjusted_probability(self, base_prob: Dict[str, float], 
                                          team1_form: str, team2_form: str) -> Dict[str, float]:
        """Adjust probabilities based on recent form"""
        def form_to_score(form: str) -> float:
            """Convert form string (WWDLW) to numerical score"""
            points = {"W": 1.0, "D": 0.5, "L": 0.0}
            return sum(points.get(result, 0.5) for result in form) / len(form)
        
        team1_form_score = form_to_score(team1_form)
        team2_form_score = form_to_score(team2_form)
        
        # Form adjustment factor
        form_diff = team1_form_score - team2_form_score
        adjustment = form_diff * 0.1  # 10% max adjustment
        
        adjusted_prob = base_prob.copy()
        adjusted_prob["team1_win"] = max(0.05, min(0.95, base_prob["team1_win"] + adjustment))
        adjusted_prob["team2_win"] = max(0.05, min(0.95, base_prob["team2_win"] - adjustment))
        
        # Normalize to ensure sum = 1
        total = sum(adjusted_prob.values())
        for key in adjusted_prob:
            adjusted_prob[key] /= total
        
        return adjusted_prob
    
    def calculate_betting_value(self, probabilities: Dict[str, float], 
                               odds: Dict[str, float]) -> Dict[str, float]:
        """Calculate betting value (Kelly Criterion)"""
        value = {}
        for outcome in probabilities:
            if outcome in odds:
                implied_prob = 1 / odds[outcome]
                true_prob = probabilities[outcome]
                value[outcome] = (true_prob * odds[outcome] - 1) / (odds[outcome] - 1)
        
        return value
    
    def monte_carlo_simulation(self, team1_strength: float, team2_strength: float, 
                              num_simulations: int = 10000) -> Dict[str, Any]:
        """Run Monte Carlo simulation for match outcomes"""
        results = {"team1_wins": 0, "team2_wins": 0, "draws": 0}
        scores = []
        
        for _ in range(num_simulations):
            # Simulate goals using Poisson distribution
            team1_goals = np.random.poisson(team1_strength)
            team2_goals = np.random.poisson(team2_strength)
            
            scores.append((team1_goals, team2_goals))
            
            if team1_goals > team2_goals:
                results["team1_wins"] += 1
            elif team2_goals > team1_goals:
                results["team2_wins"] += 1
            else:
                results["draws"] += 1
        
        # Convert to probabilities
        probabilities = {
            "team1_win": results["team1_wins"] / num_simulations,
            "team2_win": results["team2_wins"] / num_simulations,
            "draw": results["draws"] / num_simulations
        }
        
        # Calculate statistics
        team1_goals_avg = np.mean([score[0] for score in scores])
        team2_goals_avg = np.mean([score[1] for score in scores])
        
        return {
            "probabilities": probabilities,
            "expected_goals": {
                "team1": float(team1_goals_avg),
                "team2": float(team2_goals_avg)
            },
            "most_likely_score": self._find_most_likely_score(scores),
            "confidence_interval": self._calculate_confidence_interval(probabilities)
        }
    
    def _find_most_likely_score(self, scores: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Find the most likely score from simulation results"""
        score_counts = {}
        for score in scores:
            score_counts[score] = score_counts.get(score, 0) + 1
        
        return max(score_counts, key=score_counts.get)
    
    def _calculate_confidence_interval(self, probabilities: Dict[str, float], 
                                     confidence: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for probabilities"""
        # Simplified confidence interval calculation
        z_score = 1.96  # 95% confidence
        intervals = {}
        
        for outcome, prob in probabilities.items():
            margin = z_score * math.sqrt(prob * (1 - prob) / 10000)  # Assuming 10k simulations
            intervals[outcome] = (
                max(0, prob - margin),
                min(1, prob + margin)
            )
        
        return intervals
    
    def as_tool(self) -> Tool:
        """Convert calculator to LangChain tool"""
        def calculate_probabilities(input_str: str) -> str:
            """Calculate match outcome probabilities from team data"""
            try:
                # Parse input (simplified - in production, use more robust parsing)
                lines = input_str.strip().split('\n')
                data = {}
                
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        data[key.strip().lower()] = value.strip()
                
                # Extract team strengths (goals per game)
                team1_strength = float(data.get('team1_goals_per_game', 1.5))
                team2_strength = float(data.get('team2_goals_per_game', 1.3))
                
                # Run simulation
                result = self.monte_carlo_simulation(team1_strength, team2_strength)
                
                # Format output
                output = f"""
Probability Analysis:
- Team 1 Win: {result['probabilities']['team1_win']:.3f} ({result['probabilities']['team1_win']*100:.1f}%)
- Draw: {result['probabilities']['draw']:.3f} ({result['probabilities']['draw']*100:.1f}%)
- Team 2 Win: {result['probabilities']['team2_win']:.3f} ({result['probabilities']['team2_win']*100:.1f}%)

Expected Goals:
- Team 1: {result['expected_goals']['team1']:.2f}
- Team 2: {result['expected_goals']['team2']:.2f}

Most Likely Score: {result['most_likely_score'][0]}-{result['most_likely_score'][1]}

Confidence Intervals (95%):
- Team 1 Win: {result['confidence_interval']['team1_win'][0]:.3f} - {result['confidence_interval']['team1_win'][1]:.3f}
- Draw: {result['confidence_interval']['draw'][0]:.3f} - {result['confidence_interval']['draw'][1]:.3f}
- Team 2 Win: {result['confidence_interval']['team2_win'][0]:.3f} - {result['confidence_interval']['team2_win'][1]:.3f}
                """
                
                return output.strip()
                
            except Exception as e:
                return f"Error calculating probabilities: {str(e)}"
        
        return Tool(
            name="calculate_probabilities",
            description="Calculate match outcome probabilities using statistical models and Monte Carlo simulation",
            func=calculate_probabilities
        )