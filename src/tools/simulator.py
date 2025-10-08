"""
Strategy Simulator Tool
Simulates betting and trading strategies with risk analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from langchain.tools import Tool
import random


class StrategySimulator:
    """Simulator for betting and trading strategies"""
    
    def __init__(self):
        self.initial_bankroll = 1000
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
    
    def simulate_betting_strategy(self, strategy_config: Dict[str, Any], 
                                 num_matches: int = 100) -> Dict[str, Any]:
        """Simulate a betting strategy over multiple matches"""
        
        bankroll = self.initial_bankroll
        bet_history = []
        bankroll_history = [bankroll]
        
        strategy_type = strategy_config.get('type', 'fixed')
        base_bet = strategy_config.get('base_bet', 50)
        
        for match_num in range(num_matches):
            # Generate random match outcome and odds
            true_prob = random.uniform(0.3, 0.7)  # True win probability
            bookmaker_prob = true_prob + random.uniform(-0.1, 0.1)  # Bookmaker's estimate
            bookmaker_prob = max(0.1, min(0.9, bookmaker_prob))
            
            odds = 1 / bookmaker_prob
            
            # Determine bet size based on strategy
            if strategy_type == 'fixed':
                bet_size = base_bet
            elif strategy_type == 'kelly':
                # Kelly Criterion
                edge = true_prob - bookmaker_prob
                if edge > 0:
                    kelly_fraction = edge / (odds - 1)
                    bet_size = min(bankroll * kelly_fraction, bankroll * 0.1)  # Max 10% of bankroll
                else:
                    bet_size = 0
            elif strategy_type == 'martingale':
                # Martingale system (double after loss)
                if bet_history and not bet_history[-1]['won']:
                    bet_size = min(bet_history[-1]['bet_size'] * 2, bankroll * 0.5)
                else:
                    bet_size = base_bet
            else:
                bet_size = base_bet
            
            # Skip if insufficient funds
            if bet_size > bankroll:
                bet_size = 0
            
            # Simulate match outcome
            won = random.random() < true_prob
            
            # Calculate profit/loss
            if bet_size > 0:
                if won:
                    profit = bet_size * (odds - 1)
                    bankroll += profit
                else:
                    profit = -bet_size
                    bankroll += profit
            else:
                profit = 0
            
            # Record bet
            bet_history.append({
                'match': match_num + 1,
                'bet_size': bet_size,
                'odds': odds,
                'true_prob': true_prob,
                'bookmaker_prob': bookmaker_prob,
                'won': won,
                'profit': profit,
                'bankroll': bankroll
            })
            
            bankroll_history.append(bankroll)
            
            # Stop if bankrupt
            if bankroll <= 0:
                break
        
        return self._analyze_strategy_results(bet_history, bankroll_history)
    
    def simulate_portfolio_strategy(self, assets: List[Dict[str, Any]], 
                                   strategy: str = 'equal_weight') -> Dict[str, Any]:
        """Simulate a portfolio trading strategy"""
        
        num_days = 252  # Trading days in a year
        initial_value = 10000
        
        # Generate synthetic price data
        portfolio_data = {}
        for asset in assets:
            name = asset['name']
            volatility = asset.get('volatility', 0.2)
            expected_return = asset.get('expected_return', 0.08)
            
            # Generate price series using geometric Brownian motion
            dt = 1/252
            prices = [100]  # Starting price
            
            for _ in range(num_days):
                drift = expected_return * dt
                shock = volatility * np.sqrt(dt) * np.random.normal()
                price = prices[-1] * np.exp(drift + shock)
                prices.append(price)
            
            portfolio_data[name] = prices
        
        # Calculate portfolio performance
        if strategy == 'equal_weight':
            weights = {name: 1/len(assets) for name in portfolio_data.keys()}
        elif strategy == 'momentum':
            # Simple momentum strategy (buy winners, sell losers)
            weights = self._calculate_momentum_weights(portfolio_data)
        else:
            weights = {name: 1/len(assets) for name in portfolio_data.keys()}
        
        portfolio_values = []
        for day in range(len(list(portfolio_data.values())[0])):
            daily_value = 0
            for asset_name, prices in portfolio_data.items():
                daily_value += weights[asset_name] * (prices[day] / prices[0]) * initial_value
            portfolio_values.append(daily_value)
        
        return self._analyze_portfolio_results(portfolio_values, portfolio_data)
    
    def _calculate_momentum_weights(self, portfolio_data: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate momentum-based weights"""
        lookback = 20  # 20-day momentum
        weights = {}
        
        for name, prices in portfolio_data.items():
            if len(prices) >= lookback:
                momentum = (prices[-1] - prices[-lookback]) / prices[-lookback]
                weights[name] = max(0, momentum)  # Only positive momentum
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight/total_weight for name, weight in weights.items()}
        else:
            weights = {name: 1/len(portfolio_data) for name in portfolio_data.keys()}
        
        return weights
    
    def _analyze_strategy_results(self, bet_history: List[Dict], 
                                 bankroll_history: List[float]) -> Dict[str, Any]:
        """Analyze betting strategy results"""
        
        if not bet_history:
            return {"error": "No bets placed"}
        
        df = pd.DataFrame(bet_history)
        
        # Calculate metrics
        total_bets = len(df)
        winning_bets = len(df[df['won'] == True])
        win_rate = winning_bets / total_bets if total_bets > 0 else 0
        
        total_profit = df['profit'].sum()
        roi = (total_profit / self.initial_bankroll) * 100
        
        # Risk metrics
        profits = df['profit'].values
        sharpe_ratio = np.mean(profits) / np.std(profits) if np.std(profits) > 0 else 0
        max_drawdown = self._calculate_max_drawdown(bankroll_history)
        
        # Betting efficiency
        avg_bet_size = df['bet_size'].mean()
        avg_odds = df[df['bet_size'] > 0]['odds'].mean()
        
        return {
            "summary": {
                "total_bets": total_bets,
                "winning_bets": winning_bets,
                "win_rate": win_rate,
                "total_profit": total_profit,
                "roi_percent": roi,
                "final_bankroll": bankroll_history[-1]
            },
            "risk_metrics": {
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "volatility": np.std(profits)
            },
            "betting_stats": {
                "avg_bet_size": avg_bet_size,
                "avg_odds": avg_odds,
                "largest_win": df['profit'].max(),
                "largest_loss": df['profit'].min()
            },
            "bankroll_progression": bankroll_history[-10:]  # Last 10 values
        }
    
    def _analyze_portfolio_results(self, portfolio_values: List[float], 
                                  portfolio_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze portfolio strategy results"""
        
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Performance metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        annualized_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        var_95 = np.percentile(returns, 5)  # 5% VaR
        
        return {
            "performance": {
                "total_return": total_return,
                "annualized_return": annualized_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio
            },
            "risk_metrics": {
                "max_drawdown": max_drawdown,
                "var_95": var_95,
                "downside_deviation": np.std(returns[returns < 0])
            },
            "portfolio_value": {
                "initial": portfolio_values[0],
                "final": portfolio_values[-1],
                "peak": max(portfolio_values)
            }
        }
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def as_tool(self) -> Tool:
        """Convert simulator to LangChain tool"""
        def simulate_strategy(input_str: str) -> str:
            """Simulate betting or trading strategies with risk analysis"""
            try:
                # Parse strategy configuration
                lines = input_str.strip().split('\n')
                config = {}
                
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        config[key.strip().lower()] = value.strip()
                
                strategy_type = config.get('strategy_type', 'betting')
                
                if strategy_type == 'betting':
                    # Betting strategy simulation
                    betting_config = {
                        'type': config.get('betting_type', 'fixed'),
                        'base_bet': float(config.get('base_bet', 50))
                    }
                    
                    num_matches = int(config.get('num_matches', 100))
                    results = self.simulate_betting_strategy(betting_config, num_matches)
                    
                    output = f"""
Betting Strategy Simulation Results:

Summary:
- Total Bets: {results['summary']['total_bets']}
- Win Rate: {results['summary']['win_rate']:.2%}
- Total Profit: ${results['summary']['total_profit']:.2f}
- ROI: {results['summary']['roi_percent']:.2f}%
- Final Bankroll: ${results['summary']['final_bankroll']:.2f}

Risk Analysis:
- Sharpe Ratio: {results['risk_metrics']['sharpe_ratio']:.3f}
- Max Drawdown: {results['risk_metrics']['max_drawdown']:.2%}
- Volatility: ${results['risk_metrics']['volatility']:.2f}

Betting Statistics:
- Average Bet Size: ${results['betting_stats']['avg_bet_size']:.2f}
- Average Odds: {results['betting_stats']['avg_odds']:.2f}
- Largest Win: ${results['betting_stats']['largest_win']:.2f}
- Largest Loss: ${results['betting_stats']['largest_loss']:.2f}
                    """
                    
                else:
                    # Portfolio strategy simulation
                    assets = [
                        {'name': 'Stock A', 'volatility': 0.25, 'expected_return': 0.10},
                        {'name': 'Stock B', 'volatility': 0.20, 'expected_return': 0.08},
                        {'name': 'Bond', 'volatility': 0.05, 'expected_return': 0.04}
                    ]
                    
                    strategy = config.get('portfolio_strategy', 'equal_weight')
                    results = self.simulate_portfolio_strategy(assets, strategy)
                    
                    output = f"""
Portfolio Strategy Simulation Results:

Performance:
- Total Return: {results['performance']['total_return']:.2%}
- Annualized Return: {results['performance']['annualized_return']:.2%}
- Volatility: {results['performance']['volatility']:.2%}
- Sharpe Ratio: {results['performance']['sharpe_ratio']:.3f}

Risk Metrics:
- Max Drawdown: {results['risk_metrics']['max_drawdown']:.2%}
- 95% VaR: {results['risk_metrics']['var_95']:.2%}
- Downside Deviation: {results['risk_metrics']['downside_deviation']:.3f}

Portfolio Value:
- Initial: ${results['portfolio_value']['initial']:,.2f}
- Final: ${results['portfolio_value']['final']:,.2f}
- Peak: ${results['portfolio_value']['peak']:,.2f}
                    """
                
                return output.strip()
                
            except Exception as e:
                return f"Error simulating strategy: {str(e)}"
        
        return Tool(
            name="simulate_strategy",
            description="Simulate betting or trading strategies with comprehensive risk analysis and performance metrics",
            func=simulate_strategy
        )