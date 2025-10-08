"""
Data Fetcher Tool
Fetches real-time sports and market data from external APIs
"""

import requests
import json
from typing import Dict, List, Any, Optional
from langchain.tools import Tool
import os
from datetime import datetime, timedelta


class DataFetcher:
    """Fetcher for real-time sports and market data"""
    
    def __init__(self):
        self.football_api_key = os.getenv('FOOTBALL_API_KEY')
        self.sports_api_key = os.getenv('SPORTS_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        
        # Base URLs
        self.football_api_base = "https://api.football-data.org/v4"
        self.sports_api_base = "https://api.the-odds-api.com/v4"
        self.alpha_vantage_base = "https://www.alphavantage.co/query"
        self.finnhub_base = "https://finnhub.io/api/v1"
    
    def fetch_team_stats(self, team_name: str, league: str = "premier-league") -> Dict[str, Any]:
        """Fetch team statistics and recent form"""
        try:
            # Mock data for demonstration - replace with real API calls
            mock_data = {
                "Manchester United": {
                    "league_position": 6,
                    "points": 45,
                    "games_played": 28,
                    "wins": 13,
                    "draws": 6,
                    "losses": 9,
                    "goals_for": 42,
                    "goals_against": 35,
                    "goal_difference": 7,
                    "form": "WDLWW",
                    "recent_matches": [
                        {"opponent": "Arsenal", "result": "2-1", "date": "2024-01-15"},
                        {"opponent": "Chelsea", "result": "1-1", "date": "2024-01-08"},
                        {"opponent": "Liverpool", "result": "0-2", "date": "2024-01-01"}
                    ],
                    "key_players": [
                        {"name": "Bruno Fernandes", "goals": 8, "assists": 6},
                        {"name": "Marcus Rashford", "goals": 12, "assists": 4}
                    ],
                    "injuries": ["Luke Shaw", "Mason Mount"],
                    "home_record": {"wins": 8, "draws": 3, "losses": 3},
                    "away_record": {"wins": 5, "draws": 3, "losses": 6}
                },
                "Arsenal": {
                    "league_position": 2,
                    "points": 58,
                    "games_played": 28,
                    "wins": 18,
                    "draws": 4,
                    "losses": 6,
                    "goals_for": 55,
                    "goals_against": 28,
                    "goal_difference": 27,
                    "form": "WWWDW",
                    "recent_matches": [
                        {"opponent": "Manchester United", "result": "1-2", "date": "2024-01-15"},
                        {"opponent": "Tottenham", "result": "3-1", "date": "2024-01-10"},
                        {"opponent": "Brighton", "result": "2-0", "date": "2024-01-05"}
                    ],
                    "key_players": [
                        {"name": "Bukayo Saka", "goals": 11, "assists": 9},
                        {"name": "Martin Odegaard", "goals": 6, "assists": 12}
                    ],
                    "injuries": ["Gabriel Jesus"],
                    "home_record": {"wins": 11, "draws": 2, "losses": 1},
                    "away_record": {"wins": 7, "draws": 2, "losses": 5}
                }
            }
            
            return mock_data.get(team_name, {
                "error": f"No data found for {team_name}",
                "available_teams": list(mock_data.keys())
            })
            
        except Exception as e:
            return {"error": f"Failed to fetch team stats: {str(e)}"}
    
    def fetch_match_odds(self, team1: str, team2: str, sport: str = "soccer") -> Dict[str, Any]:
        """Fetch current betting odds for a match"""
        try:
            # Mock odds data - replace with real API calls
            mock_odds = {
                f"{team1} vs {team2}": {
                    "bookmakers": [
                        {
                            "name": "Bet365",
                            "odds": {
                                team1: 2.10,
                                "Draw": 3.40,
                                team2: 3.20
                            }
                        },
                        {
                            "name": "William Hill",
                            "odds": {
                                team1: 2.05,
                                "Draw": 3.50,
                                team2: 3.30
                            }
                        },
                        {
                            "name": "Betfair",
                            "odds": {
                                team1: 2.15,
                                "Draw": 3.30,
                                team2: 3.10
                            }
                        }
                    ],
                    "average_odds": {
                        team1: 2.10,
                        "Draw": 3.40,
                        team2: 3.20
                    },
                    "implied_probabilities": {
                        team1: 0.476,
                        "Draw": 0.294,
                        team2: 0.313
                    },
                    "market_margin": 0.083,
                    "last_updated": datetime.now().isoformat()
                }
            }
            
            match_key = f"{team1} vs {team2}"
            return mock_odds.get(match_key, {
                "error": f"No odds found for {match_key}",
                "suggestion": "Try with different team names or check spelling"
            })
            
        except Exception as e:
            return {"error": f"Failed to fetch odds: {str(e)}"}
    
    def fetch_weather_data(self, venue: str, match_date: str) -> Dict[str, Any]:
        """Fetch weather forecast for match venue"""
        try:
            # Mock weather data - replace with real weather API
            weather_conditions = [
                {"condition": "Clear", "temperature": 15, "humidity": 45, "wind_speed": 8},
                {"condition": "Cloudy", "temperature": 12, "humidity": 65, "wind_speed": 12},
                {"condition": "Light Rain", "temperature": 8, "humidity": 85, "wind_speed": 15},
                {"condition": "Heavy Rain", "temperature": 6, "humidity": 95, "wind_speed": 20}
            ]
            
            import random
            weather = random.choice(weather_conditions)
            
            return {
                "venue": venue,
                "date": match_date,
                "condition": weather["condition"],
                "temperature_celsius": weather["temperature"],
                "humidity_percent": weather["humidity"],
                "wind_speed_kmh": weather["wind_speed"],
                "impact_assessment": self._assess_weather_impact(weather)
            }
            
        except Exception as e:
            return {"error": f"Failed to fetch weather: {str(e)}"}
    
    def fetch_market_data(self, symbol: str, period: str = "1d") -> Dict[str, Any]:
        """Fetch financial market data"""
        try:
            # Mock market data - replace with real financial API
            mock_data = {
                "AAPL": {
                    "symbol": "AAPL",
                    "price": 175.50,
                    "change": 2.30,
                    "change_percent": 1.33,
                    "volume": 45678900,
                    "market_cap": 2800000000000,
                    "pe_ratio": 28.5,
                    "52_week_high": 198.23,
                    "52_week_low": 124.17
                },
                "GOOGL": {
                    "symbol": "GOOGL",
                    "price": 142.80,
                    "change": -1.20,
                    "change_percent": -0.83,
                    "volume": 23456789,
                    "market_cap": 1800000000000,
                    "pe_ratio": 25.2,
                    "52_week_high": 151.55,
                    "52_week_low": 83.34
                }
            }
            
            return mock_data.get(symbol.upper(), {
                "error": f"No data found for symbol {symbol}",
                "available_symbols": list(mock_data.keys())
            })
            
        except Exception as e:
            return {"error": f"Failed to fetch market data: {str(e)}"}
    
    def _assess_weather_impact(self, weather: Dict[str, Any]) -> str:
        """Assess weather impact on match outcome"""
        condition = weather["condition"]
        temperature = weather["temperature"]
        wind_speed = weather["wind_speed"]
        
        impacts = []
        
        if "Rain" in condition:
            impacts.append("Wet conditions may favor defensive play and reduce scoring")
        
        if temperature < 5:
            impacts.append("Cold weather may affect player performance and ball control")
        elif temperature > 30:
            impacts.append("Hot weather may lead to fatigue and more substitutions")
        
        if wind_speed > 20:
            impacts.append("Strong winds may affect passing accuracy and set pieces")
        
        return "; ".join(impacts) if impacts else "Weather conditions should have minimal impact"
    
    def as_tool(self) -> Tool:
        """Convert data fetcher to LangChain tool"""
        def fetch_live_data(input_str: str) -> str:
            """Fetch real-time sports and market data"""
            try:
                lines = input_str.strip().split('\n')
                data_type = lines[0].lower() if lines else "team_stats"
                
                if data_type == "team_stats":
                    team_name = lines[1] if len(lines) > 1 else "Manchester United"
                    data = self.fetch_team_stats(team_name)
                    
                    if "error" in data:
                        return f"Error: {data['error']}"
                    
                    output = f"""
Team Statistics for {team_name}:

League Performance:
- Position: {data.get('league_position', 'N/A')}
- Points: {data.get('points', 'N/A')}
- Games Played: {data.get('games_played', 'N/A')}
- Record: {data.get('wins', 0)}W-{data.get('draws', 0)}D-{data.get('losses', 0)}L
- Goals: {data.get('goals_for', 0)} for, {data.get('goals_against', 0)} against
- Goal Difference: {data.get('goal_difference', 0)}

Recent Form: {data.get('form', 'N/A')}

Key Players:
{chr(10).join([f"- {player['name']}: {player['goals']} goals, {player['assists']} assists" 
               for player in data.get('key_players', [])])}

Injuries: {', '.join(data.get('injuries', ['None']))}

Home Record: {data.get('home_record', {}).get('wins', 0)}W-{data.get('home_record', {}).get('draws', 0)}D-{data.get('home_record', {}).get('losses', 0)}L
Away Record: {data.get('away_record', {}).get('wins', 0)}W-{data.get('away_record', {}).get('draws', 0)}D-{data.get('away_record', {}).get('losses', 0)}L
                    """
                    
                elif data_type == "odds":
                    team1 = lines[1] if len(lines) > 1 else "Manchester United"
                    team2 = lines[2] if len(lines) > 2 else "Arsenal"
                    data = self.fetch_match_odds(team1, team2)
                    
                    if "error" in data:
                        return f"Error: {data['error']}"
                    
                    output = f"""
Betting Odds for {team1} vs {team2}:

Average Odds:
- {team1} Win: {data['average_odds'][team1]:.2f}
- Draw: {data['average_odds']['Draw']:.2f}
- {team2} Win: {data['average_odds'][team2]:.2f}

Implied Probabilities:
- {team1} Win: {data['implied_probabilities'][team1]:.1%}
- Draw: {data['implied_probabilities']['Draw']:.1%}
- {team2} Win: {data['implied_probabilities'][team2]:.1%}

Market Margin: {data['market_margin']:.1%}
Last Updated: {data['last_updated']}
                    """
                    
                elif data_type == "weather":
                    venue = lines[1] if len(lines) > 1 else "Old Trafford"
                    date = lines[2] if len(lines) > 2 else datetime.now().strftime("%Y-%m-%d")
                    data = self.fetch_weather_data(venue, date)
                    
                    if "error" in data:
                        return f"Error: {data['error']}"
                    
                    output = f"""
Weather Forecast for {venue} on {date}:

Conditions: {data['condition']}
Temperature: {data['temperature_celsius']}°C
Humidity: {data['humidity_percent']}%
Wind Speed: {data['wind_speed_kmh']} km/h

Impact Assessment: {data['impact_assessment']}
                    """
                    
                else:
                    output = "Unknown data type. Available types: team_stats, odds, weather"
                
                return output.strip()
                
            except Exception as e:
                return f"Error fetching data: {str(e)}"
        
        return Tool(
            name="fetch_live_data",
            description="Fetch real-time sports data including team statistics, betting odds, weather conditions, and market data",
            func=fetch_live_data
        )