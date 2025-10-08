"""
Streamlit Web Interface
Interactive web application for the match forecasting assistant
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agent.core import MatchForecastingAgent


def run_web(model: str = "gpt-4"):
    """Run the Streamlit web interface"""
    
    st.set_page_config(
        page_title="LLM Match Forecasting Assistant",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = MatchForecastingAgent(model_name=model)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Header
    st.title("🏆 LLM Match Forecasting Assistant")
    st.markdown("*AI-powered predictions with explainable reasoning*")
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Configuration")
        
        # Model selection
        selected_model = st.selectbox(
            "LLM Model",
            ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet"],
            index=0 if model == "gpt-4" else 1
        )
        
        if selected_model != model:
            st.session_state.agent = MatchForecastingAgent(model_name=selected_model)
        
        # Temperature
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        
        st.divider()
        
        # Quick actions
        st.header("🚀 Quick Actions")
        
        if st.button("📊 Sample Prediction"):
            sample_query = "Predict Manchester United vs Arsenal this weekend"
            st.session_state.current_query = sample_query
        
        if st.button("💰 Strategy Simulation"):
            sample_query = "Simulate a Kelly criterion betting strategy for Premier League matches"
            st.session_state.current_query = sample_query
        
        if st.button("🌤️ Weather Analysis"):
            sample_query = "How would heavy rain affect Liverpool vs Chelsea at Anfield?"
            st.session_state.current_query = sample_query
        
        st.divider()
        
        # Statistics
        st.header("📈 Session Stats")
        st.metric("Predictions Made", len(st.session_state.chat_history))
        st.metric("Model", selected_model)
        
        # Clear history
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Chat input
        query = st.text_input(
            "Ask me about match predictions, strategy simulations, or what-if scenarios:",
            value=getattr(st.session_state, 'current_query', ''),
            placeholder="e.g., What are the chances of Manchester City beating Liverpool?"
        )
        
        col_send, col_clear = st.columns([1, 1])
        
        with col_send:
            if st.button("🚀 Send", type="primary") and query:
                process_query(query)
        
        with col_clear:
            if st.button("🔄 Clear Input"):
                st.session_state.current_query = ""
                st.rerun()
        
        # Chat history
        st.subheader("💭 Conversation History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Query {len(st.session_state.chat_history) - i}: {chat['query'][:50]}..."):
                st.markdown(f"**Query:** {chat['query']}")
                st.markdown(f"**Response:** {chat['response']}")
                
                if 'reasoning_steps' in chat and chat['reasoning_steps']:
                    st.markdown("**Reasoning Steps:**")
                    for step in chat['reasoning_steps']:
                        st.markdown(f"- {step}")
                
                if 'confidence' in chat:
                    st.metric("Confidence", f"{chat['confidence']:.1%}")
    
    with col2:
        st.header("📊 Analytics Dashboard")
        
        # Sample visualization
        if st.session_state.chat_history:
            # Confidence over time
            confidences = [chat.get('confidence', 0.5) for chat in st.session_state.chat_history]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=confidences,
                mode='lines+markers',
                name='Confidence',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Prediction Confidence Over Time",
                xaxis_title="Prediction Number",
                yaxis_title="Confidence",
                yaxis=dict(range=[0, 1]),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Sample team comparison
        st.subheader("⚽ Team Comparison")
        
        teams_data = {
            'Team': ['Man United', 'Arsenal', 'Liverpool', 'Man City'],
            'Win Probability': [0.65, 0.72, 0.78, 0.85],
            'Goals Per Game': [1.8, 2.1, 2.3, 2.7],
            'Defense Rating': [0.8, 0.85, 0.75, 0.9]
        }
        
        df = pd.DataFrame(teams_data)
        
        fig = px.scatter(
            df, 
            x='Goals Per Game', 
            y='Win Probability',
            size='Defense Rating',
            color='Team',
            title="Team Performance Matrix"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Strategy performance
        st.subheader("💰 Strategy Performance")
        
        strategy_data = {
            'Strategy': ['Fixed Bet', 'Kelly Criterion', 'Martingale', 'Value Betting'],
            'ROI': [5.2, 12.8, -8.5, 15.3],
            'Win Rate': [0.52, 0.58, 0.48, 0.61]
        }
        
        strategy_df = pd.DataFrame(strategy_data)
        
        fig = px.bar(
            strategy_df,
            x='Strategy',
            y='ROI',
            color='Win Rate',
            title="Betting Strategy Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)


def process_query(query: str):
    """Process user query and update chat history"""
    
    with st.spinner("🤔 Thinking..."):
        try:
            # Determine query type
            if any(word in query.lower() for word in ['simulate', 'strategy', 'betting', 'trading']):
                result = st.session_state.agent.simulate_strategy(query)
            elif any(word in query.lower() for word in ['what if', 'scenario', 'weather', 'injury']):
                result = st.session_state.agent.what_if_analysis(query)
            else:
                result = st.session_state.agent.predict_match(query)
            
            if result['success']:
                # Add to chat history
                chat_entry = {
                    'query': query,
                    'response': result.get('prediction', result.get('simulation', result.get('analysis', ''))),
                    'reasoning_steps': result.get('reasoning_steps', result.get('steps', result.get('scenarios', []))),
                    'confidence': result.get('confidence', 0.5),
                    'timestamp': pd.Timestamp.now()
                }
                
                st.session_state.chat_history.append(chat_entry)
                
                # Show success message
                st.success("✅ Analysis complete!")
                
            else:
                st.error(f"❌ Error: {result.get('error', 'Unknown error occurred')}")
        
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
    
    # Clear current query
    if hasattr(st.session_state, 'current_query'):
        del st.session_state.current_query
    
    st.rerun()


if __name__ == "__main__":
    run_web()