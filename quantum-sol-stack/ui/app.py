"""
QuantumSol Stack - Streamlit UI

This module provides a web-based user interface for monitoring and controlling
the QuantumSol trading system.
"""
import os
import sys
import json
import time
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import ccxt
from loguru import logger
import plotly.express as px

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import components
from components.config_editor import render_config_editor
from components.model_generator import render_model_generator
from components.agent_controls import render_agent_controls, render_profit_targets, render_kill_switch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="QuantumSol Stack",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DEFAULT_SYMBOL = "SOL-USD"
DEFAULT_INTERVAL = "1d"
DEFAULT_PERIOD = "6mo"

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.current_page = "dashboard"
    st.session_state.symbol = DEFAULT_SYMBOL
    st.session_state.interval = DEFAULT_INTERVAL
    st.session_state.period = DEFAULT_PERIOD
    st.session_state.trading_active = False
    st.session_state.kill_switch_active = False

# Authentication
def check_credentials(username: str, password: str) -> bool:
    """Check if the provided credentials are valid"""
    # In a real application, use proper authentication
    # This is just a simple example
    return (username == os.getenv("WEB_USERNAME", "admin") and 
            password == os.getenv("WEB_PASSWORD", "changeme123"))

def login():
    """Render the login form"""
    st.title("🔒 QuantumSol Login")
    st.write("Please enter your credentials to access the trading dashboard.")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if check_credentials(username, password):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid username or password")

def logout():
    """Log out the current user"""
    st.session_state.authenticated = False
    st.rerun()

# Data fetching
def fetch_market_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Fetch historical market data"""
    try:
        # Convert symbol format if needed (e.g., SOL-USD to SOL/USDT)
        yf_symbol = symbol.replace("/", "-")
        if not yf_symbol.endswith("-USD") and not yf_symbol.endswith("-USDT"):
            yf_symbol += "-USD"
        
        # Download data
        df = yf.download(
            yf_symbol, 
            period=period,
            interval=interval,
            progress=False
        )
        
        # Reset index to get Date as a column
        df = df.reset_index()
        
        # Calculate technical indicators
        if not df.empty:
            df = add_technical_indicators(df)
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return pd.DataFrame()

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the DataFrame"""
    if df.empty:
        return df
    
    try:
        # Convert column names to lowercase for consistency
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure required columns exist
        required_columns = ['close', 'high', 'low', 'open', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return df
        
        # Calculate moving averages
        df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
        df['ema_12'] = EMAIndicator(close=df['close'], window=12).ema_indicator()
        df['ema_26'] = EMAIndicator(close=df['close'], window=26).ema_indicator()
        
        # Calculate RSI
        df['rsi'] = RSIIndicator(close=df['close']).rsi()
        
        # Calculate MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # Calculate Bollinger Bands
        bb = BollingerBands(close=df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        
        # Calculate ATR
        df['atr'] = AverageTrueRange(
            high=df['high'], 
            low=df['low'], 
            close=df['close']
        ).average_true_range()
        
        # Calculate Stochastic Oscillator
        stoch = StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close']
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        return df
    
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        return df

# Visualization functions
def plot_candlestick_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create a candlestick chart with volume"""
    if df.empty:
        return go.Figure()
    
    try:
        # Create subplots
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{symbol} Price", "Volume")
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Price",
                increasing_line_color='#2ecc71',
                decreasing_line_color='#e74c3c'
            ),
            row=1, col=1
        )
        
        # Add moving averages if they exist
        if 'sma_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['sma_20'],
                    name="SMA 20",
                    line=dict(color='#3498db', width=1.5)
                ),
                row=1, col=1
            )
        
        if 'sma_50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['sma_50'],
                    name="SMA 50",
                    line=dict(color='#f39c12', width=1.5)
                ),
                row=1, col=1
            )
        
        # Add Bollinger Bands if they exist
        if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['bb_upper'],
                    name="BB Upper",
                    line=dict(color='#95a5a6', width=1, dash='dash'),
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['bb_middle'],
                    name="BB Middle",
                    line=dict(color='#7f8c8d', width=1),
                    opacity=0.7,
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['bb_lower'],
                    name="BB Lower",
                    line=dict(color='#95a5a6', width=1, dash='dash'),
                    opacity=0.7,
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Volume chart
        fig.add_trace(
            go.Bar(
                x=df['date'],
                y=df['volume'],
                name="Volume",
                marker_color='#3498db',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(rangeslider=dict(visible=False)),
            xaxis2=dict(title="Date"),
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Volume"),
            margin=dict(l=50, r=50, t=50, b=50),
            template="plotly_dark",
            plot_bgcolor='#1e293b',
            paper_bgcolor='#0f172a',
            font=dict(color='#f8fafc'),
            hovermode="x unified"
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating candlestick chart: {e}")
        return go.Figure()

def plot_technical_indicators(df: pd.DataFrame) -> go.Figure:
    """Create a figure with technical indicators"""
    if df.empty:
        return go.Figure()
    
    try:
        # Create subplots
        fig = make_subplots(
            rows=3, 
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.4, 0.3, 0.3],
            subplot_titles=("RSI", "MACD", "Stochastic")
        )
        
        # RSI
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['rsi'],
                    name="RSI",
                    line=dict(color='#3498db', width=1.5)
                ),
                row=1, col=1
            )
            
            # Add RSI levels
            fig.add_hline(
                y=70, 
                line_dash="dash", 
                line_color="#e74c3c",
                opacity=0.7,
                row=1, 
                col=1
            )
            
            fig.add_hline(
                y=30, 
                line_dash="dash", 
                line_color="#2ecc71",
                opacity=0.7,
                row=1, 
                col=1
            )
        
        # MACD
        if all(col in df.columns for col in ['macd', 'macd_signal', 'macd_hist']):
            # MACD Line
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['macd'],
                    name="MACD",
                    line=dict(color='#3498db', width=1.5)
                ),
                row=2, col=1
            )
            
            # Signal Line
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['macd_signal'],
                    name="Signal",
                    line=dict(color='#f39c12', width=1.5)
                ),
                row=2, col=1
            )
            
            # Histogram
            colors = ['#2ecc71' if val >= 0 else '#e74c3c' 
                     for val in df['macd_hist']]
            
            fig.add_trace(
                go.Bar(
                    x=df['date'],
                    y=df['macd_hist'],
                    name="Histogram",
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Stochastic Oscillator
        if all(col in df.columns for col in ['stoch_k', 'stoch_d']):
            # %K line
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['stoch_k'],
                    name="%K",
                    line=dict(color='#3498db', width=1.5)
                ),
                row=3, col=1
            )
            
            # %D line
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['stoch_d'],
                    name="%D",
                    line=dict(color='#f39c12', width=1.5)
                ),
                row=3, col=1
            )
            
            # Add overbought/oversold levels
            for level in [80, 20]:
                fig.add_hline(
                    y=level,
                    line_dash="dash",
                    line_color="#95a5a6",
                    opacity=0.7,
                    row=3,
                    col=1
                )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis3=dict(title="Date"),
            margin=dict(l=50, r=50, t=50, b=50),
            template="plotly_dark",
            plot_bgcolor='#1e293b',
            paper_bgcolor='#0f172a',
            font=dict(color='#f8fafc'),
            hovermode="x unified"
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating technical indicators chart: {e}")
        return go.Figure()

# Page rendering
def render_dashboard():
    """Render the main dashboard page"""
    st.title("📊 QuantumSol Dashboard")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        
        # Symbol selection
        symbol = st.text_input("Symbol", value=st.session_state.symbol)
        
        # Timeframe selection
        interval = st.selectbox(
            "Interval",
            ["1m", "5m", "15m", "1h", "1d", "1wk"],
            index=["1m", "5m", "15m", "1h", "1d", "1wk"].index(st.session_state.interval)
        )
        
        # Period selection
        period = st.selectbox(
            "Period",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
            index=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"].index(st.session_state.period)
        )
        
        # Update button
        if st.button("Update Data"):
            st.session_state.symbol = symbol
            st.session_state.interval = interval
            st.session_state.period = period
            st.rerun()
        
        st.divider()
        
        # Navigation
        st.header("Navigation")
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state.current_page = "dashboard"
            st.rerun()
            
        if st.button("⚙️ Configuration", use_container_width=True):
            st.session_state.current_page = "config"
            st.rerun()
            
        if st.button("🤖 Model Generator", use_container_width=True):
            st.session_state.current_page = "model_generator"
            st.rerun()
            
        if st.button("📈 Backtesting", use_container_width=True):
            st.session_state.current_page = "backtesting"
            st.rerun()
            
        if st.button("📊 Analytics", use_container_width=True):
            st.session_state.current_page = "analytics"
    
    with tab1:  # Dashboard Tab
        # Market data controls
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.symbol = st.text_input("Symbol", st.session_state.symbol)
        with col2:
            st.session_state.interval = st.selectbox(
                "Interval",
                ["1m", "5m", "15m", "1h", "1d", "1wk", "1mo"],
                index=3
            )
        with col3:
            st.session_state.period = st.selectbox(
                "Period",
                ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
                index=5
            )
        
        # Fetch and display market data
        df = fetch_market_data(
            st.session_state.symbol,
            st.session_state.interval,
            st.session_state.period
        )
        
        if not df.empty:
            # Display charts
            st.plotly_chart(
                plot_candlestick_chart(df, st.session_state.symbol),
                use_container_width=True
            )
            
            st.plotly_chart(
                plot_technical_indicators(df),
                use_container_width=True
            )
        else:
            st.warning("No data available for the selected symbol and time period.")
    
    with tab2:  # Agents Tab
        render_agent_controls()
    
    with tab3:  # Performance Tab
        render_profit_targets()
        
        # Add performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Win Rate", "64.2%", "+2.1%")
        with col2:
            st.metric("Avg. Trade P&L", "$42.50", "-$5.20")
        with col3:
            st.metric("Max Drawdown", "8.7%", "-1.2%")
        
        # Grafana Dashboard
        st.subheader("Trading Dashboard")
        st.markdown("""
        <iframe src="http://localhost:3000/d/quantumsol-dashboard/quantumsol-trading-dashboard?orgId=1&refresh=5s&kiosk" 
                width="100%" 
                height="800" 
                frameborder="0"
                style="border: 1px solid #ddd; border-radius: 5px;">
        </iframe>
        """, unsafe_allow_html=True)
        
        # Recent trades table
        st.subheader("Recent Trades")
        trades = [
            {"Date": "2023-07-26 11:30", "Symbol": "SOL/USDT", "Side": "BUY", "Size": "10", "Price": "24.56", "P&L": "+2.8%"},
            {"Date": "2023-07-26 10:45", "Symbol": "BTC/USDT", "Side": "SELL", "Size": "0.5", "Price": "29,450", "P&L": "-1.2%"},
            {"Date": "2023-07-26 09:15", "Symbol": "ETH/USDT", "Side": "BUY", "Size": "5", "Price": "1,845", "P&L": "+1.5%"},
            {"Date": "2023-07-25 15:20", "Symbol": "BONK/USDT", "Side": "SELL", "Size": "50000", "Price": "0.000012", "P&L": "+8.3%"},
            {"Date": "2023-07-25 11:10", "Symbol": "SOL/USDT", "Side": "SELL", "Size": "8", "Price": "25.10", "P&L": "+3.1%"},
        ]
        st.dataframe(trades, use_container_width=True, hide_index=True)
    
    with tab4:  # Settings Tab
        render_kill_switch()
        
        st.subheader("System Settings")
        
        # API Configuration
        with st.expander("🔑 API Configuration", expanded=False):
            st.text_input("Exchange API Key", type="password")
            st.text_input("Exchange API Secret", type="password")
            st.text_input("OpenAI API Key", type="password")
            if st.button("Save API Keys", type="primary"):
                st.success("API keys updated successfully!")
        
        # Notification Settings
        with st.expander("🔔 Notifications", expanded=False):
            email = st.text_input("Email Address", "your.email@example.com")
            st.checkbox("Email Alerts", True)
            st.checkbox("SMS Alerts", False)
            st.checkbox("Push Notifications", True)
            st.slider("Alert Threshold (%)", 0.1, 10.0, 2.5, 0.1)
            if st.button("Save Notification Settings", type="primary"):
                st.success("Notification settings saved!")
        
        # System Info
        with st.expander("ℹ️ System Information", expanded=False):
            st.write("**Version:** 1.0.0")
            st.write("**Last Updated:** 2023-07-26")
            st.write("**Status:** Operational")
            st.write("**Uptime:** 3d 12h 45m")

# Main application
def main():
    """Main application entry point"""
    # Check authentication
    if not st.session_state.authenticated:
        login()
        return
    
    # Render the appropriate page based on navigation
    if st.session_state.current_page == "dashboard":
        render_dashboard()
    elif st.session_state.current_page == "config":
        render_config_editor()
    elif st.session_state.current_page == "model_generator":
        render_model_generator()
    elif st.session_state.current_page == "backtesting":
        st.title("🔍 Backtesting")
        st.write("Backtesting functionality coming soon...")
    elif st.session_state.current_page == "analytics":
        st.title("📊 Analytics")
        st.write("Analytics dashboard coming soon...")
    else:
        st.session_state.current_page = "dashboard"
        st.rerun()
    
    # Add footer
    st.sidebar.divider()
    st.sidebar.caption(
        f"© {datetime.datetime.now().year} QuantumSol Stack | "
        f"v0.1.0 | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

if __name__ == "__main__":
    main()
