#!/usr/bin/env python3
"""
Sniper Bot Dashboard

A Streamlit-based dashboard for monitoring the SniperBot's activity.
"""
import os
import json
import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
from loguru import logger

# Set page config
st.set_page_config(
    page_title="Sniper Bot Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #1E2130;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
    st.session_state.trades = []
    st.session_state.balance_history = []
    st.session_state.logs = []

# Sidebar for controls
with st.sidebar:
    st.title("🚀 Sniper Bot")
    
    st.header("Settings")
    
    # Trading parameters
    st.subheader("Trading Parameters")
    target_profit = st.slider(
        "Target Profit (%)", 
        min_value=0.1, 
        max_value=50.0, 
        value=5.0, 
        step=0.1,
        help="Target profit percentage before selling"
    )
    
    stop_loss = st.slider(
        "Stop Loss (%)", 
        min_value=0.1, 
        max_value=20.0, 
        value=2.0, 
        step=0.1,
        help="Stop loss percentage"
    )
    
    max_slippage = st.slider(
        "Max Slippage (%)", 
        min_value=0.1, 
        max_value=10.0, 
        value=1.0, 
        step=0.1,
        help="Maximum allowed slippage percentage"
    )
    
    min_liquidity = st.number_input(
        "Minimum Liquidity (ETH)", 
        min_value=0.1, 
        max_value=100.0, 
        value=5.0, 
        step=0.1,
        help="Minimum liquidity required to trade a token"
    )
    
    # Bot controls
    st.subheader("Bot Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start Bot", use_container_width=True):
            # TODO: Implement bot start
            st.success("Bot started!")
    with col2:
        if st.button("⏹️ Stop Bot", use_container_width=True):
            # TODO: Implement bot stop
            st.warning("Bot stopped!")
    
    # Status
    st.subheader("Status")
    status = st.empty()
    status.info("❌ Bot is not running")
    
    # Version info
    st.markdown("---")
    st.markdown("**Version:** 0.1.0")
    st.markdown("**Last Updated:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Main content
st.title("📊 Sniper Bot Dashboard")

# Metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Trades", "0")
    
with col2:
    st.metric("Win Rate", "0%")
    
with col3:
    st.metric("Total Profit (ETH)", "0.00")
    
with col4:
    st.metric("Active Trades", "0")

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Portfolio", "📊 Trades", "📝 Logs"])

with tab1:
    # Portfolio value over time
    st.subheader("Portfolio Value")
    
    # Dummy data - replace with real data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    values = [1000 + i * 10 + (i % 5) * 5 for i in range(30)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, 
        y=values,
        mode='lines+markers',
        name='Portfolio Value (ETH)'
    ))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value (ETH)",
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Current positions
    st.subheader("Current Positions")
    
    # Dummy data - replace with real data
    positions = pd.DataFrame({
        'Token': ['ETH', 'UNI', 'LINK'],
        'Amount': [1.5, 50.0, 100.0],
        'Value (ETH)': [1.5, 0.5, 0.3],
        'Entry Price (ETH)': [1.0, 0.01, 0.003],
        'Current Price (ETH)': [1.0, 0.01, 0.003],
        'P&L (%)': [0.0, 0.0, 0.0],
    })
    
    st.dataframe(
        positions,
        column_config={
            'Token': st.column_config.TextColumn("Token"),
            'Amount': st.column_config.NumberColumn("Amount", format="%.4f"),
            'Value (ETH)': st.column_config.NumberColumn("Value (ETH)", format="%.4f"),
            'Entry Price (ETH)': st.column_config.NumberColumn("Entry Price (ETH)", format="%.6f"),
            'Current Price (ETH)': st.column_config.NumberColumn("Current Price (ETH)", format="%.6f"),
            'P&L (%)': st.column_config.ProgressColumn("P&L (%)", format="%.2f%%", min_value=-100, max_value=100),
        },
        hide_index=True,
        use_container_width=True
    )

with tab2:
    # Trade history
    st.subheader("Trade History")
    
    # Dummy data - replace with real data
    trades = pd.DataFrame({
        'Date': [datetime.now() - pd.Timedelta(days=i) for i in range(10, 0, -1)],
        'Type': ['BUY' if i % 2 == 0 else 'SELL' for i in range(10)],
        'Token': [f'TOKEN{i}' for i in range(10, 0, -1)],
        'Amount': [100 - i*2 for i in range(10)],
        'Price (ETH)': [0.001 * (1 + i*0.1) for i in range(10)],
        'Value (ETH)': [(100 - i*2) * 0.001 * (1 + i*0.1) for i in range(10)],
        'Status': ['COMPLETED'] * 10,
    })
    
    st.dataframe(
        trades,
        column_config={
            'Date': st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD HH:mm:ss"),
            'Type': st.column_config.TextColumn("Type"),
            'Token': st.column_config.TextColumn("Token"),
            'Amount': st.column_config.NumberColumn("Amount", format="%.4f"),
            'Price (ETH)': st.column_config.NumberColumn("Price (ETH)", format="%.6f"),
            'Value (ETH)': st.column_config.NumberColumn("Value (ETH)", format="%.6f"),
            'Status': st.column_config.TextColumn("Status"),
        },
        hide_index=True,
        use_container_width=True
    )

with tab3:
    # Logs
    st.subheader("Logs")
    
    # Log level filter
    log_level = st.selectbox(
        "Log Level",
        ["ALL", "INFO", "WARNING", "ERROR", "CRITICAL"],
        index=0
    )
    
    # Dummy logs - replace with real logs
    log_entries = [
        ("INFO", "2023-11-01 10:00:00", "Bot started"),
        ("INFO", "2023-11-01 10:01:23", "Connected to Ethereum node"),
        ("WARNING", "2023-11-01 10:05:45", "High gas price detected: 150 Gwei"),
        ("INFO", "2023-11-01 10:10:12", "New token detected: 0x1234...5678"),
        ("INFO", "2023-11-01 10:10:15", "Liquidity check passed: 10.5 ETH"),
        ("INFO", "2023-11-01 10:10:20", "Buy order placed: 0.1 ETH for 1000 TOKEN"),
        ("SUCCESS", "2023-11-01 10:10:25", "Buy order executed successfully"),
        ("INFO", "2023-11-01 10:15:30", "Price update: +5.2% from buy price"),
        ("INFO", "2023-11-01 10:20:45", "Target profit reached (+5.0%)"),
        ("INFO", "2023-11-01 10:20:47", "Sell order placed: 1000 TOKEN"),
        ("SUCCESS", "2023-11-01 10:20:50", "Sell order executed successfully"),
        ("PROFIT", "2023-11-01 10:20:51", "Trade completed: +0.0052 ETH (+5.2%)"),
    ]
    
    # Filter logs by level
    filtered_logs = [
        log for log in log_entries 
        if log_level == "ALL" or log[0] == log_level
    ]
    
    # Display logs with color coding
    log_container = st.container()
    with log_container:
        for level, timestamp, message in filtered_logs:
            if level == "INFO":
                st.info(f"{timestamp} - {message}")
            elif level == "WARNING":
                st.warning(f"{timestamp} - {message}")
            elif level == "ERROR" or level == "CRITICAL":
                st.error(f"{timestamp} - {message}")
            elif level == "SUCCESS":
                st.success(f"{timestamp} - {message}")
            elif level == "PROFIT":
                st.success(f"{timestamp} - 💰 {message}")
            else:
                st.text(f"{timestamp} - {message}")

# Auto-refresh every 5 seconds
st_autorefresh = st.container()
with st_autorefresh:
    st_autorefresh.empty()  # This is a hack to make the autorefresh work
    time.sleep(5)
    st.rerun()
