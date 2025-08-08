import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
import json
import os
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Grok4Trades - Minimal Dashboard",
    page_icon="📊",
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
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        font-weight: 500;
    }
    .stTextInput>div>div>input {
        border-radius: 0.5rem;
    }
    .stSelectbox>div>div>div {
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

class TradingDashboard:
    """
    A minimal Streamlit dashboard for monitoring and controlling the trading bot.
    """
    
    def __init__(self):
        self.state = self._init_session_state()
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize dashboard
        self._setup_sidebar()
        self._setup_main_content()
    
    def _init_session_state(self) -> Dict[str, Any]:
        """Initialize the Streamlit session state"""
        if 'bot_running' not in st.session_state:
            st.session_state.bot_running = False
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if 'trades' not in st.session_state:
            st.session_state.trades = []
        if 'balance_history' not in st.session_state:
            st.session_state.balance_history = []
            
        return st.session_state
    
    def _setup_sidebar(self) -> None:
        """Setup the sidebar with controls and status"""
        with st.sidebar:
            st.title("🎛️ Controls")
            
            # Bot status
            status_color = "green" if self.state.bot_running else "red"
            st.markdown(f"### Status: <span style='color:{status_color}'>{'🟢 Running' if self.state.bot_running else '🔴 Stopped'}</span>", 
                       unsafe_allow_html=True)
            
            # Control buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶️ Start", disabled=self.state.bot_running, 
                           help="Start the trading bot"):
                    self._start_bot()
            with col2:
                if st.button("⏹️ Stop", disabled=not self.state.bot_running,
                           help="Stop the trading bot"):
                    self._stop_bot()
            
            st.markdown("---")
            
            # Strategy settings
            st.subheader("⚙️ Strategy Settings")
            
            # Symbol selection
            self.symbol = st.selectbox(
                "Trading Pair",
                ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"],
                index=0,
                help="Select the trading pair to trade"
            )
            
            # Strategy parameters
            self.strategy = st.selectbox(
                "Strategy",
                ["Simple MA Crossover", "Mean Reversion", "Breakout"],
                index=0,
                help="Select the trading strategy"
            )
            
            # Risk settings
            st.subheader("⚠️ Risk Settings")
            self.max_risk = st.slider(
                "Max Risk per Trade (%)",
                min_value=0.1,
                max_value=10.0,
                value=2.0,
                step=0.1,
                help="Maximum percentage of account to risk per trade"
            )
            
            self.max_daily_loss = st.slider(
                "Max Daily Loss (%)",
                min_value=1.0,
                max_value=20.0,
                value=5.0,
                step=0.5,
                help="Maximum percentage of account to lose in a day"
            )
            
            # Trading mode
            self.trading_mode = st.radio(
                "Trading Mode",
                ["Paper Trading", "Live Trading"],
                index=0,
                help="Paper trading uses simulated funds, live trading uses real money"
            )
            
            st.markdown("---")
            
            # Performance summary
            st.subheader("📊 Performance")
            
            # Today's P&L
            today_pnl = self._get_todays_pnl()
            pnl_color = "green" if today_pnl >= 0 else "red"
            st.markdown(f"**Today's P&L:** <span style='color:{pnl_color}'>${today_pnl:,.2f}</span>", 
                       unsafe_allow_html=True)
            
            # Total trades today
            total_trades = len(self._get_todays_trades())
            st.markdown(f"**Trades Today:** {total_trades}")
            
            # Win rate
            win_rate = self._calculate_win_rate()
            st.markdown(f"**Win Rate:** {win_rate:.1f}%")
            
            # Last update
            st.markdown(f"*Last updated: {self.state.last_update.strftime('%H:%M:%S')}*")
    
    def _setup_main_content(self) -> None:
        """Setup the main content area with charts and logs"""
        st.title("📊 Grok4Trades - Minimal Dashboard")
        
        # Metrics row
        self._display_metrics()
        
        # Charts row
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Price chart
            self._display_price_chart()
            
            # Recent trades
            st.subheader("📝 Recent Trades")
            self._display_recent_trades()
        
        with col2:
            # Account balance
            st.subheader("💰 Account Balance")
            self._display_balance_chart()
            
            # Open positions
            st.subheader("📊 Open Positions")
            self._display_open_positions()
            
            # System logs
            st.subheader("📜 System Logs")
            self._display_system_logs()
    
    def _display_metrics(self) -> None:
        """Display key metrics in a row"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("<div class='metric-card'>"
                       "<h3>Current Price</h3>"
                       f"<h2>${self._get_current_price():,.2f}</h2>"
                       "<p>Last updated: Just now</p>"
                       "</div>", unsafe_allow_html=True)
        
        with col2:
            pnl_today = self._get_todays_pnl()
            pnl_color = "green" if pnl_today >= 0 else "red"
            st.markdown(f"<div class='metric-card'>"
                       f"<h3>Today's P&L</h3>"
                       f"<h2 style='color:{pnl_color}'>${pnl_today:+,.2f}</h2>"
                       f"<p>{'↑' if pnl_today >= 0 else '↓'} {abs(pnl_today/1000*100):.1f}%</p>"
                       "</div>", unsafe_allow_html=True)
        
        with col3:
            total_trades = len(self._get_todays_trades())
            win_rate = self._calculate_win_rate()
            st.markdown("<div class='metric-card'>"
                       "<h3>Performance</h3>"
                       f"<h2>{win_rate:.1f}% Win Rate</h2>"
                       f"<p>{total_trades} trades today</p>"
                       "</div>", unsafe_allow_html=True)
        
        with col4:
            account_balance = self._get_account_balance()
            st.markdown("<div class='metric-card'>"
                       "<h3>Account Balance</h3>"
                       f"<h2>${account_balance:,.2f}</h2>"
                       "<p>Available: ${account_balance * 0.95:,.2f}</p>"
                       "</div>", unsafe_allow_html=True)
    
    def _display_price_chart(self) -> None:
        """Display the price chart with indicators"""
        # In a real implementation, this would fetch real market data
        # For now, we'll generate some sample data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='15min')
        prices = np.cumsum(np.random.randn(100) * 10 + 50000)
        
        # Create a simple line chart
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add moving averages (sample)
        fig.add_trace(go.Scatter(
            x=dates[-50:],
            y=prices[-50:].rolling(20).mean(),
            mode='lines',
            name='MA(20)',
            line=dict(color='#ff7f0e', width=1.5, dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis_rangeslider_visible=False,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_balance_chart(self) -> None:
        """Display the account balance chart"""
        # Generate sample balance data if none exists
        if not self.state.balance_history:
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            balance = 10000 + np.cumsum(np.random.randn(30) * 100)
            self.state.balance_history = list(zip(dates, balance))
        
        # Create balance chart
        dates, balances = zip(*self.state.balance_history)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=balances,
            mode='lines+markers',
            name='Balance',
            line=dict(color='#2ecc71', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False,
            xaxis_showgrid=False,
            yaxis_showgrid=True,
            yaxis_title="Balance (USD)",
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_recent_trades(self) -> None:
        """Display a table of recent trades"""
        # In a real implementation, this would fetch from the database
        if not self.state.trades:
            # Generate sample trades
            symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
            actions = ["BUY", "SELL"]
            
            for _ in range(10):
                trade = {
                    "time": (datetime.now() - timedelta(minutes=np.random.randint(1, 1440))).strftime("%H:%M:%S"),
                    "symbol": np.random.choice(symbols),
                    "action": np.random.choice(actions),
                    "size": round(np.random.uniform(0.01, 0.5), 4),
                    "price": round(np.random.uniform(100, 100000), 2),
                    "pnl": round(np.random.uniform(-100, 100), 2)
                }
                trade["value"] = trade["size"] * trade["price"]
                self.state.trades.append(trade)
        
        # Display trades in a table
        if self.state.trades:
            df = pd.DataFrame(self.state.trades)
            
            # Format the table
            df["P&L"] = df["pnl"].apply(
                lambda x: f"<span style='color: {'green' if x >= 0 else 'red'}'>{x:+,.2f}</span>"
            )
            
            st.markdown(
                df[["time", "symbol", "action", "size", "price", "P&L"]]
                .to_html(escape=False, index=False),
                unsafe_allow_html=True
            )
        else:
            st.info("No trades yet today.")
    
    def _display_open_positions(self) -> None:
        """Display current open positions"""
        # In a real implementation, this would fetch from the exchange
        positions = [
            {"symbol": "BTC/USDT", "size": 0.05, "entry": 45000, "current": 45230, "pnl": 11.5, "pnl_pct": 0.51},
            {"symbol": "ETH/USDT", "size": 0.5, "entry": 2500, "current": 2480, "pnl": -10.0, "pnl_pct": -0.8}
        ]
        
        if positions:
            for pos in positions:
                pnl_color = "green" if pos["pnl"] >= 0 else "red"
                st.markdown(f"""
                <div style="margin-bottom: 1rem; padding: 0.75rem; background-color: #f8f9fa; border-radius: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; font-weight: 500;">
                        <span>{pos['symbol']}</span>
                        <span>{pos['size']} {pos['symbol'].split('/')[0]}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.9rem;">
                        <span>Entry: ${pos['entry']:,.2f}</span>
                        <span>Current: ${pos['current']:,.2f}</span>
                    </div>
                    <div style="margin-top: 0.5rem; text-align: right; color: {pnl_color};">
                        ${pos['pnl']:+,.2f} ({pos['pnl_pct']:+,.2f}%)
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No open positions.")
    
    def _display_system_logs(self) -> None:
        """Display system logs"""
        # In a real implementation, this would read from a log file
        logs = [
            {"time": "10:23:15", "level": "INFO", "message": "Bot started"},
            {"time": "10:23:16", "level": "INFO", "message": "Connected to exchange"},
            {"time": "10:25:30", "level": "TRADE", "message": "BUY 0.05 BTC @ $45,230.50"},
            {"time": "10:30:45", "level": "INFO", "message": "Checking market conditions..."},
            {"time": "10:35:20", "level": "WARNING", "message": "High volatility detected"}
        ]
        
        log_html = "<div style='max-height: 200px; overflow-y: auto; font-family: monospace; font-size: 0.85rem;'>"
        
        for log in logs[-10:]:  # Show only the last 10 logs
            level_color = {
                "INFO": "#3498db",
                "WARNING": "#f39c12",
                "ERROR": "#e74c3c",
                "TRADE": "#2ecc71"
            }.get(log["level"], "#7f8c8d")
            
            log_html += f"""
            <div style="margin-bottom: 0.25rem;">
                <span style="color: #7f8c8d;">[{log['time']}]</span>
                <span style="color: {level_color}; font-weight: 500;">{log['level']}</span>
                <span>{log['message']}</span>
            </div>
            """
        
        log_html += "</div>"
        st.markdown(log_html, unsafe_allow_html=True)
    
    def _start_bot(self) -> None:
        """Start the trading bot"""
        if not self.state.bot_running:
            self.state.bot_running = True
            self._log_event("INFO", "Trading bot started")
            st.experimental_rerun()
    
    def _stop_bot(self) -> None:
        """Stop the trading bot"""
        if self.state.bot_running:
            self.state.bot_running = False
            self._log_event("INFO", "Trading bot stopped")
            st.experimental_rerun()
    
    def _log_event(self, level: str, message: str) -> None:
        """Log an event to the system logs"""
        # In a real implementation, this would write to a log file
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {level}: {message}")
    
    def _get_current_price(self) -> float:
        """Get the current price of the selected symbol"""
        # In a real implementation, this would fetch from the exchange
        return 45230.50  # Sample price
    
    def _get_account_balance(self) -> float:
        """Get the current account balance"""
        # In a real implementation, this would fetch from the exchange
        return 10000.0  # Sample balance
    
    def _get_todays_trades(self) -> List[Dict]:
        """Get today's trades"""
        # In a real implementation, this would filter trades from the database
        today = datetime.now().date()
        return [t for t in self.state.trades if 
                datetime.strptime(t["time"], "%H:%M:%S").date() == today]
    
    def _get_todays_pnl(self) -> float:
        """Calculate today's P&L"""
        # In a real implementation, this would calculate from the database
        return sum(t["pnl"] for t in self._get_todays_trades())
    
    def _calculate_win_rate(self) -> float:
        """Calculate the win rate of trades"""
        # In a real implementation, this would calculate from the database
        if not self.state.trades:
            return 0.0
        
        winning_trades = sum(1 for t in self.state.trades if t["pnl"] > 0)
        return (winning_trades / len(self.state.trades)) * 100

# Run the dashboard
if __name__ == "__main__":
    dashboard = TradingDashboard()
