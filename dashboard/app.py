import os
import time
import json
import redis
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Grok4Trades Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Redis connection
@st.cache_resource
def get_redis_connection():
    return redis.Redis.from_url(
        os.getenv('REDIS_URL', 'redis://redis-cache:6379'),
        decode_responses=True
    )

redis_client = get_redis_connection()

# Custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stStatusWidget {
        display: none;
    }
    .status-card {
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .status-good {
        background-color: #e6f7e6;
        border-left: 5px solid #4CAF50;
    }
    .status-warning {
        background-color: #fff3cd;
        border-left: 5px solid #FFC107;
    }
    .status-danger {
        background-color: #f8d7da;
        border-left: 5px solid #DC3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def get_status():
    """Get current trading status from Redis"""
    try:
        status = redis_client.hgetall('trading:status')
        if not status:
            return {
                'running': 'False',
                'emergency_stop': 'False',
                'balance': '0',
                'last_updated': datetime.utcnow().isoformat()
            }
        return status
    except Exception as e:
        st.error(f"Error fetching status: {str(e)}")
        return None

def get_trades(limit=20):
    """Get recent trades from Redis"""
    try:
        # Get most recent trade IDs
        trade_ids = redis_client.zrevrange('trading:trade_history', 0, limit - 1)
        
        trades = []
        for trade_id in trade_ids:
            trade = redis_client.hgetall(f'trading:trades:{trade_id}')
            if trade:
                trades.append(trade)
        
        if trades:
            df = pd.DataFrame(trades)
            # Convert numeric columns
            for col in ['price', 'amount', 'cost', 'confidence']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            # Convert timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df.sort_values('timestamp', ascending=False)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching trades: {str(e)}")
        return pd.DataFrame()

def get_performance_metrics():
    """Calculate performance metrics from trades"""
    trades = get_trades(limit=1000)  # Get more trades for better metrics
    if trades.empty:
        return {}
    
    # Calculate basic metrics
    trades['pnl'] = trades.apply(
        lambda x: float(x['cost']) * (-1 if x['side'] == 'buy' else 1),
        axis=1
    )
    
    # Calculate win rate
    winning_trades = trades[trades['pnl'] > 0]
    win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
    
    # Calculate profit factor
    gross_profit = winning_trades['pnl'].sum()
    gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Calculate drawdown
    trades = trades.sort_values('timestamp')
    trades['cumulative_pnl'] = trades['pnl'].cumsum()
    trades['running_max'] = trades['cumulative_pnl'].cummax()
    trades['drawdown'] = trades['running_max'] - trades['cumulative_pnl']
    max_drawdown = trades['drawdown'].max() if not trades.empty else 0
    
    return {
        'total_trades': len(trades),
        'win_rate': win_rate,
        'total_pnl': trades['pnl'].sum(),
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'avg_win': winning_trades['pnl'].mean() if not winning_trades.empty else 0,
        'avg_loss': trades[trades['pnl'] < 0]['pnl'].mean() if len(trades[trades['pnl'] < 0]) > 0 else 0
    }

# Dashboard layout
def main():
    st.title("🚀 Grok4Trades Dashboard")
    
    # Auto-refresh every 30 seconds
    st_autorefresh = st.empty()
    if st_autorefresh.button("🔄 Refresh"):
        st.experimental_rerun()
    
    # Get current status
    status = get_status()
    
    # Status cards
    if status:
        # Top row: Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_class = "status-warning" if status['running'] == 'True' else "status-danger"
            if status['emergency_stop'] == 'True':
                status_class = "status-danger"
            
            st.markdown(
                f"""
                <div class="status-card {status_class}">
                    <h3>Status</h3>
                    <h2>{'🟢 Running' if status['running'] == 'True' else '🔴 Stopped'}</h2>
                    <p>Last updated: {datetime.fromisoformat(status['last_updated']).strftime('%Y-%m-%d %H:%M:%S') if 'last_updated' in status else 'N/A'}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            emergency_class = "status-danger" if status.get('emergency_stop') == 'True' else "status-good"
            st.markdown(
                f"""
                <div class="status-card {emergency_class}">
                    <h3>Emergency Stop</h3>
                    <h2>{'🔴 Active' if status.get('emergency_stop') == 'True' else '🟢 Inactive'}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            balance = float(status.get('balance', 0))
            st.markdown(
                f"""
                <div class="status-card">
                    <h3>Account Balance</h3>
                    <h2>${balance:,.2f} USDT</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col4:
            st.markdown(
                """
                <div class="status-card">
                    <h3>Mode</h3>
                    <h2>📝 Paper Trading</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Performance metrics
    st.markdown("## 📈 Performance Metrics")
    metrics = get_performance_metrics()
    
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Total P&L</div>
                    <div class="metric-value" style="color: {'#4CAF50' if metrics['total_pnl'] >= 0 else '#DC3545'}">
                        ${metrics['total_pnl']:+,.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value">
                        {metrics['win_rate']*100:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Profit Factor</div>
                    <div class="metric-value">
                        {metrics['profit_factor']:.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value" style="color: {'#4CAF50' if metrics['max_drawdown'] == 0 else '#DC3545'}">
                        ${metrics['max_drawdown']:,.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Recent trades
    st.markdown("## 📝 Recent Trades")
    trades = get_trades()
    
    if not trades.empty:
        # Display trades in a table
        st.dataframe(
            trades[['timestamp', 'side', 'price', 'amount', 'cost', 'reason']].head(20),
            column_config={
                'timestamp': st.column_config.DatetimeColumn('Time'),
                'side': st.column_config.TextColumn('Side'),
                'price': st.column_config.NumberColumn('Price', format="$%.2f"),
                'amount': st.column_config.NumberColumn('Amount'),
                'cost': st.column_config.NumberColumn('Cost', format="$%.2f"),
                'reason': st.column_config.TextColumn('Reason')
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Plot PnL over time
        trades_pnl = trades.sort_values('timestamp')
        trades_pnl['cumulative_pnl'] = trades_pnl['pnl'].cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trades_pnl['timestamp'],
            y=trades_pnl['cumulative_pnl'],
            mode='lines+markers',
            name='Cumulative P&L',
            line=dict(color='#4CAF50')
        ))
        
        fig.update_layout(
            title='Cumulative P&L Over Time',
            xaxis_title='Date',
            yaxis_title='P&L (USDT)',
            template='plotly_white',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trades recorded yet.")
    
    # Control panel
    st.sidebar.title("⚙️ Control Panel")
    
    st.sidebar.markdown("### Trading Controls")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("▶️ Start Trading", type="primary"):
            try:
                redis_client.hset('trading:status', 'running', 'True')
                st.sidebar.success("Trading started!")
                time.sleep(1)
                st.experimental_rerun()
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
    
    with col2:
        if st.button("⏹️ Stop Trading"):
            try:
                redis_client.hset('trading:status', 'running', 'False')
                st.sidebar.warning("Trading stopped.")
                time.sleep(1)
                st.experimental_rerun()
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
    
    st.sidebar.markdown("### Emergency Controls")
    
    if st.sidebar.button("🛑 Emergency Stop", type="secondary"):
        try:
            redis_client.hmset('trading:status', {
                'running': 'False',
                'emergency_stop': 'True'
            })
            st.sidebar.error("EMERGENCY STOP ACTIVATED!")
            time.sleep(1)
            st.experimental_rerun()
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")
    
    # Debug information
    if st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.markdown("### Debug Information")
        st.sidebar.json(status or {})

if __name__ == "__main__":
    main()
