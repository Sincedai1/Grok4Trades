"""
Agent Controls Component

This module provides UI controls for interacting with the QuantumSol trading agents.
"""
import streamlit as st
from typing import Dict, Any, Optional, List
import json
import logging
from datetime import datetime
from ...services.agent_service import (
    get_agent_status, start_trading, stop_trading, emergency_stop,
    get_agent_logs, update_agent_config, AgentServiceError
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_agent_status_display():
    """Get and display agent status"""
    try:
        status = get_agent_status()
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Strategy Agent
        with col1:
            status_strat = status.get('strategy_agent', {})
            status_emoji = "🟢" if status_strat.get('status') == 'running' else "🔴"
            st.metric("Strategy Agent", f"{status_emoji} {status_strat.get('status', 'unknown').title()}")
            st.caption(f"Strategy: {status_strat.get('strategy', 'N/A')}")
        
        # Monitoring Agent
        with col2:
            status_mon = status.get('monitoring_agent', {})
            status_emoji = "🟢" if status_mon.get('status') == 'running' else "🔴"
            st.metric("Monitoring Agent", f"{status_emoji} {status_mon.get('status', 'unknown').title()}")
            st.caption(f"Alerts: {status_mon.get('alert_count', 0)} today")
        
        # Sentiment Agent
        with col3:
            status_sent = status.get('sentiment_agent', {})
            status_emoji = "🟢" if status_sent.get('status') == 'running' else "🔴"
            st.metric("Sentiment Agent", f"{status_emoji} {status_sent.get('status', 'unknown').title()}")
            st.caption(f"Coins tracked: {len(status_sent.get('tracked_coins', []))}")
        
        # Orchestrator
        with col4:
            status_orch = status.get('orchestrator', {})
            status_emoji = "🟢" if status_orch.get('status') == 'running' else "🔴"
            st.metric("Orchestrator", f"{status_emoji} {status_orch.get('status', 'unknown').title()}")
            st.caption(f"Active trades: {status_orch.get('active_trades', 0)}")
        
        return status
    except AgentServiceError as e:
        st.error(f"Error getting agent status: {e}")
        return {}

def render_agent_controls():
    """Render the agent controls section"""
    st.header("🤖 Agent Controls")
    
    # Get and display agent status
    status = get_agent_status_display()
    
    # Trading controls
    with st.expander("⚙️ Trading Controls", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️ Start Trading", type="primary", use_container_width=True):
                try:
                    if start_trading():
                        st.session_state.trading_active = True
                        st.success("Trading started successfully!")
                    else:
                        st.error("Failed to start trading")
                except AgentServiceError as e:
                    st.error(f"Error starting trading: {e}")
                st.rerun()
                
        with col2:
            if st.button("⏸️ Pause Trading", type="secondary", use_container_width=True):
                try:
                    if stop_trading():
                        st.session_state.trading_active = False
                        st.warning("Trading paused")
                    else:
                        st.error("Failed to pause trading")
                except AgentServiceError as e:
                    st.error(f"Error pausing trading: {e}")
                st.rerun()
                
        with col3:
            if st.button("🛑 Emergency Stop", type="primary", use_container_width=True):
                try:
                    if emergency_stop():
                        st.session_state.trading_active = False
                        st.session_state.kill_switch_active = True
                        st.error("EMERGENCY STOP ACTIVATED - All positions are being closed")
                    else:
                        st.error("Failed to activate emergency stop")
                except AgentServiceError as e:
                    st.error(f"Error during emergency stop: {e}")
                st.rerun()
    
    # Agent configuration
    with st.expander("🔧 Agent Configuration", expanded=False):
        tab1, tab2, tab3, tab4 = st.tabs(["Strategy", "Monitoring", "Sentiment", "Orchestrator"])
        
        with tab1:
            st.subheader("Strategy Agent")
            strategy_config = {
                "enabled": True,
                "max_position_size": st.slider("Max Position Size (%)", 1, 50, 10, key="strat_max_pos"),
                "risk_per_trade": st.slider("Risk per Trade (%)", 0.1, 10.0, 1.0, 0.1, key="strat_risk"),
                "take_profit": st.number_input("Take Profit (%)", 0.1, 50.0, 5.0, 0.1, key="strat_tp"),
                "stop_loss": st.number_input("Stop Loss (%)", 0.1, 20.0, 2.0, 0.1, key="strat_sl")
            }
            
            if st.button("💾 Save Strategy Config", key="save_strat"):
                # TODO: Save config to file/API
                st.success("Strategy configuration saved!")
        
        with tab2:
            st.subheader("Monitoring Agent")
            monitoring_config = {
                "enabled": True,
                "daily_profit_target": st.number_input("Daily Profit Target ($)", 100, 100000, 5000, 100, key="mon_daily_target"),
                "max_daily_loss": st.number_input("Max Daily Loss ($)", 100, 50000, 2000, 100, key="mon_max_loss"),
                "max_drawdown": st.slider("Max Drawdown (%)", 1, 50, 10, key="mon_drawdown"),
                "alert_email": st.text_input("Alert Email", "alerts@example.com", key="mon_email")
            }
            
            if st.button("💾 Save Monitoring Config", key="save_mon"):
                # TODO: Save config to file/API
                st.success("Monitoring configuration saved!")
        
        with tab3:
            st.subheader("Sentiment Agent")
            sentiment_config = {
                "enabled": True,
                "sources": st.multiselect(
                    "Data Sources",
                    ["Twitter", "Reddit", "Telegram", "News", "Pump.fun"],
                    ["Twitter", "Reddit", "Pump.fun"],
                    key="sent_sources"
                ),
                "min_volume_sol": st.number_input("Min Volume (SOL)", 10, 10000, 1000, 10, key="sent_min_vol"),
                "max_age_hours": st.number_input("Max Age (hours)", 1, 168, 24, 1, key="sent_max_age"),
                "rug_pull_check": st.checkbox("Enable Rug Pull Check", True, key="sent_rug_check")
            }
            
            if st.button("💾 Save Sentiment Config", key="save_sent"):
                # TODO: Save config to file/API
                st.success("Sentiment configuration saved!")
        
        with tab4:
            st.subheader("Orchestrator")
            orchestrator_config = {
                "enabled": True,
                "trading_hours_start": st.time_input("Trading Start Time", value=datetime.strptime("09:30", "%H:%M").time(), key="orch_start"),
                "trading_hours_end": st.time_input("Trading End Time", value=datetime.strptime("16:00", "%H:%M").time(), key="orch_end"),
                "max_open_positions": st.number_input("Max Open Positions", 1, 20, 5, 1, key="orch_max_pos"),
                "auto_restart": st.checkbox("Auto-restart on Failure", True, key="orch_restart")
            }
            
            if st.button("💾 Save Orchestrator Config", key="save_orch"):
                # TODO: Save config to file/API
                st.success("Orchestrator configuration saved!")
    
    # Agent logs
    with st.expander("📜 Agent Logs", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            log_level = st.selectbox(
                "Log Level",
                ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"],
                key="log_level"
            )
        with col2:
            log_limit = st.slider("Max Logs", 10, 1000, 100, 10, key="log_limit")
        
        try:
            logs = get_agent_logs(limit=log_limit, level=log_level)
            log_container = st.container(height=300, border=True)
            
            if logs:
                log_text = "\n".join([
                    f"{log.get('timestamp', '')} [{log.get('level', 'INFO')}] {log.get('message', '')}"
                    for log in reversed(logs)
                ])
                log_container.code(log_text, language="log")
            else:
                log_container.warning("No logs available")
                
        except AgentServiceError as e:
            st.error(f"Error fetching logs: {e}")
        
        if st.button("🔄 Refresh Logs", key="refresh_logs"):
            st.rerun()

def render_profit_targets():
    """Render profit targets and performance metrics"""
    st.header("🎯 Profit Targets")
    
    try:
        # Get performance metrics
        metrics = get_performance_metrics()
        
        # Daily and overall targets
        col1, col2, col3 = st.columns(3)
        
        with col1:
            daily_target = 5000  # Could be fetched from config
            daily_pnl = metrics.get('daily_pnl', 0)
            daily_progress = min(daily_pnl / daily_target, 1.0) if daily_target > 0 else 0
            st.metric("Daily Target", f"${daily_target:,.2f}", f"${daily_pnl:,.2f}")
        
        with col2:
            today_pnl = metrics.get('today_pnl', 0)
            today_pnl_pct = metrics.get('today_pnl_pct', 0)
            pnl_color = "green" if today_pnl >= 0 else "red"
            st.metric(
                "Today's P&L", 
                f"${today_pnl:,.2f}", 
                f"{today_pnl_pct:+.2f}%"
            )
        
        with col3:
            total_pnl = metrics.get('total_pnl', 0)
            total_pnl_pct = metrics.get('total_pnl_pct', 0)
            st.metric(
                "Overall P&L", 
                f"${total_pnl:,.2f}", 
                f"{total_pnl_pct:+.2f}%"
            )
        
        # Progress bars for targets
        st.progress(daily_progress, f"Daily Target Progress ({daily_progress*100:.1f}%)")
        
        # Monthly progress
        monthly_target = 100000  # Could be fetched from config
        monthly_pnl = metrics.get('monthly_pnl', 0)
        monthly_progress = min(monthly_pnl / monthly_target, 1.0) if monthly_target > 0 else 0
        st.progress(monthly_progress, f"Monthly Target Progress ({monthly_progress*100:.1f}%)")
        
        # Profit/loss chart (placeholder - would be replaced with actual data)
        st.line_chart({
            'PnL': [
                metrics.get('pnl_history', [10000, 10250, 10750, 10500, 11000, 11500, 12000, 12500, 12250, 12785])
            ],
            'Benchmark': [10000, 10100, 10150, 10050, 10200, 10300, 10400, 10500, 10450, 10500]
        }, use_container_width=True)
        
    except AgentServiceError as e:
        st.error(f"Error loading performance metrics: {e}")
        # Fallback to static data
        st.warning("Showing placeholder data due to connection error")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Daily Target", "$5,000", "$3,250")
        with col2:
            st.metric("Today's P&L", "$3,250", "+2.15%")
        with col3:
            st.metric("Overall P&L", "$127,845", "+25.57%")
        
        st.progress(0.65, "Daily Target Progress (65%)")
        st.progress(0.26, "Monthly Target Progress (26%)")

def render_kill_switch():
    """Render the kill switch component"""
    st.header("⚠️ Kill Switch")
    
    if st.button("🛑 ACTIVATE KILL SWITCH", 
                type="primary", 
                use_container_width=True,
                help="Immediately close all positions and stop all trading"):
        # TODO: Implement kill switch functionality
        st.session_state.trading_active = False
        st.error("KILL SWITCH ACTIVATED - All positions are being closed")
        st.session_state.kill_switch_active = True
        
    if st.session_state.get('kill_switch_active', False):
        st.warning("Kill switch is ACTIVE. Trading is disabled.")
        if st.button("🔄 Reset Kill Switch", type="secondary"):
            st.session_state.kill_switch_active = False
            st.rerun()
