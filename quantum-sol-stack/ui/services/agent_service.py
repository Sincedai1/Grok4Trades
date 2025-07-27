"""
Agent Service

This module provides functions to interact with the QuantumSol agent system.
"""
import os
import requests
import json
from typing import Dict, Any, Optional, List
from loguru import logger

# Configuration
AGENT_API_URL = os.getenv("AGENT_API_URL", "http://agent:8502")
TIMEOUT = 10  # seconds

class AgentServiceError(Exception):
    """Custom exception for agent service errors"""
    pass

def get_agent_status() -> Dict[str, Any]:
    """Get the current status of all agents"""
    try:
        response = requests.get(f"{AGENT_API_URL}/status", timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting agent status: {e}")
        raise AgentServiceError(f"Failed to get agent status: {e}")

def start_trading() -> bool:
    """Start the trading system"""
    try:
        response = requests.post(
            f"{AGENT_API_URL}/trading/start",
            json={"dry_run": False},
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json().get("success", False)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error starting trading: {e}")
        raise AgentServiceError(f"Failed to start trading: {e}")

def stop_trading() -> bool:
    """Stop the trading system"""
    try:
        response = requests.post(
            f"{AGENT_API_URL}/trading/stop",
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json().get("success", False)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error stopping trading: {e}")
        raise AgentServiceError(f"Failed to stop trading: {e}")

def emergency_stop() -> bool:
    """Immediately stop all trading and close all positions"""
    try:
        response = requests.post(
            f"{AGENT_API_URL}/emergency/stop",
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json().get("success", False)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in emergency stop: {e}")
        raise AgentServiceError(f"Emergency stop failed: {e}")

def get_trades(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent trades"""
    try:
        response = requests.get(
            f"{AGENT_API_URL}/trades",
            params={"limit": limit},
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json().get("trades", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting trades: {e}")
        raise AgentServiceError(f"Failed to get trades: {e}")

def get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics"""
    try:
        response = requests.get(
            f"{AGENT_API_URL}/performance",
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise AgentServiceError(f"Failed to get performance metrics: {e}")

def update_agent_config(agent_type: str, config: Dict[str, Any]) -> bool:
    """Update agent configuration"""
    try:
        response = requests.post(
            f"{AGENT_API_URL}/config/{agent_type}",
            json=config,
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json().get("success", False)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error updating {agent_type} config: {e}")
        raise AgentServiceError(f"Failed to update {agent_type} config: {e}")

def get_agent_logs(limit: int = 100, level: str = "INFO") -> List[Dict[str, Any]]:
    """Get agent logs"""
    try:
        response = requests.get(
            f"{AGENT_API_URL}/logs",
            params={"limit": limit, "level": level},
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json().get("logs", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting logs: {e}")
        raise AgentServiceError(f"Failed to get logs: {e}")
