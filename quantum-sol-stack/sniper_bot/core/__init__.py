"""
Core functionality for the Uniswap V2 Sniper Bot.

This package contains the main components for the bot including:
- Web3 connection and contract interaction
- Event monitoring
- Trading strategies
- Risk management
""__

__version__ = "0.1.0"

# Import key components to make them available at the package level
from .web3_manager import Web3Manager
from .sniper_bot import SniperBot
from .config import Config

__all__ = [
    'Web3Manager',
    'SniperBot',
    'Config',
]
