#!/usr/bin/env python3
"""
Sniper Bot - A high-frequency trading bot for Uniswap V2

This script initializes and starts the SniperBot to monitor and trade on Uniswap V2.
"""
import asyncio
import signal
import sys
from loguru import logger

from .core.sniper_bot import SniperBot

def handle_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info("Shutdown signal received. Stopping SniperBot...")
    sys.exit(0)

def main():
    """Main entry point for the SniperBot application."""
    try:
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)
        
        # Initialize the bot
        bot = SniperBot()
        
        # Run the bot
        logger.info("Starting SniperBot...")
        asyncio.run(bot.start())
        
    except KeyboardInterrupt:
        logger.info("SniperBot stopped by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
