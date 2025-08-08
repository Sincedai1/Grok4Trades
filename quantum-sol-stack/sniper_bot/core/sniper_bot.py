import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from web3 import Web3
from web3.types import TxReceipt, ChecksumAddress
from loguru import logger

from .web3_manager import Web3Manager
from .config import config

class SniperBot:
    def __init__(self):
        """Initialize the SniperBot with Web3 connection and configuration."""
        self.web3_manager = Web3Manager()
        self.w3 = self.web3_manager.w3
        self.running = False
        self.tracked_tokens: Dict[str, Dict] = {}
        self.last_block_processed = self.w3.eth.block_number
        
        # Validate configuration
        config.validate()
        
        logger.info("SniperBot initialized")
        logger.info(f"Target profit: {config.TARGET_PROFIT_PERCENT}%")
        logger.info(f"Stop loss: {config.STOP_LOSS_PERCENT}%")
        logger.info(f"Max slippage: {config.MAX_SLIPPAGE}%")
        logger.info(f"Min liquidity: {config.MIN_LIQUIDITY_ETH} ETH")
    
    async def start(self):
        """Start the main event loop for the sniper bot."""
        self.running = True
        logger.info("Starting SniperBot...")
        
        try:
            # Start monitoring for new pairs
            await self.monitor_new_pairs()
            
            # Start the main event loop
            while self.running:
                try:
                    # Process tracked tokens
                    await self.process_tracked_tokens()
                    
                    # Small delay to prevent high CPU usage
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(5)  # Wait before retrying
                    
        except KeyboardInterrupt:
            logger.info("Shutting down SniperBot...")
        except Exception as e:
            logger.critical(f"Fatal error: {e}")
        finally:
            self.running = False
