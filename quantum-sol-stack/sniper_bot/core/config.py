"""
Configuration module for the Uniswap V2 Sniper Bot.

This module handles loading and validating configuration from environment variables.
"""
import os
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    """Configuration class for the Sniper Bot."""
    # Required configuration
    rpc_url: str
    private_key: str
    wallet_address: str
    
    # Contract addresses
    factory_address: str
    router_address: str
    weth_address: str
    
    # Trading parameters
    target_profit_percent: Decimal = Decimal('5.0')
    stop_loss_percent: Decimal = Decimal('2.0')
    max_slippage: Decimal = Decimal('1.0')
    max_gas_price_gwei: Decimal = Decimal('100.0')
    min_liquidity_eth: Decimal = Decimal('0.1')
    
    # Logging
    log_level: str = 'INFO'
    log_file: str = 'logs/sniper_bot.log'
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create a Config instance from environment variables."""
        # Required variables
        rpc_url = os.getenv('RPC_URL_GOERLI')
        private_key = os.getenv('PRIVATE_KEY')
        wallet_address = os.getenv('WALLET_ADDRESS')
        
        # Contract addresses
        factory_address = os.getenv('FACTORY_ADDRESS')
        router_address = os.getenv('ROUTER_ADDRESS')
        weth_address = os.getenv('WETH_ADDRESS')
        
        # Trading parameters with defaults
        target_profit = Decimal(os.getenv('TARGET_PROFIT_PERCENT', '5.0'))
        stop_loss = Decimal(os.getenv('STOP_LOSS_PERCENT', '2.0'))
        max_slippage = Decimal(os.getenv('MAX_SLIPPAGE', '1.0'))
        max_gas = Decimal(os.getenv('MAX_GAS_PRICE_GWEI', '100.0'))
        min_liquidity = Decimal(os.getenv('MIN_LIQUIDITY_ETH', '0.1'))
        
        # Logging
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        log_file = os.getenv('LOG_FILE', 'logs/sniper_bot.log')
        
        # Validate required variables
        if not all([rpc_url, private_key, wallet_address, factory_address, 
                   router_address, weth_address]):
            missing = []
            if not rpc_url:
                missing.append('RPC_URL_GOERLI')
            if not private_key:
                missing.append('PRIVATE_KEY')
            if not wallet_address:
                missing.append('WALLET_ADDRESS')
            if not factory_address:
                missing.append('FACTORY_ADDRESS')
            if not router_address:
                missing.append('ROUTER_ADDRESS')
            if not weth_address:
                missing.append('WETH_ADDRESS')
                
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        return cls(
            rpc_url=rpc_url,
            private_key=private_key,
            wallet_address=wallet_address,
            factory_address=factory_address,
            router_address=router_address,
            weth_address=weth_address,
            target_profit_percent=target_profit,
            stop_loss_percent=stop_loss,
            max_slippage=max_slippage,
            max_gas_price_gwei=max_gas,
            min_liquidity_eth=min_liquidity,
            log_level=log_level,
            log_file=log_file
        )
    
    def validate(self) -> bool:
        """Validate the configuration values."""
        # Add any additional validation logic here
        if not self.wallet_address.startswith('0x'):
            raise ValueError("Wallet address must start with '0x'")
        
        if not self.private_key.startswith('0x'):
            raise ValueError("Private key must start with '0x'")
            
        if not 0 < self.target_profit_percent <= 100:
            raise ValueError("Target profit percent must be between 0 and 100")
            
        if not 0 < self.stop_loss_percent <= 100:
            raise ValueError("Stop loss percent must be between 0 and 100")
            
        if not 0 < self.max_slippage <= 100:
            raise ValueError("Max slippage must be between 0 and 100")
            
        if self.max_gas_price_gwei <= 0:
            raise ValueError("Max gas price must be greater than 0")
            
        if self.min_liquidity_eth <= 0:
            raise ValueError("Minimum liquidity must be greater than 0")
            
        return True

# Global config instance
config = Config.from_env()
