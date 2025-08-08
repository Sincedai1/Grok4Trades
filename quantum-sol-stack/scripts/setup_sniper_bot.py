#!/usr/bin/env python3
"""
Uniswap V2 Sniper Bot Setup Script

This script provides a basic setup for monitoring Uniswap V2 for new token pairs
and executing trades based on configurable parameters.
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from web3 import Web3
from web3.exceptions import ContractLogicError
from loguru import logger

# Configure logging
logger.add("logs/setup.log", rotation="10 MB", level="INFO")

class SniperBotSetup:
    """Handles the setup and configuration of the Uniswap V2 Sniper Bot."""
    
    def __init__(self):
        """Initialize the setup with configuration from environment variables."""
        self.config = self._load_config()
        self.web3 = self._connect_to_web3()
        self._verify_wallet()
    
    def _load_config(self):
        """Load and validate environment variables for the bot."""
        load_dotenv()
        
        required_vars = [
            "RPC_URL_GOERLI", "PRIVATE_KEY", "WALLET_ADDRESS",
            "FACTORY_ADDRESS", "ROUTER_ADDRESS", "WETH_ADDRESS"
        ]
        
        config = {}
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                raise ValueError(f"Missing required environment variable: {var}")
            config[var] = value
            
        # Optional variables with defaults
        config["TARGET_PROFIT_PERCENT"] = float(os.getenv("TARGET_PROFIT_PERCENT", 5.0))
        config["STOP_LOSS_PERCENT"] = float(os.getenv("STOP_LOSS_PERCENT", 2.0))
        config["MAX_SLIPPAGE"] = float(os.getenv("MAX_SLIPPAGE", 1.0))
        config["MAX_GAS_PRICE_GWEI"] = float(os.getenv("MAX_GAS_PRICE_GWEI", 100.0))
        config["MIN_LIQUIDITY_ETH"] = float(os.getenv("MIN_LIQUIDITY_ETH", 0.1))
        
        return config
    
    def _connect_to_web3(self):
        """Establish connection to the Ethereum network."""
        web3 = Web3(Web3.HTTPProvider(self.config["RPC_URL_GOERLI"]))
        if not web3.is_connected():
            raise ConnectionError("Failed to connect to the Ethereum network")
        logger.info(f"Connected to {self.config['RPC_URL_GOERLI']}")
        return web3
    
    def _verify_wallet(self):
        """Verify that the provided private key matches the wallet address."""
        account = self.web3.eth.account.from_key(self.config["PRIVATE_KEY"])
        if account.address.lower() != self.config["WALLET_ADDRESS"].lower():
            raise ValueError("Wallet address does not match private key")
        logger.info(f"Using wallet: {account.address}")
        self.account = account
    
    def check_contracts(self):
        """Verify that all required contracts are accessible."""
        contracts = {
            "Factory": self.config["FACTORY_ADDRESS"],
            "Router": self.config["ROUTER_ADDRESS"],
            "WETH": self.config["WETH_ADDRESS"]
        }
        
        for name, address in contracts.items():
            try:
                code = self.web3.eth.get_code(address)
                if not code or code.hex() == '0x':
                    logger.warning(f"No code found at {name} address: {address}")
                else:
                    logger.info(f"Successfully connected to {name} contract")
            except Exception as e:
                logger.error(f"Error connecting to {name} contract: {str(e)}")
    
    def get_eth_balance(self):
        """Get the ETH balance of the configured wallet."""
        balance_wei = self.web3.eth.get_balance(self.account.address)
        balance_eth = self.web3.from_wei(balance_wei, 'ether')
        logger.info(f"Wallet balance: {balance_eth:.6f} ETH")
        return balance_eth
    
    def monitor_new_pairs(self):
        """Monitor for new token pair creation events."""
        factory_abi = [{
            "anonymous": False,
            "inputs": [
                {"indexed": True, "name": "token0", "type": "address"},
                {"indexed": True, "name": "token1", "type": "address"},
                {"indexed": False, "name": "pair", "type": "address"},
                {"indexed": False, "name": "", "type": "uint256"}
            ],
            "name": "PairCreated",
            "type": "event"
        }]
        
        factory = self.web3.eth.contract(
            address=self.config["FACTORY_ADDRESS"],
            abi=factory_abi
        )
        
        logger.info("Starting to monitor for new token pairs...")
        # This is a simplified example - in a real implementation, you would
        # set up event filters and handle them asynchronously
        return factory

def create_env_file():
    """Create a .env.example file if it doesn't exist."""
    env_example = """# Required Configuration
RPC_URL_GOERLI=https://goerli.infura.io/v3/YOUR_INFURA_PROJECT_ID
PRIVATE_KEY=your_goerli_private_key
WALLET_ADDRESS=0xYourGoerliAddress

# Contract Addresses (Uniswap V2)
FACTORY_ADDRESS=0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f
ROUTER_ADDRESS=0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D
WETH_ADDRESS=0xB4FBF271143F4FBf7B91A5ded31805e42b2208d6

# Trading Parameters
TARGET_PROFIT_PERCENT=5.0
STOP_LOSS_PERCENT=2.0
MAX_SLIPPAGE=1.0
MAX_GAS_PRICE_GWEI=100
MIN_LIQUIDITY_ETH=0.1
"""
    env_path = Path(".env.example")
    if not env_path.exists():
        with open(env_path, "w") as f:
            f.write(env_example)
        logger.info("Created .env.example file. Please copy it to .env and update with your values.")
    else:
        logger.info(".env.example already exists.")

def main():
    """Main entry point for the setup script."""
    try:
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)
        
        logger.info("Starting Uniswap V2 Sniper Bot setup...")
        
        # Create .env.example if it doesn't exist
        create_env_file()
        
        # Check if .env exists
        if not Path(".env").exists():
            logger.error("Error: .env file not found. Please create it from .env.example")
            return
        
        # Initialize setup
        setup = SniperBotSetup()
        
        # Check wallet balance
        setup.get_eth_balance()
        
        # Verify contract connections
        setup.check_contracts()
        
        logger.info("Setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
