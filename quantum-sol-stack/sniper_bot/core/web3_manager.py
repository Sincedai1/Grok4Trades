import json
import time
from typing import Dict, Optional, Tuple, Any
from web3 import Web3, HTTPProvider
from web3.contract import Contract
from web3.types import TxReceipt, Wei
from eth_account import Account
from eth_typing import ChecksumAddress
from loguru import logger

from .config import config

# Enable unverified transaction signing
Account.enable_unaudited_hdwallet_features()

class Web3Manager:
    def __init__(self):
        # Initialize Web3 connection
        self.w3 = Web3(HTTPProvider(config.RPC_URL))
        
        # Set up the account
        self.account = Account.from_key(config.PRIVATE_KEY)
        if self.account.address.lower() != config.WALLET_ADDRESS.lower():
            raise ValueError("Wallet address does not match private key")
            
        # Load contract ABIs
        self.abis = self._load_abis()
        
        # Initialize contracts
        self.router = self._init_contract(
            config.ROUTER_ADDRESS,
            self.abis['IUniswapV2Router02']
        )
        self.factory = self._init_contract(
            config.FACTORY_ADDRESS,
            self.abis['IUniswapV2Factory']
        )
        self.weth = self._init_contract(
            config.WETH_ADDRESS,
            self.abis['IWETH']
        )
        
        logger.info(f"Connected to {self.w3.provider.endpoint_uri}")
        logger.info(f"Using wallet: {self.account.address}")
        
    def _load_abis(self) -> Dict[str, Any]:
        """Load contract ABIs from the abis directory."""
        abis_dir = Path(__file__).parent.parent / 'abis'
        abis = {}
        
        # Load common ABIs
        for abi_file in abis_dir.glob('*.json'):
            with open(abi_file, 'r') as f:
                abis[abi_file.stem] = json.load(f)
                
        return abis
        
    def _init_contract(self, address: str, abi: list) -> Contract:
        """Initialize a contract instance."""
        if not self.w3.is_address(address):
            raise ValueError(f"Invalid contract address: {address}")
            
        checksum_address = self.w3.to_checksum_address(address)
        return self.w3.eth.contract(address=checksum_address, abi=abi)
    
    async def get_gas_price(self) -> Wei:
        """Get current gas price with a cap from config."""
        try:
            current_gas = self.w3.eth.gas_price
            max_gas = self.w3.to_wei(config.MAX_GAS_PRICE_GWEI, 'gwei')
            return min(current_gas, max_gas)
        except Exception as e:
            logger.error(f"Error getting gas price: {e}")
            return self.w3.to_wei(config.MAX_GAS_PRICE_GWEI, 'gwei')
    
    async def get_token_balance(self, token_address: str) -> int:
        """Get token balance for the given token address."""
        if token_address.lower() == 'eth':
            return self.w3.eth.get_balance(self.account.address)
            
        token = self._init_contract(token_address, self.abis['IERC20'])
        return token.functions.balanceOf(self.account.address).call()
    
    async def get_token_info(self, token_address: str) -> Tuple[str, str, int]:
        """Get token name, symbol, and decimals."""
        token = self._init_contract(token_address, self.abis['IERC20'])
        
        name = token.functions.name().call()
        symbol = token.functions.symbol().call()
        decimals = token.functions.decimals().call()
        
        return name, symbol, decimals
    
    async def send_transaction(self, transaction: Dict) -> Optional[TxReceipt]:
        """Send a signed transaction and wait for receipt."""
        try:
            # Set transaction parameters
            nonce = self.w3.eth.get_transaction_count(self.account.address, 'pending')
            transaction.update({
                'nonce': nonce,
                'from': self.account.address,
                'gasPrice': await self.get_gas_price(),
                'chainId': config.CHAIN_ID
            })
            
            # Estimate gas if not provided
            if 'gas' not in transaction:
                try:
                    gas_estimate = self.w3.eth.estimate_gas(transaction)
                    transaction['gas'] = int(gas_estimate * 1.2)  # Add 20% buffer
                except Exception as e:
                    logger.warning(f"Gas estimation failed: {e}, using default")
                    transaction['gas'] = 300000  # Default gas limit
            
            # Sign and send transaction
            signed_txn = self.w3.eth.account.sign_transaction(
                transaction, private_key=config.PRIVATE_KEY
            )
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for transaction receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if receipt.status == 1:
                logger.info(f"Transaction successful: {self.w3.to_hex(tx_hash)}")
            else:
                logger.error(f"Transaction failed: {self.w3.to_hex(tx_hash)}")
                
            return receipt
            
        except Exception as e:
            logger.error(f"Transaction error: {e}")
            return None
    
    async def get_liquidity_pool(self, token_address: str) -> Optional[ChecksumAddress]:
        """Get the liquidity pool address for a token/WETH pair."""
        try:
            return self.factory.functions.getPair(
                self.w3.to_checksum_address(token_address),
                self.w3.to_checksum_address(config.WETH_ADDRESS)
            ).call()
        except Exception as e:
            logger.error(f"Error getting liquidity pool: {e}")
            return None
