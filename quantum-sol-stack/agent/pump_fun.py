""
Pump.fun integration for the QuantumSol trading system.

This module provides tools to interact with Pump.fun, a platform for launching and trading meme coins.
It includes functionality to scan for new launches, analyze token metrics, and execute trades.
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any

import aiohttp
from pydantic import BaseModel, Field, validator
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.transaction import Transaction
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solders.system_program import TransferParams, transfer
from solders.transaction import Transaction

from .guardrails_utils import RiskAssessment, RiskLevel
from .kill_switch import KillSwitchReason, kill_switch

logger = logging.getLogger(__name__)

class PumpFunTokenStatus(str, Enum):
    """Status of a Pump.fun token."""
    ACTIVE = "ACTIVE"  # Token is actively trading
    COMPLETED = "COMPLETED"  # Token has completed its pump phase
    FAILED = "FAILED"  # Token failed to launch
    UPCOMING = "UPCOMING"  # Token is scheduled for launch

class PumpFunToken(BaseModel):
    """Represents a token on Pump.fun."""
    id: str
    name: str
    symbol: str
    mint_address: str
    creator: str
    created_at: datetime
    status: PumpFunTokenStatus
    
    # Token metrics
    price_sol: float = 0.0
    price_usd: float = 0.0
    market_cap: float = 0.0
    liquidity: float = 0.0
    volume_24h: float = 0.0
    holders: int = 0
    
    # Social metrics
    telegram_members: Optional[int] = None
    twitter_followers: Optional[int] = None
    
    # Risk assessment
    risk_score: float = 0.0
    risk_assessment: Optional[RiskAssessment] = None
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
    
    @validator('mint_address', 'creator', pre=True)
    def validate_solana_address(cls, v):
        if not v:
            return v
        try:
            # This will raise if the address is invalid
            Pubkey.from_string(str(v))
            return str(v)
        except Exception as e:
            raise ValueError(f"Invalid Solana address: {v}") from e

class PumpFunAPI:
    """Client for interacting with the Pump.fun API."""
    
    BASE_URL = "https://api.pump.fun"
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Pump.fun API client.
        
        Args:
            api_key: Optional API key for authenticated endpoints.
        """
        self.api_key = api_key
        self.session = None
        self._last_request = 0
        self._rate_limit_delay = 1.0  # seconds between requests
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def connect(self):
        """Initialize the HTTP session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={
                    "User-Agent": "QuantumSol/1.0",
                    "Accept": "application/json",
                    **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})
                },
                timeout=aiohttp.ClientTimeout(total=30)
            )
    
    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request to the Pump.fun API."""
        await self.connect()
        
        # Rate limiting
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - elapsed)
        
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                response.raise_for_status()
                
                # Update last request time
                self._last_request = time.time()
                
                # Handle empty responses
                content = await response.text()
                if not content.strip():
                    return {}
                    
                return await response.json()
                
        except aiohttp.ClientError as e:
            logger.error(f"API request failed: {e}")
            raise
    
    async def get_active_tokens(self, limit: int = 100) -> List[PumpFunToken]:
        """Get a list of active tokens on Pump.fun."""
        data = await self._request("GET", f"/tokens/active?limit={limit}")
        return [PumpFunToken(**token) for token in data.get("tokens", [])]
    
    async def get_token_info(self, mint_address: str) -> Optional[PumpFunToken]:
        """Get detailed information about a specific token."""
        try:
            data = await self._request("GET", f"/tokens/{mint_address}")
            return PumpFunToken(**data) if data else None
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                return None
            raise
    
    async def get_token_metrics(self, mint_address: str) -> Dict[str, Any]:
        """Get metrics for a specific token."""
        return await self._request("GET", f"/tokens/{mint_address}/metrics")
    
    async def get_token_trades(
        self,
        mint_address: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get recent trades for a token."""
        return await self._request(
            "GET",
            f"/tokens/{mint_address}/trades",
            params={"limit": limit, "offset": offset}
        )
    
    async def get_upcoming_launches(self, limit: int = 50) -> List[PumpFunToken]:
        """Get upcoming token launches."""
        data = await self._request("GET", f"/launches/upcoming?limit={limit}")
        return [PumpFunToken(**token) for token in data.get("tokens", [])]
    
    async def search_tokens(
        self,
        query: str,
        limit: int = 20
    ) -> List[PumpFunToken]:
        """Search for tokens by name or symbol."""
        data = await self._request(
            "GET",
            "/tokens/search",
            params={"q": query, "limit": limit}
        )
        return [PumpFunToken(**token) for token in data.get("tokens", [])]

class PumpFunTrader:
    """Handles trading operations on Pump.fun."""
    
    def __init__(
        self,
        private_key: str,
        rpc_url: str = "https://api.mainnet-beta.solana.com",
        api_key: Optional[str] = None
    ):
        """Initialize the Pump.fun trader.
        
        Args:
            private_key: Base58-encoded private key for the trading wallet.
            rpc_url: Solana RPC endpoint URL.
            api_key: Optional Pump.fun API key.
        """
        self.private_key = private_key
        self.rpc_url = rpc_url
        self.api = PumpFunAPI(api_key)
        self.client = AsyncClient(rpc_url, commitment=Confirmed)
        self.wallet = None  # Will be initialized in connect()
        self._connected = False
    
    async def connect(self):
        """Initialize the connection to Solana and Pump.fun."""
        if self._connected:
            return
        
        try:
            # Initialize wallet from private key
            from solders.keypair import Keypair
            from solders.system_program import TransferParams, transfer
            
            keypair = Keypair.from_base58_string(self.private_key)
            self.wallet = keypair
            
            # Connect to RPC
            await self.client.is_connected()
            
            # Connect to Pump.fun API
            await self.api.connect()
            
            self._connected = True
            logger.info(f"Connected to Solana and Pump.fun. Wallet: {self.wallet.pubkey()}")
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            await self.close()
            raise
    
    async def close(self):
        """Close all connections."""
        if hasattr(self, 'client') and self.client:
            await self.client.close()
        if hasattr(self, 'api') and self.api:
            await self.api.close()
        self._connected = False
    
    async def get_balance(self) -> float:
        """Get the SOL balance of the wallet."""
        await self.connect()
        balance = await self.client.get_balance(self.wallet.pubkey())
        return balance.value / 1e9  # Convert lamports to SOL
    
    async def get_token_balance(self, mint_address: str) -> float:
        """Get the balance of a specific token."""
        await self.connect()
        from solders.token.associated import get_associated_token_address
        
        # Get the associated token account
        token_account = get_associated_token_address(
            owner=self.wallet.pubkey(),
            mint=Pubkey.from_string(mint_address)
        )
        
        # Get token balance
        try:
            balance = await self.client.get_token_account_balance(token_account)
            return float(balance.value.amount) / (10 ** balance.value.decimals)
        except Exception as e:
            # Token account might not exist yet
            if "could not find account" in str(e).lower():
                return 0.0
            raise
    
    async def buy_token(
        self,
        mint_address: str,
        amount_sol: float,
        slippage: float = 5.0,
        max_retries: int = 3
    ) -> Optional[str]:
        """Buy tokens on Pump.fun.
        
        Args:
            mint_address: The token's mint address.
            amount_sol: Amount of SOL to spend.
            slippage: Maximum allowed slippage percentage.
            max_retries: Maximum number of retry attempts.
            
        Returns:
            Transaction signature if successful, None otherwise.
        """
        await self.connect()
        
        # Check if kill switch is active
        if kill_switch.is_active:
            logger.warning("Buy operation blocked: Kill switch is active")
            return None
        
        # Validate amount
        balance = await self.get_balance()
        if amount_sol > balance:
            logger.error(f"Insufficient balance: {balance} SOL available, {amount_sol} SOL requested")
            return None
        
        # Get token info
        token = await self.api.get_token_info(mint_address)
        if not token:
            logger.error(f"Token not found: {mint_address}")
            return None
        
        # Check token status
        if token.status != PumpFunTokenStatus.ACTIVE:
            logger.error(f"Token is not active for trading. Status: {token.status}")
            return None
        
        # Check risk assessment
        if token.risk_assessment and not token.risk_assessment.is_acceptable():
            logger.warning(
                f"High risk token detected: {token.name} ({token.symbol}). "
                f"Risk level: {token.risk_assessment.risk_level}"
            )
            # Potentially trigger kill switch based on risk level
            if token.risk_assessment.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.EXTREME]:
                await kill_switch.activate(
                    reason=KillSwitchReason.RISK_LIMIT,
                    description=f"High risk token detected: {token.name} ({token.symbol})"
                )
                return None
        
        # Simulate transaction (dry-run)
        try:
            # This is a placeholder for the actual transaction simulation
            # In a real implementation, you would construct and simulate the transaction
            # using the Pump.fun program ID and accounts
            
            # For demonstration, we'll just log the buy attempt
            logger.info(
                f"[SIMULATED] Buying {amount_sol} SOL worth of {token.name} ({token.symbol}) at "
                f"{token.price_sol} SOL per token (~{token.price_usd} USD)"
            )
            
            # In a real implementation, you would:
            # 1. Get the current price and calculate the expected amount of tokens
            # 2. Build the transaction with the appropriate instructions
            # 3. Simulate the transaction to check for errors
            # 4. If simulation passes, sign and send the transaction
            # 5. Return the transaction signature
            
            # For now, return a dummy signature
            return "simulated_tx_signature"
            
        except Exception as e:
            logger.error(f"Failed to execute buy order: {e}", exc_info=True)
            return None
    
    async def sell_token(
        self,
        mint_address: str,
        amount: Optional[float] = None,
        percentage: float = 100.0,
        slippage: float = 5.0,
        max_retries: int = 3
    ) -> Optional[str]:
        """Sell tokens on Pump.fun.
        
        Args:
            mint_address: The token's mint address.
            amount: Amount of tokens to sell. If None, sells all.
            percentage: Percentage of balance to sell (ignored if amount is specified).
            slippage: Maximum allowed slippage percentage.
            max_retries: Maximum number of retry attempts.
            
        Returns:
            Transaction signature if successful, None otherwise.
        """
        await self.connect()
        
        # Check if kill switch is active
        if kill_switch.is_active:
            logger.warning("Sell operation blocked: Kill switch is active")
            return None
        
        # Get token balance
        balance = await self.get_token_balance(mint_address)
        if balance <= 0:
            logger.error(f"No balance for token: {mint_address}")
            return None
        
        # Calculate amount to sell
        if amount is None:
            amount = balance * (percentage / 100.0)
        
        if amount > balance:
            logger.warning(
                f"Requested amount ({amount}) exceeds balance ({balance}). "
                f"Selling entire balance."
            )
            amount = balance
        
        # Get token info
        token = await self.api.get_token_info(mint_address)
        if not token:
            logger.error(f"Token not found: {mint_address}")
            return None
        
        # Simulate transaction (dry-run)
        try:
            # This is a placeholder for the actual transaction simulation
            # In a real implementation, you would construct and simulate the transaction
            
            # For demonstration, we'll just log the sell attempt
            logger.info(
                f"[SIMULATED] Selling {amount} {token.symbol} at "
                f"{token.price_sol} SOL per token (~{token.price_usd} USD)"
            )
            
            # In a real implementation, you would:
            # 1. Get the current price and calculate the expected SOL amount
            # 2. Build the transaction with the appropriate instructions
            # 3. Simulate the transaction to check for errors
            # 4. If simulation passes, sign and send the transaction
            # 5. Return the transaction signature
            
            # For now, return a dummy signature
            return "simulated_tx_signature"
            
        except Exception as e:
            logger.error(f"Failed to execute sell order: {e}", exc_info=True)
            return None

# Global instance
pump_fun_trader = None

def get_pump_fun_trader(
    private_key: Optional[str] = None,
    rpc_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> PumpFunTrader:
    """Get or create a global PumpFunTrader instance."""
    global pump_fun_trader
    
    if pump_fun_trader is None and private_key is not None:
        pump_fun_trader = PumpFunTrader(
            private_key=private_key,
            rpc_url=rpc_url or "https://api.mainnet-beta.solana.com",
            api_key=api_key
        )
    
    return pump_fun_trader
