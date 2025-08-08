import asyncio
import ccxt.async_support as ccxt
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import aiohttp
from solana.rpc.async_api import AsyncClient
from solana.publickey import PublicKey
from solders.pubkey import Pubkey as SoldersPubkey
from solders.rpc.requests import GetTokenSupply, GetTokenAccountsByOwner
from solders.rpc.config import RpcTokenAccountsFilter
from solana.rpc.types import TokenAccountOpts
from solana.rpc.commitment import Confirmed
import base58
import time
from loguru import logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ArbitrageOpportunity:
    """Data class representing an arbitrage opportunity"""
    base_asset: str
    quote_asset: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread: float
    spread_pct: float
    timestamp: float
    
    def to_dict(self):
        return {
            'base_asset': self.base_asset,
            'quote_asset': self.quote_asset,
            'buy_exchange': self.buy_exchange,
            'sell_exchange': self.sell_exchange,
            'buy_price': self.buy_price,
            'sell_price': self.sell_price,
            'spread': self.spread,
            'spread_pct': self.spread_pct,
            'timestamp': self.timestamp,
            'timestamp_iso': datetime.utcfromtimestamp(self.timestamp).isoformat()
        }

class SolanaTokenScanner:
    """Scans Solana blockchain for DEX liquidity pools"""
    
    def __init__(self, rpc_url: str = "https://api.mainnet-beta.solana.com"):
        self.rpc_url = rpc_url
        self.client = AsyncClient(rpc_url)
        self.raydium_program_id = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
        self.serum_program_id = "9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin"
        
    async def get_token_balance(self, wallet_address: str, token_mint: str) -> float:
        """Get token balance for a specific wallet and token mint"""
        try:
            # Convert wallet address to PublicKey
            wallet_pubkey = PublicKey(wallet_address)
            
            # Get token accounts for the wallet
            token_accounts = await self.client.get_token_accounts_by_owner(
                wallet_pubkey,
                TokenAccountOpts(
                    mint=PublicKey(token_mint),
                    program_id=PublicKey("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
                ),
                commitment=Confirmed
            )
            
            if not token_accounts.value:
                return 0.0
                
            # Get token account info
            token_account = token_accounts.value[0]
            account_info = await self.client.get_token_account_balance(
                token_account.pubkey,
                commitment=Confirmed
            )
            
            # Convert balance to human-readable format (assuming 9 decimals for most SPL tokens)
            balance = float(account_info.value.amount) / 1e9
            return balance
            
        except Exception as e:
            logger.error(f"Error getting token balance: {str(e)}")
            return 0.0
    
    async def get_raydium_pools(self, token_mint: str) -> List[Dict]:
        """Find Raydium pools for a given token mint"""
        # In a production environment, you'd query Raydium's API or on-chain data
        # This is a simplified example
        return [
            {
                'pool_address': '...',
                'base_mint': token_mint,
                'quote_mint': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',  # USDC
                'liquidity': 1000000.0,
                'price': 1.5  # Price in quote token (e.g., USDC)
            }
        ]
    
    async def get_serum_markets(self, token_mint: str) -> List[Dict]:
        """Find Serum markets for a given token mint"""
        # In a production environment, you'd query Serum's API or on-chain data
        return [
            {
                'market_address': '...',
                'base_mint': token_mint,
                'quote_mint': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',  # USDC
                'bids': [],
                'asks': [],
                'price': 1.52  # Current market price
            }
        ]

class CrossChainArbitrageDetector:
    """Detects arbitrage opportunities across CEX and DEX markets"""
    
    def __init__(self, redis_url: str = "redis://redis-cache:6379"):
        self.exchanges = {
            'binance': ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                }
            }),
            'kraken': ccxt.kraken({
                'enableRateLimit': True
            }),
            # Add more exchanges as needed
        }
        self.solana_scanner = SolanaTokenScanner()
        self.min_profit_pct = 0.5  # Minimum profit percentage to consider an arbitrage opportunity
        self.min_liquidity_usd = 10000  # Minimum liquidity required in USD
        self.redis_url = redis_url
        self.redis_client = None
        self.supported_pairs = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'SOL/USDC',
            'BTC/USDC', 'ETH/USDC', 'SOL/BTC', 'ETH/BTC'
        ]
    
    async def initialize(self):
        """Initialize connections and load markets"""
        # Initialize Redis client
        import redis
        self.redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        
        # Load markets for all exchanges
        for exchange_name, exchange in self.exchanges.items():
            try:
                await exchange.load_markets()
                logger.info(f"Loaded markets for {exchange_name}")
            except Exception as e:
                logger.error(f"Error loading markets for {exchange_name}: {str(e)}")
    
    async def close(self):
        """Close all connections"""
        for exchange in self.exchanges.values():
            await exchange.close()
        
        if hasattr(self, 'solana_scanner') and hasattr(self.solana_scanner, 'client'):
            await self.solana_scanner.client.close()
    
    async def get_cex_orderbook(self, exchange: ccxt.Exchange, symbol: str) -> Optional[Dict]:
        """Fetch order book from a CEX"""
        try:
            orderbook = await exchange.fetch_order_book(symbol)
            return {
                'bids': orderbook['bids'][:5],  # Top 5 bids
                'asks': orderbook['asks'][:5],  # Top 5 asks
                'timestamp': orderbook['timestamp']
            }
        except Exception as e:
            logger.warning(f"Error fetching order book for {symbol} from {exchange.id}: {str(e)}")
            return None
    
    async def find_arbitrage_opportunities(self) -> List[ArbitrageOpportunity]:
        """Find cross-exchange arbitrage opportunities"""
        opportunities = []
        
        for symbol in self.supported_pairs:
            # Get order books from all exchanges
            orderbooks = {}
            for exchange_name, exchange in self.exchanges.items():
                if exchange.has['fetchOrderBook'] and symbol in exchange.markets:
                    orderbook = await self.get_cex_orderbook(exchange, symbol)
                    if orderbook:
                        orderbooks[exchange_name] = orderbook
            
            # Compare order books to find arbitrage opportunities
            for buy_exchange, buy_book in orderbooks.items():
                if not buy_book['bids']:
                    continue
                    
                for sell_exchange, sell_book in orderbooks.items():
                    if buy_exchange == sell_exchange or not sell_book['asks']:
                        continue
                    
                    # Get best bid from buy exchange and best ask from sell exchange
                    best_bid = buy_book['bids'][0][0]  # (price, amount)
                    best_ask = sell_book['asks'][0][0]  # (price, amount)
                    
                    # Calculate spread
                    spread = best_bid - best_ask
                    spread_pct = (spread / best_ask) * 100
                    
                    # Check if spread is profitable after fees
                    if spread_pct > self.min_profit_pct:
                        base, quote = symbol.split('/')
                        opportunity = ArbitrageOpportunity(
                            base_asset=base,
                            quote_asset=quote,
                            buy_exchange=sell_exchange,
                            sell_exchange=buy_exchange,
                            buy_price=best_ask,
                            sell_price=best_bid,
                            spread=spread,
                            spread_pct=spread_pct,
                            timestamp=time.time()
                        )
                        
                        # Cache the opportunity
                        cache_key = f"arbitrage:{base}_{quote}:{int(time.time())}"
                        self.redis_client.setex(
                            cache_key,
                            300,  # 5 minute TTL
                            json.dumps(opportunity.to_dict())
                        )
                        
                        opportunities.append(opportunity)
                        
                        logger.info(f"Arbitrage opportunity found: {opportunity}")
        
        return opportunities
    
    async def monitor_opportunities(self, interval: int = 10):
        """Continuously monitor for arbitrage opportunities"""
        logger.info("Starting arbitrage monitor...")
        
        try:
            while True:
                try:
                    await self.find_arbitrage_opportunities()
                    await asyncio.sleep(interval)
                except Exception as e:
                    logger.error(f"Error in monitor_opportunities: {str(e)}")
                    await asyncio.sleep(5)  # Wait before retrying
        except asyncio.CancelledError:
            logger.info("Arbitrage monitor stopped")
        finally:
            await self.close()

# Example usage
async def main():
    detector = CrossChainArbitrageDetector()
    await detector.initialize()
    
    try:
        # Run for 1 minute as an example
        await asyncio.wait_for(detector.monitor_opportunities(), timeout=60)
    except asyncio.TimeoutError:
        print("Monitoring completed")
    finally:
        await detector.close()

if __name__ == "__main__":
    asyncio.run(main())
