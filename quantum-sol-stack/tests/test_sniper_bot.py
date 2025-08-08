import pytest
from web3 import Web3
from web3.exceptions import ContractLogicError
from pathlib import Path
import json
import time

# Import the SniperBot components
from sniper_bot.core.sniper_bot import SniperBot
from sniper_bot.core.web3_manager import Web3Manager
from sniper_bot.core.config import Config

# Test token details for Goerli (using existing test tokens)
GOERLI_TEST_TOKENS = {
    'DAI': '0x11fE444095b2b4fB4E3E9Ea00F555417Ae56d3Dd',
    'USDC': '0xD87Ba7A50B2E7E660f678A895E4B72E7CB4CCd9C',
    'UNI': '0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984',
}

class TestSniperBot:
    """Test suite for the SniperBot on Goerli testnet."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self, web3, config, account):
        """Setup test environment for each test method."""
        self.web3 = web3
        self.config = config
        self.account = account
        
        # Initialize Web3Manager with test config
        self.web3_manager = Web3Manager()
        
        # Initialize SniperBot with test config
        self.bot = SniperBot()
        
        # Set a test token for trading
        self.test_token = GOERLI_TEST_TOKENS['DAI']  # Using DAI on Goerli for testing
    
    def test_web3_connection(self):
        """Test Web3 connection to Goerli testnet."""
        assert self.web3.is_connected(), "Failed to connect to the Ethereum network"
        
        # Check chain ID (5 is Goerli)
        assert self.web3.eth.chain_id == 5, "Not connected to Goerli testnet"
    
    def test_account_balance(self, ensure_test_eth):
        """Test that the test account has sufficient ETH balance."""
        balance_eth = self.web3.from_wei(ensure_test_eth, 'ether')
        print(f"Test account balance: {balance_eth} ETH")
        assert balance_eth > 0, "Test account has no ETH"
    
    def test_token_balance(self):
        """Test token balance check functionality."""
        # Get token contract
        token_abi = json.loads((Path(__file__).parent.parent / 'sniper_bot' / 'abis' / 'IERC20.json').read_text())
        token_contract = self.web3.eth.contract(address=self.test_token, abi=token_abi)
        
        # Get token balance
        balance = token_contract.functions.balanceOf(self.account.address).call()
        token_decimals = token_contract.functions.decimals().call()
        token_balance = balance / (10 ** token_decimals)
        
        print(f"Token balance: {token_balance} DAI")
        assert balance >= 0, "Failed to get token balance"
    
    @pytest.mark.skip(reason="This is a long-running test that monitors for new pairs")
    def test_monitor_new_pairs(self):
        """Test monitoring for new pairs (this test is skipped by default)."""
        print("Monitoring for new pairs for 30 seconds...")
        
        # Start monitoring in a separate thread
        import threading
        from queue import Queue
        
        event_queue = Queue()
        
        def monitor():
            try:
                event_filter = self.web3_manager.factory.events.PairCreated.create_filter(
                    fromBlock='latest'
                )
                
                while True:
                    for event in event_filter.get_new_entries():
                        event_queue.put(event)
                    time.sleep(2)
            except Exception as e:
                event_queue.put(e)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        
        # Wait for events or timeout
        try:
            for _ in range(15):  # Check for 30 seconds (15 * 2s)
                if not event_queue.empty():
                    event = event_queue.get()
                    if isinstance(event, Exception):
                        raise event
                    
                    print(f"New pair created: {event}")
                    assert 'pair' in event.args, "Pair address not found in event"
                    break
                time.sleep(2)
            else:
                print("No new pairs detected in the last 30 seconds")
                pytest.skip("No new pairs detected in the monitoring period")
        finally:
            monitor_thread.join(timeout=1)
    
    def test_get_token_price(self):
        """Test getting token price from Uniswap V2."""
        # Test with WETH-DAI pair (should always have liquidity on testnet)
        weth = self.config['WETH_ADDRESS']
        
        # Get the router contract
        router_abi = json.loads((Path(__file__).parent.parent / 'sniper_bot' / 'abis' / 'IUniswapV2Router02.json').read_text())
        router = self.web3.eth.contract(address=self.config['ROUTER_ADDRESS'], abi=router_abi)
        
        # Get price of 1 DAI in WETH
        path = [self.test_token, weth]
        amounts = router.functions.getAmountsOut(
            int(1 * (10 ** 18)),  # 1 DAI (18 decimals)
            path
        ).call()
        
        price_in_eth = amounts[1] / (10 ** 18)  # Convert from wei to ETH
        print(f"1 DAI = {price_in_eth} WETH")
        
        assert price_in_eth > 0, "Invalid token price"
    
    @pytest.mark.skip(reason="This test executes actual trades and requires ETH")
    def test_buy_sell_flow(self, ensure_test_eth):
        """Test the complete buy/sell flow (skipped by default as it requires ETH)."""
        # This test is a placeholder for actual buy/sell tests
        # It's skipped by default as it requires ETH and affects the blockchain state
        
        # Example of how a test trade would work:
        # 1. Get initial balances
        # 2. Execute buy
        # 3. Verify balances updated
        # 4. Execute sell
        # 5. Verify final balances
        
        pytest.skip("Skipping actual trade test to avoid spending ETH")

def test_sniper_bot_integration(web3, config, account):
    """Test the full SniperBot integration with a test token."""
    # This is a placeholder for a more comprehensive integration test
    # that would test the full bot flow with a test token
    
    # For now, just verify we can initialize the bot
    bot = SniperBot()
    assert bot is not None, "Failed to initialize SniperBot"
    
    # Verify the bot's Web3 connection is working
    assert bot.w3.is_connected(), "Bot's Web3 connection is not active"
    
    # Verify the bot's configuration
    assert hasattr(bot, 'config'), "Bot is missing config"
    assert hasattr(bot, 'web3_manager'), "Bot is missing web3_manager"
    
    print("SniperBot integration test passed")
