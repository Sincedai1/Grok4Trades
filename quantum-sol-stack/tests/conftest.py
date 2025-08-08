import os
import pytest
from dotenv import load_dotenv
from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider

# Load environment variables from .env file
load_dotenv()

# Test configuration
TEST_CONFIG = {
    'GOERLI_RPC_URL': os.getenv('GOERLI_RPC_URL', 'https://goerli.infura.io/v3/YOUR_PROJECT_ID'),
    'PRIVATE_KEY': os.getenv('PRIVATE_KEY', ''),
    'WALLET_ADDRESS': os.getenv('WALLET_ADDRESS', ''),
    'FACTORY_ADDRESS': '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f',  # Uniswap V2 Factory
    'ROUTER_ADDRESS': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',  # Uniswap V2 Router 02
    'WETH_ADDRESS': '0xB4FBF271143F4FBf7B91A5ded31805e42b2208d6',  # WETH on Goerli
    'TARGET_PROFIT_PERCENT': 5.0,
    'STOP_LOSS_PERCENT': 2.0,
    'MAX_SLIPPAGE': 1.0,
    'MAX_GAS_PRICE_GWEI': 100,
    'MIN_LIQUIDITY_ETH': 0.1,  # Lower for testnet
}

@pytest.fixture(scope='module')
def web3():
    """Fixture providing a Web3 instance connected to the Goerli testnet."""
    if os.getenv('USE_ETHEREUM_TESTER', 'false').lower() == 'true':
        # Use eth-tester for local testing
        return Web3(EthereumTesterProvider())
    else:
        # Connect to Goerli testnet
        return Web3(Web3.HTTPProvider(TEST_CONFIG['GOERLI_RPC_URL']))

@pytest.fixture(scope='module')
def config():
    """Fixture providing test configuration."""
    return TEST_CONFIG

@pytest.fixture(scope='module')
def account(web3):
    """Fixture providing the test account."""
    if not TEST_CONFIG['PRIVATE_KEY']:
        pytest.skip("PRIVATE_KEY not set in environment variables")
    
    account = web3.eth.account.from_key(TEST_CONFIG['PRIVATE_KEY'])
    if TEST_CONFIG['WALLET_ADDRESS'] and account.address.lower() != TEST_CONFIG['WALLET_ADDRESS'].lower():
        pytest.fail("PRIVATE_KEY does not match WALLET_ADDRESS")
    
    return account

@pytest.fixture(scope='module')
def ensure_test_eth(web3, account):
    """Check if test account has sufficient ETH for testing."""
    balance = web3.eth.get_balance(account.address)
    min_balance = web3.to_wei(0.01, 'ether')  # Minimum 0.01 ETH for testing
    
    if balance < min_balance:
        pytest.skip(
            f"Insufficient test ETH. Need at least {web3.from_wei(min_balance, 'ether')} ETH, "
            f"but only have {web3.from_wei(balance, 'ether')} ETH. "
            "Get test ETH from a Goerli faucet."
        )
    
    return balance
