# QuantumSol Trading System

QuantumSol is a high-frequency crypto trading system built with Python, leveraging AI agents for strategy execution, monitoring, and risk management. The system is designed to trade on both CEX (via Freqtrade) and DEX (via Hummingbot) platforms, with a focus on the Solana ecosystem.

## Features

- **AI-Powered Trading Agents**:
  - Strategy Agent: Implements and optimizes trading strategies
  - Monitoring Agent: Enforces risk management and profit targets
  - Sentiment Agent: Analyzes market sentiment and social signals
  - Orchestrator: Coordinates all agents and manages the trading workflow

- **Multi-Exchange Support**:
  - CEX: Kraken, Binance, and other major exchanges via Freqtrade
  - DEX: Raydium and other Solana DEXs via Hummingbot

- **Advanced Monitoring**:
  - Real-time metrics with Prometheus
  - Beautiful dashboards with Grafana
  - Custom alerts and notifications

- **Risk Management**:
  - Daily profit/loss limits
  - Maximum drawdown protection
  - Emergency stop functionality

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│  Streamlit UI   │◄──►│  Agent System   │◄──►│  Trading Bots   │
│                 │    │                 │    │  (Freqtrade/    │
└─────────────────┘    └─────────────────┘    │   Hummingbot)   │
       ▲                     ▲                └────────┬────────┘
       │                     │                         │
       ▼                     ▼                         ▼
┌─────────────────┐  ┌───────────────┐    ┌─────────────────────┐
│                 │  │               │    │                     │
│   Grafana       │  │   Prefect     │    │   Exchanges        │
│   Dashboards    │  │   Workflows   │    │   (CEX & DEX)      │
│                 │  │               │    │                     │
└─────────────────┘  └───────────────┘    └─────────────────────┘
```

## Prerequisites

- Docker and Docker Compose
- Python 3.10+
- Node.js (for UI development)
- Solana CLI tools (for DEX operations)

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/quantum-sol-stack.git
   cd quantum-sol-stack
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Build and start the services**
   ```bash
   docker-compose up -d --build
   ```

4. **Access the applications**
   - Streamlit UI: http://localhost:8501
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090
   - Prefect UI: http://localhost:4200

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```ini
# API Keys
EXCHANGE_API_KEY=your_exchange_api_key
EXCHANGE_SECRET=your_exchange_secret
OPENAI_API_KEY=your_openai_api_key

# Agent Configuration
AGENT_MODEL_ORCHESTRATOR=gpt-4o
AGENT_MODEL_STRATEGY=gpt-4o
AGENT_MODEL_MONITORING=gpt-4o
AGENT_MODEL_SENTIMENT=gpt-4o

# Trading Parameters
DAILY_PROFIT_TARGET=5000
MAX_DAILY_LOSS=2000
MAX_DRAWDOWN_PCT=10

# Monitoring
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_URL=http://grafana:3000
```

### Freqtrade Configuration

Edit `freqtrade/user_data/config.json` to configure your trading pairs, strategies, and exchange settings.

### Hummingbot Configuration

Edit `hummingbot/conf/conf_global.yml` and other configuration files in the `hummingbot/conf` directory.

## Usage

### Starting the System

```bash
docker-compose up -d
```

### Accessing the Web UI

Open your browser to http://localhost:8501 to access the Streamlit dashboard.

### Monitoring

Access Grafana at http://localhost:3000 to view trading metrics and system performance.

### Managing Trades

Use the web interface to:
- Start/stop trading
- Monitor positions
- View performance metrics
- Configure trading parameters

## Agent System

The agent system consists of four main components:

1. **Strategy Agent**: Develops and optimizes trading strategies
2. **Monitoring Agent**: Enforces risk management rules
3. **Sentiment Agent**: Analyzes market sentiment
4. **Orchestrator**: Coordinates all agents and manages the trading workflow

## Development

### Setting Up Development Environment

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

### Running Tests

```bash
pytest tests/
```

### Code Style

We use `black` for code formatting and `flake8` for linting:

```bash
black .
flake8
```

## Deployment

### Production Deployment

For production deployment, it's recommended to:

1. Set up HTTPS with a reverse proxy (Nginx, Traefik, etc.)
2. Configure proper logging and monitoring
3. Set up alerts for critical events
4. Use a proper secrets management solution

### Kubernetes

For Kubernetes deployment, see the `k8s/` directory for example manifests.

## Security

- Never commit API keys or secrets to version control
- Use strong, unique passwords for all services
- Regularly rotate API keys and credentials
- Monitor system access and activity

## License

This project is proprietary and confidential. All rights reserved.

## Support

For support, please contact [support@quantumsol.ai](mailto:support@quantumsol.ai)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Roadmap

- [ ] Implement additional trading strategies
- [ ] Add support for more exchanges
- [ ] Enhance sentiment analysis with NLP
- [ ] Improve backtesting capabilities
- [ ] Add machine learning-based price prediction

# Uniswap V2 Sniper Bot

A high-frequency trading bot that monitors Uniswap V2 for new token listings and executes trades based on customizable parameters.

## Features

- Real-time monitoring of new token pairs on Uniswap V2
- Automated buying of new tokens that meet liquidity requirements
- Automated selling based on profit targets and stop-loss conditions
- Configurable trading parameters (slippage, gas price, etc.)
- Comprehensive logging and error handling
- Lightweight Streamlit dashboard for monitoring

## Prerequisites

- Python 3.8+
- Ethereum node or Web3 provider (Infura, Alchemy, etc.)
- Ethereum wallet with ETH for gas fees
- Environment variables set up (see Configuration)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sniper-bot.git
   cd sniper-bot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Configuration

1. Copy the example environment file:
   ```bash
   cp sniper_bot/config/.env.example .env
   ```

2. Edit the `.env` file with your configuration:
   ```ini
   # Ethereum Node Configuration
   RPC_URL_MAINNET=your_mainnet_rpc_url
   RPC_URL_GOERLI=your_goerli_rpc_url
   
   # Wallet Configuration
   PRIVATE_KEY=your_private_key
   WALLET_ADDRESS=your_wallet_address
   
   # Trading Parameters
   TARGET_PROFIT_PERCENT=5.0
   STOP_LOSS_PERCENT=2.0
   MAX_SLIPPAGE=1.0
   MAX_GAS_PRICE_GWEI=100
   MIN_LIQUIDITY_ETH=5.0
   
   # Contract Addresses (Uniswap V2)
   FACTORY_ADDRESS=0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f
   ROUTER_ADDRESS=0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D
   WETH_ADDRESS=0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2
   
   # Logging
   LOG_LEVEL=INFO
   LOG_FILE=logs/sniper_bot.log
   ```

## Testing

### Prerequisites for Testing

1. **Testnet ETH**: You'll need testnet ETH to run integration tests. Get some from:
   - [Goerli Faucet](https://goerlifaucet.com/)
   - [Sepolia Faucet](https://sepoliafaucet.com/)

2. **Environment Variables**: Create a `.env` file based on `.env.example` and fill in your details:
   ```bash
   cp .env.example .env
   # Edit .env with your private key and RPC URL
   ```

### Running Tests

#### 1. Run All Tests
```bash
# Run all tests on Goerli testnet (default)
python run_tests.py

# Run with verbose output
python run_tests.py -v
```

#### 2. Run Specific Test Types
```bash
# Run only unit tests
python run_tests.py --test-type unit

# Run only integration tests (requires testnet ETH)
python run_tests.py --test-type integration

# Run tests on a specific network
python run_tests.py --network sepolia
```

#### 3. Run Tests Directly with pytest
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_sniper_bot.py -v

# Run tests with coverage report
pytest --cov=sniper_bot tests/
```

### Test Coverage

To generate a coverage report:
```bash
pytest --cov=sniper_bot --cov-report=html tests/
# Open htmlcov/index.html in your browser
```

### Local Testing with eth-tester

For testing without using a live network, you can use eth-tester:
```bash
# Install eth-tester
pip install eth-tester py-evm

# Run tests with local Ethereum tester
USE_ETHEREUM_TESTER=true pytest tests/ -v
```

### Writing Tests

1. **Unit Tests**: Test individual components in isolation
   - Place in `tests/unit/`
   - Use `@pytest.mark.unit` decorator

2. **Integration Tests**: Test interactions with the blockchain
   - Place in `tests/integration/`
   - Use `@pytest.mark.integration` decorator
   - May require testnet ETH

3. **Fixtures**: Common test utilities are in `tests/conftest.py`

### Test Tokens on Goerli

The test suite uses these tokens on Goerli by default:
- DAI: `0x11fE444095b2b4fB4E3E9Ea00F555417Ae56d3Dd`
- USDC: `0xD87Ba7A50B2E7E660f678A895E4B72E7CB4CCd9C`
- UNI: `0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984`

### Troubleshooting

1. **Insufficient Funds**:
   ```
   Error: Insufficient test ETH. Need at least 0.01 ETH
   ```
   - Get test ETH from a faucet

2. **Connection Issues**:
   ```
   Failed to connect to the Ethereum network
   ```
   - Check your RPC URL
   - Ensure your internet connection is stable

3. **Transaction Failures**:
   - Check gas prices
   - Verify contract addresses for the correct network
   - Ensure your account has enough ETH for gas

## Usage

### Running the Bot

```bash
# Run in development mode
python -m sniper_bot

# Or after installation
sniper-bot
```

### Running with Docker

```bash
# Build the image
docker build -t sniper-bot .

# Run the container
docker run --env-file .env sniper-bot
```

### Monitoring Dashboard

Start the Streamlit dashboard:

```bash
streamlit run sniper_bot/ui/dashboard.py
```

## Trading Strategy

The bot follows these steps:

1. Monitors for new token pairs on Uniswap V2
2. When a new WETH pair is detected, it checks the liquidity
3. If liquidity is above the minimum threshold, it buys the token
4. It then monitors the token's price
5. Sells the token when either:
   - The price reaches the target profit percentage
   - The price drops to the stop-loss percentage

## Security Considerations

- **Never share your private key**
- Use a dedicated wallet with only the funds you're willing to risk
- Test with small amounts first
- Consider using a hardware wallet for additional security
- Be aware of the risks associated with automated trading

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational purposes only. Use at your own risk. The authors are not responsible for any financial losses incurred while using this software. Always do your own research and understand the risks involved in cryptocurrency trading.
