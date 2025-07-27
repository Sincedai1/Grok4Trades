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
