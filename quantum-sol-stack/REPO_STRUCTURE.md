# QuantumSol Repository Structure

This document outlines the complete structure of the QuantumSol trading system repository.

## Root Directory

```
.
├── .env.example               # Example environment variables
├── .gitignore                 # Git ignore rules
├── README.md                  # Project documentation
├── REPO_STRUCTURE.md          # This file
├── docker-compose.yml         # Docker Compose configuration
├── requirements-dev.txt       # Development dependencies
└── setup.py                   # Python package configuration
```

## Agent System

```
agent/
├── Dockerfile                # Agent service container definition
├── requirements.txt          # Python dependencies
├── agent_runner.py           # Main agent orchestration
├── guardrails_utils.py       # Risk management utilities
├── kill_switch.py            # Emergency stop functionality
├── telegram.py               # Telegram notification integration
└── utils/                    # Utility modules
    ├── __init__.py
    ├── config.py             # Configuration management
    ├── logger.py             # Logging configuration
    └── helpers.py            # Helper functions
```

## User Interface

```
ui/
├── Dockerfile                # Streamlit UI container definition
├── requirements.txt          # Python dependencies
├── app.py                   # Main Streamlit application
├── services/                # Backend services
│   ├── __init__.py
│   └── agent_service.py     # Agent communication service
└── components/              # UI components
    ├── __init__.py
    ├── agent_controls.py    # Agent control panel
    ├── config_editor.py     # Configuration editor
    └── model_generator.py   # Model generation interface
```

## Monitoring

```
monitoring/
├── grafana/
│   ├── provisioning/
│   │   ├── dashboards/
│   │   │   └── dashboard.yml
│   │   └── datasources/
│   │       └── datasource.yml
│   └── quantumsol-dashboard.json
└── prometheus/
    ├── alerts.yml           # Alerting rules
    └── prometheus.yml       # Prometheus configuration
```

## Trading Bots

```
freqtrade/
└── user_data/
    ├── config.json          # Main configuration
    └── strategies/          # Trading strategies
        ├── SmaCrossSol.py
        ├── RsiBounceSol.py
        ├── LstmPredictor.py
        ├── Momentum.py
        ├── VolatilityBreakout.py
        └── MemeCoinPumpDetector.py

hummingbot/
└── conf/
    ├── conf_global.yml      # Global configuration
    ├── conf_connectors.yml  # Exchange connectors
    └── strategies/          # Strategy configurations
        └── pure_market_making_1.yml
```

## Workflows

```
flows/
├── nightly_flow.py          # Nightly maintenance workflow
└── deploy_nightly_flow.py   # Workflow deployment script
```

## Documentation

```
docs/
├── ARCHITECTURE.md          # System architecture
├── API_REFERENCE.md         # API documentation
└── DEVELOPMENT.md           # Development guidelines
```

## Testing

```
tests/
├── __init__.py
├── test_agent.py
├── test_guardrails.py
└── test_integration.py
```

## Operations

```
ops/
├── cron_examples.txt        # Example cron jobs
└── prefect_profile.toml     # Prefect configuration
```

## Kubernetes (Optional)

```
k8s/
├── agent-deployment.yaml
├── ui-deployment.yaml
├── grafana-deployment.yaml
├── prometheus-deployment.yaml
└── ingress.yaml
```

## Getting Started

1. Copy `.env.example` to `.env` and update the values
2. Run `docker-compose up -d` to start all services
3. Access the UI at http://localhost:8501
4. Monitor the system at http://localhost:3000 (Grafana)

## License

Proprietary - All rights reserved.
