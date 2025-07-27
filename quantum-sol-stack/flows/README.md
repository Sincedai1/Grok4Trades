# QuantumSol Flows

This directory contains Prefect flows for the QuantumSol trading system, including the nightly maintenance and optimization flow.

## Nightly Flow

The nightly flow performs the following tasks during off-market hours:

1. **Daily Report Generation**
   - Compiles performance metrics
   - Generates risk assessment
   - Provides trading recommendations

2. **Strategy Optimization**
   - Optimizes trading strategy parameters
   - Backtests strategies with recent market data
   - Updates strategy configurations

3. **Market Data Update**
   - Fetches and stores latest market data
   - Updates historical price data
   - Ensures data consistency

4. **Market Structure Analysis**
   - Analyzes current market conditions
   - Identifies key support/resistance levels
   - Scans for new meme coins on Pump.fun

5. **System Maintenance**
   - Cleans up temporary files
   - Optimizes databases
   - Performs system health checks

## Deployment

### Prerequisites

- Python 3.8+
- Prefect 2.0+
- All dependencies from `requirements.txt`

### Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up Prefect (if not already done):
   ```bash
   prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
   prefect orion start
   ```

### Running the Flow

1. **Local Development**
   ```bash
   python -m nightly_flow
   ```

2. **Create Deployment**
   ```bash
   python deploy_nightly_flow.py
   ```

3. **Start Agent** (in a separate terminal)
   ```bash
   prefect agent start -q production
   ```

## Configuration

Environment variables can be set in a `.env` file:

```
PREFECT_API_URL=http://127.0.0.1:4200/api
QUANTUMSOL_ENV=production
# Add other environment variables as needed
```

## Monitoring

Monitor flow runs in the Prefect UI at http://127.0.0.1:4200

## Testing

To run the flow in test mode with reduced data:

```bash
python -m nightly_flow --test-mode
```

## License

Proprietary - All rights reserved.
